#!/usr/bin/env python3
# ruff: noqa: E501
"""
DiffusionGemma / Gemma-4 vision tower -- INDEPENDENT pure-numpy reference implementation.

This file is the NUMERICAL GROUND TRUTH that TensorSharp's C# vision encoder is diffed
against.  It is a DIRECT transcription of the HuggingFace reference sources:

    transformers/models/gemma4/modeling_gemma4.py
        Gemma4RMSNorm, Gemma4VisionPatchEmbedder, Gemma4VisionPooler, Gemma4VisionMLP,
        Gemma4VisionRotaryEmbedding, apply_multidimensional_rope, Gemma4VisionAttention,
        Gemma4VisionEncoderLayer, Gemma4VisionModel, Gemma4MultimodalEmbedder
    transformers/models/gemma4/image_processing_gemma4.py
        get_aspect_ratio_preserving_size, convert_image_to_patches
    transformers/models/gemma4/configuration_gemma4.py
        Gemma4VisionConfig (default_rope_type="axial", default_theta=100.0)

It deliberately does NOT follow TensorSharp's C# encoder or llama.cpp's
tools/mtmd/models/gemma4v.cpp, because both are known to contain at least one bug.
Where a divergence exists it is called out in a "DIVERGENCE:" comment.

Correctness is the only goal; speed does not matter.

Dependencies: numpy + PIL only (no torch, no safetensors, no gguf).

Usage:
    python3 eng/diffusiongemma-vision-oracle.py \
        --src   ~/work/models/diffusiongemma-vision/model-00011-of-00011.safetensors \
        --image ~/work/models/testmedia/image.png \
        --max-soft-tokens 280 --act gelu_tanh \
        --dump-dir /tmp/oracle_gelu_tanh --out /tmp/oracle_gelu_tanh/final.npy
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# Minimal safetensors reader (BF16 / F16 / F32 -> float32 ndarray)
# ---------------------------------------------------------------------------
class ST:
    """Read-only safetensors accessor.  BF16 is widened to float32 exactly (bit shift)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
            self.base = 8 + n
        self.meta = self.hdr.pop("__metadata__", None)

    def names(self):
        return list(self.hdr.keys())

    def info(self, k):
        return self.hdr[k]

    def get(self, k):
        e = self.hdr[k]
        s, t = e["data_offsets"]
        dt = e["dtype"]
        shape = e["shape"]
        with open(self.path, "rb") as f:
            f.seek(self.base + s)
            raw = f.read(t - s)
        if dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            a = u.view(np.float32)
        elif dt == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        elif dt == "F16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError("unsupported dtype " + dt)
        return np.ascontiguousarray(a.reshape(shape), dtype=np.float32)


# ---------------------------------------------------------------------------
# Config (Gemma4VisionConfig as materialized by the DiffusionGemma checkpoint)
# ---------------------------------------------------------------------------
class VisionConfig:
    hidden_size = 1152
    intermediate_size = 4304
    num_hidden_layers = 27
    num_attention_heads = 16
    num_key_value_heads = 16  # == num_attention_heads, so repeat_kv is the identity
    head_dim = 72
    patch_size = 16
    pooling_kernel_size = 3
    position_embedding_size = 10240
    rms_norm_eps = 1e-6
    hidden_activation = "gelu_pytorch_tanh"
    standardize = True
    use_clipped_linears = False
    # DIVERGENCE (config file vs code): the checkpoint's vision_config.json says
    #   "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}
    # but Gemma4VisionRotaryEmbedding.__init__ raises unless rope_type == "axial".
    # rope_utils.py:809-810 resolves this: a config whose class sets
    # `default_rope_type != "default"` has a stored "default" rewritten to that value, and
    # Gemma4VisionConfig.default_rope_type == "axial" (g4_configuration_gemma4.py:271).
    # So the effective rope is AXIAL 2-D with theta = 100.0.
    rope_theta = 100.0
    text_hidden_size = 2816  # embedding_projection output


# ---------------------------------------------------------------------------
# Stage 1: preprocessing  (g4_image_processing_gemma4.py)
# ---------------------------------------------------------------------------
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    """Verbatim transcription of HF get_aspect_ratio_preserving_size.

    NOTE: this is *anisotropic* -- each side is independently floored to a multiple of
    `pooling_kernel_size * patch_size` (= 48), so the aspect ratio is only approximately
    preserved.  There is NO letterbox and NO pixel padding.
    """
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
            "`pooling_kernel_size * patch_size`=%d." % side_mult
        )

    max_side_length = (max_patches // pooling_kernel_size ** 2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            "Resizing [%dx%d] to [%dx%d] but this exceeds %d patches with patch_size %d"
            % (height, width, target_height, target_width, max_patches, patch_size)
        )

    return target_height, target_width


def convert_image_to_patches(image_chw, patch_size):
    """Verbatim: reshape(C,nH,p,nW,p).permute(1,3,2,4,0).reshape(nH*nW,-1).

    The in-patch element order is therefore [ky][kx][c]  -- CHANNEL FASTEST.
    (llama.cpp implements the same projection as a conv2d whose OIHW weight is [c][ky][kx],
    so a GGUF converter must permute input_proj: w.reshape(1152,16,16,3).transpose(0,3,1,2).)
    """
    num_channels, image_height, image_width = image_chw.shape
    nH = image_height // patch_size
    nW = image_width // patch_size
    patched = image_chw.reshape(num_channels, nH, patch_size, nW, patch_size)
    patched = patched.transpose(1, 3, 2, 4, 0)
    patched = patched.reshape(nH * nW, -1)
    return np.ascontiguousarray(patched)


def _to_rgb_chw(pil_image, alpha="white"):
    """do_convert_rgb.

    HF's image_transforms.convert_to_rgb alpha-composites an RGBA image over a WHITE
    background (a plain .convert("RGB") would leave garbage under transparent pixels).
    `alpha="drop"` reproduces the naive .convert("RGB") for comparison.
    """
    if pil_image.mode == "RGB":
        rgb = pil_image
    elif alpha == "white":
        src = pil_image.convert("RGBA")
        bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, src).convert("RGB")
    else:
        rgb = pil_image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)  # HWC
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW uint8


def _resize_chw(img_chw_u8, target_h, target_w, mode="float"):
    """BICUBIC resize.

    HF's default image processor for this checkpoint is `Gemma4ImageProcessor`
    (TorchvisionBackend); it calls tvF.resize(..., InterpolationMode.BICUBIC, antialias=True)
    on a *float* tensor still in [0,255], and only rescales by 1/255 afterwards.

    PIL's bicubic is the same separable, support-scaled (antialiased) filter, so the two agree
    to within float rounding -- but this is a KNOWN, BOUNDED source of divergence between this
    oracle and a real torchvision run, and no exact agreement should be expected.

    mode="float"  : resize each channel as a PIL 'F' (float32) image.  No [0,255] clamping and
                    no uint8 requantization, which is the closest match to torchvision.
                    Bicubic overshoot below 0 / above 255 is preserved, exactly as torchvision
                    does (tvF.resize does not clamp float input).
    mode="uint8"  : resize the uint8 image directly.  This is what the sibling
                    `Gemma4ImageProcessorPil` (PilBackend) effectively does, since
                    image_transforms.resize round-trips through to_pil_image().  Values are
                    rounded and clamped to [0,255].
    """
    c, h, w = img_chw_u8.shape
    if (h, w) == (target_h, target_w):
        return img_chw_u8.astype(np.float32)
    if mode == "uint8":
        hwc = np.ascontiguousarray(img_chw_u8.transpose(1, 2, 0))
        im = Image.fromarray(hwc, mode="RGB")
        im = im.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        out = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)
        return np.ascontiguousarray(out)
    out = np.empty((c, target_h, target_w), dtype=np.float32)
    for ci in range(c):
        # float32 array -> PIL 'F' mode is inferred; passing mode= is deprecated in Pillow 12.
        im = Image.fromarray(img_chw_u8[ci].astype(np.float32))
        im = im.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        out[ci] = np.asarray(im, dtype=np.float32)
    return out


def preprocess(pil_image, max_soft_tokens=280, patch_size=16, pooling_kernel_size=3,
               resize_mode="float", alpha="white"):
    """Stage 1.

    Returns (patches [N,768] float32 in [0,1], pos_xy [N,2] int32, (n_rows, n_cols), n_soft).

    DIVERGENCE (intentional, numerically inert): HF pads every image up to `max_patches`
    rows with zero patches at position (-1,-1), and the model zeroes their embeddings, masks
    them out of attention, and strips their pooled rows afterwards.  A single unpadded image
    produces bit-identical outputs for the real rows, so the oracle omits the padding.
    """
    if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
        raise ValueError("max_soft_tokens must be one of %s" % (_SUPPORTED_SOFT_TOKENS,))
    max_patches = max_soft_tokens * pooling_kernel_size ** 2

    img = _to_rgb_chw(pil_image, alpha=alpha)
    src_h, src_w = img.shape[1], img.shape[2]
    target_h, target_w = get_aspect_ratio_preserving_size(
        height=src_h, width=src_w, patch_size=patch_size,
        max_patches=max_patches, pooling_kernel_size=pooling_kernel_size,
    )
    resized = _resize_chw(img, target_h, target_w, mode=resize_mode)

    # do_rescale (1/255); do_normalize is FALSE -- image_mean=0, image_std=1, so there is
    # deliberately NO mean/std normalization here.
    resized = resized * np.float32(1.0 / 255.0)

    patches = convert_image_to_patches(resized, patch_size).astype(np.float32)
    n_rows = target_h // patch_size
    n_cols = target_w // patch_size
    n_soft = patches.shape[0] // pooling_kernel_size ** 2

    # position ids: meshgrid(arange(W), arange(H), indexing="xy") -> (H, W) grids,
    # stacked as (x, y) and flattened row-major, matching the patch order above.
    gx, gy = np.meshgrid(np.arange(n_cols), np.arange(n_rows), indexing="xy")
    pos_xy = np.stack([gx, gy], axis=-1).reshape(patches.shape[0], 2).astype(np.int32)

    assert patches.shape == (n_rows * n_cols, patch_size * patch_size * 3)
    assert n_rows % pooling_kernel_size == 0 and n_cols % pooling_kernel_size == 0
    assert (n_rows // pooling_kernel_size) * (n_cols // pooling_kernel_size) == n_soft
    return patches, pos_xy, (n_rows, n_cols), n_soft, (target_h, target_w)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def rms_norm(x, weight=None, eps=1e-6):
    """Gemma4RMSNorm.

    VERIFIED against the HF source (g4_modeling_gemma4.py:197-215): the class has a
    `with_scale` flag and its forward is

        normed = x * (mean(x**2) + eps) ** -0.5
        if with_scale: normed = normed * weight

    i.e. norm(x) * w.  There is NO (1 + w) scale shift -- that is Gemma-3's RMSNorm, and
    Gemma-4's vision tower does NOT use it.  `with_scale=False` (V-norm in attention and the
    multimodal embedder's pre-projection norm) simply drops the multiply.

    Note the eps is INSIDE the sqrt (added to the mean square), matching HF.
    """
    x = x.astype(np.float32, copy=False)
    ms = np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)
    out = x * np.power(ms, -0.5, dtype=np.float32)
    if weight is not None:
        out = out * weight.astype(np.float32)
    return out


def linear(x, w):
    """nn.Linear(bias=False) with HF weight layout [out, in]."""
    return x @ w.T


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_tanh(x):
    """ACT2FN["gelu_pytorch_tanh"] == F.gelu(x, approximate="tanh")."""
    x = x.astype(np.float32, copy=False)
    inner = np.float32(_GELU_C) * (x + np.float32(0.044715) * x * x * x)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))


def quick_gelu(x):
    """ACT2FN["quick_gelu"] == x * sigmoid(1.702 * x).

    NOT what this checkpoint declares (vision_config.hidden_activation ==
    "gelu_pytorch_tanh").  Selectable only so the llama.cpp / TensorSharp QuickGELU bug can be
    quantified empirically.
    """
    x = x.astype(np.float32, copy=False)
    z = np.float32(1.702) * x
    # stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return x * out


ACT = {"gelu_tanh": gelu_tanh, "quick_gelu": quick_gelu}


# ---------------------------------------------------------------------------
# Axial 2-D RoPE  (Gemma4VisionRotaryEmbedding + apply_multidimensional_rope)
# ---------------------------------------------------------------------------
def build_rope(pos_xy, head_dim, theta):
    """cos/sin of shape [N, head_dim].

    compute_axial_rope_parameters:
        spatial_dim = head_dim // 2                       (= 36)
        inv_freq    = 1 / theta ** (arange(0, spatial_dim, 2) / spatial_dim)   -> 18 values
    forward:
        freqs = position_ids[..., None] @ inv_freq[None]  -> [N, 2, 18]
    recomposition_frequencies:
        cat([f[:, 0], f[:, 0], f[:, 1], f[:, 1]], -1)     -> [N, 72]

    Axis 0 of position_ids is x (column) and axis 1 is y (row); the HF local names
    freq_h/freq_w are a misnomer but numerically irrelevant.

    The trig is evaluated in float64 and stored as float32.  HF evaluates in float32; the
    difference is < 1e-6 and this direction is strictly the more accurate one for an oracle.
    """
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, spatial_dim, 2, dtype=np.float64) / spatial_dim))
    freqs = pos_xy.astype(np.float64)[:, :, None] * inv_freq[None, None, :]  # [N,2,18]
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.concatenate([cos[:, 0], cos[:, 0], cos[:, 1], cos[:, 1]], axis=-1)
    sin = np.concatenate([sin[:, 0], sin[:, 0], sin[:, 1], sin[:, 1]], axis=-1)
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_axial_rope(x, cos, sin):
    """apply_multidimensional_rope with ndim == 2.

    x: [N, heads, head_dim].  num_rotated_channels_per_dim = 2 * (72 // 4) = 36, so the head
    dim splits into TWO 36-wide halves: the first is rotated by the x position, the second by
    the y position.  Inside each half rotate_half pairs channel i with channel i+18
    (NEOX-style / split-half pairing), NOT adjacent pairs.
    """
    n, heads, d = x.shape
    half = d // 2
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = np.empty_like(x)
    for k in range(2):
        sl = slice(k * half, (k + 1) * half)
        p = x[..., sl]
        q = half // 2
        rot = np.concatenate([-p[..., q:], p[..., :q]], axis=-1)
        out[..., sl] = p * c[..., sl] + rot * s[..., sl]
    return out


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def load_weights(src, cfg, verbose=False):
    st = ST(src)
    names = st.names()

    def find_suffix(suffix):
        hits = [n for n in names if n.endswith(suffix)]
        if len(hits) != 1:
            raise KeyError("expected exactly one tensor ending in %r, got %d" % (suffix, len(hits)))
        return hits[0]

    ip = find_suffix("patch_embedder.input_proj.weight")
    tower_prefix = ip[: -len("patch_embedder.input_proj.weight")]
    proj = find_suffix("embed_vision.embedding_projection.weight")

    W = {}
    W["input_proj"] = st.get(ip)                                              # [1152,768]
    W["pos_table"] = st.get(tower_prefix + "patch_embedder.position_embedding_table")  # [2,10240,1152]
    W["std_bias"] = st.get(tower_prefix + "std_bias")                         # [1152]
    W["std_scale"] = st.get(tower_prefix + "std_scale")                       # [1152]
    W["embedding_projection"] = st.get(proj)                                  # [2816,1152]

    # `.linear.` is the Gemma4ClippableLinear wrapper.  use_clipped_linears == False for this
    # checkpoint (and there are no input_min/max buffers in the shard), so the wrapper is a
    # plain bias-free Linear and the segment is semantically empty.
    layers = []
    for i in range(cfg.num_hidden_layers):
        p = "%sencoder.layers.%d." % (tower_prefix, i)
        L = {
            "q": st.get(p + "self_attn.q_proj.linear.weight"),
            "k": st.get(p + "self_attn.k_proj.linear.weight"),
            "v": st.get(p + "self_attn.v_proj.linear.weight"),
            "o": st.get(p + "self_attn.o_proj.linear.weight"),
            "q_norm": st.get(p + "self_attn.q_norm.weight"),
            "k_norm": st.get(p + "self_attn.k_norm.weight"),
            "gate": st.get(p + "mlp.gate_proj.linear.weight"),
            "up": st.get(p + "mlp.up_proj.linear.weight"),
            "down": st.get(p + "mlp.down_proj.linear.weight"),
            "input_ln": st.get(p + "input_layernorm.weight"),
            "post_attn_ln": st.get(p + "post_attention_layernorm.weight"),
            "pre_ffw_ln": st.get(p + "pre_feedforward_layernorm.weight"),
            "post_ffw_ln": st.get(p + "post_feedforward_layernorm.weight"),
        }
        layers.append(L)
    W["layers"] = layers

    # There is no v_norm.weight in the shard: Gemma4VisionAttention builds
    #   self.v_norm = Gemma4RMSNorm(head_dim, with_scale=False)
    # so V gets a WEIGHTLESS RMSNorm.  Assert the absence, because a stray v_norm tensor would
    # mean this transcription is wrong.
    assert not [n for n in names if n.endswith("v_norm.weight")], "unexpected v_norm.weight"

    if verbose:
        print("# tower prefix : %s" % tower_prefix, file=sys.stderr)
        print("# proj tensor  : %s" % proj, file=sys.stderr)
    return W


# ---------------------------------------------------------------------------
# Stage 2: the tower
# ---------------------------------------------------------------------------
def forward_tower(W, patches, pos_xy, cfg=VisionConfig, act="gelu_tanh", dump=None, stats=None,
                  num_layers=None):
    """Stage 2 -> hidden [N, 1152]."""
    act_fn = ACT[act]
    eps = cfg.rms_norm_eps
    heads = cfg.num_attention_heads
    hd = cfg.head_dim
    n = patches.shape[0]
    nl = cfg.num_hidden_layers if num_layers is None else num_layers

    # ---- patch embedder -------------------------------------------------
    # Gemma4 applies no image normalization and instead scales in model code.
    x = np.float32(2.0) * (patches.astype(np.float32) - np.float32(0.5))
    h = linear(x, W["input_proj"])                               # [N,1152]
    _dump(dump, "post_input_proj", h)

    pos_x = pos_xy[:, 0].astype(np.int64)
    pos_y = pos_xy[:, 1].astype(np.int64)
    assert pos_x.max() < cfg.position_embedding_size and pos_y.max() < cfg.position_embedding_size
    pe = W["pos_table"][0][pos_x] + W["pos_table"][1][pos_y]      # [N,1152]
    _dump(dump, "pos_embed", pe)
    h = h + pe
    _dump(dump, "post_posembd", h)

    cos, sin = build_rope(pos_xy, hd, cfg.rope_theta)
    _dump(dump, "rope_cos", cos)
    _dump(dump, "rope_sin", sin)

    # No attention mask: all patches are valid (the oracle does not pad), and
    # create_bidirectional_mask over an all-valid sequence is the all-ones mask.
    for i in range(nl):
        L = W["layers"][i]
        residual = h

        # ---- attention block (sandwich norm) ----------------------------
        hn = rms_norm(h, L["input_ln"], eps)

        q = linear(hn, L["q"]).reshape(n, heads, hd)
        q = rms_norm(q, L["q_norm"], eps)                        # per-head RMSNorm, weight[72]
        q = apply_axial_rope(q, cos, sin)

        k = linear(hn, L["k"]).reshape(n, heads, hd)
        k = rms_norm(k, L["k_norm"], eps)
        k = apply_axial_rope(k, cos, sin)

        v = linear(hn, L["v"]).reshape(n, heads, hd)
        v = rms_norm(v, None, eps)                               # WEIGHTLESS RMSNorm on V

        if stats is not None:
            stats.setdefault("q_absmax", []).append(float(np.abs(q).max()))
            stats.setdefault("k_absmax", []).append(float(np.abs(k).max()))
            stats.setdefault("v_absmax", []).append(float(np.abs(v).max()))

        # scaling == 1.0 EXACTLY (Gemma4VisionAttention.__init__ sets self.scaling = 1.0),
        # i.e. NOT 1/sqrt(head_dim).  num_key_value_heads == num_attention_heads so repeat_kv
        # is the identity.
        attn = np.empty((n, heads, hd), dtype=np.float32)
        for hh in range(heads):
            scores = q[:, hh, :] @ k[:, hh, :].T                 # [N,N], scaling 1.0
            scores -= scores.max(axis=-1, keepdims=True)         # softmax in fp32
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            attn[:, hh, :] = scores @ v[:, hh, :]
        attn = attn.reshape(n, heads * hd)
        attn = linear(attn, L["o"])
        if stats is not None:
            stats.setdefault("attn_out_absmax", []).append(float(np.abs(attn).max()))

        attn = rms_norm(attn, L["post_attn_ln"], eps)
        h = residual + attn

        # ---- MLP block (sandwich norm) ----------------------------------
        residual = h
        hn = rms_norm(h, L["pre_ffw_ln"], eps)
        gate = linear(hn, L["gate"])
        up = linear(hn, L["up"])
        ff = act_fn(gate) * up
        if stats is not None:
            stats.setdefault("ffn_absmax", []).append(float(np.abs(ff).max()))
        ff = linear(ff, L["down"])
        ff = rms_norm(ff, L["post_ffw_ln"], eps)
        h = residual + ff

        assert np.isfinite(h).all(), "non-finite hidden states at layer %d" % i
        if stats is not None:
            stats.setdefault("layer_absmax", []).append(float(np.abs(h).max()))
        _dump(dump, "layer%02d_out" % i, h)

    # There is NO final post-layernorm in the vision encoder.
    assert np.isfinite(h).all(), "non-finite hidden states"
    return h


# ---------------------------------------------------------------------------
# Stage 3: pooling, standardization, projection
# ---------------------------------------------------------------------------
def pool_and_project(W, hidden, n_rows, n_cols, pos_xy=None, cfg=VisionConfig, dump=None):
    """Stage 3 -> soft [n_soft, 2816]."""
    k = cfg.pooling_kernel_size
    n, c = hidden.shape
    assert n == n_rows * n_cols
    length = (n_rows // k) * (n_cols // k)

    if pos_xy is None:
        gx, gy = np.meshgrid(np.arange(n_cols), np.arange(n_rows), indexing="xy")
        pos_xy = np.stack([gx, gy], axis=-1).reshape(n, 2).astype(np.int32)

    # Gemma4VisionPooler._avg_pool_by_positions, transcribed:
    #   max_x       = max(pos_x) + 1                     (= n_cols)
    #   kernel_idxs = pos // k
    #   kernel      = kernel_idxs.x + (max_x // k) * kernel_idxs.y
    #   weights     = one_hot(kernel, length) / k**2
    #   output      = weights.T @ hidden
    # i.e. a plain 3x3 average over the (n_rows x n_cols) patch grid, emitted row-major over
    # the pooled grid of shape (n_rows//3, n_cols//3).
    kk = int((n // length) ** 0.5)
    assert kk == k and kk * kk * length == n, "pool geometry mismatch"
    max_x = int(pos_xy[:, 0].max()) + 1
    kidx = pos_xy.astype(np.int64) // k
    kernel = kidx[:, 0] + (max_x // k) * kidx[:, 1]
    assert kernel.min() == 0 and kernel.max() == length - 1
    weights = np.zeros((n, length), dtype=np.float32)
    weights[np.arange(n), kernel] = np.float32(1.0) / np.float32(k * k)
    pooled = weights.T @ hidden.astype(np.float32)
    assert np.bincount(kernel, minlength=length).min() == k * k

    # Scale by sqrt(hidden_size); HF does this in float32 precisely because it can leave the
    # float16 range.
    pooled = pooled * np.float32(math.sqrt(cfg.hidden_size))
    _dump(dump, "pooled", pooled)

    # standardize == True for this checkpoint.
    std = (pooled - W["std_bias"].astype(np.float32)) * W["std_scale"].astype(np.float32)
    _dump(dump, "standardized", std)

    # Gemma4MultimodalEmbedder / DiffusionGemmaMultimodalEmbedder:
    #   embs_normed = embedding_pre_projection_norm(inputs_embeds)   # weightless RMSNorm
    #   return embedding_projection(embs_normed)                     # Linear, no bias
    # NORM BEFORE PROJECTION.  TensorSharp's Gemma4VisionEncoder.cs does it the other way
    # round; that is bug D1.
    normed = rms_norm(std, None, cfg.rms_norm_eps)
    _dump(dump, "pre_projection_norm", normed)
    out = linear(normed, W["embedding_projection"])
    assert out.shape == (length, cfg.text_hidden_size)
    assert np.isfinite(out).all(), "non-finite soft tokens"

    # The vision embeddings are NOT multiplied by the text embedding normalizer sqrt(2816);
    # only Gemma4TextScaledWordEmbedding scales, and soft tokens bypass it.
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dump(dump_dir, name, arr):
    if dump_dir:
        np.save(os.path.join(dump_dir, name + ".npy"), np.ascontiguousarray(arr))


def describe(a):
    a64 = a.astype(np.float64)
    return {
        "shape": list(a.shape),
        "mean": float(a64.mean()),
        "std": float(a64.std()),
        "absmax": float(np.abs(a64).max()),
        "min": float(a64.min()),
        "max": float(a64.max()),
        "sum_f64": float(a64.sum()),
        "abs_sum_f64": float(np.abs(a64).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="safetensors shard holding the vision tower")
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--act", choices=sorted(ACT.keys()), default="gelu_tanh",
                    help="MLP activation; gelu_tanh is what the checkpoint declares")
    ap.add_argument("--resize-mode", choices=("float", "uint8"), default="float")
    ap.add_argument("--alpha", choices=("white", "drop"), default="white")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, default=None, help="debug: run only the first N layers")
    ap.add_argument("--json", action="store_true", help="also print a machine-readable report")
    args = ap.parse_args(argv)

    if Image is None:
        raise SystemExit("PIL is required")

    # numpy 2.0.2 on arm64/Accelerate raises spurious FP status flags out of `matmul`
    # ("divide by zero / overflow / invalid value encountered in matmul") even for a matmul
    # of two finite random matrices -- reproducible with a 3-line snippet.  Suppress them and
    # rely on the explicit np.isfinite assertions after every stage instead.
    with np.errstate(all="ignore"):
        return _run(args)


def _run(args):
    t0 = time.time()
    cfg = VisionConfig
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    pil = Image.open(os.path.expanduser(args.image))
    src_size = pil.size  # (W, H)
    patches, pos_xy, (n_rows, n_cols), n_soft, (th, tw) = preprocess(
        pil, max_soft_tokens=args.max_soft_tokens, patch_size=cfg.patch_size,
        pooling_kernel_size=cfg.pooling_kernel_size, resize_mode=args.resize_mode,
        alpha=args.alpha,
    )
    _dump(args.dump_dir, "patches", patches)
    _dump(args.dump_dir, "pos", pos_xy.astype(np.int32))
    t_pre = time.time()

    W = load_weights(os.path.expanduser(args.src), cfg, verbose=True)
    t_load = time.time()

    stats = {}
    hidden = forward_tower(W, patches, pos_xy, cfg=cfg, act=args.act,
                           dump=args.dump_dir, stats=stats, num_layers=args.layers)
    t_tower = time.time()

    soft = pool_and_project(W, hidden, n_rows, n_cols, pos_xy=pos_xy, cfg=cfg, dump=args.dump_dir)
    _dump(args.dump_dir, "final", soft)
    t_end = time.time()

    if args.out:
        outp = os.path.expanduser(args.out)
        d = os.path.dirname(outp)
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(outp, soft)

    rep = {
        "image": args.image,
        "source_size_wh": list(src_size),
        "target_size_hw": [th, tw],
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_patches": int(patches.shape[0]),
        "n_soft": n_soft,
        "act": args.act,
        "resize_mode": args.resize_mode,
        "patches": describe(patches),
        "hidden": describe(hidden),
        "final": describe(soft),
        "per_layer": {k: v for k, v in stats.items()},
        "timing_s": {
            "preprocess": round(t_pre - t0, 3),
            "load_weights": round(t_load - t_pre, 3),
            "tower": round(t_tower - t_load, 3),
            "pool_project": round(t_end - t_tower, 3),
            "total": round(t_end - t0, 3),
        },
    }

    print("image            : %s  (%dx%d %s)" % (args.image, src_size[0], src_size[1], pil.mode))
    print("target size (HxW): %d x %d" % (th, tw))
    print("patch grid       : n_rows=%d  n_cols=%d  N=%d patches" % (n_rows, n_cols, patches.shape[0]))
    print("soft tokens      : %d   (pooled grid %d x %d)" % (n_soft, n_rows // 3, n_cols // 3))
    print("activation       : %s" % args.act)
    f = rep["final"]
    print("final            : shape=%s mean=%.9g std=%.9g absmax=%.9g" %
          (f["shape"], f["mean"], f["std"], f["absmax"]))
    print("checksum (f64sum): %.12g    |sum|=%.12g" % (f["sum_f64"], f["abs_sum_f64"]))
    print("hidden (pre-pool): mean=%.9g std=%.9g absmax=%.9g" %
          (rep["hidden"]["mean"], rep["hidden"]["std"], rep["hidden"]["absmax"]))
    print("per-layer absmax (layer / q / k / v / attn_out / ffn):")
    for i in range(len(stats.get("layer_absmax", []))):
        print("  L%02d  %12.4f  %8.4f  %8.4f  %8.4f  %10.4f  %12.4f" % (
            i, stats["layer_absmax"][i], stats["q_absmax"][i], stats["k_absmax"][i],
            stats["v_absmax"][i], stats["attn_out_absmax"][i], stats["ffn_absmax"][i]))
    print("timing           : %s" % rep["timing_s"])
    if args.json:
        print("JSON " + json.dumps(rep))
    return rep


if __name__ == "__main__":
    main()
