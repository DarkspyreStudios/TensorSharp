#!/usr/bin/env python3
"""Convert the DiffusionGemma-26B-A4B vision tower into a single-file mmproj GGUF.

The upstream checkpoint ships no mmproj. Every vision weight lives in one shard
(``model-00011-of-00011.safetensors``, 356 BF16 tensors); this script reads them
directly and emits a ``clip`` / ``mmproj`` GGUF with ``projector_type = gemma4v``
that both llama.cpp's clip loader (tools/mtmd/clip.cpp + models/gemma4v.cpp) and
TensorSharp's ``Gemma4VisionEncoder`` can load.

    python3 eng/diffusiongemma-mmproj.py \\
        --src  ~/work/models/diffusiongemma-vision/model-00011-of-00011.safetensors \\
        --out  ~/work/models/diffusiongemma-vision/mmproj-diffusiongemma-26B-A4B-it-F16.gguf

No third-party packages: numpy only (no torch, no safetensors, no gguf module).

Ground truth for the layout below is the shipped, known-good gemma4v mmproj
``mmproj-gemma-4-E4B-it-{Q8_0,BF16}.gguf``; every tensor name, dimension order
and per-tensor dtype here was cross-checked against a dump of that file and
against clip.cpp / clip-impl.h / gemma4v.cpp.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32

GGML_F32 = 0
GGML_F16 = 1

# general.file_type (llama_ftype); only the two we can emit
FTYPE_ALL_F32 = 0
FTYPE_MOSTLY_F16 = 1

KV_UINT32 = 4
KV_INT32 = 5
KV_FLOAT32 = 6
KV_BOOL = 7
KV_STRING = 8
KV_ARRAY = 9

_TYPE_SIZE = {GGML_F32: 4, GGML_F16: 2}
_TYPE_NAME = {GGML_F32: "F32", GGML_F16: "F16"}


class GgufWriter:
    """Minimal GGUF v3 writer.

    Tensors are registered with a *producer* callable rather than a payload, so
    the ~1.2 GB of converted weights are never all resident at once: the header
    only needs (name, dims, type), and each payload is materialised, written and
    dropped in turn.
    """

    def __init__(self, path, alignment=ALIGNMENT):
        self.path = path
        self.alignment = alignment
        self.kv = []
        self.tensors = []  # (name, dims_ne, ggml_type, nbytes, producer)

    # -- metadata ----------------------------------------------------------
    def add_string(self, key, value):
        self.kv.append((key, KV_STRING, str(value)))

    def add_uint32(self, key, value):
        self.kv.append((key, KV_UINT32, int(value)))

    def add_int32(self, key, value):
        self.kv.append((key, KV_INT32, int(value)))

    def add_float32(self, key, value):
        self.kv.append((key, KV_FLOAT32, float(value)))

    def add_bool(self, key, value):
        self.kv.append((key, KV_BOOL, bool(value)))

    def add_string_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_STRING, [str(v) for v in values])))

    def add_float32_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_FLOAT32, [float(v) for v in values])))

    def add_int32_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_INT32, [int(v) for v in values])))

    # -- tensors -----------------------------------------------------------
    def add_tensor(self, name, dims_ne, ggml_type, producer):
        """`dims_ne` is in GGUF ne order (ne[0] fastest), i.e. reversed numpy shape."""
        if ggml_type not in _TYPE_SIZE:
            raise ValueError("%s: unsupported GGML type %d" % (name, ggml_type))
        if not dims_ne or len(dims_ne) > 4:
            raise ValueError("%s: bad dims %r" % (name, dims_ne))
        if any(type(d) is not int or d <= 0 for d in dims_ne):
            raise ValueError("%s: bad dims %r" % (name, dims_ne))
        nbytes = int(np.prod(dims_ne)) * _TYPE_SIZE[ggml_type]
        self.tensors.append((name, list(dims_ne), int(ggml_type), nbytes, producer))

    # -- serialisation -----------------------------------------------------
    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _kv_value_bytes(self, vtype, value):
        if vtype == KV_STRING:
            return self._str(value)
        if vtype == KV_UINT32:
            return struct.pack("<I", value)
        if vtype == KV_INT32:
            return struct.pack("<i", value)
        if vtype == KV_FLOAT32:
            return struct.pack("<f", value)
        if vtype == KV_BOOL:
            return struct.pack("<B", 1 if value else 0)
        if vtype == KV_ARRAY:
            etype, items = value
            out = bytearray(struct.pack("<IQ", etype, len(items)))
            for it in items:
                out += self._kv_value_bytes(etype, it)
            return bytes(out)
        raise ValueError("unsupported kv type %d" % vtype)

    def _kv_bytes(self):
        out = bytearray()
        for key, vtype, value in self.kv:
            out += self._str(key)
            out += struct.pack("<I", vtype)
            out += self._kv_value_bytes(vtype, value)
        return bytes(out)

    def write(self):
        header = bytearray()
        header += struct.pack("<II", GGUF_MAGIC, GGUF_VERSION)
        header += struct.pack("<QQ", len(self.tensors), len(self.kv))
        header += self._kv_bytes()

        infos = bytearray()
        offset = 0
        for name, dims, ttype, nbytes, _ in self.tensors:
            infos += self._str(name)
            infos += struct.pack("<I", len(dims))
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", ttype)
            infos += struct.pack("<Q", offset)
            offset += nbytes
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment

        pre = bytes(header) + bytes(infos)
        pad = (-len(pre)) % self.alignment
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(pre)
            f.write(b"\0" * pad)
            for name, dims, ttype, nbytes, producer in self.tensors:
                payload = producer()
                if len(payload) != nbytes:
                    raise ValueError("%s: produced %d bytes, header says %d"
                                     % (name, len(payload), nbytes))
                f.write(payload)
                f.write(b"\0" * ((-nbytes) % self.alignment))
                del payload
        os.replace(tmp, self.path)
        return len(pre) + pad


# ---------------------------------------------------------------------------
# safetensors reading (BF16 / F16 / F32 -> float32)
# ---------------------------------------------------------------------------

class SafeTensorShard:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def __contains__(self, name):
        return name in self.header

    def names(self):
        return list(self.header.keys())

    def shape(self, name):
        return list(self.header[name]["shape"])

    def get(self, name):
        """Return the tensor as float32 with its checkpoint shape."""
        e = self.header[name]
        beg, end = e["data_offsets"]
        # np.array() copies, which also guarantees the alignment .view() needs
        # (safetensors data_offsets are not required to be 2- or 4-byte aligned).
        raw = np.array(self._mm[self.base + beg: self.base + end])
        dt = e["dtype"]
        if dt == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << np.uint32(16)
            a = u.view(np.float32)
        elif dt == "F16":
            a = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            a = raw.view(np.float32)
        else:
            raise ValueError("%s: unsupported safetensors dtype %s" % (name, dt))
        return np.ascontiguousarray(a.reshape(e["shape"]))


def to_payload(arr, ggml_type, name):
    """float32 ndarray -> little-endian GGUF payload bytes, with range checks."""
    if ggml_type == GGML_F32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if ggml_type == GGML_F16:
        absmax = float(np.abs(arr).max()) if arr.size else 0.0
        if absmax > 65504.0:
            raise ValueError("%s: |max| = %g overflows F16 (65504); write it as F32"
                             % (name, absmax))
        out = np.ascontiguousarray(arr, dtype="<f2")
        if not np.isfinite(np.asarray(out, dtype=np.float32)).all():
            raise ValueError("%s: F16 conversion produced a non-finite value" % name)
        return out.tobytes()
    raise ValueError("%s: unsupported GGML type %d" % (name, ggml_type))


# ---------------------------------------------------------------------------
# Vision-tower constants (all verified against the safetensors header)
# ---------------------------------------------------------------------------

N_LAYER = 27
N_EMBD = 1152
N_FF = 4304
N_HEAD = 16           # head_dim = 1152 / 16 = 72, matches q_norm/k_norm [72]
PATCH_SIZE = 16
PROJECTION_DIM = 2816
EPS = 1e-6
N_MERGE = 3           # Gemma4VisionPooler pooling_kernel_size (3x3 average pool)
POS_TABLE_ROWS = 10240

# clip.vision.image_size.
#
# GEMMA4V is a *variable-resolution* tower: clip.cpp reads this key (it is
# mandatory, clip.cpp:1312) but for PROJECTOR_TYPE_GEMMA4V it only ever uses it
# as the initial `warmup_image_size` (clip.cpp:1366), which is immediately
# overwritten twice inside the GEMMA4V case -- first by
# set_limit_image_tokens(70, 1120) and then by set_warmup_n_tokens(256)
# (clip.cpp:1649-1650). The only other constraints are the sanity checks
# 0 <= image_size <= 8192 (clip.cpp:2022-2028); the preprocessor uses
# image_min_pixels / image_max_pixels instead. So this value is inert for the
# forward pass and is only a nominal/log value.
#
# 768 is picked to be self-consistent with the soft-token budget: with
# patch_size 16 and n_merge 3 one soft token covers 48x48 px, so a 768x768
# image is 16x16 = 256 soft tokens -- just under the 280-soft-token working
# budget, and exactly the value llama.cpp itself computes for the warmup
# (set_warmup_n_tokens(256) -> 16 * 16 * 3 = 768).
IMAGE_SIZE = 768

SRC_PREFIX = "model.encoder.vision_tower"
SRC_PROJ = "model.encoder.embed_vision.embedding_projection.weight"

# Upstream layer sub-name -> (gguf suffix, force F32?).
# `use_clipped_linears=false` in this checkpoint, so every projection carries a
# semantically empty ".linear." segment that has to be stripped. (The E4B
# checkpoint has use_clipped_linears=true, which is why its mmproj additionally
# carries `.input_min/.input_max/.output_min/.output_max` scalars per linear --
# clip.cpp reads those with a FLT_MAX default (clip.cpp:2607-2626) and
# Gemma4VisionEncoder.cs leaves the clamp disabled when they are absent, so
# omitting them here is correct, not a gap.)
LAYER_MAP = [
    ("input_layernorm.weight",             "ln1.weight",            True),
    ("self_attn.q_proj.linear.weight",     "attn_q.weight",         False),
    ("self_attn.k_proj.linear.weight",     "attn_k.weight",         False),
    ("self_attn.v_proj.linear.weight",     "attn_v.weight",         False),
    ("self_attn.o_proj.linear.weight",     "attn_out.weight",       False),
    ("self_attn.q_norm.weight",            "attn_q_norm.weight",    True),
    ("self_attn.k_norm.weight",            "attn_k_norm.weight",    True),
    ("post_attention_layernorm.weight",    "attn_post_norm.weight", True),
    ("pre_feedforward_layernorm.weight",   "ln2.weight",            True),
    ("mlp.gate_proj.linear.weight",        "ffn_gate.weight",       False),
    ("mlp.up_proj.linear.weight",          "ffn_up.weight",         False),
    ("mlp.down_proj.linear.weight",        "ffn_down.weight",       False),
    ("post_feedforward_layernorm.weight",  "ffn_post_norm.weight",  True),
]


# ---------------------------------------------------------------------------
# The patch-embedding permute
# ---------------------------------------------------------------------------

def permute_patch_embed(w):
    """HF flat-patch Linear weight -> ggml_conv_2d OIHW kernel.

    HF patchifies in the image processor, not in the model:

        patched = image.reshape(C, nH, p, nW, p).permute(1, 3, 2, 4, 0)
                       .reshape(nH * nW, -1)            # g4_image_processing_gemma4.py:93-99

    so the 768-wide input row of `patch_embedder.input_proj` is indexed
    ``ky * p * C + kx * C + c`` -- CHANNEL FASTEST.

    llama.cpp instead runs a real convolution, ``ggml_conv_2d(model.patch_
    embeddings_0, inp_raw, patch_size, patch_size, ...)`` (gemma4v.cpp:16), whose
    kernel is OIHW: ne = [KW, KH, IC, OC], i.e. numpy [OC, IC, KH, KW], so its
    im2col row is indexed ``c * p * p + ky * p + kx`` -- CHANNEL SLOWEST.
    TensorSharp's own im2col (Gemma4VisionEncoder.cs:426-438) builds rows in the
    same channel-slowest order.

    The two orders differ, so the weight must be permuted once here:

        [OC, ky*p*C + kx*C + c]  ->  reshape(OC, p, p, C)  ->  transpose(0,3,1,2)
                                 ->  [OC, C, p, p]

    and written with 4 dims (GGUF ne [p, p, C, OC]), matching the known-good
    E4B mmproj whose v.patch_embd.weight dumps as ne [16, 16, 3, 768].
    """
    out_dim, flat = w.shape
    p = PATCH_SIZE
    assert flat == 3 * p * p, "input_proj input width %d != 3*%d*%d" % (flat, p, p)
    conv = np.ascontiguousarray(w.reshape(out_dim, p, p, 3).transpose(0, 3, 1, 2))
    assert conv.shape == (out_dim, 3, p, p), conv.shape
    return conv


def prove_patch_embed_permute(w, conv):
    """Numerically prove the permute: HF flat-dot == conv im2col-dot, same patch.

    Builds one synthetic 16x16x3 patch, runs it through both conventions and
    requires bit-comparable results. A wrong transpose fails this immediately.
    """
    p = PATCH_SIZE
    rng = np.random.default_rng(0)
    patch_chw = rng.standard_normal((3, p, p), dtype=np.float32)

    # HF convention: reshape(C,nH,p,nW,p).permute(1,3,2,4,0) with nH=nW=1
    # collapses to [ky][kx][c], i.e. transpose(1, 2, 0) of a [C,ky,kx] patch.
    hf_row = patch_chw.transpose(1, 2, 0).reshape(-1).astype(np.float64)
    # elementwise, not matmul: keeps the check off BLAS so it stays an
    # independent reimplementation of the dot product
    hf_out = (w.astype(np.float64) * hf_row).sum(axis=1)

    # conv/im2col convention: [c][ky][kx], straight flatten of [C,ky,kx].
    conv_row = patch_chw.reshape(-1).astype(np.float64)
    conv_out = (conv.reshape(conv.shape[0], -1).astype(np.float64) * conv_row).sum(axis=1)

    err = float(np.abs(hf_out - conv_out).max())
    scale = float(np.abs(hf_out).max()) + 1e-12
    if err / scale > 1e-12:
        raise AssertionError("patch-embed permute is wrong: rel err %g" % (err / scale))
    return err / scale


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def build(st, writer, dtype_code, rows):
    """Register every tensor. `rows` collects the summary table."""

    def register(name, arr, ggml_type, note=""):
        """Register an already-materialised array (kept alive by the closure)."""
        dims_ne = list(reversed(list(arr.shape)))
        writer.add_tensor(name, dims_ne, ggml_type,
                          (lambda a=arr, t=ggml_type, n=name: to_payload(a, t, n)))
        rows.append((name, tuple(arr.shape), tuple(dims_ne), _TYPE_NAME[ggml_type], note))

    def register_lazy(name, src, ggml_type, shape, note=""):
        """Register by checkpoint name; the payload is read and converted only
        at write time, so the ~1.1 GB of weights never all sit in memory."""
        dims_ne = list(reversed(list(shape)))

        def produce(src=src, t=ggml_type, n=name):
            return to_payload(st.get(src), t, n)

        writer.add_tensor(name, dims_ne, ggml_type, produce)
        rows.append((name, tuple(shape), tuple(dims_ne), _TYPE_NAME[ggml_type], note))

    # --- patch embedding (permuted to a conv2d kernel) --------------------
    pe_src = SRC_PREFIX + ".patch_embedder.input_proj.weight"
    pe = st.get(pe_src)
    conv = permute_patch_embed(pe)
    rel = prove_patch_embed_permute(pe, conv)
    print("patch-embed permute verified (HF flat-dot vs conv im2col-dot, "
          "rel err %.3g)" % rel)
    # F32 like the reference mmproj (both the Q8_0 and the BF16 E4B files keep
    # v.patch_embd.weight at F32; ggml_conv_2d takes its im2col dtype from the
    # kernel, so F32 also keeps the conv in the well-trodden F32 path).
    register("v.patch_embd.weight", conv, GGML_F32,
             "permuted [OC,ky,kx,C]->[OC,C,ky,kx]")
    del pe, conv

    # --- 2D position lookup tables ---------------------------------------
    # gemma4v.cpp:30 reads `pos_size = model.position_embeddings->ne[1]` and
    # then takes two ggml_view_2d row-ranges at offsets 0 and pos_size*nb1, so
    # the tensor MUST stay 3-D with ne = [n_embd, pos_size, 2]. Flattening it to
    # a 2-D [n_embd, 2*pos_size] would make pos_size 20480 and push the y-table
    # view past the end of the buffer. TensorSharp reads it the same way
    # (Gemma4VisionEncoder.cs:365-370: maxPos = Sizes[1], yTable = ptr +
    # maxPos*hidden). The E4B reference dumps as ne [768, 10240, 2]; ours is
    # ne [1152, 10240, 2]. F32, matching the reference.
    pos_src = SRC_PREFIX + ".patch_embedder.position_embedding_table"
    pos_shape = st.shape(pos_src)
    assert pos_shape == [2, POS_TABLE_ROWS, N_EMBD], pos_shape
    register_lazy("v.position_embd.weight", pos_src, GGML_F32, pos_shape,
                  "x-table then y-table")

    # --- pooled-feature standardisation ----------------------------------
    # std_bias reaches |5.4e4|: inside F16's range but only ~32 apart there, and
    # the value is *subtracted* from the pooled features, so F16 rounding leaks
    # straight into the projector input. Both stay F32. (Absmax is printed in
    # the summary below.)
    for short, key in (("std_bias", "v.std_bias"), ("std_scale", "v.std_scale")):
        src = "%s.%s" % (SRC_PREFIX, short)
        arr = st.get(src)
        register(key, arr, GGML_F32, "absmax %.6g" % float(np.abs(arr).max()))
        del arr

    # --- transformer blocks ----------------------------------------------
    for il in range(N_LAYER):
        base = "%s.encoder.layers.%d" % (SRC_PREFIX, il)
        for src_suffix, dst_suffix, force_f32 in LAYER_MAP:
            src = "%s.%s" % (base, src_suffix)
            if src not in st:
                raise KeyError("missing checkpoint tensor %s" % src)
            t = GGML_F32 if force_f32 else dtype_code
            register_lazy("v.blk.%d.%s" % (il, dst_suffix), src, t, st.shape(src))

    # --- multimodal projector --------------------------------------------
    # clip.cpp:5973 returns mm_input_proj_w->ne[1] as n_mmproj_embd, so ne must
    # be [n_embd, projection_dim] = [1152, 2816] -> numpy [2816, 1152], which is
    # exactly the checkpoint layout (no transpose). Reference E4B dumps as
    # ne [768, 2560].
    proj_shape = st.shape(SRC_PROJ)
    assert proj_shape == [PROJECTION_DIM, N_EMBD], proj_shape
    register_lazy("mm.input_projection.weight", SRC_PROJ, dtype_code, proj_shape,
                  "ne[1]=%d is n_mmproj_embd" % PROJECTION_DIM)


def add_metadata(writer, dtype_code, name):
    ftype = FTYPE_MOSTLY_F16 if dtype_code == GGML_F16 else FTYPE_ALL_F32

    writer.add_uint32("general.alignment", ALIGNMENT)
    writer.add_string("general.architecture", "clip")
    writer.add_string("general.type", "mmproj")
    writer.add_string("general.name", name)
    writer.add_uint32("general.file_type", ftype)
    writer.add_uint32("general.quantization_version", 2)

    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)

    # clip.cpp:1263-1272 reads KEY_PROJ_TYPE ("clip.projector_type") first and
    # only falls back to the per-modality KEY_VISION_PROJ_TYPE
    # ("clip.vision.projector_type") when it is empty. The shipped E4B mmproj is
    # a mixed-modality file so it carries only the latter. This file is
    # vision-only, so both are unambiguous; TensorSharp reads
    # "clip.vision.projector_type" (Gemma4VisionEncoder.cs:112). Emit both.
    writer.add_string("clip.projector_type", "gemma4v")
    writer.add_string("clip.vision.projector_type", "gemma4v")

    writer.add_uint32("clip.vision.embedding_length", N_EMBD)
    writer.add_uint32("clip.vision.feed_forward_length", N_FF)
    writer.add_uint32("clip.vision.block_count", N_LAYER)
    writer.add_uint32("clip.vision.attention.head_count", N_HEAD)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", EPS)
    writer.add_uint32("clip.vision.patch_size", PATCH_SIZE)
    writer.add_uint32("clip.vision.projection_dim", PROJECTION_DIM)
    writer.add_uint32("clip.vision.image_size", IMAGE_SIZE)

    # do_normalize is false in the HF processor; llama.cpp still requires both
    # arrays to exist and to hold >= 3 entries (clip.cpp:1398-1406).
    writer.add_float32_array("clip.vision.image_mean", [0.0, 0.0, 0.0])
    writer.add_float32_array("clip.vision.image_std", [1.0, 1.0, 1.0])

    # CRITICAL. vision_config.hidden_activation is "gelu_pytorch_tanh"
    # (g4_configuration_gemma4.py:159,279) but clip.cpp:1376-1386 falls back to
    # FFN_GELU_QUICK whenever neither clip.use_gelu nor clip.use_silu is set --
    # and the shipped E4B mmproj carries neither. Setting use_gelu selects
    # FFN_GELU, the tanh-approximate GELU ggml implements.
    writer.add_bool("clip.use_gelu", True)

    # Pooling / merge factor, value 3 (Gemma4VisionPooler's 3x3 average pool).
    # llama.cpp's canonical key is "clip.vision.projector.scale_factor"
    # (clip-impl.h:61, read at clip.cpp:1648); TensorSharp reads
    # "clip.vision.projector_scale_factor" (Gemma4VisionEncoder.cs:108). Both
    # default to 3 when absent, but emit both spellings so neither runtime has
    # to rely on its default.
    writer.add_uint32("clip.vision.projector.scale_factor", N_MERGE)
    writer.add_uint32("clip.vision.projector_scale_factor", N_MERGE)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(path, expect_tensors, expect_kv):
    """Re-read the file we just wrote with an independent parser."""
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        ntensor, nkv = struct.unpack("<QQ", f.read(16))
    if magic != GGUF_MAGIC:
        raise AssertionError("bad magic 0x%08x" % magic)
    if ver != GGUF_VERSION:
        raise AssertionError("bad version %d" % ver)
    if ntensor != expect_tensors:
        raise AssertionError("tensor count %d != %d" % (ntensor, expect_tensors))
    if nkv != expect_kv:
        raise AssertionError("kv count %d != %d" % (nkv, expect_kv))
    return ntensor, nkv


# ---------------------------------------------------------------------------

DEFAULT_SRC = os.path.expanduser(
    "~/work/models/diffusiongemma-vision/model-00011-of-00011.safetensors")
DEFAULT_OUT_TMPL = os.path.expanduser(
    "~/work/models/diffusiongemma-vision/mmproj-diffusiongemma-26B-A4B-it-%s.gguf")
MODEL_NAME = "DiffusionGemma-26B-A4B-it"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert the DiffusionGemma vision tower to an mmproj GGUF.")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="safetensors shard holding the vision tensors "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output GGUF (default: " + DEFAULT_OUT_TMPL % "<DTYPE>" + ")")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16",
                    help="dtype for the 2-D matmul weights; norms, the patch "
                         "kernel, the position tables and std_bias/std_scale "
                         "are always F32 (default: %(default)s)")
    ap.add_argument("--name", default=MODEL_NAME, help="general.name")
    args = ap.parse_args(argv)

    dtype_code = GGML_F16 if args.dtype == "f16" else GGML_F32
    out = args.out or DEFAULT_OUT_TMPL % args.dtype.upper()
    out = os.path.expanduser(out)
    src = os.path.expanduser(args.src)

    if not os.path.isfile(src):
        ap.error("no such file: %s" % src)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("src : %s (%.2f GB)" % (src, os.path.getsize(src) / 1e9))
    print("out : %s" % out)
    print("dtype: %s (matmul weights)\n" % args.dtype)

    st = SafeTensorShard(src)
    writer = GgufWriter(out)
    add_metadata(writer, dtype_code, args.name)

    rows = []
    build(st, writer, dtype_code, rows)

    expect = 5 + N_LAYER * len(LAYER_MAP)
    if len(rows) != expect:
        raise AssertionError("built %d tensors, expected %d" % (len(rows), expect))

    print("\nwriting %d tensors, %d metadata keys ..." % (len(rows), len(writer.kv)))
    writer.write()

    size = os.path.getsize(out)
    verify(out, len(rows), len(writer.kv))

    # -- summary table -----------------------------------------------------
    w_name = max(len(r[0]) for r in rows)
    print("\n%-*s  %-22s  %-24s  %-4s  %s"
          % (w_name, "tensor", "numpy shape", "gguf ne (reversed)", "type", "note"))
    print("-" * (w_name + 70))
    for name, shape, ne, tname, note in rows:
        print("%-*s  %-22s  %-24s  %-4s  %s"
              % (w_name, name, str(list(shape)), str(list(ne)), tname, note))

    print("\nmetadata:")
    for key, vtype, value in writer.kv:
        if vtype == KV_ARRAY:
            value = value[1]
        print("  %-42s = %r" % (key, value))

    print("\nOK  %s" % out)
    print("    %d tensors, %d KV, %.1f MiB" % (len(rows), len(writer.kv), size / (1 << 20)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
