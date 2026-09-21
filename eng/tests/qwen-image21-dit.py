#!/usr/bin/env python3
"""Independent NumPy reference for the Qwen-Image-2.1 native transformer.

Runs small, deterministic full forwards (including the conditioning projections,
time embedding, segmented causal attention and final projection) on CPU or Metal.
No model downloads required. This validates operators, not image quality.
Example: python3 eng/tests/qwen-image21-dit.py --backend metal
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import time
import numpy as np


class Weight(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("type", ct.c_int32), ("reserved", ct.c_int32),
                ("ne0", ct.c_int64), ("ne1", ct.c_int64), ("bytes", ct.c_int64)]


class Block(ct.Structure):
    _fields_ = [(n, Weight) for n in ("q", "k", "v", "out", "gate", "up", "down")] + [
        ("norm_q", ct.c_void_p), ("norm_k", ct.c_void_p)]


class Segment(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in ("start", "end", "source_start", "is_image")]


class Desc(ct.Structure):
    _fields_ = [(n, ct.c_void_p) for n in ("images", "text", "time_embedding", "cos", "sin", "output")] + [
        (n, Weight) for n in ("image_in", "text_in", "text_out", "time_in", "time_out", "modulation", "norm_out", "proj_out")] + [
        (n, ct.c_void_p) for n in ("text_norm", "blocks", "segments")] + [
        (n, ct.c_int32) for n in ("struct_bytes", "dim", "heads", "head_dim", "channels", "text_dim", "image_seq", "text_seq", "total_seq", "prefix_seq", "num_layers", "num_segments")] + [("eps", ct.c_float)]


def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def norm(x):
    return rms(x - np.mean(x, axis=-1, keepdims=True))


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def linear(x, w):
    # Explicit contraction keeps the oracle independent of platform BLAS kernels.
    return np.einsum("...i,oi->...o", x, w, optimize=False)


def reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads):
    time = silu(linear(silu(linear(ts, w["time_in"])), w["time_out"]))
    mods = np.split(linear(time, w["modulation"]), 4, axis=-1)
    text = linear(gelu(linear(rms(text) * (1 + w["text_norm"]), w["text_in"])), w["text_out"])
    images = linear(images, w["image_in"])
    x = np.concatenate([(images if s.is_image else text)[s.source_start:s.source_start+s.end-s.start] for s in segments])
    seq, dim = x.shape
    hd = dim // heads

    def modulate(h, mod, gate=False):
        factors = np.concatenate([np.repeat(mod[1:2], prefix, axis=0), np.repeat(mod[:1], seq-prefix, axis=0)])
        return h * (np.tanh(factors) if gate else 1 + factors)

    def rope(h):
        e, o = h[..., ::2], h[..., 1::2]
        # Interleaved reference intentionally differs from the native half-split
        # optimization: attention must remain invariant to that channel permutation.
        out = np.empty_like(h)
        out[..., ::2] = e * cos[:, None, :] - o * sin[:, None, :]
        out[..., 1::2] = o * cos[:, None, :] + e * sin[:, None, :]
        return out

    for b in blocks:
        h = modulate(norm(x), mods[0])
        q = rope(rms((linear(h, b["q"])).reshape(seq, heads, hd)) * b["norm_q"])
        k = rope(rms((linear(h, b["k"])).reshape(seq, heads, hd)) * b["norm_k"])
        v = (linear(h, b["v"])).reshape(seq, heads, hd)
        outputs = []
        for s in segments:
            scores = np.einsum("qhd,khd->hqk", q[s.start:s.end], k[:s.end]) / np.sqrt(hd)
            if not s.is_image:
                mask = np.arange(s.end)[None, :] > np.arange(s.start, s.end)[:, None]
                scores = np.where(mask[None, :, :], -np.inf, scores)
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            outputs.append(np.einsum("hqk,khd->qhd", probs, v[:s.end]).reshape(s.end-s.start, dim))
        x += modulate(linear(np.concatenate(outputs), b["out"]), mods[1], True)
        h = modulate(norm(x), mods[2])
        h = silu(linear(h, b["gate"])) * (linear(h, b["up"]))
        x += modulate(linear(h, b["down"]), mods[3], True)
    return linear(norm(x[prefix:]) * (1 + linear(time[:1], w["norm_out"])), w["proj_out"])


def run_case(lib, backend, case, fused, flash, benchmark_tokens=0):
    rng = np.random.default_rng(2141)
    keep = []

    def array(shape, scale=1):
        a = (rng.standard_normal(shape) * scale).astype(np.float32)
        keep.append(a)
        return a

    def weight(a):
        return Weight(a.ctypes.data, 0, 0, a.shape[-1], a.shape[0], a.nbytes)

    dim, hd, heads, channels, td, ff = 256, 128, 2, 64, 96, 320
    if benchmark_tokens:
        specs = [(0, 64, 0, 0), (64, 64 + benchmark_tokens, 0, 1)]
        tsq, isq, prefix = 64, benchmark_tokens, 64
    elif case == "text":
        specs = [(0, 5, 0, 0), (5, 11, 0, 1)]
        tsq, isq, prefix = 5, 6, 5
    elif case == "edit":
        specs = [(0, 2, 0, 0), (2, 6, 0, 1), (6, 9, 3, 0), (9, 15, 4, 1)]
        tsq, isq, prefix = 6, 10, 9
    elif case == "multi-edit":
        specs = [(0, 2, 0, 0), (2, 6, 0, 1), (6, 8, 3, 0), (8, 12, 4, 1), (12, 15, 6, 0), (15, 21, 8, 1)]
        tsq, isq, prefix = 9, 14, 15
    else:
        # Text and image boundaries straddle flash-attention tile lengths;
        # the shape mutation below changes 512 KV/256 query rows to 511/255.
        specs = [(0, 63, 0, 0), (63, 255, 0, 1), (255, 256, 63, 0), (256, 512, 192, 1)]
        tsq, isq, prefix = 64, 448, 256
    segments = (Segment * len(specs))(*(Segment(*s) for s in specs))
    seq = specs[-1][1]
    images, text = array((isq, channels)), array((tsq, td))
    ts = array((2, 256), .5)
    angles = array((seq, hd//2))
    cos, sin = np.cos(angles), np.sin(angles)
    w = {}
    for name, shape in {
        "image_in": (dim, channels), "text_in": (dim, td), "text_out": (dim, dim),
        "time_in": (dim, 256), "time_out": (dim, dim), "modulation": (4*dim, dim),
        "norm_out": (dim, dim), "proj_out": (channels, dim),
    }.items():
        w[name] = array(shape, .6 / np.sqrt(shape[-1]))
    w["text_norm"] = array((td,), .2)
    blocks, native_blocks = [], (Block * 2)()
    for i in range(2):
        b = {n: array(shape, .6 / np.sqrt(shape[-1])) for n, shape in {
            "q": (dim, dim), "k": (dim, dim), "v": (dim, dim), "out": (dim, dim),
            "gate": (ff, dim), "up": (ff, dim), "down": (dim, ff),
        }.items()}
        b["norm_q"] = array((hd,), .2) + 1
        b["norm_k"] = array((hd,), .2) + 1
        blocks.append(b)
        for name in ("q", "k", "v", "out", "gate", "up", "down"):
            setattr(native_blocks[i], name, weight(b[name]))
        if fused:
            gu = np.concatenate([b["gate"], b["up"]])
            keep.append(gu)
            native_blocks[i].gate, native_blocks[i].up = weight(gu), Weight()
        native_blocks[i].norm_q, native_blocks[i].norm_k = b["norm_q"].ctypes.data, b["norm_k"].ctypes.data
    expected = None if benchmark_tokens else reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads)
    output = np.zeros((seq - prefix, channels), dtype=np.float32)
    d = Desc()
    for name, a in dict(images=images, text=text, time_embedding=ts, cos=cos, sin=sin, output=output).items():
        setattr(d, name, a.ctypes.data)
    for name in ("image_in", "text_in", "text_out", "time_in", "time_out", "modulation", "norm_out", "proj_out"):
        setattr(d, name, weight(w[name]))
    d.text_norm = w["text_norm"].ctypes.data
    d.blocks, d.segments = ct.addressof(native_blocks), ct.addressof(segments)
    for name, value in dict(struct_bytes=ct.sizeof(Desc), dim=dim, heads=heads, head_dim=hd, channels=channels,
                            text_dim=td, image_seq=isq, text_seq=tsq, total_seq=seq, prefix_seq=prefix,
                            num_layers=2, num_segments=len(specs), eps=1e-6).items():
        setattr(d, name, value)
    os.environ["TS_QWEN21_FLASH"] = "1" if flash else "0"
    # The larger boundary fixture uses Metal's half-precision matrix kernels;
    # the unchanged baseline has ~1.4e-3 absolute error against the F32 oracle.
    tolerance = (.002 if case == "tile-boundaries" else .001) if backend == "metal" else .0001
    errors, seconds = [], []
    first_output = None
    for repeat in range(6 if benchmark_tokens else 5):
        # Changed latent input exercises allocation/cache reuse with new data.
        if repeat == 2 and not benchmark_tokens:
            images *= 1.1
            expected = reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads)
        if repeat == 3 and not benchmark_tokens:
            # Real CFG alternates prompt shapes through the shared allocator.
            # Shrink then restore the target with resident weights untouched.
            segments[-1].end -= 1
            d.total_seq -= 1
            d.image_seq -= 1
            expected = reference(w, blocks, images[:-1], text, ts, cos[:-1], sin[:-1], segments, prefix, heads)
        if repeat == 4 and not benchmark_tokens:
            segments[-1].end += 1
            d.total_seq += 1
            d.image_seq += 1
            expected = reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads)
        output.fill(np.nan)
        start = time.perf_counter()
        if lib.TSGgml_QwenImage21Forward(ct.byref(d)) != 1:
            raise RuntimeError(lib.TSGgml_GetLastError().decode())
        seconds.append(time.perf_counter() - start)
        if benchmark_tokens:
            assert np.isfinite(output).all(), "Non-finite benchmark output"
            if first_output is None:
                first_output = output.copy()
            else:
                assert np.array_equal(first_output, output), "Repeated input produced a different output"
            continue
        actual = output[:len(expected)]
        error = float(np.max(np.abs(actual - expected)))
        assert np.isfinite(actual).all() and error < tolerance, (case, fused, flash, repeat, error)
        assert np.isnan(output[len(expected):]).all(), "Native output wrote beyond target shape"
        errors.append(error)
    # Reject a malformed segment before any ggml assertion or output write.
    segments[0].start = 1
    assert lib.TSGgml_QwenImage21Forward(ct.byref(d)) == 0
    lib.TSGgml_ClearHostBufferCache()
    if benchmark_tokens:
        return dict(case=case, fused_mlp=fused, flash=flash, target_tokens=benchmark_tokens,
                    prefix_tokens=prefix, dim=dim, head_dim=hd, layers=len(blocks), weight_type="F32",
                    first_seconds=seconds[0], warm_seconds=seconds[1:], warm_median_seconds=float(np.median(seconds[1:])),
                    reference_validated=False,
                    limitation="Synthetic two-layer graph; excludes model loading, conditioning, scheduler and VAE. No image quality validation.")
    return dict(case=case, fused_mlp=fused, flash=flash, tolerance=tolerance, max_absolute_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--library", type=Path, default=Path(__file__).resolve().parents[2]/"TensorSharp.GGML.Native/build/libGgmlOps.dylib")
    parser.add_argument("--benchmark-target-tokens", type=int, default=0,
                        help="Run one synthetic F32/fused/flash timing workload with this many target tokens; skips the NumPy oracle.")
    args = parser.parse_args()
    if args.benchmark_target_tokens < 0:
        parser.error("--benchmark-target-tokens must be positive when supplied")
    lib = ct.CDLL(str(args.library.resolve()))
    lib.TSGgml_GetLastError.restype = ct.c_char_p
    lib.TSGgml_QwenImage21Forward.argtypes = [ct.POINTER(Desc)]
    assert lib.TSGgml_IsBackendAvailable({"cpu": 2, "metal": 1, "cuda": 3}[args.backend]) == 1
    try:
        if args.benchmark_target_tokens:
            rows = [run_case(lib, args.backend, "text", True, True, args.benchmark_target_tokens)]
        else:
            rows = [run_case(lib, args.backend, case, fused, flash)
                    for case in ("text", "edit", "multi-edit", "tile-boundaries") for fused in (False, True) for flash in (False, True)]
    finally:
        lib.TSGgml_ReleaseReuseComputeBuffers()
        lib.TSGgml_ClearHostBufferCache()
    print(json.dumps(dict(backend=args.backend, passed=0 if args.benchmark_target_tokens else len(rows), cases=rows), indent=2))


if __name__ == "__main__":
    main()
