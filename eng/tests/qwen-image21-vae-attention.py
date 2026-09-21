#!/usr/bin/env python3
"""Check native Qwen VAE spatial attention against an independent NumPy oracle.

Exercises the real 768/1152-channel head widths, query-tile tails, repeated
calls, allocator shape changes and rejected arguments. Requires NumPy, no model
weights. This is operator validation, not a full-model quality or speed test.

Example:
  python3 eng/tests/qwen-image21-vae-attention.py --backend metal \
    --output docs/validation/qwen-image-2.1/vae-attention-metal.json
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FLOAT_PTR = ct.POINTER(ct.c_float)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reference(qkv):
    """Full, untiled, FP64 attention over every spatial key for each query."""
    q, k, v = qkv.astype(np.float64)
    scores = np.einsum("cq,ck->qk", q, k, optimize=False) / np.sqrt(q.shape[0])
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.einsum("qk,ck->cq", probabilities, v, optimize=False)


def error_message(lib):
    return (lib.TSGgml_GetLastError() or b"unknown native error").decode(errors="replace")


def invoke(lib, qkv, expected, label, atol, rtol):
    channels, sequence = qkv.shape[1:]
    original = qkv.copy()
    # Distinct finite sentinels detect writes both before and after the buffer;
    # NaNs within it also detect tiles or channels the graph forgot to copy.
    storage = np.full(channels * sequence + 32, np.float32(123456.25))
    actual = storage[16:-16].reshape(channels, sequence)
    actual.fill(np.nan)
    success = lib.TSGgml_QwenVaeAttention(
        qkv.ctypes.data_as(FLOAT_PTR), actual.ctypes.data_as(FLOAT_PTR), channels, sequence)
    require(success == 1, f"{label}: {error_message(lib)}")
    require(np.isfinite(actual).all(), f"{label}: missing or non-finite output")
    require(np.array_equal(qkv, original), f"{label}: input was modified")
    require(np.all(storage[:16] == 123456.25) and np.all(storage[-16:] == 123456.25),
            f"{label}: write outside output buffer")
    absolute = np.abs(actual.astype(np.float64) - expected)
    allowed = atol + rtol * np.abs(expected)
    relative_l2 = float(np.linalg.norm(actual.astype(np.float64) - expected) /
                        max(np.linalg.norm(expected), np.finfo(np.float64).tiny))
    require(np.all(absolute <= allowed),
            f"{label}: max absolute error {absolute.max():.8g}, "
            f"max tolerance ratio {(absolute / allowed).max():.8g}")
    require(relative_l2 <= 0.002, f"{label}: relative L2 error {relative_l2:.8g} exceeds 0.002")
    return {"case": label, "channels": channels, "sequence": sequence,
            "max_absolute_error": float(absolute.max()),
            "rmse": float(np.sqrt(np.mean(absolute ** 2))),
            "relative_l2_error": relative_l2,
            "max_tolerance_ratio": float((absolute / allowed).max())}


def run_cases(lib, atol, rtol):
    rng = np.random.default_rng(2141768)
    rows = []
    # Alternating large and small shapes exercises persistent gallocr re-planning.
    for channels, sequence in [(7, 1), (13, 19), (768, 257), (1152, 511),
                               (768, 769), (5, 7), (1152, 257)]:
        qkv = rng.standard_normal((3, channels, sequence)).astype(np.float32)
        expected = reference(qkv)
        label = f"random-{channels}x{sequence}"
        rows.append(invoke(lib, qkv, expected, label, atol, rtol))
        # Same shape with different values catches stale uploads/cache reuse.
        qkv[0] *= np.float32(.7)
        qkv[1] += np.float32(.17)
        qkv[2] = np.roll(qkv[2], 1, axis=1) - np.float32(.2)
        rows.append(invoke(lib, qkv, reference(qkv), label + "-changed", atol, rtol))
    # Uniform attention has a closed-form expected result and catches layout
    # errors independently of the dense oracle's contractions.
    qkv = rng.standard_normal((3, 768, 513)).astype(np.float32)
    qkv[0].fill(0)
    expected = np.repeat(qkv[2].astype(np.float64).mean(axis=1, keepdims=True), 513, axis=1)
    rows.append(invoke(lib, qkv, expected, "zero-query-uniform", atol, rtol))
    # Larger logits exercise max-subtracted softmax without overflowing exp.
    qkv = rng.standard_normal((3, 23, 273)).astype(np.float32)
    qkv[:2] *= np.float32(4)
    rows.append(invoke(lib, qkv, reference(qkv), "large-logits-tail", atol, rtol))
    return rows


def invalid_arguments(lib):
    data = np.zeros(3, dtype=np.float32)
    output = np.full(1, np.float32(137))
    inp, out = data.ctypes.data_as(FLOAT_PTR), output.ctypes.data_as(FLOAT_PTR)
    cases = [(None, out, 1, 1), (inp, None, 1, 1), (inp, out, 0, 1),
             (inp, out, 1, 0), (inp, out, -1, 1), (inp, out, 1, -1),
             (inp, out, 2**31 - 1, 2**31 - 1)]
    for index, arguments in enumerate(cases):
        require(lib.TSGgml_QwenVaeAttention(*arguments) == 0,
                f"invalid argument case {index} was accepted")
        require(output[0] == 137, f"invalid argument case {index} wrote output")
        require(bool(lib.TSGgml_GetLastError()), f"invalid argument case {index} lost error message")
    # A rejected invocation must not poison the next valid request.
    data[:] = [2, 3, 5]
    require(lib.TSGgml_QwenVaeAttention(inp, out, 1, 1) == 1, error_message(lib))
    require(output[0] == 5, "valid invocation after invalid arguments failed")
    return len(cases)


def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--library", type=Path,
                        default=ROOT / "TensorSharp.GGML.Native/build/libGgmlOps.dylib")
    parser.add_argument("--ggml-source", type=Path, default=ROOT / "ExternalProjects/ggml")
    parser.add_argument("--output", type=Path, help="Write generated JSON evidence (use docs/validation/ or artifacts/).")
    args = parser.parse_args()
    library = args.library.resolve()
    lib = ct.CDLL(str(library))
    lib.TSGgml_GetLastError.restype = ct.c_char_p
    lib.TSGgml_IsBackendAvailable.argtypes = [ct.c_int]
    lib.TSGgml_QwenVaeAttention.argtypes = [FLOAT_PTR, FLOAT_PTR, ct.c_int, ct.c_int]
    lib.TSGgml_QwenVaeAttention.restype = ct.c_int
    require(lib.TSGgml_IsBackendAvailable({"cpu": 2, "metal": 1, "cuda": 3, "vulkan": 4}[args.backend]) == 1,
            f"Requested backend unavailable: {args.backend}: {error_message(lib)}")
    # Unchanged upstream ggml-metal/kernels/mul_mm.metal uses half operand tiles
    # for kernel_mul_mm_f32_f32 with float accumulation. Metal measured up to
    # .00238 absolute error on the deliberately large-logit case vs the FP64
    # oracle; the real 768/1152-channel cases measured <=.00103. An initial
    # 3e-4 check failed; keep that finding explicit instead of claiming FP32
    # parity. Also require relative L2 <=.002 independently of this max bound.
    atol, rtol = (3e-3, 0.0) if args.backend == "metal" else (3e-5, 3e-5)
    try:
        rows = run_cases(lib, atol, rtol)
        rejected = invalid_arguments(lib)
    finally:
        lib.TSGgml_ReleaseReuseComputeBuffers()
        lib.TSGgml_ClearHostBufferCache()
    report = {"backend": args.backend, "passed": len(rows), "rejected_invalid_arguments": rejected,
              "absolute_tolerance": atol, "relative_tolerance": rtol,
              "relative_l2_tolerance": 0.002,
              "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "ggml_revision": git(args.ggml_source, "rev-parse", "HEAD"),
              "ggml_worktree_status": git(args.ggml_source, "status", "--porcelain"),
              "precision_note": "Metal wide matmul uses reduced-precision operand tiles with F32 accumulation. Initial 3e-4 pointwise tolerance failed; final bound is 3e-3 absolute plus 0.002 relative L2 against FP64. This is not bit-exact FP32 validation.",
              "cases": rows,
              "limitations": "Synthetic operator tests only; no model weights, full-resolution generation, quality evaluation or speed benchmark."}
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
