#!/usr/bin/env python3
"""Compare native V4.1 inference with the independent PyTorch fixture oracle.

Generate the fixture with eng/dsv41-fixture.py DIRECTORY --f32, then run this
script DIRECTORY --library /absolute/path/to/libGgmlOps.so. CPU checks use
strict tolerances; GPU tolerances can be explicitly selected if necessary.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu-moe", type=int, default=0, help="Number of initial routed-MoE layers placed on CPU")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--report", type=Path, help="Write a distinct report when comparing native precision modes")
    args = parser.parse_args()
    config = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not config.get("fixture"):
        raise ValueError("This test is for the small deterministic fixture, not downloaded model weights")
    spec = importlib.util.spec_from_file_location("dsv41_reference", Path(__file__).parents[1] / "dsv41-reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    weights = reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
    engram = reference.load_engram(args.fixture_dir / "deepseek41.engram.bin")
    tokens = np.array(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    other = np.array([0, 9, 21, 85, 11, 19, 6, 44, 35, 72, 11], dtype=np.int32)
    def oracle(ids):
        model = reference.Reference(weights, config["config"], engram, "model")
        return model.forward(ids.tolist()).cpu().numpy()
    expected, expected_other = oracle(tokens), oracle(other)
    lib = ctypes.CDLL(str(args.library.resolve()))
    signatures = {
        "LoadModel": ([ctypes.c_char_p] + [ctypes.c_int] * 5 + [ctypes.c_char_p], ctypes.c_void_p),
        "Forward": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
        "Reset": ([ctypes.c_void_p], None), "Free": ([ctypes.c_void_p], None),
        "SlotAlloc": ([ctypes.c_void_p], ctypes.c_int),
        "SetActiveSlot": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "SlotFree": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "NPast": ([ctypes.c_void_p], ctypes.c_int),
        "Rewind": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    }
    api = {}
    for name, (parameters, result) in signatures.items():
        function = getattr(lib, "TSGgml_Dsv4" + name)
        function.argtypes, function.restype = parameters, result
        api[name] = function
    handle = api["LoadModel"](str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                                args.gpus, 256, 32, 2, args.cpu_moe, args.backend.encode())
    if not handle:
        raise RuntimeError("Native fixture model load failed")
    checks = []
    def forward(ids):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty(config["config"]["text_config"]["vocab_size"], dtype=np.float32)
        result = api["Forward"](handle, ids.ctypes.data, len(ids), output.ctypes.data)
        if result:
            raise RuntimeError(f"Native forward returned {result}")
        return output
    def compare(name, output, target):
        checks.append(dict(name=name, max_absolute_error=float(np.max(np.abs(output - target))),
                           relative_l2=float(np.linalg.norm(output - target) / np.linalg.norm(target)),
                           argmax=int(output.argmax()), reference_argmax=int(target.argmax()),
                           passed=bool(np.allclose(output, target, atol=args.atol, rtol=args.rtol))))
    try:
        for chunk in (len(tokens), 1, 3, 5):
            api["Reset"](handle)
            for start in range(0, len(tokens), chunk):
                stop = min(start + chunk, len(tokens))
                compare(f"chunk_{chunk}_position_{stop}", forward(tokens[start:stop]), expected[stop - 1])
                assert api["NPast"](handle) == stop
        # Reset must clear Engram token history as well as raw/compressed KV.
        api["Reset"](handle)
        compare("reset_other_prompt", forward(other), expected_other[-1])
        # V4.1's rolling compressor state cannot be restored by moving only
        # n_past. Rejected rewind must preserve both position and continuation.
        api["Reset"](handle)
        forward(tokens[:4])
        assert api["Rewind"](handle, 1) == 0, "V4.1 rewind must be rejected"
        assert api["NPast"](handle) == 4, "Rejected rewind changed position"
        compare("rejected_rewind_continuation", forward(tokens[4:8]), expected[7])
        assert api["NPast"](handle) == 8
        api["Reset"](handle)
        slot = api["SlotAlloc"](handle)
        assert slot > 0
        positions = [0, 0]
        while positions[0] < len(tokens) or positions[1] < len(other):
            for stream, (slot_id, prompt, target, chunk) in enumerate(((0, tokens, expected, 3), (slot, other, expected_other, 2))):
                start = positions[stream]
                if start == len(prompt):
                    continue
                assert api["SetActiveSlot"](handle, slot_id) == 0
                stop = min(start + chunk, len(prompt))
                compare(f"interleaved_slot_{stream}_position_{stop}", forward(prompt[start:stop]), target[stop - 1])
                assert api["NPast"](handle) == stop
                positions[stream] = stop
        assert api["SetActiveSlot"](handle, 0) == 0
        assert api["SlotFree"](handle, slot) == 0
    finally:
        api["Free"](handle)
    environment = {name: os.environ.get(name) for name in ("TS_DSV4_FA", "TS_DSV4_GATHER", "TS_DSV41_TP", "TS_DSV41_SPARSE_FA",
                  "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_WARM", "NVIDIA_TF32_OVERRIDE")}
    result = dict(backend=args.backend, gpus=args.gpus, cpu_moe=args.cpu_moe, atol=args.atol, rtol=args.rtol,
                  environment=environment, checks=checks)
    path = args.report or args.fixture_dir / f"validation-{args.backend.lower()}-{args.gpus}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    passed = sum(check["passed"] for check in checks)
    print(f"Passed {passed}/{len(checks)} native/reference checks; {path}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
