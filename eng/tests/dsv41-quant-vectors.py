#!/usr/bin/env python3
"""Hold the managed V4.1 cache quantization to the PyTorch reference.

Writes a vector file, runs the C# harness test over it, and compares each mode
against eng/dsv41-reference.py's own cache() implementation. Run it as:

    python3 eng/tests/dsv41-quant-vectors.py --repo . --work /tmp/dsv41q
"""
import argparse, importlib.util, subprocess, sys
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path("."))
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--count", type=int, default=1 << 16)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("dsv41_reference", args.repo / "eng" / "dsv41-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

rng = np.random.default_rng(20260911)
# Mix scales and include exact FP4/FP8 bin midpoints, which is where the two
# implementations can disagree without disagreeing anywhere else.
values = np.concatenate([
    rng.normal(0, 1, args.count // 2),
    rng.normal(0, 1e-3, args.count // 4),
    rng.choice([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 448.0, 0.0], args.count // 8),
    rng.normal(0, 40, args.count // 8),
]).astype("<f4")
path = args.work / "vectors.f32"
values.tofile(path)

env = {"TS_DSV41_QUANT_VECTORS": str(path)}
run = subprocess.run([
    "dotnet", "test", str(args.repo / "InferenceWeb.Tests" / "InferenceWeb.Tests.csproj"),
    "-c", "Release", "--nologo", "--filter", "QuantizesTheHarnessVectorsForEveryMode"],
    capture_output=True, text=True, env={**__import__("os").environ, **env})
if run.returncode != 0:
    sys.exit(run.stdout[-3000:] + run.stderr[-2000:])

kinds = {0: "raw", 1: "index", 2: "compressed"}
failures = 0
for mode, kind in kinds.items():
    actual = np.fromfile(f"{path}.cs.mode{mode}.f32", dtype="<f4")
    model = reference.Reference.__new__(reference.Reference)
    model.cache_type = "model"
    expected = model.cache(torch.from_numpy(values.copy()).reshape(1, -1), kind).numpy().reshape(-1)
    bad = np.nonzero(actual != expected)[0]
    print(f"mode {mode} ({kind}): {len(bad)} of {len(values)} values differ")
    for i in bad[:5]:
        print(f"   [{i}] input={values[i]!r} managed={actual[i]!r} reference={expected[i]!r}")
    failures += len(bad)
sys.exit(1 if failures else 0)
