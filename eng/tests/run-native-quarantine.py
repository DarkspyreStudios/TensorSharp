#!/usr/bin/env python3
"""Fresh-process managed authority checks; never builds or loads a native backend."""

import argparse
import json
import os
from pathlib import Path
import subprocess


MODES = (
    "healthy", "promotion", "wildcard", "mlx", "lease-negatives", "race",
    "causes", "constructor", "initialize", "concurrent-faults", "queued",
    "protocol", "malformed-cells", "roles", "cross-nesting", "lock-order", "mismatched-lease",
    "resolved-cuda-nesting",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("Debug", "Release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = root / "eng/tests/native-quarantine/native-quarantine.csproj"
    env = dict(os.environ, TENSORSHARP_GGML_NATIVE_SKIP="true", TENSORSHARP_MLX_NATIVE_SKIP="true")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in MODES:
        command = [
            "dotnet", "run", "--project", str(project), "-c", args.configuration,
            "--no-build", "--no-restore", "-p:TensorSharpSkipGgmlNative=true",
            "-p:TensorSharpSkipMlxNative=true", "-p:TensorSharpSkipCudaNative=true",
            "-p:GeneratePackageOnBuild=false", "-p:PublishAot=false", "--", mode,
        ]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=20)
        (args.output / f"{mode}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{mode} failed ({result.returncode}); see {args.output / (mode + '.log')}")
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report["command"] = command
        reports.append(report)
    (args.output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"{args.configuration}: {len(reports)} fresh-process managed-only modes passed")


if __name__ == "__main__":
    main()
