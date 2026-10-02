#!/usr/bin/env python3
"""Run fresh-process actual managed CUDA classes with injected calls, never native execution."""

import argparse
import json
import os
from pathlib import Path
import subprocess


MODES = (
    "reference-overflow", "reference-gate-admission", "reference-gate-refusal",
    "reference-parallel-release", "reference-tensor-intents",
    "context-clean", "context-release-refusal", "context-drain-refusal",
    "context-construction-rollback", "context-construction-rollback-refusal",
    "foreign-context-clean", "foreign-context-release-refusal",
    "stream-clean", "stream-ambient-clean", "stream-sync-refusal", "stream-destroy-refusal",
    "stream-construction-rollback", "stream-construction-rollback-refusal",
    "foreign-stream-clean", "foreign-stream-sync-refusal",
    "module-clean", "module-drain-refusal", "module-unload-refusal",
    "module-construction-rollback", "module-construction-rollback-refusal",
    "foreign-module-clean", "foreign-module-drain-refusal",
    "context-composed-clean", "context-composed-external", "context-composed-drain-refusal",
    "context-composed-restore-refusal", "context-composed-thread-refusal",
    "foreign-composed-clean", "foreign-composed-drain-refusal",
    "context-independent-postfault-release",
    "blas-clean", "blas-drain-refusal", "blas-destroy-refusal",
    "blas-construction-rollback", "blas-construction-rollback-refusal",
    "foreign-blas-clean", "foreign-blas-drain-refusal",
    "kernels-construction-rollback", "kernels-construction-rollback-refusal",
    "kernels-clean", "kernels-drain-refusal", "kernels-free-refusal", "kernels-resize-refusal",
    "kernels-allocation-rollback", "kernels-allocation-rollback-refusal", "kernels-ordinary-faults",
    "foreign-kernels-clean", "foreign-kernels-drain-refusal",
    "pool-small-failed-free-retention", "pool-large-failed-free-retention",
    "pool-small-partial-free-retention", "pool-large-partial-free-retention",
    "kernels-diagnostic-outside-effects",
    "dyn-clean", "dyn-ambient-stream-clean", "dyn-drain-refusal", "dyn-device-free-refusal", "dyn-host-free-refusal",
    "dyn-construction-rollback", "dyn-construction-rollback-refusal",
    "dyn-device-allocation-fallback", "dyn-host-allocation-fallback", "dyn-context-mismatch",
    "dyn-allocation-fallback-refusal", "dyn-retired-stream",
    "dyn-consumer-disposal", "dyn-consumer-mismatch",
    "dyn-ordinary-upload-fault", "dyn-nested-release-refusal", "foreign-dyn-clean", "foreign-dyn-drain-refusal",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("Debug", "Release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    binary = root / "eng/tests/cuda-quarantine/bin" / args.configuration / "net10.0/TensorSharp.CudaQuarantineFixture.dll"
    env = dict(os.environ, TENSORSHARP_GGML_NATIVE_SKIP="true", TENSORSHARP_MLX_NATIVE_SKIP="true",
               TensorSharpSkipGgmlNative="true", TensorSharpSkipMlxNative="true", TensorSharpSkipCudaNative="true")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in MODES:
        command = ["dotnet", str(binary), mode]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=20)
        (args.output / f"{mode}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{mode} failed ({result.returncode}); see {args.output / (mode + '.log')}")
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report["command"] = command
        reports.append(report)
    (args.output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"{args.configuration}: {len(reports)} fresh-process actual managed CUDA modes passed (native execution false)")


if __name__ == "__main__":
    main()
