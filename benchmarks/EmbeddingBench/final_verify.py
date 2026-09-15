#!/usr/bin/env python3
"""Run the final Release test lanes and native-free embedding/context proof.

Run from the repository root after performance validation and source freeze.
This runner stops before tests if an incremental build changes any production
host binary, so the recorded performance cannot silently refer to older code.
Earlier JSON/TRX evidence is retained with a before-final-verification suffix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


PROJECT = "InferenceWeb.Tests/InferenceWeb.Tests.csproj"
HOST = Path("TensorSharp.Server.Host/bin")
TEST_BIN = Path("InferenceWeb.Tests/bin/Release/net10.0")
TEST_RESULTS = Path("InferenceWeb.Tests/TestResults")
UNIT_FILTER = "Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx"
FOCUSED_CLASSES = ["EmbeddingModelTests", "EmbeddingHostingTests", "EmbeddingEndpointTests",
                   "EmbeddingTokenizerTests", "ManagedQuantizedOpsTests", "ManagedEmbeddingQ8MatrixTests",
                   "CpuWorkerPoolTests", "ManagedEmbeddingMathTests", "NativeLongEmbeddingTests",
                   "EmbeddingRequestDispatcherTests"]
FOCUSED_FILTER = "|".join("FullyQualifiedName~" + name for name in FOCUSED_CLASSES)
JSON_NAMES = ["unit-tests.json", "managed-focused-tests.json", "managed-native-free.json",
              "managed-context-boundaries.json", "final-verification.json"]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve(path):
    path = Path(path)
    if not path.exists():
        return None
    suffix = "-before-final-verification"
    backup = path.with_name(path.stem + suffix + path.suffix)
    attempt = 2
    while backup.exists():
        backup = path.with_name(path.stem + suffix + "-" + str(attempt) + path.suffix)
        attempt += 1
    shutil.copy2(path, backup)
    return str(backup)


def binaries():
    paths = set(HOST.glob("TensorSharp*.dll"))
    paths.update(HOST.glob("*GgmlOps*"))
    paths.update(HOST.glob("*.dylib"))
    paths.update(HOST.glob("*.so*"))
    return {path.name: dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
            for path in sorted(paths) if path.is_file()}


def read_trx(path):
    ns = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
    root = ET.parse(path).getroot()
    counters = root.find(".//t:Counters", ns)
    tests = root.findall(".//t:UnitTestResult", ns)
    failed = [dict(name=test.attrib["testName"], outcome=test.attrib["outcome"],
                   message=test.findtext(".//t:Message", default="", namespaces=ns))
              for test in tests if test.attrib["outcome"] not in ("Passed", "NotExecuted")]
    hosting = sum(any("." + name + "." in test.attrib["testName"]
                      for name in ("EmbeddingEndpointTests", "EmbeddingHostingTests", "EmbeddingRequestDispatcherTests"))
                  for test in tests)
    return dict(counts={key: int(value) for key, value in counters.attrib.items()}, failed_tests=failed,
                run_times=root.find("t:Times", ns).attrib, embedding_http_and_hosting_tests=hosting)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minilm-model", type=Path, required=True)
    parser.add_argument("--snowflake-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/embeddings-2026-09"))
    parser.add_argument("--native-free-copy", type=Path, default=Path("/tmp/tensorsharp-embedding-managed-host"))
    parser.add_argument("--proof-port", type=int, default=18383)
    args = parser.parse_args()
    if not HOST.joinpath("TensorSharp.Server.Host.dll").is_file():
        parser.error("Run from the repository root with the final Release Host already built")
    models = {"minilm": args.minilm_model.resolve(), "snowflake": args.snowflake_model.resolve()}
    if any(not path.is_file() for path in models.values()):
        parser.error("Both GGUF models must exist")
    args.output.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS.mkdir(parents=True, exist_ok=True)
    prior_unit = json.loads((args.output / "unit-tests.json").read_text()) if (args.output / "unit-tests.json").exists() else {}
    prior_focus = json.loads((args.output / "managed-focused-tests.json").read_text()) if (args.output / "managed-focused-tests.json").exists() else {}
    preserved = [backup for name in JSON_NAMES if (backup := preserve(args.output / name))]
    destination = args.output / "final-verification.json"
    report = dict(started=utc(), command=[sys.executable, *sys.argv], preserved_evidence=preserved,
                  status="running", stages=[], production_binaries_before=binaries(),
                  models={name: dict(path=str(path), sha256=sha256(path)) for name, path in models.items()})

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, overrides=None, timeout=900):
        log = TEST_RESULTS / ("final-verification-" + name + ".log")
        stage = dict(name=name, command=command, environment_overrides=overrides or {},
                     started=utc(), log=str(log))
        report["stages"].append(stage)
        save()
        print(name, "started", flush=True)
        with log.open("w") as output:
            completed = subprocess.run(command, env=dict(os.environ, **(overrides or {})),
                                       stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        stage.update(finished=utc(), exit_code=completed.returncode)
        save()
        return completed.returncode

    def check_binaries(stage):
        current = binaries()
        before = report["production_binaries_before"]
        changed = {name: dict(before=before.get(name), after=current.get(name))
                   for name in before.keys() | current.keys() if before.get(name) != current.get(name)}
        report["production_binaries_after"] = current
        if changed:
            report["unexpected_production_binary_changes"] = dict(stage=stage, binaries=changed)
            save()
            raise RuntimeError("Production binaries changed; stop and notify the benchmark owner before proceeding")
        # The correctness lane must execute the same assemblies/native engine
        # as the already benchmarked Host, even when the build was incremental.
        mismatches = [name for name, artifact in current.items()
                      if (TEST_BIN / name).exists() and sha256(TEST_BIN / name) != artifact["sha256"]]
        if mismatches:
            report["test_host_binary_mismatches"] = mismatches
            save()
            raise RuntimeError("Test/Host binary hashes differ: " + ", ".join(mismatches))
        report["production_hashes_unchanged_through"] = stage
        save()

    try:
        anchor_path = args.output / "dispatcher-focused-tests.json"
        if anchor_path.exists():
            anchor = json.loads(anchor_path.read_text())["host_binaries"]
            mismatches = [name for name, item in anchor.items()
                          if report["production_binaries_before"].get(name, {}).get("sha256") != item["sha256"]]
            report["validated_binary_anchor"] = dict(path=str(anchor_path), sha256=sha256(anchor_path), mismatches=mismatches)
            if mismatches:
                raise RuntimeError("Host differs from the frozen dispatcher build: " + ", ".join(mismatches))
        build = ["dotnet", "build", PROJECT, "-c", "Release", "--no-restore",
                 "-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true", "-clp:ErrorsOnly"]
        if run("build", build) != 0:
            raise RuntimeError("Final Release test build failed")
        check_binaries("incremental Release build")
        specifications = [
            ("unit", UNIT_FILTER, "embedding-final.trx", "unit-tests.json", prior_unit, {}),
            ("focused", FOCUSED_FILTER, "embedding-managed-focused.trx", "managed-focused-tests.json", prior_focus,
             {"TS_TEST_EMBEDDING_BACKEND": "GGML_CPU", "TENSORSHARP_MINILM_EMBED_MODEL": str(models["minilm"]),
              "TENSORSHARP_SNOWFLAKE_EMBED_MODEL": str(models["snowflake"])})]
        for name, test_filter, trx_name, json_name, previous, environment in specifications:
            trx = TEST_RESULTS / trx_name
            backup = preserve(trx)
            if backup:
                report["preserved_evidence"].append(backup)
                trx.unlink()
            command = ["dotnet", "test", PROJECT, "-c", "Release", "--no-build", "--no-restore",
                       "--filter", test_filter, "--logger", "trx;LogFileName=" + trx_name,
                       "--results-directory", str(TEST_RESULTS)]
            code = run(name, command, environment)
            summary = dict(previous)
            if trx.exists():
                summary.update(read_trx(trx))
            summary.update(configuration="Release", result="passed" if code == 0 else "failed",
                           filter=test_filter, command=command, test_command=command, build_command=build,
                           environment=environment, trx=str(trx), test_assembly_sha256=sha256(TEST_BIN / "InferenceWeb.Tests.dll"),
                           managed_assembly_sha256={key: value["sha256"] for key, value in report["production_binaries_before"].items()
                                                    if key.endswith(".dll")},
                           production_binaries=report["production_binaries_before"], final_verification_artifact=str(destination))
            if name == "focused":
                summary["suite"] = "Complete embedding/model/kernel/pool/server suite, including real GGUF models and queued-request batching"
                summary["focused_classes"] = FOCUSED_CLASSES
            (args.output / json_name).write_text(json.dumps(summary, indent=2) + "\n")
            report["stages"][-1]["summary"] = str(args.output / json_name)
            save()
            if code != 0:
                raise RuntimeError(name + " correctness tests failed; inspect the saved TRX and log")
            print(name, summary["counts"], flush=True)
            check_binaries(name + " correctness lane")
        focused = json.loads((args.output / "managed-focused-tests.json").read_text())
        unit_path = args.output / "unit-tests.json"
        unit = json.loads(unit_path.read_text())
        unit["final_focused_suite"] = dict(artifact="managed-focused-tests.json", counts=focused["counts"], includes_real_models=True)
        unit["model_and_performance_validation"] = "See the final focused model/kernel/pool suite, managed-native-free.json, managed-context-boundaries.json, and final HTTP/concurrency benchmark artifacts."
        unit_path.write_text(json.dumps(unit, indent=2) + "\n")
        native_free = args.output / "managed-native-free.json"
        contexts = args.output / "managed-context-boundaries.json"
        command = [sys.executable, "benchmarks/EmbeddingBench/native_free_smoke.py", "--source", str(HOST),
                   "--copy", str(args.native_free_copy), "--model", str(models["minilm"]),
                   "--model", str(models["snowflake"]), "--port", str(args.proof_port), "--threads", "8",
                   "--output", str(native_free), "--context-output", str(contexts)]
        if run("native-free-and-contexts", command, timeout=1200) != 0:
            raise RuntimeError("Native-free API/full-context proof failed; inspect its saved log")
        for path in (native_free, contexts):
            proof = json.loads(path.read_text())
            proof["model_files"] = report["models"]
            proof["production_binaries"] = report["production_binaries_before"]
            proof["final_verification_artifact"] = str(destination)
            proof["verified_at"] = utc()
            path.write_text(json.dumps(proof, indent=2) + "\n")
        check_binaries("native-free API and full-context proofs")
        report.update(status="passed", finished=utc(), owned_proof_hosts_stopped=True)
        print("Final verification passed; production binary hashes are unchanged.", flush=True)
    except BaseException as error:
        report.update(status="failed", finished=utc(), error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
