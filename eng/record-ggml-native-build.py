#!/usr/bin/env python3
"""Record verified native build provenance without loading or executing the bridge."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ggml_packer", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
CUDA_ARCHITECTURES = pack.CUDA_ARCHITECTURES
INPUTS = ["TensorSharp.GGML.Native", "TensorSharp.Backends.GGML", "eng/GgmlNativeIdentity.cmake",
          "eng/GgmlNativeIdentity.targets", "eng/build-ggml-natives.sh", "eng/record-ggml-native-build.py",
          "eng/pack-ggml-natives.py", "eng/native-artifact-manifest.py", "eng/guard-ggml-interop",
          "eng/ggml-required-exports.json", "eng/relink-ggml-native-identity.py", "eng/ggml-revision", "Directory.Build.props", "LICENSE"]


def command(arguments):
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def read_cache(path):
    return pack.read_cmake_cache(pack.read_build_file(path))


def verify_clean_source(root, source):
    if command(["git", "-C", str(root), "rev-parse", "HEAD"]) != source:
        raise ValueError("TensorSharp HEAD differs from the recorded source commit")
    if command(["git", "-C", str(root), "status", "--porcelain", "--", *INPUTS]):
        raise ValueError("native source, managed ABI or build inputs changed during the release build")
    upstream = root / "ExternalProjects/ggml"
    pinned = (root / "eng/ggml-revision").read_text(encoding="utf-8").strip()
    if command(["git", "-C", str(upstream), "rev-parse", "HEAD"]) != pinned:
        raise ValueError("ggml HEAD differs from the pinned revision")
    if command(["git", "-C", str(upstream), "status", "--porcelain"]):
        raise ValueError("the pinned ggml source has local changes")
    return pinned


def stage_components(root, binary, build, redist_dir=None, redist_manifest=None):
    directory = binary.parent
    components, contents = [], {}
    for expected in pack.core_component_specs(build, binary.name):
        source = root / ("LICENSE" if expected["id"] == "tensorsharp" else "ExternalProjects/ggml/LICENSE")
        text = pack.read_build_file(source)
        if not text.strip():
            raise ValueError("source license is empty: " + str(source))
        data = source.read_bytes()
        evidence = {"path": expected["evidencePaths"][0], "size": len(data), "sha256": pack.sha256_bytes(data)}
        components.append({key: value for key, value in expected.items() if key not in ("binaryPaths", "evidencePaths")} | {
            "binaryFiles": [{"path": binary.name, "size": binary.stat().st_size, "sha256": build["bridgeSha256"]}],
            "evidenceFiles": [evidence]})
        contents[evidence["path"]] = (source, evidence)
    if (redist_dir is None) != (redist_manifest is None):
        raise ValueError("--redist-dir requires an explicit --redist-manifest and vice versa")
    if redist_dir is not None:
        redist_dir, redist_manifest = redist_dir.absolute(), redist_manifest.absolute()
        manifest = json.loads(pack.read_build_file(redist_manifest.absolute()))
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "components"}
                or manifest["schema"] != "tensorsharp-native-redistribution/1"):
            raise ValueError("malformed redistribution mapping")
        supplied = manifest["components"]
        pack.validate_component_shape(supplied)
        expected_files = set()
        for component in supplied:
            if component["id"] in ("tensorsharp", "ggml") or component["kind"] != "redistributed":
                raise ValueError("redistribution mapping cannot override core components")
            for reference in component["binaryFiles"] + component["evidenceFiles"]:
                name = reference["path"]
                if name in {binary.name, *pack.REQUIRED_LICENSES}:
                    raise ValueError("redistribution mapping cannot overwrite bridge or core licenses")
                pack.verify_component_file(redist_dir, reference)
                if name in contents and contents[name][1] != reference:
                    raise ValueError("conflicting supplied component evidence: " + name)
                contents[name] = (redist_dir / name, reference)
                expected_files.add(name)
        actual_files = {path.relative_to(redist_dir).as_posix() for path in pack.regular_files(redist_dir.absolute())}
        try:
            actual_files.discard(redist_manifest.absolute().relative_to(redist_dir.absolute()).as_posix())
        except ValueError:
            pass
        if actual_files != expected_files:
            raise ValueError("unmapped redistribution files: " + ", ".join(sorted(actual_files - expected_files)))
        components += supplied
    pack.validate_component_shape(components)
    for name, (source, reference) in contents.items():
        destination = directory / name
        pack.require_unlinked(destination.absolute())
        if destination.exists():
            raise ValueError("component staging refuses existing evidence or runtime file: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        pack.verify_component_file(directory, reference)
    files = [pack.file_record(directory, path) for path in pack.regular_files(directory.absolute())]
    described = [pack.INVENTORY.describe(item["path"], build["rid"], build["variant"], directory / item["path"]) for item in files]
    validated = pack.check_components(directory, build | {"components": components}, files, described, binary.name)
    errors = pack.check_artifact(build["rid"], build["variant"], directory, {"files": files}, described)
    if errors:
        raise ValueError("; ".join(errors))
    return validated


def create_record(root, build_dir, binary, rid, variant, source, redist_dir=None, redist_manifest=None):
    binary = binary.absolute()
    if rid not in pack.VARIANTS or variant not in pack.VARIANTS[rid]:
        raise ValueError("unsupported release RID/variant")
    pack.require_unlinked(binary.absolute())
    if not binary.is_file() or binary.name != pack.ENTRY[rid.split("-")[0]]:
        raise ValueError("the release bridge must be an ordinary RID-specific entry library")
    if not re.fullmatch(r"[a-f0-9]{40}", source):
        raise ValueError("the source commit must be an exact Git identity")
    pinned = verify_clean_source(root, source)
    version = ET.parse(root / "Directory.Build.props").findtext(".//TensorSharpVersion")
    abi = pack.native_abi(root)
    cache_path = build_dir / "CMakeCache.txt"
    settings = read_cache(cache_path)
    identity = pack.read_identity(binary)
    expected = {"format": "1", "tensorsharp": version, "source": source, "ggml": pinned,
                "rid": rid, "variant": variant, "abi": abi}
    if not identity or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("the binary build identity differs from the release source, ABI or target")
    deployment = pack.validate_release_profile(settings, identity)
    facts = pack.INVENTORY.describe(binary.name, rid, variant, binary)
    os_part, arch = rid.split("-")
    if (facts.get("format") != pack.INVENTORY.RID_FORMAT[os_part]
            or facts.get("identity", {}).get("arch") != pack.INVENTORY.RID_ARCH[arch]
            or facts.get("tsggmlBuildIdentityExport") is not True):
        raise ValueError("the bridge header, architecture or identity export does not match the release")
    compiler = settings.get("CMAKE_CXX_COMPILER")
    if not compiler:
        raise ValueError("the CMake cache has no C++ compiler")
    record = {"tensorSharpBuild": version, "sourceCommit": source, "ggmlCommit": pinned, "nativeAbi": abi,
              "rid": rid, "variant": variant, "cpuProfile": identity["cpu"], "macosDeploymentTarget": deployment,
              "host": platform.platform(), "compiler": compiler, "compilerVersion": command([compiler, "--version"]),
              "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "bridgeSha256": pack.sha256_bytes(binary.read_bytes()),
              "cmakeCacheSha256": pack.sha256_bytes(cache_path.read_bytes()),
              "cmakeConfiguration": settings,
              "qualification": {"targetExecution": "not-recorded", "gpuExecution": "not-recorded"}}
    if variant == "cuda13":
        compiler = settings.get("CMAKE_CUDA_COMPILER")
        compiler_version = command([compiler, "--version"])
        record["cuda"] = pack.cuda_profile_evidence(settings, compiler_version)
    pack.validate_recorded_profile(record, identity, settings, pack.sha256_bytes(binary.read_bytes()))
    record["components"] = stage_components(root, binary, record, redist_dir, redist_manifest)
    verify_clean_source(root, source)
    return record


def write_build_record(out, record, cache_path):
    # Preserve the observed bytes as evidence, never as a future CMake input.
    snapshot = pack.observed_cache_bytes(cache_path, record)
    settings = record["cmakeConfiguration"]
    normalized = "".join(f"{key}={settings[key]}\n" for key in sorted(settings))
    if pack.read_cmake_settings(normalized) != settings:
        raise ValueError("the observed cache cannot be represented by normalized CMake settings")
    pack.require_unlinked(out.absolute())
    out.mkdir(parents=True, exist_ok=True)
    for name in ("build-identity.json", "cmake-settings.txt", pack.CMAKE_CACHE_SNAPSHOT):
        pack.require_unlinked(out / name)
        if (out / name).exists() and not (out / name).is_file():
            raise ValueError("build record output must be an ordinary file: " + name)
    (out / pack.CMAKE_CACHE_SNAPSHOT).write_bytes(snapshot)
    (out / "cmake-settings.txt").write_text(normalized, encoding="utf-8")
    (out / "build-identity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rid", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--redist-dir", type=Path)
    parser.add_argument("--redist-manifest", type=Path,
                        help="explicit component, binary and license/notice mapping with exact input hashes")
    args = parser.parse_args()
    try:
        record = create_record(ROOT, args.build_dir, args.binary, args.rid, args.variant, args.source_commit,
                               args.redist_dir, args.redist_manifest)
        write_build_record(args.out, record, args.build_dir / "CMakeCache.txt")
        print("Recorded verified build identity; target/GPU execution is not asserted: " + str(args.out))
        return 0
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError, ET.ParseError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
