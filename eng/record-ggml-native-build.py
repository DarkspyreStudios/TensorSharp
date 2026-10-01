#!/usr/bin/env python3
"""Record verified native build provenance without loading or executing the bridge."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ggml_packer", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
CUDA_ARCHITECTURES = "75-real;80-real;86-real;89-real;120-real;120-virtual"
INPUTS = ["TensorSharp.GGML.Native", "TensorSharp.Backends.GGML", "eng/GgmlNativeIdentity.cmake",
          "eng/GgmlNativeIdentity.targets", "eng/build-ggml-natives.sh", "eng/record-ggml-native-build.py",
          "eng/pack-ggml-natives.py", "eng/native-artifact-manifest.py", "eng/ggml-revision", "Directory.Build.props"]


def command(arguments):
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def read_cache(path):
    settings = {}
    for line in pack.read_build_file(path).splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        match = re.fullmatch(r"([^:=]+):([A-Z]+)=(.*)", line)
        if not match or match[1] in settings:
            raise ValueError("malformed or duplicate CMake cache entry: " + line)
        settings[match[1]] = match[3]
    return settings


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


def create_record(root, build_dir, binary, rid, variant, source):
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
    for key, value in (("TENSORSHARP_NATIVE_ABI", abi), ("TENSORSHARP_NATIVE_RID", rid),
                       ("TENSORSHARP_NATIVE_VARIANT", variant), ("TENSORSHARP_GGML_NATIVE_PORTABLE", "ON"),
                       ("GGML_NATIVE", "OFF")):
        if settings.get(key) != value:
            raise ValueError("the CMake cache differs from the release profile: " + key)
    profiles = {"osx-arm64": "apple-m1", "linux-x64": "x86-64", "win-x64": "x86-64",
                "linux-arm64": "armv8.2-a+dotprod", "win-arm64": "armv8.2-a+dotprod"}
    if identity.get("cpu") != profiles[rid]:
        raise ValueError("the binary CPU profile differs from the portable release floor")
    for backend in ("metal", "cuda13", "vulkan"):
        flag = "GGML_" + ("CUDA" if backend == "cuda13" else backend.upper())
        if settings.get(flag) != ("ON" if variant == backend else "OFF"):
            raise ValueError("the CMake backend differs from the declared variant: " + flag)
    if rid.startswith("linux-"):
        if settings.get("CMAKE_INSTALL_RPATH") != "$ORIGIN" or settings.get("CMAKE_BUILD_WITH_INSTALL_RPATH") != "ON":
            raise ValueError("Linux release libraries require the selected directory's $ORIGIN runpath")
    deployment = settings.get("CMAKE_OSX_DEPLOYMENT_TARGET") if rid.startswith("osx-") else None
    if rid.startswith("osx-") and (not deployment or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", deployment)):
        raise ValueError("macOS release libraries require an explicit deployment target")
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
        if settings.get("CMAKE_CUDA_ARCHITECTURES") != CUDA_ARCHITECTURES:
            raise ValueError("the full release CUDA SASS/PTX profile is required")
        compiler = settings.get("CMAKE_CUDA_COMPILER")
        if not compiler:
            raise ValueError("the CMake cache has no CUDA compiler")
        compiler_version = command([compiler, "--version"])
        match = re.search(r"\brelease ([0-9]+)\.([0-9]+)\b", compiler_version)
        if not match or match[1] != "13":
            raise ValueError("the cuda13 variant requires an observed CUDA 13 compiler")
        record["cuda"] = {"compiler": compiler, "compilerVersion": compiler_version,
                          "toolkitVersion": match[1] + "." + match[2],
                          "architectures": CUDA_ARCHITECTURES.split(";")}
    verify_clean_source(root, source)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rid", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = create_record(ROOT, args.build_dir, args.binary, args.rid, args.variant, args.source_commit)
        pack.require_unlinked(args.out.absolute())
        args.out.mkdir(parents=True, exist_ok=True)
        for name in ("build-identity.json", "cmake-settings.txt"):
            pack.require_unlinked(args.out / name)
        (args.out / "build-identity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        settings = record["cmakeConfiguration"]
        (args.out / "cmake-settings.txt").write_text("".join(f"{key}={settings[key]}\n" for key in sorted(settings)), encoding="utf-8")
        print("Recorded verified build identity; target/GPU execution is not asserted: " + str(args.out))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
