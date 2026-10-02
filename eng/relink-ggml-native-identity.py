#!/usr/bin/env python3
"""Rebuild only the Mac bridge identity object, preserving all other link inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ggml_record", ROOT / "eng/record-ggml-native-build.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)
pack = record.pack


def copy_evidence(source, directory, name):
    if not isinstance(name, str) or not pack.portable_path(name) or not name.startswith("identity-relink/"):
        raise ValueError("relink evidence requires an allowlisted portable relative path")
    source = source.absolute()
    pack.require_unlinked(source)
    if not source.is_file():
        raise ValueError("relink evidence must be an ordinary file: " + str(source))
    digest = pack.sha256_bytes(source.read_bytes())
    destination = directory / name
    pack.require_unlinked(destination.absolute())
    if destination.exists():
        raise ValueError("relink evidence output already exists: " + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    reference = {"path": name, "size": destination.stat().st_size, "sha256": digest}
    pack.verify_component_file(directory, reference)
    if pack.sha256_bytes(source.read_bytes()) != digest:
        raise ValueError("original relink evidence changed while copying")
    return reference


def relink(fixture, original_record_path, build_dir, stage):
    fixture, original_record_path, build_dir, stage = [path.absolute() for path in (fixture, original_record_path, build_dir, stage)]
    source = record.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    ggml = record.verify_clean_source(ROOT, source)
    original_record = json.loads(pack.read_build_file(original_record_path))
    old_bridge = fixture / "libGgmlOps.dylib"
    original_identity = pack.read_identity(old_bridge)
    if not original_identity:
        raise ValueError("original bridge has no exact identity")
    native_tree = pack.native_source_tree(source)
    if native_tree != pack.native_source_tree(original_identity["source"]):
        raise ValueError("native implementation sources differ from the preserved bridge")
    original_cache = fixture / "CMakeCache.txt"
    pack.observed_cache_bytes(original_cache, original_record)
    pack.validate_recorded_profile(original_record, original_identity, record.read_cache(original_cache),
                                   pack.sha256_bytes(old_bridge.read_bytes()))
    if (original_identity["rid"] != "osx-arm64" or original_identity["variant"] != "metal"
            or original_identity["ggml"] != ggml or original_identity["cpu"] != "apple-m1"):
        raise ValueError("original bridge is not the pinned Mac baseline")
    settings = record.read_cache(build_dir / "CMakeCache.txt")
    identity = original_identity | {"source": source, "abi": pack.native_abi(ROOT)}
    pack.validate_release_profile(settings, identity)
    if settings["CMAKE_CXX_COMPILER"] != original_record["compiler"]:
        raise ValueError("fresh configuration changes the original compiler")
    original_link_path = fixture / "CMakeFiles/GgmlOps.dir/link.txt"
    original_flags_path = fixture / "CMakeFiles/GgmlOps.dir/flags.make"
    fresh_link_path = build_dir / "CMakeFiles/GgmlOps.dir/link.txt"
    fresh_flags_path = build_dir / "CMakeFiles/GgmlOps.dir/flags.make"
    link = shlex.split(pack.read_build_file(original_link_path))
    if link != shlex.split(pack.read_build_file(fresh_link_path)):
        raise ValueError("fresh configure changes the preserved linker command")
    if pack.relink_compile_flags(pack.read_build_file(original_flags_path)) != pack.relink_compile_flags(pack.read_build_file(fresh_flags_path)):
        raise ValueError("fresh configure changes the preserved compiler flags")
    directory = stage / "build/osx-arm64-metal"
    binary = stage / "runtimes/osx-arm64/native/metal/libGgmlOps.dylib"
    pack.require_unlinked(directory.absolute())
    pack.require_unlinked(binary.absolute())
    if directory.exists() or binary.parent.exists():
        raise ValueError("identity relink requires a fresh staging directory")
    directory.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    proof = {"schema": "tensorsharp-identity-relink/1", "originalIdentity": original_identity, "nativeSourceTree": native_tree,
             "compileIncludes": pack.identity_include_paths()}
    evidence = {"originalBuildRecord": (original_record_path, "original-build.json"),
                "originalCache": (original_cache, "original-cache.txt"),
                "originalBridge": (old_bridge, "original-bridge.dylib"),
                "originalLink": (original_link_path, "original-link.txt"),
                "originalFlags": (original_flags_path, "original-flags.txt"),
                "freshLink": (fresh_link_path, "fresh-link.txt"), "freshFlags": (fresh_flags_path, "fresh-flags.txt"),
                "identitySource": (ROOT / "TensorSharp.GGML.Native/ggml_ops_build_identity.cpp", "identity-source.cpp")}
    for name, (path, filename) in evidence.items():
        proof[name] = copy_evidence(path, directory, "identity-relink/" + filename)
    replacement_token = "CMakeFiles/GgmlOps.dir/ggml_ops_build_identity.cpp.o"
    tokens = [token for token in link if token.endswith((".o", ".a")) and token != replacement_token]
    if link.count(replacement_token) != 1 or len(tokens) != len(set(tokens)) or len(tokens) != 59:
        raise ValueError("the preserved Mac fixture does not have the verified 59 unchanged link inputs")
    proof["inputs"] = [copy_evidence(fixture / token, directory, "identity-relink/inputs/" + token) for token in tokens]
    replacement = directory / "identity-relink/replacement/identity.o"
    replacement.parent.mkdir()
    compiler = original_record["compiler"]
    proof["compileArgv"] = pack.identity_compile_arguments(directory / proof["identitySource"]["path"], replacement,
                                                         identity, compiler, proof["compileIncludes"])
    resolved = {token: str(directory / reference["path"]) for token, reference in zip(tokens, proof["inputs"])}
    proof["linkArgv"] = [resolved.get(token, str(replacement) if token == replacement_token else token) for token in link]
    proof["linkArgv"][proof["linkArgv"].index("-o") + 1] = str(binary)
    (directory / "identity-relink/commands.json").write_text(json.dumps({"compileArgv": proof["compileArgv"], "linkArgv": proof["linkArgv"]}, indent=2) + "\n")
    for name in ("compileArgv", "linkArgv"):
        completed = subprocess.run(proof[name], capture_output=True, text=True, timeout=60, check=True)
        (directory / ("identity-relink/" + name + ".log")).write_text(completed.stdout + completed.stderr)
    proof["replacementObject"] = {key: value for key, value in pack.file_record(directory, replacement).items() if key != "executable"}
    for reference, token in zip(proof["inputs"], tokens):
        if pack.sha256_bytes((fixture / token).read_bytes()) != reference["sha256"]:
            raise ValueError("preserved original link input changed during relink")
    build = record.create_record(ROOT, build_dir, binary, "osx-arm64", "metal", source)
    build["identityRelink"] = proof
    pack.validate_identity_relink(build, pack.read_identity(binary), directory, binary)
    record.write_build_record(directory, build, build_dir / "CMakeCache.txt")
    print("Verified identity-only Mac relink; 59 inputs unchanged; no upstream/backend build or target execution claimed.")
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-fixture", type=Path, required=True)
    parser.add_argument("--original-record", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True, help="fresh configure-only output; never a historical cache")
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    try:
        relink(args.original_fixture, args.original_record, args.build_dir, args.stage)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
