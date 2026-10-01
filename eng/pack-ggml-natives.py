#!/usr/bin/env python3
"""Package staged GgmlOps builds and write the GGML native artifact manifest.

Input is a staging root written by eng/build-ggml-natives.sh or
eng/build-ggml-natives.ps1:

  <stage>/runtimes/<rid>/native/<variant>/   the library, bundled runtime libraries
                                             and licenses/
  <stage>/build/<rid>-<variant>/             build-identity.json, cmake-settings.txt

Output, in --out (default: <stage>/dist):

  Darkspyre.TensorSharp.Backends.GGML.Native.<rid>.<version>.nupkg
      The RID's baseline build. Native files sit in runtimes/<rid>/native/, where
      the runtime's default probing finds them. License and notice files sit in
      licenses/ and NOTICE.md at the package root.
  Darkspyre.TensorSharp.Backends.GGML.Native.<rid>.<variant>.<version>.nupkg
      Optional developer package for a non-baseline variant. Files sit in
      ggml/<variant>/; buildTransitive/<id>.targets copies them to
      <output>/ggml/<variant>/, a directory a GgmlNativeLoader candidate can name.
      Packages above --developer-package-limit bytes are not written unless
      --allow-oversized is passed.
  ggml-<version>-<rid>-<variant>.zip
      Immutable variant archive of the whole artifact directory. Entries are
      sorted, carry a fixed 1980-01-01 timestamp and mode 0644, so the same
      files always give the same archive bytes.
  ggml-native-artifacts.json
      The artifact manifest (schema tensorsharp-native-artifacts/1). It records
      every package and archive with size and SHA-256, and every artifact with
      its file list, binary identity, build record, dependency closure and
      notices. Catalog generation and static hosting consume it.

Every staged variant gets an archive. The baseline variants are osx-arm64
metal, linux-x64 cpu, linux-arm64 cpu, win-x64 cpu and win-arm64 cpu. Optional
variants are Linux Vulkan/CUDA13, Windows x64 Vulkan/CUDA13 and Windows ARM64
Vulkan. Unsupported RID/variant pairs fail validation.

The script refuses: a file whose format or architecture does not match its RID
folder, a TSGgml library without the TSGgml_GetBuildIdentity export or with an
identity that differs from the build record, an unresolved dependency, a
build-machine RUNPATH, an artifact without the TensorSharp and ggml licenses,
and a bundled MSVC runtime without its notice. It validates the complete staging
input before writing output. --validate-only performs that validation without
creating packages, archives, a manifest or an output directory.

Usage:
  eng/pack-ggml-natives.py [--stage DIR] [--out DIR] [--managed-package NUPKG]
                           [--allow-oversized] [--developer-package-limit BYTES]
                           [--validate-only]
"""

import argparse
import hashlib
import importlib.util
import io
import json
import re
import stat
import struct
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "tensorsharp-native-artifacts/1"
DRIVER_ID = "ggml"
PACKAGE_PREFIX = "Darkspyre.TensorSharp.Backends.GGML.Native"
BASELINE = {"osx-arm64": "metal", "linux-x64": "cpu", "linux-arm64": "cpu", "win-x64": "cpu", "win-arm64": "cpu"}
VARIANTS = {
    "osx-arm64": {"metal"},
    "linux-x64": {"cpu", "vulkan", "cuda13"},
    "linux-arm64": {"cpu", "vulkan", "cuda13"},
    "win-x64": {"cpu", "vulkan", "cuda13"},
    "win-arm64": {"cpu", "vulkan"},
}
ENTRY = {"osx": "libGgmlOps.dylib", "linux": "libGgmlOps.so", "win": "GgmlOps.dll"}
BACKENDS = {"cpu": ["cpu"], "metal": ["metal", "cpu"], "vulkan": ["vulkan", "cpu"], "cuda13": ["cuda", "cpu"]}
REQUIRED_LICENSES = ("licenses/TensorSharp-LICENSE.txt", "licenses/ggml-LICENSE.txt")
MSVC_NOTICE = "licenses/MSVC-runtime-NOTICE.txt"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
DEFAULT_DEVELOPER_LIMIT = 250 * 1024 * 1024
REPOSITORY_URL = "https://github.com/DarkspyreStudios/TensorSharp"


def load_inventory():
    spec = importlib.util.spec_from_file_location("native_artifact_manifest", REPO_ROOT / "eng" / "native-artifact-manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INVENTORY = load_inventory()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_record(root, path):
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size": len(data), "sha256": sha256_bytes(data), "executable": False}


def zip_bytes(entries):
    """entries: list of (name, bytes). Returns deterministic zip bytes."""
    names = [name for name, _ in entries]
    if len(names) != len(set(names)) or any(not portable_path(name) for name in names):
        raise ValueError("archive entries must have unique portable relative paths")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(entries, key=lambda e: e[0]):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def xml_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def nupkg(package_id, version, description, tags, commit, files):
    """files: list of (package path, bytes). Returns deterministic .nupkg bytes."""
    nuspec = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
  <metadata>
    <id>{package_id}</id>
    <version>{version}</version>
    <authors>Darkspyre Studios; Zhongkai Fu</authors>
    <license type="expression">BSD-3-Clause AND MIT</license>
    <licenseUrl>https://licenses.nuget.org/BSD-3-Clause%20AND%20MIT</licenseUrl>
    <projectUrl>{REPOSITORY_URL}</projectUrl>
    <description>{xml_escape(description)}</description>
    <tags>{tags}</tags>
    <repository type="git" url="{REPOSITORY_URL}" commit="{commit}" />
  </metadata>
</package>
"""
    extensions = sorted({Path(name).suffix.lstrip(".").lower() for name, _ in files if Path(name).suffix} - {"rels", "psmdcp", "nuspec"})
    content_types = ('<?xml version="1.0" encoding="utf-8"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />'
                     '<Default Extension="psmdcp" ContentType="application/vnd.openxmlformats-package.core-properties+xml" />'
                     '<Default Extension="nuspec" ContentType="application/octet" />'
                     + "".join(f'<Default Extension="{e}" ContentType="application/octet" />' for e in extensions)
                     + '</Types>')
    rels = ('<?xml version="1.0" encoding="utf-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Type="http://schemas.microsoft.com/packaging/2010/07/manifest" Target="/{package_id}.nuspec" Id="R0" />'
            '<Relationship Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
            'Target="/package/services/metadata/core-properties/0.psmdcp" Id="R1" />'
            '</Relationships>')
    core = ('<?xml version="1.0" encoding="utf-8"?>'
            '<coreProperties xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns="http://schemas.openxmlformats.org/package/2006/metadata/core-properties">'
            f'<dc:creator>Darkspyre Studios; Zhongkai Fu</dc:creator><dc:description>{xml_escape(description)}</dc:description>'
            f'<dc:identifier>{package_id}</dc:identifier><version>{version}</version><keywords>{tags}</keywords>'
            '<lastModifiedBy>eng/pack-ggml-natives.py</lastModifiedBy></coreProperties>')
    entries = [("[Content_Types].xml", content_types.encode()), ("_rels/.rels", rels.encode()),
               (f"{package_id}.nuspec", nuspec.encode()),
               ("package/services/metadata/core-properties/0.psmdcp", core.encode())]
    return zip_bytes(entries + list(files))


def notice_markdown(title, artifact_files, has_msvc):
    lines = [f"# {title}", "", "This package contains GgmlOps, the TensorSharp native bridge, which statically links ggml.", "",
             "| Component | License | File |", "|---|---|---|",
             "| TensorSharp (GgmlOps) | BSD-3-Clause | licenses/TensorSharp-LICENSE.txt |",
             "| ggml | MIT | licenses/ggml-LICENSE.txt |"]
    if has_msvc:
        lines.append("| Microsoft Visual C++ runtime | Microsoft Visual Studio license terms, Distributable Code | licenses/MSVC-runtime-NOTICE.txt |")
    for extra in sorted(p for p in artifact_files if p.startswith("licenses/") and p not in REQUIRED_LICENSES and p != MSVC_NOTICE):
        lines.append(f"| third-party runtime library | see file | {extra} |")
    return ("\n".join(lines) + "\n").encode()


def developer_targets(package_id, variant):
    return f"""<Project>
  <!-- Copies the {variant} GgmlOps artifact to $(OutDir)ggml/{variant}/ and to the publish directory.
       Pass that directory to GgmlNativeLoader.Select as a candidate. -->
  <ItemGroup>
    <None Include="$(MSBuildThisFileDirectory)../ggml/{variant}/**"
          Link="ggml/{variant}/%(RecursiveDir)%(Filename)%(Extension)"
          CopyToOutputDirectory="PreserveNewest"
          CopyToPublishDirectory="PreserveNewest"
          Visible="false" />
  </ItemGroup>
</Project>
""".encode()


def check_artifact(rid, variant, directory, record, described, required_exports=None):
    errors = []
    if variant not in VARIANTS.get(rid, ()):
        return [f"{rid}/{variant}: unsupported RID/variant pair"]
    names = {f["path"] for f in record["files"]}
    for required in REQUIRED_LICENSES:
        if required not in names:
            errors.append(f"{rid}/{variant}: {required} is missing")
    entry = ENTRY[rid.split("-")[0]]
    if entry not in names:
        errors.append(f"{rid}/{variant}: {entry} is missing")
    primary = next((f for f in described if f["path"] == entry), {})
    if primary.get("format") != INVENTORY.RID_FORMAT[rid.split("-")[0]]:
        errors.append(f"{rid}/{variant}: {entry} is not a native bridge for this RID")
    if primary.get("tsggmlBuildIdentityExport") is not True:
        errors.append(f"{rid}/{variant}: {entry} does not export TSGgml_GetBuildIdentity")
    if required_exports is not None:
        symbols = primary.get("functionExports")
        if not isinstance(symbols, list) or any(not isinstance(name, str) for name in symbols):
            errors.append(f"{rid}/{variant}: bridge exports were not inspected")
        else:
            for name in sorted(set(required_exports) - set(symbols)):
                errors.append(f"{rid}/{variant}: missing required managed entrypoint {name}")
    os_part, _, arch_part = rid.partition("-")
    siblings = {}
    for item in described:
        path = item["path"]
        if "/" in path:
            continue
        key = path.lower() if os_part == "win" else path
        if key in siblings:
            errors.append(f"{rid}/{variant}: competing native sibling name {path}")
        siblings[key] = item
    for f in described:
        if f["format"] == "other":
            continue
        if INVENTORY.RID_FORMAT.get(os_part) != f["format"]:
            errors.append(f"{f['path']}: {f['format']} binary under {rid}")
        arch = f.get("identity", {}).get("arch")
        if INVENTORY.RID_ARCH.get(arch_part) != arch:
            errors.append(f"{f['path']}: {arch} binary under {rid}")
        if not isinstance(f.get("dependencies"), list):
            errors.append(f"{f['path']}: native dependencies were not inspected")
        for dep in f.get("dependencies", []) if isinstance(f.get("dependencies"), list) else []:
            resolution = INVENTORY.classify(f["format"], dep["name"], set(siblings))
            if resolution.startswith("unresolved"):
                errors.append(f"{f['path']}: dependency {dep['name']} is {resolution}")
            elif resolution == "bundled":
                name = INVENTORY.dependency_basename(f["format"], dep["name"])
                sibling = siblings[name.lower() if os_part == "win" else name]
                if (sibling.get("format") != INVENTORY.RID_FORMAT[os_part]
                        or sibling.get("identity", {}).get("arch") != INVENTORY.RID_ARCH[arch_part]
                        or not isinstance(sibling.get("dependencies"), list)):
                    errors.append(f"{f['path']}: dependency {dep['name']} is not an inspected native sibling for {rid}")
                if f["format"] == "elf" and "$ORIGIN" not in f.get("runpath", []):
                    errors.append(f"{f['path']}: bundled dependency {dep['name']} has no selected-directory $ORIGIN runpath")
        for rp in f.get("runpath", []):
            if rp != "$ORIGIN":
                errors.append(f"{f['path']}: build-machine search path {rp}")
    if any(INVENTORY.MSVC_REDIST.match(Path(n).name) for n in names) and MSVC_NOTICE not in names:
        errors.append(f"{rid}/{variant}: bundled MSVC runtime without {MSVC_NOTICE}")
    for forbidden in ("libcuda.so", "libcuda.so.1", "nvcuda.dll"):
        if forbidden in {Path(n).name.lower() for n in names}:
            errors.append(f"{rid}/{variant}: the NVIDIA driver library {forbidden} must not ship")
    return errors


def check_release_matrix(artifacts):
    required = {(rid, variant) for rid, variants in VARIANTS.items() for variant in variants}
    present = [(artifact["rid"], artifact["variant"]) for artifact in artifacts]
    errors = [f"complete release: missing {rid}/{variant}" for rid, variant in sorted(required - set(present))]
    errors += [f"complete release: unsupported {rid}/{variant}" for rid, variant in sorted(set(present) - required)]
    if len(present) != len(set(present)):
        errors.append("complete release: duplicate RID/variant artifacts")
    return errors


def read_identity(path):
    text = path.read_bytes()
    match = re.search(rb"format=1;tensorsharp=[\x20-\x7e]*", text)
    if not match:
        return None
    return dict(pair.split("=", 1) for pair in match.group(0).decode().split(";") if "=" in pair)


def native_abi(root):
    """Mirror the CMake/MSBuild bridge identity for artifact validation without a build tool."""
    extensions = {".cpp", ".h", ".hpp", ".inc", ".cu", ".cuh"}
    files = [p for p in (root / "TensorSharp.GGML.Native").iterdir() if p.is_file() and p.suffix in extensions]
    files += list((root / "TensorSharp.Backends.GGML").glob("*.cs"))
    files.append(root / "eng" / "ggml-revision")
    rows = []
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        rows.append(path.relative_to(root).as_posix() + "=" + digest + "\n")
    manifest = "".join(rows)
    return sha256_bytes(manifest.encode("utf-8"))


def read_required_exports(root):
    inventory = json.loads(read_build_file(root / "eng/ggml-required-exports.json"))
    if (not isinstance(inventory, dict) or set(inventory) != {"schema", "nativeAbi", "ggmlCommit", "exports"}
            or inventory["schema"] != "tensorsharp-ggml-required-exports/1"
            or inventory["nativeAbi"] != native_abi(root)
            or inventory["ggmlCommit"] != (root / "eng/ggml-revision").read_text(encoding="utf-8").strip()):
        raise ValueError("the required-export inventory does not match the exact managed/native ABI and upstream")
    names = inventory["exports"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII) for name in names)
            or len(set(names)) != len(names) or names != sorted(names)):
        raise ValueError("the required-export inventory must contain unique sorted literal entrypoint names")
    return names


def portable_path(name):
    if not isinstance(name, str):
        return False
    parts = name.split("/")
    return (bool(name) and not PurePosixPath(name).is_absolute() and "\\" not in name and ":" not in name
            and not any(ord(char) < 32 or ord(char) == 127 for char in name)
            and all(part not in ("", ".", "..") and not part.endswith((".", " ")) for part in parts))


def require_unlinked(path):
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"linked staging path: {ancestor}")


def directories(path):
    require_unlinked(path)
    result = []
    for child in sorted(path.iterdir()):
        mode = child.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"expected an ordinary staging directory: {child}")
        result.append(child)
    return result


def regular_files(directory):
    require_unlinked(directory)
    files = []
    pending = [directory]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            name = path.relative_to(directory).as_posix()
            if not portable_path(name):
                raise ValueError(f"nonportable artifact path: {name}")
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise ValueError(f"artifact contains a link or special file: {name}")
    return sorted(files)


def read_build_file(path):
    require_unlinked(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"expected an ordinary build record file: {path}")
    return path.read_text(encoding="utf-8")


def collect_artifacts(stage, version, ggml_commit):
    artifacts = []
    errors = []
    expected_abi = native_abi(REPO_ROOT)
    required_exports = read_required_exports(REPO_ROOT)
    for rid_dir in directories(stage / "runtimes"):
        rid = rid_dir.name
        if rid not in VARIANTS:
            errors.append(f"{rid}: unsupported RID")
            continue
        for variant_dir in directories(rid_dir / "native"):
            variant = variant_dir.name
            if variant not in VARIANTS[rid]:
                errors.append(f"{rid}/{variant}: unsupported RID/variant pair")
                continue
            try:
                paths = regular_files(variant_dir)
                build_dir = stage / "build" / f"{rid}-{variant}"
                build = json.loads(read_build_file(build_dir / "build-identity.json"))
                if not isinstance(build, dict):
                    raise ValueError(f"{rid}/{variant}: build record is not an object")
                for key, expected in (("tensorSharpBuild", version), ("rid", rid), ("variant", variant), ("ggmlCommit", ggml_commit),
                                      ("nativeAbi", expected_abi)):
                    if build.get(key) != expected:
                        errors.append(f"{rid}/{variant}: build record {key}={build.get(key)}, expected {expected}")
                source = build.get("sourceCommit")
                if not isinstance(source, str) or not re.fullmatch(r"[a-f0-9]{40}", source):
                    errors.append(f"{rid}/{variant}: build record has no exact source commit")
                settings_path = build_dir / "cmake-settings.txt"
                build["cmakeSettings"] = read_build_file(settings_path).splitlines() if settings_path.exists() or settings_path.is_symlink() else []
                files = [file_record(variant_dir, path) for path in paths]
                described = [INVENTORY.describe(f["path"], rid, variant, variant_dir / f["path"]) for f in files]
                errors += check_artifact(rid, variant, variant_dir, {"files": files}, described, required_exports)
                entry = ENTRY[rid.split("-")[0]]
                identity = read_identity(variant_dir / entry) if entry in {f["path"] for f in files} else None
                if identity is None:
                    errors.append(f"{rid}/{variant}: {entry} has no build identity")
                else:
                    for key, expected in (("tensorsharp", version), ("rid", rid), ("variant", variant),
                                          ("ggml", ggml_commit), ("source", source), ("abi", expected_abi)):
                        if identity.get(key) != expected:
                            errors.append(f"{rid}/{variant}: binary identity {key}={identity.get(key)}, expected {expected}")
                artifacts.append({"rid": rid, "variant": variant, "directory": variant_dir, "build": build,
                                  "files": files, "inventory": described, "identity": identity})
            except (OSError, ValueError, KeyError, IndexError, struct.error) as error:
                errors.append(f"{rid}/{variant}: artifact inspection failed: {error}")
    if not artifacts and not errors:
        errors.append("no native artifacts staged")
    return artifacts, errors


def managed_package_record(path, version):
    require_unlinked(path.absolute())
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("the managed package must be an ordinary file")
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG):
                raise ValueError("the managed package contains a link or special file")
        entries = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if len(entries) != len(set(entries)) or any(not portable_path(name) for name in entries):
            raise ValueError("the managed package has duplicate or nonportable entries")
        natives = [name for name in entries if name.startswith("runtimes/") or name.lower().endswith((".so", ".dylib"))
                   or re.search(r"(^|/)GgmlOps\.dll$", name, re.I)]
        if natives:
            raise ValueError(f"the managed package carries native files: {', '.join(natives)}")
        specs = [name for name in entries if name.endswith(".nuspec")]
        if len(specs) != 1:
            raise ValueError("the managed package must contain exactly one nuspec")
        metadata = ET.fromstring(archive.read(specs[0])).find("{*}metadata")
        if metadata is None or metadata.findtext("{*}id") != "Darkspyre.TensorSharp.Backends.GGML" or metadata.findtext("{*}version") != version:
            raise ValueError("the managed package identity/version differs from the release")
    return {"id": "Darkspyre.TensorSharp.Backends.GGML", "version": version, "kind": "managed",
            "fileName": path.name, "size": len(data), "sha256": sha256_bytes(data), "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--managed-package", type=Path,
                        help="the Darkspyre.TensorSharp.Backends.GGML .nupkg; it must carry no native file")
    parser.add_argument("--allow-oversized", action="store_true")
    parser.add_argument("--developer-package-limit", type=int, default=DEFAULT_DEVELOPER_LIMIT)
    parser.add_argument("--validate-only", action="store_true", help="validate staged inputs without writing release artifacts")
    parser.add_argument("--complete-release", action="store_true",
                        help="require the entire canonical release matrix during validation-only checks; output generation always requires it")
    args = parser.parse_args()

    version = ET.parse(REPO_ROOT / "Directory.Build.props").findtext(".//TensorSharpVersion")
    if not version:
        parser.error("Directory.Build.props has no TensorSharpVersion")
    stage = (args.stage or REPO_ROOT / "artifacts" / "ggml-natives" / version).absolute()
    out = (args.out or stage / "dist").absolute()
    ggml_commit = (REPO_ROOT / "eng" / "ggml-revision").read_text().strip()
    try:
        require_unlinked(stage)
        require_unlinked(out)
        stage, out = stage.resolve(), out.resolve()
        if any(root == out or root in out.parents for root in (stage / "runtimes", stage / "build")):
            raise ValueError("output directory overlaps staged artifacts or build records")
        staged, errors = collect_artifacts(stage, version, ggml_commit)
        if args.complete_release or not args.validate_only:
            errors += check_release_matrix(staged)
        managed = managed_package_record(args.managed_package, version) if args.managed_package else None
    except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"error: staging validation failed: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print(f"validated {len(staged)} staged artifacts; no output written")
        return 0
    ggml_cmake = (REPO_ROOT / "ExternalProjects" / "ggml" / "CMakeLists.txt").read_text()
    ggml_version = ".".join(re.search(rf"set\(GGML_VERSION_{part}\s+(\d+)\)", ggml_cmake).group(1)
                            for part in ("MAJOR", "MINOR", "PATCH"))
    package_commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)

    artifacts = []
    packages = [managed] if managed else []
    for candidate in staged:
        rid, variant, variant_dir = candidate["rid"], candidate["variant"], candidate["directory"]
        build, files = candidate["build"], candidate["files"]
        described, identity = candidate["inventory"], candidate["identity"]
        entry = ENTRY[rid.split("-")[0]]

        archive_name = f"ggml-{version}-{rid}-{variant}.zip"
        archive = zip_bytes([(f["path"], (variant_dir / f["path"]).read_bytes()) for f in files])
        (out / archive_name).write_bytes(archive)

        entry_info = next((d for d in described if d["path"] == entry), {})
        requires = {}
        for d in described:
            for dep in d.get("dependencies", []) if isinstance(d.get("dependencies"), list) else []:
                if dep["resolution"] not in ("bundled", "os"):
                    requires.setdefault(dep["resolution"], set()).add(Path(dep["name"]).name)
        baseline = BASELINE.get(rid) == variant
        native_files = [f for f in files if not f["path"].startswith("licenses/")]
        artifact = {
            "driverId": DRIVER_ID,
            "rid": rid,
            "variant": variant,
            "version": version,
            "tensorSharpBuild": version,
            "nativeAbi": identity["abi"],
            "tensorSharp": {"packageVersion": version, "packageCommit": package_commit,
                            "nativeSourceCommit": build["sourceCommit"]},
            "ggml": {"version": ggml_version, "commit": ggml_commit},
            "backends": BACKENDS.get(variant, [variant]),
            "bundled": baseline,
            "entryLibrary": entry,
            "files": files,
            "fileListFingerprint": sha256_bytes("".join(f"{f['path']}\n{f['size']}\n{f['sha256']}\n" for f in files).encode()),
            "totalSize": sum(f["size"] for f in files),
            "archive": {"name": archive_name, "format": "zip", "size": len(archive), "sha256": sha256_bytes(archive)},
            "baselinePackage": None,
            "developerPackage": None,
            "binaryIdentity": {
                "native": identity,
                "elfBuildId": entry_info.get("identity", {}).get("buildId"),
                "machoUuid": entry_info.get("identity", {}).get("uuid"),
                "machoMinOs": entry_info.get("identity", {}).get("minOs"),
                "peTimestamp": entry_info.get("identity", {}).get("timestamp"),
                "tsggmlExports": entry_info.get("tsggmlExports"),
                "tsggmlBuildIdentityExport": entry_info.get("tsggmlBuildIdentityExport"),
                "maxSymbolVersions": entry_info.get("maxSymbolVersions"),
                "runpath": entry_info.get("runpath"),
            },
            "build": build,
            "requires": {k: sorted(v) for k, v in sorted(requires.items())},
            "notices": [{k: f[k] for k in ("path", "size", "sha256")} for f in files if f["path"].startswith("licenses/")],
            "inventory": described,
        }

        has_msvc = any(INVENTORY.MSVC_REDIST.match(Path(f["path"]).name) for f in files)
        licenses = [(f["path"], (variant_dir / f["path"]).read_bytes()) for f in files if f["path"].startswith("licenses/")]
        if baseline:
            package_id = f"{PACKAGE_PREFIX}.{rid}"
            title = f"{package_id} {version}"
            contents = [(f"runtimes/{rid}/native/{f['path']}", (variant_dir / f["path"]).read_bytes()) for f in native_files]
            contents += licenses + [("NOTICE.md", notice_markdown(title, [f["path"] for f in files], has_msvc))]
            description = (f"GgmlOps baseline native library for {rid} ({', '.join(BACKENDS[variant])}), TensorSharp build {version}. "
                           "Use with Darkspyre.TensorSharp.Backends.GGML.")
            data = nupkg(package_id, version, description, f"darkspyre tensorsharp ggml native {rid}", build["sourceCommit"], contents)
            file_name = f"{package_id}.{version}.nupkg"
            (out / file_name).write_bytes(data)
            entries = sorted(name for name, _ in contents)
            packages.append({"id": package_id, "version": version, "kind": "baseline", "rid": rid, "variant": variant,
                             "fileName": file_name, "size": len(data), "sha256": sha256_bytes(data), "entries": entries})
            artifact["baselinePackage"] = {"id": package_id, "version": version, "nativeDirectory": f"runtimes/{rid}/native",
                                           "files": [f for f in native_files]}
        else:
            package_id = f"{PACKAGE_PREFIX}.{rid}.{variant}"
            title = f"{package_id} {version}"
            contents = [(f"ggml/{variant}/{f['path']}", (variant_dir / f["path"]).read_bytes()) for f in files]
            contents += [("NOTICE.md", notice_markdown(title, [f["path"] for f in files], has_msvc)),
                         (f"buildTransitive/{package_id}.targets", developer_targets(package_id, variant)),
                         (f"ggml/{variant}.artifact.json", json.dumps({k: artifact[k] for k in (
                             "driverId", "rid", "variant", "version", "tensorSharpBuild", "backends", "entryLibrary", "files")},
                             indent=2).encode())]
            file_name = f"{package_id}.{version}.nupkg"
            size_estimate = len(archive) + 65536
            if size_estimate > args.developer_package_limit and not args.allow_oversized:
                artifact["developerPackage"] = {"id": package_id, "version": version, "withheld":
                                                f"archive is {len(archive)} bytes, above the developer package limit of "
                                                f"{args.developer_package_limit} bytes; pass --allow-oversized after the owner decides"}
            else:
                description = (f"GgmlOps {variant} native library for {rid}, TensorSharp build {version}, for development. "
                               f"Copies the artifact to ggml/{variant}/ in the build output.")
                data = nupkg(package_id, version, description, f"darkspyre tensorsharp ggml native {rid} {variant}",
                             build["sourceCommit"], contents)
                (out / file_name).write_bytes(data)
                packages.append({"id": package_id, "version": version, "kind": "developer", "rid": rid, "variant": variant,
                                 "fileName": file_name, "size": len(data), "sha256": sha256_bytes(data),
                                 "entries": sorted(name for name, _ in contents)})
                artifact["developerPackage"] = {"id": package_id, "version": version, "directory": f"ggml/{variant}"}
        if not baseline:
            artifact["delivery"] = "variant-archive"
        artifacts.append(artifact)

    missing_baselines = sorted(rid for rid in BASELINE if not any(a["rid"] == rid and a["bundled"] for a in artifacts))
    manifest = {
        "schema": SCHEMA,
        "driverId": DRIVER_ID,
        "tensorSharpBuild": version,
        "tensorSharp": {"packageVersion": version, "packageCommit": package_commit},
        "ggml": {"version": ggml_version, "commit": ggml_commit},
        "baselineVariants": BASELINE,
        "missingBaselines": missing_baselines,
        "packages": packages,
        "artifacts": artifacts,
    }
    (out / "ggml-native-artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for rid in missing_baselines:
        print(f"note: no baseline staged for {rid}", file=sys.stderr)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
