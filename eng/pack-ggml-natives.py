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
metal, linux-x64 cpu, linux-arm64 cpu and win-x64 cpu. A variant staged for any
other RID gets an archive and, when it is not a baseline, a developer package.

The script refuses: a file whose format or architecture does not match its RID
folder, a TSGgml library without the TSGgml_GetBuildIdentity export or with an
identity that differs from the build record, an unresolved dependency, a
build-machine RUNPATH, an artifact without the TensorSharp and ggml licenses,
and a bundled MSVC runtime without its notice.

Usage:
  eng/pack-ggml-natives.py [--stage DIR] [--out DIR] [--managed-package NUPKG]
                           [--allow-oversized] [--developer-package-limit BYTES]
"""

import argparse
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "tensorsharp-native-artifacts/1"
DRIVER_ID = "ggml"
PACKAGE_PREFIX = "Darkspyre.TensorSharp.Backends.GGML.Native"
BASELINE = {"osx-arm64": "metal", "linux-x64": "cpu", "linux-arm64": "cpu", "win-x64": "cpu"}
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


def check_artifact(rid, variant, directory, record, described):
    errors = []
    names = {f["path"] for f in record["files"]}
    for required in REQUIRED_LICENSES:
        if required not in names:
            errors.append(f"{rid}/{variant}: {required} is missing")
    entry = ENTRY[rid.split("-")[0]]
    if entry not in names:
        errors.append(f"{rid}/{variant}: {entry} is missing")
    os_part, _, arch_part = rid.partition("-")
    for f in described:
        if f["format"] == "other":
            continue
        if INVENTORY.RID_FORMAT.get(os_part) != f["format"]:
            errors.append(f"{f['path']}: {f['format']} binary under {rid}")
        arch = f.get("identity", {}).get("arch")
        if arch and INVENTORY.RID_ARCH.get(arch_part) != arch:
            errors.append(f"{f['path']}: {arch} binary under {rid}")
        for dep in f.get("dependencies", []) if isinstance(f.get("dependencies"), list) else []:
            if dep["resolution"].startswith("unresolved"):
                errors.append(f"{f['path']}: dependency {dep['name']} is {dep['resolution']}")
        for rp in f.get("runpath", []):
            if rp != "$ORIGIN":
                errors.append(f"{f['path']}: build-machine search path {rp}")
    if any(INVENTORY.MSVC_REDIST.match(Path(n).name) for n in names) and MSVC_NOTICE not in names:
        errors.append(f"{rid}/{variant}: bundled MSVC runtime without {MSVC_NOTICE}")
    for forbidden in ("libcuda.so", "libcuda.so.1", "nvcuda.dll"):
        if forbidden in {Path(n).name for n in names}:
            errors.append(f"{rid}/{variant}: the NVIDIA driver library {forbidden} must not ship")
    return errors


def read_identity(path):
    text = path.read_bytes()
    match = re.search(rb"format=1;tensorsharp=[\x20-\x7e]*", text)
    if not match:
        return None
    return dict(pair.split("=", 1) for pair in match.group(0).decode().split(";") if "=" in pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--managed-package", type=Path,
                        help="the Darkspyre.TensorSharp.Backends.GGML .nupkg; it must carry no native file")
    parser.add_argument("--allow-oversized", action="store_true")
    parser.add_argument("--developer-package-limit", type=int, default=DEFAULT_DEVELOPER_LIMIT)
    args = parser.parse_args()

    props = (REPO_ROOT / "Directory.Build.props").read_text()
    version = re.search(r"<TensorSharpVersion>([^<]+)</TensorSharpVersion>", props).group(1)
    stage = (args.stage or REPO_ROOT / "artifacts" / "ggml-natives" / version).resolve()
    out = (args.out or stage / "dist").resolve()
    out.mkdir(parents=True, exist_ok=True)
    ggml_commit = (REPO_ROOT / "eng" / "ggml-revision").read_text().strip()
    ggml_cmake = (REPO_ROOT / "ExternalProjects" / "ggml" / "CMakeLists.txt").read_text()
    ggml_version = ".".join(re.search(rf"set\(GGML_VERSION_{part}\s+(\d+)\)", ggml_cmake).group(1)
                            for part in ("MAJOR", "MINOR", "PATCH"))
    package_commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()

    errors = []
    artifacts = []
    packages = []
    runtimes = stage / "runtimes"
    for rid_dir in sorted(p for p in runtimes.iterdir() if p.is_dir()):
        rid = rid_dir.name
        for variant_dir in sorted(p for p in (rid_dir / "native").iterdir() if p.is_dir()):
            variant = variant_dir.name
            build_dir = stage / "build" / f"{rid}-{variant}"
            build = json.loads((build_dir / "build-identity.json").read_text())
            settings_path = build_dir / "cmake-settings.txt"
            build["cmakeSettings"] = settings_path.read_text().splitlines() if settings_path.exists() else []
            files = sorted((file_record(variant_dir, p) for p in variant_dir.rglob("*")
                            if p.is_file() and not p.name.startswith(".")), key=lambda f: f["path"])
            record = {"files": files}
            described = [INVENTORY.describe(f["path"], rid, variant, variant_dir / f["path"]) for f in files]
            errors += check_artifact(rid, variant, variant_dir, record, described)

            entry = ENTRY[rid.split("-")[0]]
            identity = read_identity(variant_dir / entry) if (variant_dir / entry).exists() else None
            if identity is None:
                errors.append(f"{rid}/{variant}: {entry} has no build identity")
            else:
                for key, expected in (("tensorsharp", version), ("rid", rid), ("variant", variant),
                                      ("ggml", ggml_commit), ("source", build["sourceCommit"])):
                    if identity.get(key) != expected:
                        errors.append(f"{rid}/{variant}: binary identity {key}={identity.get(key)}, expected {expected}")
            if build.get("tensorSharpBuild") != version:
                errors.append(f"{rid}/{variant}: build record is for {build.get('tensorSharpBuild')}, not {version}")

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

    if args.managed_package:
        data = args.managed_package.read_bytes()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries = sorted(n for n in z.namelist() if not n.endswith("/"))
        natives = [n for n in entries if n.startswith("runtimes/") or n.lower().endswith((".so", ".dylib"))
                   or re.search(r"(^|/)GgmlOps\.dll$", n, re.I)]
        if natives:
            errors.append(f"{args.managed_package.name}: the managed package carries native files: {', '.join(natives)}")
        packages.insert(0, {"id": "Darkspyre.TensorSharp.Backends.GGML", "version": version, "kind": "managed",
                            "fileName": args.managed_package.name, "size": len(data), "sha256": sha256_bytes(data),
                            "entries": entries})

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
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    for rid in missing_baselines:
        print(f"note: no baseline staged for {rid}", file=sys.stderr)
    print(f"wrote {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
