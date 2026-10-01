#!/usr/bin/env python3
"""Describe GGML native artifacts and check their RID layout and dependency closure.

Each input is a .nupkg file or a directory that holds runtimes/<rid>/native/.
A package file lists as variant "package". A loose directory takes the folder
below native/ as the variant (cpu, cuda, vulkan), or "package" when the files
sit directly in native/.

For every native file the tool records size, SHA-256, binary format and
architecture, build identity (ELF build ID, Mach-O UUID and minimum OS, PE
timestamp, CodeView PDB identity and file version), direct dependencies,
ELF symbol-version needs, the count of exported TSGgml_* functions and whether
TSGgml_GetBuildIdentity is exported.
Dependencies resolve against sibling files first and then a fixed list of
operating-system, runtime and driver libraries. Anything else is reported as
unresolved.

The dependency and export listings use objdump, and otool on macOS. Apple's
objdump reads ELF, PE and Mach-O files, so one macOS host inspects every RID.
The tool does not load any library.

Usage:
  eng/native-artifact-manifest.py [--work-dir DIR] [--out FILE] INPUT [INPUT ...]

Package files are extracted under --work-dir, which defaults to tmp/native-artifact-manifest
at the repository root. The JSON manifest goes to --out or standard output.
The exit code is 1 when any check reports an error.
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

RID_FORMAT = {"linux": "elf", "win": "pe", "osx": "macho"}
RID_ARCH = {"x64": "x86_64", "arm64": "arm64", "x86": "x86"}

LINUX_OS = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
}
LINUX_OS_PACKAGE = {"libstdc++.so.6", "libgcc_s.so.1", "libgomp.so.1"}
LINUX_GPU_RUNTIME = {"libcuda.so.1": "nvidia-driver", "libvulkan.so.1": "vulkan-loader"}
WINDOWS_OS = {
    "kernel32.dll", "advapi32.dll", "user32.dll", "ntdll.dll", "bcrypt.dll",
    "ole32.dll", "oleaut32.dll", "shell32.dll", "ws2_32.dll", "dbghelp.dll",
}
WINDOWS_GPU_RUNTIME = {"nvcuda.dll": "nvidia-driver", "vulkan-1.dll": "vulkan-loader"}
MSVC_REDIST = re.compile(r"^(msvcp140(_\w+)?|vcruntime140(_\w+)?|vcomp140|concrt140)\.dll$", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", check=False)
        return result.stdout if result.returncode == 0 else None
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- binary headers

def read_header(data):
    if data[:4] == b"\x7fELF":
        return "elf"
    if data[:2] == b"MZ" and len(data) > 0x40:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] == b"PE\0\0":
            return "pe"
    if data[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
        return "macho"
    return "other"


def elf_info(data):
    info = {"arch": {62: "x86_64", 183: "arm64"}.get(struct.unpack_from("<H", data, 18)[0], "unknown")}
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    strtab = sections[shstrndx]
    for name_off, _, _, _, offset, size, *_ in sections:
        start = strtab[4] + name_off
        name = data[start:data.index(b"\0", start)].decode()
        if name == ".note.gnu.build-id":
            namesz, descsz = struct.unpack_from("<II", data, offset)
            desc = offset + 12 + ((namesz + 3) & ~3)
            info["buildId"] = data[desc:desc + descsz].hex()
    return info


def macho_version(value):
    return f"{value >> 16}.{(value >> 8) & 0xFF}.{value & 0xFF}".removesuffix(".0")


def macho_info(data):
    if data[:4] == b"\xca\xfe\xba\xbe":
        return {"arch": "universal"}
    cputype, _, _, ncmds = struct.unpack_from("<iiII", data, 4)
    info = {"arch": {0x0100000C: "arm64", 0x01000007: "x86_64"}.get(cputype, "unknown")}
    offset = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, offset)
        if cmd == 0x1B:
            raw = data[offset + 8:offset + 24].hex().upper()
            info["uuid"] = f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
        elif cmd == 0x32:
            _, minos, sdk = struct.unpack_from("<III", data, offset + 8)
            info["minOs"] = macho_version(minos)
            info["sdk"] = macho_version(sdk)
        offset += size
    return info


def pe_info(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    machine, nsections, timestamp, _, _, optsize = struct.unpack_from("<HHIIIH", data, pe + 4)
    info = {
        "arch": {0x8664: "x86_64", 0xAA64: "arm64", 0x14C: "x86"}.get(machine, "unknown"),
        "timestamp": timestamp,
    }
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dirs = opt + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(nsections):
        s = opt + optsize + i * 40
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, s + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def rva_to_offset(rva):
        for vaddr, size, rawptr in sections:
            if vaddr <= rva < vaddr + size:
                return rawptr + rva - vaddr
        return None

    # Directory 4 holds the Authenticode certificate table. Its presence is recorded; its validity is not checked.
    info["certificateTableBytes"] = struct.unpack_from("<II", data, dirs + 4 * 8)[1]
    debug_rva, debug_size = struct.unpack_from("<II", data, dirs + 6 * 8)
    debug = rva_to_offset(debug_rva) if debug_rva else None
    for i in range((debug_size // 28) if debug is not None else 0):
        _, _, _, _, kind, size, _, ptr = struct.unpack_from("<IIHHIIII", data, debug + i * 28)
        if kind == 2 and data[ptr:ptr + 4] == b"RSDS":
            guid = data[ptr + 4:ptr + 20]
            d1, d2, d3 = struct.unpack_from("<IHH", guid)
            info["pdbGuid"] = f"{d1:08X}-{d2:04X}-{d3:04X}-{guid[8:10].hex().upper()}-{guid[10:].hex().upper()}"
            info["pdbAge"] = struct.unpack_from("<I", data, ptr + 20)[0]
            end = data.index(b"\0", ptr + 24)
            info["pdbPath"] = data[ptr + 24:end].decode(errors="replace")
    fixed = data.find(b"\xbd\x04\xef\xfe")
    if fixed >= 0:
        ms, ls = struct.unpack_from("<II", data, fixed + 8)
        info["fileVersion"] = f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    return info


# ---------------------------------------------------------------- tool output

def elf_dynamic(path):
    text = run(["objdump", "-p", str(path)])
    if text is None:
        return None
    needed = re.findall(r"^\s+NEEDED\s+(\S+)", text, re.M)
    runpath = re.findall(r"^\s+(?:RUNPATH|RPATH)\s+(\S+)", text, re.M)
    versions = {}
    for ns, ver in re.findall(r"\b(GLIBC|GLIBCXX|CXXABI|GCC|GOMP|OMP)_([0-9][0-9.]*)\b",
                              text.split("Version References", 1)[-1]):
        key = tuple(int(p) for p in ver.split("."))
        if ns not in versions or key > versions[ns][0]:
            versions[ns] = (key, f"{ns}_{ver}")
    exports = run(["objdump", "-T", str(path)]) or ""
    return {
        "needed": needed,
        "runpath": runpath,
        "maxSymbolVersions": sorted(v[1] for v in versions.values()),
        "tsggmlExports": len(re.findall(r"\.text\s+\S+\s+(?:Base\s+)?TSGgml_", exports)),
        "tsggmlBuildIdentityExport": bool(re.search(r"\.text\s+\S+\s+(?:Base\s+)?TSGgml_GetBuildIdentity\s*$", exports, re.M)),
    }


def pe_dynamic(path):
    text = run(["objdump", "-p", str(path)])
    if text is None:
        return None
    exports = text.split("Export Table:", 1)[1] if "Export Table:" in text else ""
    return {
        "needed": re.findall(r"DLL Name:\s+(\S+)", text),
        "tsggmlExports": len(re.findall(r"\sTSGgml_\w+", exports)),
        "tsggmlBuildIdentityExport": bool(re.search(r"\sTSGgml_GetBuildIdentity(?:\s|$)", exports)),
    }


def macho_dynamic(path):
    text = run(["otool", "-L", str(path)])
    if text is None:
        return None
    own = (run(["otool", "-D", str(path)]) or "").strip().splitlines()[1:]
    deps = [line.strip().split(" (")[0] for line in text.splitlines()[1:] if line.strip()]
    exports = run(["nm", "-gU", str(path)]) or ""
    return {
        "installName": own[0] if own else None,
        "needed": [d for d in deps if d not in own],
        "tsggmlExports": len(re.findall(r"\s_TSGgml_\w+$", exports, re.M)),
        "tsggmlBuildIdentityExport": bool(re.search(r"\s_TSGgml_GetBuildIdentity$", exports, re.M)),
    }


def dependency_basename(fmt, dep):
    if (not isinstance(dep, str) or not dep or "\\" in dep or ":" in dep
            or any(ord(char) < 32 or ord(char) == 127 for char in dep)
            or any(part in ("", ".", "..") for part in dep.lstrip("/").split("/"))):
        return None
    if fmt == "macho":
        if dep.startswith(("/System/Library/", "/usr/lib/")) and not dep.startswith("//"):
            return PurePosixPath(dep).name
        if dep.startswith("@loader_path/") and dep.count("/") == 1:
            return dep.split("/", 1)[1]
        # @rpath and bare names need an additional search scope that this
        # inventory does not establish. Only the selected directory is trusted.
        return None
    return dep if "/" not in dep and not dep.startswith("@") else None


def classify(fmt, dep, siblings):
    base = dependency_basename(fmt, dep)
    if base is None:
        return "unresolved-loader-path"
    if fmt == "macho" and dep.startswith(("/System/Library/", "/usr/lib/")):
        return "os"
    lower = {s.lower() for s in siblings}
    if fmt == "pe" and base.lower() in lower or fmt != "pe" and base in siblings:
        return "bundled"
    if fmt == "elf":
        if base in LINUX_OS:
            return "os"
        if base in LINUX_OS_PACKAGE:
            return "os-package"
        return LINUX_GPU_RUNTIME.get(base, "unresolved")
    if fmt == "pe":
        low = base.lower()
        if low in WINDOWS_OS or low.startswith("api-ms-win-"):
            return "os"
        if MSVC_REDIST.match(low):
            return "unresolved-msvc-redist"
        return WINDOWS_GPU_RUNTIME.get(low, "unresolved")
    if fmt == "macho":
        if dep.startswith("/System/Library/") or dep.startswith("/usr/lib/"):
            return "os"
        return "unresolved"
    return "unresolved"


# ---------------------------------------------------------------- inputs

def collect(root):
    """Yield (relative path, rid, variant, file path) for native files under root/runtimes."""
    for path in sorted((root / "runtimes").rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 4 or parts[2] != "native":
            yield "/".join(parts), parts[1] if len(parts) > 1 else None, None, path
            continue
        variant = parts[3] if len(parts) > 4 else "package"
        yield "/".join(parts), parts[1], variant, path


def open_input(arg, work_dir):
    path = Path(arg).resolve()
    if path.is_dir():
        return {"kind": "directory", "path": str(path)}, path
    source = {"kind": "nupkg", "path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
    target = work_dir / path.stem
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(path) as z:
        entries = [n for n in z.namelist() if not n.endswith("/")]
        source["entries"] = sorted(entries)
        for name in entries:
            dest = (target / name).resolve()
            if target.resolve() not in dest.parents:
                raise SystemExit(f"{path}: entry escapes the extraction directory: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
        nuspec = next((n for n in entries if n.endswith(".nuspec") and "/" not in n), None)
        if nuspec:
            text = z.read(nuspec).decode("utf-8-sig")
            for field in ("id", "version"):
                m = re.search(rf"<{field}>([^<]+)</{field}>", text)
                source[field] = m.group(1) if m else None
            repo = re.search(r"<repository\b([^>]*)/>", text)
            if repo:
                source["repository"] = dict(re.findall(r'(\w+)="([^"]*)"', repo.group(1)))
            lic = re.search(r"<license[^>]*>([^<]+)</license>", text)
            source["license"] = lic.group(1) if lic else None
    return source, target


def describe(rel, rid, variant, path):
    data = path.read_bytes()
    fmt = read_header(data)
    record = {"path": rel, "rid": rid, "variant": variant, "size": len(data),
              "sha256": hashlib.sha256(data).hexdigest(), "format": fmt}
    siblings = {p.name for p in path.parent.iterdir() if p.is_file()}
    if fmt == "elf":
        record["identity"] = elf_info(data)
        dyn = elf_dynamic(path)
    elif fmt == "pe":
        record["identity"] = pe_info(data)
        dyn = pe_dynamic(path)
    elif fmt == "macho":
        record["identity"] = macho_info(data)
        dyn = macho_dynamic(path)
    else:
        return record
    if dyn is None:
        record["dependencies"] = "not inspected: objdump/otool unavailable"
        return record
    record["dependencies"] = [{"name": d, "resolution": classify(fmt, d, siblings)} for d in dyn.pop("needed")]
    record.update(dyn)
    return record


def check(inputs):
    findings = []
    for source in inputs:
        files = [f for f in source["files"] if f["rid"]]
        rids = sorted({f["rid"] for f in files})
        label = source.get("id") or source["path"]
        if source["kind"] == "nupkg" and len(rids) > 1:
            findings.append({"severity": "warning", "input": label,
                             "message": f"one package carries {len(rids)} RIDs: {', '.join(rids)}"})
        for f in files:
            os_part, _, arch_part = f["rid"].partition("-")
            ident = f.get("identity", {})
            if f["format"] != "other" and RID_FORMAT.get(os_part) != f["format"]:
                findings.append({"severity": "error", "input": label, "file": f["path"],
                                 "message": f"{f['format']} binary under RID {f['rid']}"})
            if ident.get("arch") and RID_ARCH.get(arch_part) != ident["arch"]:
                findings.append({"severity": "error", "input": label, "file": f["path"],
                                 "message": f"{ident['arch']} binary under RID {f['rid']}"})
            for dep in f.get("dependencies", []) if isinstance(f.get("dependencies"), list) else []:
                if dep["resolution"].startswith("unresolved"):
                    findings.append({"severity": "warning", "input": label, "file": f["path"],
                                     "message": f"dependency {dep['name']} is {dep['resolution']}"})
            for rp in f.get("runpath", []):
                if rp == "$ORIGIN":
                    continue
                findings.append({"severity": "warning", "input": label, "file": f["path"],
                                 "message": f"build-machine search path embedded: {rp}"})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "tmp" / "native-artifact-manifest")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    inputs = []
    for arg in args.inputs:
        source, root = open_input(arg, args.work_dir)
        source["files"] = [describe(*item) for item in collect(root)]
        inputs.append(source)
    findings = check(inputs)
    manifest = {"schema": "tensorsharp-native-artifact-inventory/0", "inputs": inputs, "findings": findings}
    text = json.dumps(manifest, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)
    for f in findings:
        print(f"{f['severity']}: {f['input']}: {f.get('file', '')} {f['message']}", file=sys.stderr)
    return 1 if any(f["severity"] == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
