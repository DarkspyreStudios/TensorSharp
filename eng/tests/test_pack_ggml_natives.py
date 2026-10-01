"""Packaging validation only. The compiled fixture exports an identity, not a GGML backend."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile


spec = importlib.util.spec_from_file_location("ggml_packer", Path(__file__).parents[1] / "pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class ArtifactPolicyTests(unittest.TestCase):
    def test_packaging_and_native_build_compute_the_same_abi(self):
        result = subprocess.run(["cmake", "-P", str(pack.REPO_ROOT / "eng" / "print-ggml-native-abi.cmake")],
                                capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual("-- GgmlNativeAbi=" + pack.native_abi(pack.REPO_ROOT), result.stdout.strip())

    def test_abi_normalizes_line_endings_and_tracks_bridge_interop_and_upstream(self):
        (pack.REPO_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="abi-inputs-", dir=pack.REPO_ROOT / "tmp") as temporary:
            root = Path(temporary)
            native, managed, eng = root / "TensorSharp.GGML.Native", root / "TensorSharp.Backends.GGML", root / "eng"
            for directory in (native, managed, eng):
                directory.mkdir()
            inputs = [native / "bridge.cpp", native / "bridge.cuh", managed / "GgmlNative.cs", managed / "QwenImage21Native.cs", eng / "ggml-revision"]
            for path in inputs:
                path.write_bytes(b"\xef\xbb\xbfinput\r\n")
            loader = managed / "GgmlNativeLoader.cs"
            loader.write_bytes(b"loader excluded")
            expected = pack.native_abi(root)
            result = subprocess.run(["cmake", "-DROOT=" + str(root), "-P", str(pack.REPO_ROOT / "eng" / "print-ggml-native-abi.cmake")],
                                    capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual("-- GgmlNativeAbi=" + expected, result.stdout.strip())
            for path in inputs:
                path.write_bytes(b"\xef\xbb\xbfinput\n")
            loader.write_bytes(b"another loader version")
            self.assertEqual(expected, pack.native_abi(root))
            for path in inputs:
                path.write_bytes(b"changed")
                self.assertNotEqual(expected, pack.native_abi(root))
                path.write_bytes(b"\xef\xbb\xbfinput\n")

    def test_all_five_baselines_and_required_optional_variants_are_explicit(self):
        self.assertEqual(pack.BASELINE, {"osx-arm64": "metal", "linux-x64": "cpu", "linux-arm64": "cpu",
                                         "win-x64": "cpu", "win-arm64": "cpu"})
        self.assertEqual(pack.VARIANTS, {"osx-arm64": {"metal"}, "linux-x64": {"cpu", "vulkan", "cuda13"},
                                        "linux-arm64": {"cpu", "vulkan", "cuda13"},
                                        "win-x64": {"cpu", "vulkan", "cuda13"}, "win-arm64": {"cpu", "vulkan"}})

    def facts(self):
        return {"path": "GgmlOps.dll", "format": "pe", "identity": {"arch": "arm64"},
                "dependencies": [], "tsggmlBuildIdentityExport": True}

    def check(self, primary, extra=()):
        names = ["GgmlOps.dll", *pack.REQUIRED_LICENSES, *(item["path"] for item in extra)]
        return pack.check_artifact("win-arm64", "cpu", None, {"files": [{"path": name} for name in names]}, [primary, *extra])

    def test_valid_inspection_facts_pass_policy_without_claiming_backend_execution(self):
        self.assertEqual([], self.check(self.facts()))

    def test_missing_export_unknown_arch_wrong_format_and_uninspected_dependencies_fail(self):
        for changes in ({"tsggmlBuildIdentityExport": False}, {"tsggmlBuildIdentityExport": None},
                        {"identity": {}}, {"identity": {"arch": "x86_64"}}, {"format": "other"},
                        {"dependencies": "not inspected"}):
            with self.subTest(changes=changes):
                self.assertTrue(self.check(self.facts() | changes))

    def test_unresolved_dependencies_build_paths_missing_notices_and_driver_files_fail(self):
        for extra in ({"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"},
                       "dependencies": [{"name": "missing.dll", "resolution": "unresolved"}]},
                      {"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"},
                       "dependencies": "not inspected"},
                      {"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"},
                       "dependencies": [], "runpath": ["/build-machine/lib"]},
                      {"path": "VCRUNTIME140.dll", "format": "other"},
                      {"path": "NVCUDA.DLL", "format": "other"}):
            with self.subTest(extra=extra):
                self.assertTrue(self.check(self.facts(), [extra]))

    def test_unsupported_pair_reports_a_refusal_instead_of_guessing_a_baseline(self):
        self.assertEqual(["win-arm64/cuda13: unsupported RID/variant pair"],
                         pack.check_artifact("win-arm64", "cuda13", None, {"files": []}, []))

    def test_archive_entries_are_deterministic_regular_files(self):
        first = pack.zip_bytes([("licenses/license.txt", b"license"), ("library.bin", b"binary")])
        self.assertEqual(first, pack.zip_bytes([("library.bin", b"binary"), ("licenses/license.txt", b"license")]))
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(["library.bin", "licenses/license.txt"], archive.namelist())
            for info in archive.infolist():
                self.assertEqual(pack.FIXED_TIME, info.date_time)
                self.assertEqual(stat.S_IFREG, stat.S_IFMT(info.external_attr >> 16))

    def test_archive_paths_reject_duplicates_traversal_and_nonportable_names(self):
        for name in ("../escape", "/absolute", "C:/drive", "directory\\file", "a//b", "a/./b", "a/../b", "a\x00b", "a\x7fb", "file.", "file ", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pack.zip_bytes([(name, b"data")])
        with self.assertRaises(ValueError):
            pack.zip_bytes([("same", b"one"), ("same", b"two")])

    def test_inventory_requires_the_exact_defined_identity_export(self):
        cases = [(pack.INVENTORY.elf_dynamic, "\n NEEDED libc.so.6\n", "000000 g DF .text 00010 Base TSGgml_GetBuildIdentity\n"),
                 (pack.INVENTORY.pe_dynamic, "Export Table:\n 1 TSGgml_GetBuildIdentity\n", None),
                 (pack.INVENTORY.macho_dynamic, "fixture:\n /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n", "000 T _TSGgml_GetBuildIdentity\n")]
        for reader, dynamic, exports in cases:
            with self.subTest(reader=reader.__name__):
                def output(args):
                    if args[0] == "nm" or "-T" in args:
                        return exports
                    if "-D" in args:
                        return "fixture:\nfixture\n"
                    return dynamic
                with patch.object(pack.INVENTORY, "run", side_effect=output):
                    self.assertTrue(reader(Path("fixture"))["tsggmlBuildIdentityExport"])
                with patch.object(pack.INVENTORY, "run", side_effect=lambda args: output(args).replace("GetBuildIdentity", "GetBuildIdentityExtra")):
                    self.assertFalse(reader(Path("fixture"))["tsggmlBuildIdentityExport"])

    def test_a_failed_inspection_command_is_not_empty_successful_output(self):
        with patch.object(pack.INVENTORY.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "partial", "failed")):
            self.assertIsNone(pack.INVENTORY.run(["inspection-tool"]))


class StagingFilesystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host = (platform.system(), platform.machine().lower())
        rids = {("Darwin", "arm64"): "osx-arm64", ("Linux", "x86_64"): "linux-x64",
                ("Linux", "aarch64"): "linux-arm64"}
        cls.rid = rids.get(host)
        compiler = shutil.which("clang" if host[0] == "Darwin" else "cc")
        tools = ("otool", "nm") if host[0] == "Darwin" else ("objdump",)
        if cls.rid is None or compiler is None or any(shutil.which(tool) is None for tool in tools):
            raise unittest.SkipTest("native header/export inspection fixture needs a supported host compiler and inspection tools")
        (pack.REPO_ROOT / "tmp").mkdir(exist_ok=True)
        cls.fixture = tempfile.TemporaryDirectory(prefix="pack-inspection-fixture-", dir=pack.REPO_ROOT / "tmp")
        cls.version = ET.parse(pack.REPO_ROOT / "Directory.Build.props").findtext(".//TensorSharpVersion")
        cls.ggml = (pack.REPO_ROOT / "eng" / "ggml-revision").read_text().strip()
        cls.source = "1" * 40
        cls.variant = pack.BASELINE[cls.rid]
        cls.entry = pack.ENTRY[cls.rid.split("-")[0]]
        cls.library = Path(cls.fixture.name) / cls.entry
        cls.abi = pack.native_abi(pack.REPO_ROOT)
        identity = f"format=1;tensorsharp={cls.version};source={cls.source};ggml={cls.ggml};rid={cls.rid};variant={cls.variant};cpu=portable;abi={cls.abi}"
        source_path = Path(cls.fixture.name) / "identity.c"
        source_path.write_text(f"const char *TSGgml_GetBuildIdentity(void) {{ return {json.dumps(identity)}; }}\n", encoding="utf-8")
        flags = ["-dynamiclib"] if host[0] == "Darwin" else ["-shared", "-fPIC"]
        try:
            subprocess.run([compiler, *flags, str(source_path), "-o", str(cls.library)], check=True, capture_output=True)
        except BaseException:
            cls.fixture.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pack-staging-test-", dir=pack.REPO_ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stage = self.root / "stage"
        self.artifact = self.stage / "runtimes" / self.rid / "native" / self.variant
        self.artifact.mkdir(parents=True)
        shutil.copyfile(self.library, self.artifact / self.entry)
        for name in pack.REQUIRED_LICENSES:
            path = self.artifact / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("inspection fixture license", encoding="utf-8")
        self.build_dir = self.stage / "build" / f"{self.rid}-{self.variant}"
        self.build_dir.mkdir(parents=True)
        self.build = {"tensorSharpBuild": self.version, "sourceCommit": self.source, "ggmlCommit": self.ggml,
                      "rid": self.rid, "variant": self.variant, "nativeAbi": self.abi}
        self.write_build()
        self.out = self.root / "output"

    def write_build(self):
        (self.build_dir / "build-identity.json").write_text(json.dumps(self.build), encoding="utf-8")

    def run_cli(self, *extra):
        messages = io.StringIO()
        with patch("sys.argv", ["pack-ggml-natives.py", "--stage", str(self.stage), "--out", str(self.out), *extra]), \
                contextlib.redirect_stdout(messages), contextlib.redirect_stderr(messages):
            result = pack.main()
        self.assertFalse(self.out.exists(), "validation must not create release output")
        return result, messages.getvalue()

    def test_validate_only_inspects_real_headers_and_exports_without_creating_packages(self):
        code, output = self.run_cli("--validate-only")
        self.assertEqual(0, code, output)
        self.assertIn("validated 1 staged artifacts; no output written", output)

    def test_invalid_build_records_fail_before_packaging(self):
        for key, value in (("tensorSharpBuild", "wrong"), ("sourceCommit", "2" * 40), ("sourceCommit", "not-a-commit"),
                           ("rid", "win-arm64"), ("variant", "vulkan"), ("ggmlCommit", "wrong"), ("nativeAbi", "0" * 64)):
            with self.subTest(key=key, value=value):
                original = self.build[key]
                self.build[key] = value
                self.write_build()
                code, output = self.run_cli()
                self.assertEqual(1, code, output)
                self.build[key] = original
        self.write_build()

    def test_text_with_an_identity_is_not_a_native_bridge(self):
        (self.artifact / self.entry).write_bytes(self.library.read_bytes().split(b"format=1;tensorsharp=", 1)[1].split(b"\0", 1)[0])
        code, output = self.run_cli()
        self.assertEqual(1, code, output)
        self.assertIn("not a native bridge", output)

    def test_missing_license_missing_export_and_uninspected_dependencies_fail_before_output(self):
        (self.artifact / pack.REQUIRED_LICENSES[0]).unlink()
        self.assertEqual(1, self.run_cli()[0])
        (self.artifact / pack.REQUIRED_LICENSES[0]).write_text("license", encoding="utf-8")
        for field, value in (("tsggmlBuildIdentityExport", False), ("dependencies", "not inspected")):
            with self.subTest(field=field):
                describe = pack.INVENTORY.describe
                def altered(*args):
                    result = describe(*args)
                    if result["path"] == self.entry:
                        result[field] = value
                    return result
                with patch.object(pack.INVENTORY, "describe", side_effect=altered):
                    self.assertEqual(1, self.run_cli()[0])

    def test_linked_files_directories_records_and_stage_roots_fail_without_output(self):
        for name in (self.entry, "licenses", None):
            with self.subTest(name=name):
                path = self.artifact / name if name else self.build_dir / "build-identity.json"
                saved = self.root / "saved"
                path.rename(saved)
                path.symlink_to(saved, target_is_directory=saved.is_dir())
                self.assertEqual(1, self.run_cli()[0])
                path.unlink()
                saved.rename(path)
        alias = self.root / "stage-link"
        alias.symlink_to(self.stage, target_is_directory=True)
        self.stage = alias
        self.assertEqual(1, self.run_cli()[0])

    def test_dangling_settings_links_and_special_files_are_not_silently_ignored(self):
        settings = self.build_dir / "cmake-settings.txt"
        settings.symlink_to(self.root / "absent")
        self.assertEqual(1, self.run_cli()[0])
        settings.unlink()
        os.mkfifo(self.artifact / "fifo")
        self.assertEqual(1, self.run_cli()[0])

    def test_unknown_rids_and_variants_are_not_packaged(self):
        for kind in ("rid", "variant"):
            with self.subTest(kind=kind):
                path = self.stage / "runtimes" / self.rid if kind == "rid" else self.artifact
                renamed = path.with_name("unsupported")
                path.rename(renamed)
                self.assertEqual(1, self.run_cli()[0])
                renamed.rename(path)

    def test_output_cannot_overlap_the_artifact_tree(self):
        self.out = self.artifact / "output"
        self.assertEqual(1, self.run_cli()[0])

    def test_managed_zip_metadata_is_checked_before_any_release_output(self):
        archive_path = self.root / "managed-fixture.zip"
        metadata = (f'<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"><metadata>'
                    f'<id>Darkspyre.TensorSharp.Backends.GGML</id><version>{self.version}</version></metadata></package>')
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("fixture.nuspec", metadata)
            archive.writestr("lib/net10.0/TensorSharp.Backends.GGML.dll", b"managed fixture")
        code, output = self.run_cli("--validate-only", "--managed-package", str(archive_path))
        self.assertEqual(0, code, output)
        for name, text in (("fixture.nuspec", metadata.replace(self.version, "wrong")),
                           ("libGgmlOps.so", "native fixture"), ("../escape", "fixture")):
            with self.subTest(name=name), zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(name, text)
            self.assertEqual(1, self.run_cli("--managed-package", str(archive_path))[0])
        with zipfile.ZipFile(archive_path, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
        self.assertEqual(1, self.run_cli("--managed-package", str(archive_path))[0])


if __name__ == "__main__":
    unittest.main()
