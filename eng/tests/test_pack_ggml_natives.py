"""Packaging validation only. The compiled fixture exports ABI stubs, not a GGML backend."""
import contextlib
import copy
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

    def test_abi_normalizes_line_endings_and_tracks_bridge_backend_and_upstream(self):
        (pack.REPO_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="abi-inputs-", dir=pack.REPO_ROOT / "tmp") as temporary:
            root = Path(temporary)
            native, managed, eng = root / "TensorSharp.GGML.Native", root / "TensorSharp.Backends.GGML", root / "eng"
            for directory in (native, managed, eng):
                directory.mkdir()
            inputs = [native / "bridge.cpp", native / "bridge.cuh", managed / "GgmlNative.cs",
                      managed / "QwenImage21Native.cs", managed / "GgmlTensorParallel.cs",
                      managed / "GgmlContext.cs", managed / "GgmlNativeLoader.cs", eng / "ggml-revision"]
            for path in inputs:
                path.write_bytes(b"\xef\xbb\xbfinput\r\n")
            expected = pack.native_abi(root)
            result = subprocess.run(["cmake", "-DROOT=" + str(root), "-P", str(pack.REPO_ROOT / "eng" / "print-ggml-native-abi.cmake")],
                                    capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual("-- GgmlNativeAbi=" + expected, result.stdout.strip())
            for path in inputs:
                path.write_bytes(b"\xef\xbb\xbfinput\n")
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

    def complete_matrix(self):
        return [{"rid": rid, "variant": variant} for rid, variants in pack.VARIANTS.items() for variant in variants]

    def test_complete_release_requires_every_one_of_the_twelve_canonical_pairs(self):
        complete = self.complete_matrix()
        self.assertEqual(12, len(complete))
        self.assertEqual([], pack.check_release_matrix(complete))
        for missing in complete:
            with self.subTest(missing=missing):
                errors = pack.check_release_matrix([item for item in complete if item != missing])
                self.assertEqual([f"complete release: missing {missing['rid']}/{missing['variant']}"], errors)

    def test_complete_release_rejects_unknown_and_duplicate_pairs(self):
        complete = self.complete_matrix()
        self.assertIn("complete release: unsupported win-arm64/cuda13",
                      pack.check_release_matrix(complete + [{"rid": "win-arm64", "variant": "cuda13"}]))
        self.assertEqual(["complete release: duplicate RID/variant artifacts"],
                         pack.check_release_matrix(complete + complete[:1]))

    def facts(self):
        return {"path": "GgmlOps.dll", "format": "pe", "identity": {"arch": "arm64"},
                "dependencies": [], "tsggmlBuildIdentityExport": True}

    def check(self, primary, extra=()):
        names = ["GgmlOps.dll", *pack.REQUIRED_LICENSES, *(item["path"] for item in extra)]
        return pack.check_artifact("win-arm64", "cpu", None, {"files": [{"path": name} for name in names]}, [primary, *extra])

    def test_valid_inspection_facts_pass_policy_without_claiming_backend_execution(self):
        self.assertEqual([], self.check(self.facts()))

    def test_bundled_dependency_requires_a_matching_inspected_native_sibling(self):
        primary = self.facts() | {"dependencies": [{"name": "dependency.dll", "resolution": "bundled"}]}
        valid = {"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"}, "dependencies": []}
        self.assertEqual([], self.check(primary, [valid]))
        for sibling in (valid | {"format": "other"}, valid | {"format": "elf"},
                        valid | {"identity": {"arch": "x86_64"}}, valid | {"identity": {}},
                        valid | {"dependencies": "not inspected"}):
            with self.subTest(sibling=sibling):
                errors = self.check(primary, [sibling])
                self.assertTrue(any("not an inspected native sibling" in error for error in errors), errors)

    def test_bundled_dependency_validation_uses_actual_facts_not_claimed_resolution(self):
        primary = self.facts() | {"dependencies": [{"name": "dependency.dll", "resolution": "os"}]}
        text = {"path": "dependency.dll", "format": "other"}
        self.assertTrue(any("not an inspected native sibling" in error for error in self.check(primary, [text])))
        primary["dependencies"][0]["name"] = "missing.dll"
        self.assertTrue(any("is unresolved" in error for error in self.check(primary)))

    def test_windows_sibling_matching_is_case_insensitive_but_duplicate_names_refuse(self):
        primary = self.facts() | {"dependencies": [{"name": "DEPENDENCY.DLL", "resolution": "bundled"}]}
        valid = {"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"}, "dependencies": []}
        self.assertEqual([], self.check(primary, [valid]))
        self.assertTrue(any("competing native sibling" in error for error in
                            self.check(primary, [valid, valid | {"path": "Dependency.dll"}])))

    def test_absolute_or_traversing_dependency_names_cannot_match_a_sibling(self):
        sibling = {"path": "dependency.dll", "format": "pe", "identity": {"arch": "arm64"}, "dependencies": []}
        for name in ("/build/dependency.dll", "C:\\build\\dependency.dll", "C:/build/dependency.dll",
                     "../dependency.dll", "folder/dependency.dll", "dependency.dll\x00"):
            primary = self.facts() | {"dependencies": [{"name": name, "resolution": "bundled"}]}
            with self.subTest(name=name):
                self.assertTrue(any("unresolved-loader-path" in error for error in self.check(primary, [sibling])))

    def test_platform_dependency_paths_only_allow_verified_loader_scopes(self):
        classify = pack.INVENTORY.classify
        self.assertEqual("bundled", classify("elf", "libdependency.so", {"libdependency.so"}))
        self.assertEqual("os", classify("elf", "libc.so.6", set()))
        self.assertEqual("bundled", classify("macho", "@loader_path/libdependency.dylib", {"libdependency.dylib"}))
        self.assertEqual("os", classify("macho", "/System/Library/Frameworks/Metal.framework/Versions/A/Metal", set()))
        self.assertEqual("os", classify("macho", "/usr/lib/libSystem.B.dylib", {"libSystem.B.dylib"}))
        for fmt, name in (("elf", "/build/libdependency.so"), ("elf", "/usr/lib/libc.so.6"),
                          ("macho", "/build/libdependency.dylib"), ("macho", "@rpath/libdependency.dylib"),
                          ("macho", "libdependency.dylib"), ("macho", "@loader_path/../libdependency.dylib"),
                          ("macho", "/usr/lib/../build/libdependency.dylib")):
            with self.subTest(fmt=fmt, name=name):
                self.assertEqual("unresolved-loader-path", classify(fmt, name, {"libdependency.so", "libdependency.dylib"}))

    def test_linux_bundled_dependency_requires_an_origin_relative_search_scope(self):
        primary = {"path": "libGgmlOps.so", "format": "elf", "identity": {"arch": "arm64"},
                   "tsggmlBuildIdentityExport": True, "dependencies": [{"name": "libdependency.so", "resolution": "bundled"}]}
        sibling = {"path": "libdependency.so", "format": "elf", "identity": {"arch": "arm64"}, "dependencies": []}
        files = [{"path": name} for name in ("libGgmlOps.so", "libdependency.so", *pack.REQUIRED_LICENSES)]
        check = lambda bridge: pack.check_artifact("linux-arm64", "cpu", None, {"files": files}, [bridge, sibling])
        self.assertTrue(any("no selected-directory $ORIGIN" in error for error in check(primary)))
        self.assertEqual([], check(primary | {"runpath": ["$ORIGIN"]}))
        self.assertTrue(check(primary | {"runpath": ["/build"]}))

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


class PackageCatalogPolicyTests(unittest.TestCase):
    """Catalog/payload descriptions only; no package/archive or native build is produced."""

    def artifact(self, rid="linux-x64", variant="vulkan"):
        entry = pack.ENTRY[rid.split("-")[0]]
        return {"driverId": pack.DRIVER_ID, "rid": rid, "variant": variant,
                "version": "2.8.6.8", "tensorSharpBuild": "2.8.6.8", "nativeAbi": "a" * 64,
                "tensorSharp": {"packageVersion": "2.8.6.8", "packageCommit": "b" * 40,
                                "nativeSourceCommit": "c" * 40},
                "ggml": {"version": "0.9.0", "commit": "d" * 40},
                "backends": pack.BACKENDS[variant], "entryLibrary": entry,
                "files": [{"path": entry, "size": 7, "sha256": pack.sha256_bytes(b"fixture")},
                          {"path": "licenses/ggml-LICENSE.txt", "size": 7, "sha256": pack.sha256_bytes(b"license")}],
                "components": [{"id": "example-evidence"}],
                "archive": {"name": "not-created.zip"}, "build": {"sourceCommit": "c" * 40}}

    def test_catalog_copies_complete_existing_identity_without_recomputing_evidence(self):
        artifact = self.artifact()
        catalog = json.loads(pack.package_catalog(artifact))
        self.assertEqual(pack.SCHEMA, catalog.pop("schema"))
        self.assertEqual({key: artifact[key] for key in (
            "driverId", "rid", "variant", "version", "tensorSharpBuild", "nativeAbi",
            "tensorSharp", "ggml", "backends", "entryLibrary", "files", "components")}, catalog)
        self.assertNotIn("archive", catalog)
        self.assertNotIn("build", catalog)
        self.assertEqual(pack.package_catalog(artifact), pack.package_catalog(artifact))

    def test_catalog_file_portability_matches_runtime_validation(self):
        for name in ("wild*card", "question?mark", 'quote"name', "pipe|name", "less<name", "greater>name"):
            with self.subTest(name=name):
                self.assertFalse(pack.portable_path(name))

    def test_baseline_catalog_and_complete_closure_have_fixed_runtime_paths(self):
        for rid, variant in pack.BASELINE.items():
            with self.subTest(rid=rid):
                artifact = self.artifact(rid, variant)
                files = {item["path"]: b"fixture" for item in artifact["files"]}
                with patch.object(Path, "read_bytes", lambda path: files[path.as_posix().removeprefix("/fixture/")]):
                    contents = dict(pack.native_package_contents(artifact, Path("/fixture"), "package", True))
                for item in artifact["files"]:
                    self.assertIn(f"runtimes/{rid}/native/{item['path']}", contents)
                self.assertIn("licenses/ggml-LICENSE.txt", contents)
                self.assertEqual(json.loads(pack.package_catalog(artifact)),
                                 json.loads(contents["ggml/baseline.artifact.json"]))
                self.assertIn("buildTransitive/package.targets", contents)

    def test_developer_catalog_and_native_closure_are_siblings(self):
        artifact = self.artifact()
        files = {item["path"]: b"fixture" for item in artifact["files"]}
        with patch.object(Path, "read_bytes", lambda path: files[path.as_posix().removeprefix("/fixture/")]):
            contents = dict(pack.native_package_contents(artifact, Path("/fixture"), "package", False))
        self.assertIn("ggml/vulkan.artifact.json", contents)
        self.assertIn("ggml/vulkan/licenses/ggml-LICENSE.txt", contents)
        self.assertIn("ggml/vulkan/libGgmlOps.so", contents)

    def test_copy_targets_include_catalog_for_output_and_publish(self):
        document = ET.fromstring(pack.developer_targets("package", "vulkan", "linux-x64"))
        items = document.findall("ItemGroup/None")
        catalog = next((item for item in items if item.get("Link") == "ggml/vulkan.artifact.json"), None)
        self.assertIsNotNone(catalog)
        self.assertEqual("$(MSBuildThisFileDirectory)../ggml/vulkan.artifact.json", catalog.get("Include"))
        self.assertEqual("PreserveNewest", catalog.get("CopyToOutputDirectory"))
        self.assertEqual("PreserveNewest", catalog.get("CopyToPublishDirectory"))

    def test_baseline_targets_bind_only_selected_rid_without_architecture_mix(self):
        document = ET.fromstring(pack.baseline_targets("package", "win-arm64"))
        group = document.find("ItemGroup")
        self.assertIn("win-arm64", group.get("Condition"))
        self.assertIn("RuntimeIdentifier", group.get("Condition"))
        items = group.findall("None")
        self.assertEqual({"ggml/baseline.artifact.json", "runtimes/win-arm64/native/%(RecursiveDir)%(Filename)%(Extension)"},
                         {item.get("Link") for item in items})
        self.assertTrue(all(item.get("CopyToOutputDirectory") == "PreserveNewest" and
                            item.get("CopyToPublishDirectory") == "PreserveNewest" for item in items))

    def test_actual_msbuild_evaluation_preserves_catalog_payload_output_and_publish_links(self):
        (pack.REPO_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="catalog-copy-", dir=pack.REPO_ROOT / "tmp") as temporary:
            root = Path(temporary)
            (root / "buildTransitive").mkdir()
            (root / "ggml/vulkan").mkdir(parents=True)
            (root / "runtimes/win-arm64/native/licenses").mkdir(parents=True)
            for path in ("ggml/vulkan/libGgmlOps.so", "ggml/vulkan.artifact.json",
                         "ggml/baseline.artifact.json", "runtimes/win-arm64/native/GgmlOps.dll",
                         "runtimes/win-arm64/native/licenses/ggml-LICENSE.txt"):
                (root / path).write_text("controlled copy fixture; not native")
            (root / "buildTransitive/optional.targets").write_bytes(pack.developer_targets("optional", "vulkan", "win-arm64"))
            (root / "buildTransitive/baseline.targets").write_bytes(pack.baseline_targets("baseline", "win-arm64"))
            project = root / "copy.proj"
            project.write_text('<Project><Import Project="buildTransitive/optional.targets" />'
                               '<Import Project="buildTransitive/baseline.targets" /></Project>')
            for requested, host, include_baseline in (("win-arm64", "linux-x64", True),
                                                     ("linux-x64", "win-arm64", False),
                                                     ("", "win-arm64", True), ("", "linux-x64", False)):
                result = subprocess.run(["dotnet", "msbuild", str(project), "-nologo", "-getItem:None",
                                         "-p:RuntimeIdentifier=" + requested, "-p:NETCoreSdkRuntimeIdentifier=" + host],
                                        capture_output=True, text=True, check=True, timeout=20)
                items = json.loads(result.stdout)["Items"]["None"]
                links = {item["Link"] for item in items}
                self.assertEqual(include_baseline, "ggml/vulkan.artifact.json" in links)
                self.assertEqual(include_baseline, "ggml/vulkan/libGgmlOps.so" in links)
                self.assertEqual(include_baseline, "ggml/baseline.artifact.json" in links)
                if include_baseline:
                    self.assertIn("runtimes/win-arm64/native/licenses/ggml-LICENSE.txt", links)
                self.assertTrue(all(item["CopyToOutputDirectory"] == "PreserveNewest" and
                                    item["CopyToPublishDirectory"] == "PreserveNewest" for item in items))


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
        identity = f"format=1;tensorsharp={cls.version};source={cls.source};ggml={cls.ggml};rid={cls.rid};variant={cls.variant};cpu={pack.CPU_PROFILES[cls.rid]};abi={cls.abi}"
        source_path = Path(cls.fixture.name) / "identity.c"
        stubs = "".join(f"void {name}(void) {{}}\n" for name in pack.read_required_exports(pack.REPO_ROOT)
                        if name != "TSGgml_GetBuildIdentity")
        source_path.write_text(f"const char *TSGgml_GetBuildIdentity(void) {{ return {json.dumps(identity)}; }}\n" + stubs, encoding="utf-8")
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
        settings = {"TENSORSHARP_NATIVE_ABI": self.abi, "TENSORSHARP_NATIVE_RID": self.rid,
                    "TENSORSHARP_NATIVE_VARIANT": self.variant, "TENSORSHARP_GGML_NATIVE_PORTABLE": "ON",
                    "GGML_NATIVE": "OFF", "GGML_METAL": "ON" if self.variant == "metal" else "OFF",
                    "GGML_CUDA": "OFF", "GGML_VULKAN": "OFF", "CMAKE_CXX_COMPILER": "inspection-fixture-compiler"}
        if self.rid.startswith("linux-"):
            settings.update(CMAKE_INSTALL_RPATH="$ORIGIN", CMAKE_BUILD_WITH_INSTALL_RPATH="ON")
        if self.rid.startswith("osx-"):
            settings["CMAKE_OSX_DEPLOYMENT_TARGET"] = "14.0"
        self.build.update(cpuProfile=pack.CPU_PROFILES[self.rid], cmakeConfiguration=settings,
                          compiler=settings["CMAKE_CXX_COMPILER"], compilerVersion="inspection fixture, not release execution",
                          macosDeploymentTarget=settings.get("CMAKE_OSX_DEPLOYMENT_TARGET"),
                          bridgeSha256=pack.sha256_bytes((self.artifact / self.entry).read_bytes()))
        self.build["components"] = []
        for expected in pack.core_component_specs(self.build, self.entry):
            self.build["components"].append({key: value for key, value in expected.items() if key not in ("binaryPaths", "evidencePaths")} | {
                "binaryFiles": [self.reference(name) for name in expected["binaryPaths"]],
                "evidenceFiles": [self.reference(name) for name in expected["evidencePaths"]]})
        self.write_build()
        self.out = self.root / "output"

    def write_build(self):
        settings = self.build["cmakeConfiguration"]
        snapshot = "//Synthetic inspection fixture cache\n" + "".join(f"{key}:STRING={value}\n" for key, value in sorted(settings.items()))
        (self.build_dir / pack.CMAKE_CACHE_SNAPSHOT).write_bytes(snapshot.encode())
        self.build["cmakeCacheSha256"] = pack.sha256_bytes(snapshot.encode())
        (self.build_dir / "build-identity.json").write_text(json.dumps(self.build), encoding="utf-8")
        (self.build_dir / "cmake-settings.txt").write_text("".join(f"{key}={value}\n" for key, value in
                                                                 sorted(self.build["cmakeConfiguration"].items())), encoding="utf-8")

    def reference(self, name):
        return {key: value for key, value in pack.file_record(self.artifact, self.artifact / name).items() if key != "executable"}

    def add_runtime_component(self, name, identifier="runtime-one", evidence="licenses/provider-notice.txt"):
        shutil.copyfile(self.library, self.artifact / name)
        (self.artifact / evidence).write_text("Provider redistribution notice inspection fixture\n")
        component = {"id": identifier, "name": "Fixture runtime", "kind": "redistributed", "supplier": "Fixture provider",
                     "version": "1.2.3", "sourceId": "fixture:release-1.2.3", "binaryFiles": [self.reference(name)],
                     "evidenceFiles": [self.reference(evidence)]}
        self.build["components"].append(component)
        self.write_build()
        return component

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

    def test_supplied_components_can_share_actual_evidence_without_a_filename_legal_claim(self):
        suffix = ".dylib" if self.rid.startswith("osx-") else ".so"
        self.add_runtime_component("libfirst" + suffix)
        self.add_runtime_component("libsecond" + suffix, "runtime-two")
        code, output = self.run_cli("--validate-only")
        self.assertEqual(0, code, output)
        artifacts, errors = pack.collect_artifacts(self.stage, self.version, self.ggml)
        self.assertEqual([], errors)
        self.assertEqual(self.build["components"], artifacts[0]["components"])
        notice = pack.notice_markdown("Fixture package", artifacts[0]["components"]).decode()
        self.assertIn("Fixture provider / 1.2.3", notice)
        self.assertIn("licenses/provider-notice.txt", notice)
        self.assertIn("do not certify redistribution permission", notice)
        self.assertNotIn("Distributable Code", notice)

    def test_missing_or_forged_component_identity_refuses(self):
        original = copy.deepcopy(self.build["components"])
        for changed in (None, [], original[:1], original + original[:1],
                        [original[0] | {"sourceId": "git:" + "9" * 40}, original[1]],
                        [original[0] | {"supplier": "somebody else"}, original[1]]):
            self.build["components"] = changed
            self.write_build()
            with self.subTest(changed=changed):
                code, output = self.run_cli("--validate-only")
                self.assertEqual(1, code, output)

    def test_empty_stale_missing_non_text_or_unmapped_evidence_refuses(self):
        source = self.artifact / pack.REQUIRED_LICENSES[0]
        original = source.read_bytes()
        for data in (b"", b" \n", b"altered license", b"binary\0notice", b"\xff\xfe"):
            source.write_bytes(data)
            self.build["components"][0]["evidenceFiles"] = [self.reference(pack.REQUIRED_LICENSES[0])]
            if data == b"altered license":
                self.build["components"][0]["evidenceFiles"][0]["sha256"] = "0" * 64
            self.write_build()
            with self.subTest(data=data):
                code, output = self.run_cli("--validate-only")
                self.assertEqual(1, code, output)
        source.write_bytes(original)
        self.build["components"][0]["evidenceFiles"] = [self.reference(pack.REQUIRED_LICENSES[0])]
        self.write_build()
        (self.artifact / "licenses/unclaimed.txt").write_text("unmapped notice")
        code, output = self.run_cli("--validate-only")
        self.assertEqual(1, code, output)
        self.assertIn("unmapped staged component", output)

    def test_native_siblings_need_exhaustive_non_conflicting_mapping(self):
        suffix = ".dylib" if self.rid.startswith("osx-") else ".so"
        name = "libruntime" + suffix
        component = self.add_runtime_component(name)
        original = copy.deepcopy(self.build["components"])
        for changed in (original[:2], original + [component | {"id": "competing-runtime"}],
                        original[:2] + [component | {"kind": "static"}],
                        original[:2] + [component | {"binaryFiles": [self.reference(self.entry)]}]):
            self.build["components"] = changed
            self.write_build()
            with self.subTest(changed=changed):
                code, output = self.run_cli("--validate-only")
                self.assertEqual(1, code, output)

    def test_evidence_paths_cannot_collide_on_case_insensitive_targets(self):
        suffix = ".dylib" if self.rid.startswith("osx-") else ".so"
        component = self.add_runtime_component("libruntime" + suffix)
        duplicate = copy.deepcopy(component)
        duplicate["id"] = "case-collision"
        duplicate["binaryFiles"] = [self.reference("libruntime" + suffix) | {"path": "LIBRUNTIME" + suffix}]
        self.build["components"].append(duplicate)
        self.write_build()
        code, output = self.run_cli("--validate-only")
        self.assertEqual(1, code, output)
        self.assertIn("case-conflicting", output)

    def test_component_paths_sizes_hashes_and_supplier_identity_refuse_malformed_input(self):
        original = copy.deepcopy(self.build["components"])
        for field, value in (("path", "../outside.txt"), ("path", "/absolute.txt"), ("path", "licenses\\notice.txt"),
                             ("path", "licenses/../notice.txt"), ("size", True), ("size", -1), ("sha256", "BAD")):
            self.build["components"] = copy.deepcopy(original)
            self.build["components"][0]["evidenceFiles"][0][field] = value
            self.write_build()
            with self.subTest(field=field, value=value):
                self.assertEqual(1, self.run_cli("--validate-only")[0])
        for field in ("supplier", "version", "sourceId"):
            self.build["components"] = copy.deepcopy(original)
            self.build["components"][0][field] = " "
            self.write_build()
            with self.subTest(field=field):
                self.assertEqual(1, self.run_cli("--validate-only")[0])

    def test_complete_validation_and_release_output_refuse_a_partial_stage(self):
        for arguments in (("--validate-only", "--complete-release"), ()):
            with self.subTest(arguments=arguments):
                code, output = self.run_cli(*arguments)
                self.assertEqual(1, code, output)
                self.assertIn("complete release: missing linux-arm64/cuda13", output)
                self.assertIn("complete release: missing win-arm64/cpu", output)
                self.assertIn("complete release: missing win-arm64/vulkan", output)

    def test_complete_validation_accepts_the_full_metadata_matrix_without_output(self):
        complete = [{"rid": rid, "variant": variant} for rid, variants in pack.VARIANTS.items() for variant in variants]
        with patch.object(pack, "collect_artifacts", return_value=(complete, [])):
            code, output = self.run_cli("--validate-only", "--complete-release")
        self.assertEqual(0, code, output)
        self.assertIn("validated 12 staged artifacts; no output written", output)

    def test_real_text_sibling_cannot_satisfy_a_claimed_bundled_dependency(self):
        suffix = ".dylib" if self.rid.startswith("osx-") else ".so"
        name = "libdependency" + suffix
        (self.artifact / name).write_text("not a native library", encoding="utf-8")
        describe = pack.INVENTORY.describe
        def altered(*args):
            result = describe(*args)
            if result["path"] == self.entry:
                dependency = "@loader_path/" + name if self.rid.startswith("osx-") else name
                result["dependencies"].append({"name": dependency, "resolution": "bundled"})
            return result
        with patch.object(pack.INVENTORY, "describe", side_effect=altered):
            code, output = self.run_cli("--validate-only")
        self.assertEqual(1, code, output)
        self.assertIn("not an inspected native sibling", output)

    def test_a_missing_required_symbol_fails_before_any_release_output(self):
        describe = pack.INVENTORY.describe
        def altered(*args):
            result = describe(*args)
            if result["path"] == self.entry:
                result["functionExports"].remove("ggml_quantize_chunk")
            return result
        with patch.object(pack.INVENTORY, "describe", side_effect=altered):
            code, output = self.run_cli("--validate-only")
        self.assertEqual(1, code, output)
        self.assertIn("missing required managed entrypoint ggml_quantize_chunk", output)

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
        settings.unlink()
        settings.symlink_to(self.root / "absent")
        self.assertEqual(1, self.run_cli()[0])
        settings.unlink()
        self.write_build()
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
