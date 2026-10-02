"""Relink proof fixtures validate evidence, not actual bridge or GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("relink_pack", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class IdentityRelinkTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="identity-relink-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.directory = self.temporary / "stage/build/osx-arm64-metal"
        self.directory.mkdir(parents=True)
        self.version = pack.ET.parse(ROOT / "Directory.Build.props").findtext(".//TensorSharpVersion")
        self.ggml = (ROOT / "eng/ggml-revision").read_text().strip()
        self.original = dict(format="1", tensorsharp="2.8.6.8", source="1" * 40, ggml=self.ggml,
                             rid="osx-arm64", variant="metal", cpu="apple-m1", abi="3" * 64)
        self.identity = self.original | {"source": "4" * 40, "abi": "5" * 64, "tensorsharp": self.version}
        self.settings = {"TENSORSHARP_NATIVE_ABI": self.original["abi"], "TENSORSHARP_NATIVE_RID": "osx-arm64",
                         "TENSORSHARP_NATIVE_VARIANT": "metal", "TENSORSHARP_GGML_NATIVE_PORTABLE": "ON",
                         "GGML_NATIVE": "OFF", "GGML_METAL": "ON", "GGML_CUDA": "OFF", "GGML_VULKAN": "OFF",
                         "CMAKE_CXX_COMPILER": "/usr/bin/c++", "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0"}
        self.binary = self.temporary / "stage/runtimes/osx-arm64/native/metal/libGgmlOps.dylib"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"controlled relink output")
        old_bridge = self.file("identity-relink/original-bridge.dylib", b"controlled original bridge")
        cache_bytes = "".join(f"{key}:STRING={value}\n" for key, value in self.settings.items()).encode()
        old_cache = self.file("identity-relink/original-cache.txt", cache_bytes)
        old_build = {"tensorSharpBuild": self.original["tensorsharp"], "sourceCommit": self.original["source"], "ggmlCommit": self.original["ggml"],
                     "nativeAbi": self.original["abi"], "rid": "osx-arm64", "variant": "metal", "cpuProfile": "apple-m1",
                     "macosDeploymentTarget": "14.0", "compiler": "/usr/bin/c++", "compilerVersion": "controlled compiler",
                     "bridgeSha256": old_bridge["sha256"], "cmakeCacheSha256": old_cache["sha256"],
                     "cmakeConfiguration": self.settings}
        old_record = self.file("identity-relink/original-build.json", json.dumps(old_build).encode())
        objects = [f"CMakeFiles/GgmlOps.dir/controlled-{index}.cpp.o" for index in range(55)]
        objects += ["libggml.a", "libggml-cpu.a", "libggml-metal.a", "libggml-base.a"]
        self.inputs = [self.file("identity-relink/inputs/" + name, name.encode()) for name in objects]
        link = pack.MAC_IDENTITY_LINK_PREFIX + ["CMakeFiles/GgmlOps.dir/ggml_ops_build_identity.cpp.o"] + objects[:55] + pack.MAC_IDENTITY_LINK_SUFFIX
        original_link = self.file("identity-relink/original-link.txt", (" ".join(link) + "\n").encode())
        fresh_link = self.file("identity-relink/fresh-link.txt", (" ".join(link) + "\n").encode())
        flag_bytes = b"CXX_DEFINES = -DGGML_USE_CPU -DGGML_USE_METAL -DGgmlOps_EXPORTS -DTSG_GGML_USE_METAL=1\nCXX_FLAGS = -O3 -DNDEBUG -std=gnu++17 -arch arm64 -mmacosx-version-min=14.0 -fPIC\n"
        original_flags = self.file("identity-relink/original-flags.txt", flag_bytes)
        fresh_flags = self.file("identity-relink/fresh-flags.txt", flag_bytes)
        replacement = self.file("identity-relink/replacement/identity.o", b"new identity object")
        source = self.file("identity-relink/identity-source.cpp", (ROOT / "TensorSharp.GGML.Native/ggml_ops_build_identity.cpp").read_bytes())
        self.compile = pack.identity_compile_arguments(self.directory / source["path"], self.directory / replacement["path"],
                                                       self.identity, "/usr/bin/c++", pack.identity_include_paths())
        self.link = list(link)
        self.link[self.link.index("-o") + 1] = str(self.binary)
        for index, token in enumerate(self.link):
            if token == "CMakeFiles/GgmlOps.dir/ggml_ops_build_identity.cpp.o":
                self.link[index] = str(self.directory / replacement["path"])
            elif token in objects:
                self.link[index] = str(self.directory / "identity-relink/inputs" / token)
        proof = {"schema": "tensorsharp-identity-relink/1", "originalIdentity": self.original,
                 "originalBuildRecord": old_record, "originalCache": old_cache, "originalBridge": old_bridge,
                 "originalLink": original_link, "inputs": self.inputs, "replacementObject": replacement,
                 "identitySource": source, "nativeSourceTree": "6" * 40, "compileIncludes": pack.identity_include_paths(),
                 "originalFlags": original_flags, "freshFlags": fresh_flags, "freshLink": fresh_link,
                 "executionRoots": {"sourceRoot": str(ROOT), "buildRecordRoot": str(self.directory), "binaryPath": str(self.binary)},
                 "compileArgv": self.compile, "linkArgv": self.link}
        self.build = {"identityRelink": proof, "bridgeSha256": pack.sha256_bytes(self.binary.read_bytes())}
        patcher = patch.object(pack, "native_source_tree", return_value="6" * 40)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(pack, "read_identity", return_value=self.original)
        patcher.start()
        self.addCleanup(patcher.stop)

    def file(self, name, data):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": name, "size": len(data), "sha256": pack.sha256_bytes(data)}

    def validate(self):
        return pack.validate_identity_relink(self.build, self.identity, self.directory, self.binary)

    def test_complete_record_with_preserved_inputs_accepts(self):
        self.validate()

    def test_version_replacement_preserves_actual_original_record(self):
        self.assertNotEqual(self.original["tensorsharp"], self.identity["tensorsharp"])
        self.validate()
        reference = self.build["identityRelink"]["originalBuildRecord"]
        original = json.loads((self.directory / reference["path"]).read_text())
        original["tensorSharpBuild"] = self.version
        self.build["identityRelink"]["originalBuildRecord"] = self.file(reference["path"], json.dumps(original).encode())
        with self.assertRaisesRegex(ValueError, "original relink record"):
            self.validate()

    def test_changed_or_missing_reused_inputs_refuse(self):
        for change in ("changed", "missing"):
            with self.subTest(change=change):
                path = self.directory / self.inputs[0]["path"]
                original = path.read_bytes()
                if change == "changed":
                    path.write_bytes(b"changed")
                else:
                    path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    self.validate()
                path.write_bytes(original)

    def test_edited_original_cache_refuses(self):
        path = self.directory / self.build["identityRelink"]["originalCache"]["path"]
        path.write_bytes(path.read_bytes().replace(b"GGML_NATIVE:STRING=OFF", b"GGML_NATIVE:STRING=ON"))
        with self.assertRaises(ValueError):
            self.validate()

    def test_incompatible_build_pin_rid_variant_and_cpu_refuse(self):
        for key in ("tensorsharp", "ggml", "rid", "variant", "cpu"):
            with self.subTest(key=key):
                original = self.identity[key]
                self.identity[key] = "wrong"
                with self.assertRaises(ValueError):
                    self.validate()
                self.identity[key] = original

    def test_added_removed_duplicate_or_unsafe_input_refuses(self):
        for inputs in ([], self.inputs + [self.inputs[0]],
                       self.inputs + [self.file("identity-relink/inputs/extra.a", b"extra")],
                       [self.inputs[0] | {"path": "../escape.o"}, self.inputs[1]]):
            with self.subTest(inputs=inputs):
                self.build["identityRelink"]["inputs"] = inputs
                with self.assertRaises(ValueError):
                    self.validate()
        self.build["identityRelink"]["inputs"] = self.inputs

    def test_compile_abi_source_or_flags_and_link_mutation_refuse(self):
        proof = self.build["identityRelink"]
        for field in ("compileArgv", "linkArgv"):
            old = proof[field]
            proof[field] = old + ["-malicious-other-build"]
            with self.assertRaises(ValueError):
                self.validate()
            proof[field] = old

    def test_unknown_schema_or_field_refuses(self):
        original = copy.deepcopy(self.build)
        for update in ({"schema": "unknown"}, {"unexpected": True}):
            self.build = copy.deepcopy(original)
            self.build["identityRelink"].update(update)
            with self.assertRaises(ValueError):
                self.validate()

    def test_changed_native_source_tree_refuses(self):
        with patch.object(pack, "native_source_tree", side_effect=["6" * 40, "7" * 40]):
            with self.assertRaisesRegex(ValueError, "native implementation"):
                self.validate()

    def test_changed_observed_fresh_flags_or_link_refuses(self):
        proof = self.build["identityRelink"]
        for key in ("freshFlags", "freshLink"):
            old = proof[key]
            data = (self.directory / old["path"]).read_bytes().replace(b"-arch arm64", b"-arch x86_64")
            proof[key] = self.file(old["path"], data)
            with self.assertRaises(ValueError):
                self.validate()
            proof[key] = self.file(old["path"], data.replace(b"-arch x86_64", b"-arch arm64"))

    def test_absent_relink_preserves_ordinary_build(self):
        pack.validate_identity_relink({}, self.identity, self.directory, self.binary)

    def test_changed_identity_source_and_include_search_refuse(self):
        proof = self.build["identityRelink"]
        proof["compileIncludes"] = ["/untrusted/headers"]
        with self.assertRaises(ValueError):
            self.validate()
        proof["compileIncludes"] = pack.identity_include_paths()
        reference = proof["identitySource"]
        proof["identitySource"] = self.file(reference["path"], b"changed identity source")
        with self.assertRaisesRegex(ValueError, "current owned implementation"):
            self.validate()

    def test_evidence_symbolic_link_refuses(self):
        path = self.directory / self.inputs[0]["path"]
        other = path.with_name("backing.o")
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaises(ValueError):
            self.validate()

    def test_relocated_evidence_preserves_original_execution_arguments(self):
        relocated = self.temporary / "relocated"
        shutil.copytree(self.temporary / "stage", relocated)
        new_directory = relocated / "build/osx-arm64-metal"
        new_binary = relocated / "runtimes/osx-arm64/native/metal/libGgmlOps.dylib"
        shutil.rmtree(self.temporary / "stage")
        pack.validate_identity_relink(self.build, self.identity, new_directory, new_binary)
        other_source = self.temporary / "other-source"
        native = other_source / "TensorSharp.GGML.Native"
        native.mkdir(parents=True)
        shutil.copyfile(ROOT / "TensorSharp.GGML.Native/ggml_ops_build_identity.cpp", native / "ggml_ops_build_identity.cpp")
        shutil.copyfile(ROOT / "Directory.Build.props", other_source / "Directory.Build.props")
        with patch.object(pack, "REPO_ROOT", other_source):
            pack.validate_identity_relink(self.build, self.identity, new_directory, new_binary)

    def test_execution_root_role_or_compile_argument_change_refuses(self):
        proof = self.build["identityRelink"]
        old = proof["executionRoots"]["binaryPath"]
        proof["executionRoots"]["binaryPath"] = "/other/bridge.dylib"
        with self.assertRaises(ValueError):
            self.validate()
        proof["executionRoots"]["binaryPath"] = old
        proof["compileArgv"] = proof["compileArgv"] + ["-wrong"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_admission_refuses_existing_outputs(self):
        with self.assertRaisesRegex(ValueError, "existing output bridge"):
            pack.validate_identity_relink(self.build, self.identity, self.directory, self.binary, before_execution=True)

    def run_builder_refusal(self, module, mutation):
        fixture = self.temporary / ("original-" + mutation)
        configure = self.temporary / ("configure-" + mutation)
        stage = self.temporary / ("output-" + mutation)
        fixture.mkdir()
        configure.mkdir()
        proof = self.build["identityRelink"]
        old = json.loads((self.directory / proof["originalBuildRecord"]["path"]).read_text())
        settings = dict(old["cmakeConfiguration"])
        link_text = (self.directory / proof["originalLink"]["path"]).read_text()
        flags_text = (self.directory / proof["originalFlags"]["path"]).read_text()
        if mutation == "compiler":
            settings["CMAKE_CXX_COMPILER"] = old["compiler"] = "/bin/echo"
            link_text = link_text.replace("/usr/bin/c++", "/bin/echo")
        elif mutation == "flags":
            flags_text = flags_text.replace("-O3", "-O0")
        elif mutation == "link":
            link_text = link_text.replace("-dynamiclib", "-dynamiclib -Wl,-load,/untrusted.dylib")
        old["cmakeConfiguration"] = settings
        raw = "".join(f"{key}:STRING={value}\n" for key, value in settings.items()).encode()
        old["cmakeCacheSha256"] = pack.sha256_bytes(raw)
        (fixture / "CMakeCache.txt").write_bytes(raw)
        (fixture / "original-build.json").write_text(json.dumps(old))
        shutil.copyfile(self.directory / proof["originalBridge"]["path"], fixture / "libGgmlOps.dylib")
        fresh = settings | {"TENSORSHARP_NATIVE_ABI": self.identity["abi"]}
        (configure / "CMakeCache.txt").write_text("".join(f"{key}:STRING={value}\n" for key, value in fresh.items()))
        for directory in (fixture, configure):
            meta = directory / "CMakeFiles/GgmlOps.dir"
            meta.mkdir(parents=True)
            (meta / "flags.make").write_text(flags_text)
            (meta / "link.txt").write_text(link_text)
        for reference in proof["inputs"]:
            relative = reference["path"].removeprefix("identity-relink/inputs/")
            destination = fixture / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.directory / reference["path"], destination)
        with patch.object(module.record, "command", return_value=self.identity["source"]), \
                patch.object(module.record, "verify_clean_source", return_value=self.identity["ggml"]), \
                patch.object(module.pack, "native_source_tree", return_value="6" * 40), \
                patch.object(module.pack, "read_identity", return_value=self.original), \
                patch.object(module.pack, "native_abi", return_value=self.identity["abi"]), \
                patch.object(module.subprocess, "run", side_effect=AssertionError("external compiler/link invocation")) as execution:
            with self.assertRaises(ValueError):
                module.relink(fixture, fixture / "original-build.json", configure, stage)
            self.assertEqual(0, execution.call_count)

    def test_builder_rejects_untrusted_compiler_flags_and_link_before_execution(self):
        spec = importlib.util.spec_from_file_location("relink_builder", ROOT / "eng/relink-ggml-native-identity.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mutation in ("compiler", "flags", "link"):
            with self.subTest(mutation=mutation):
                self.run_builder_refusal(module, mutation)


if __name__ == "__main__":
    unittest.main()
