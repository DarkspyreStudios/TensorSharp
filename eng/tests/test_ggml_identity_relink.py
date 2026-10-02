"""Relink proof fixtures validate evidence, not actual bridge or GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("relink_pack", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class IdentityRelinkTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="identity-relink-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.original = dict(format="1", tensorsharp="test", source="1" * 40, ggml="2" * 40,
                             rid="osx-arm64", variant="metal", cpu="apple-m1", abi="3" * 64)
        self.identity = self.original | {"source": "4" * 40, "abi": "5" * 64}
        self.settings = {"TENSORSHARP_NATIVE_ABI": self.original["abi"], "TENSORSHARP_NATIVE_RID": "osx-arm64",
                         "TENSORSHARP_NATIVE_VARIANT": "metal", "TENSORSHARP_GGML_NATIVE_PORTABLE": "ON",
                         "GGML_NATIVE": "OFF", "GGML_METAL": "ON", "GGML_CUDA": "OFF", "GGML_VULKAN": "OFF",
                         "CMAKE_CXX_COMPILER": "/usr/bin/c++", "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0"}
        self.binary = self.directory / "libGgmlOps.dylib"
        self.binary.write_bytes(b"controlled relink output")
        old_bridge = self.file("identity-relink/original-bridge.dylib", b"controlled original bridge")
        cache_bytes = "".join(f"{key}:STRING={value}\n" for key, value in self.settings.items()).encode()
        old_cache = self.file("identity-relink/original-cache.txt", cache_bytes)
        old_build = {"tensorSharpBuild": "test", "sourceCommit": self.original["source"], "ggmlCommit": self.original["ggml"],
                     "nativeAbi": self.original["abi"], "rid": "osx-arm64", "variant": "metal", "cpuProfile": "apple-m1",
                     "macosDeploymentTarget": "14.0", "compiler": "/usr/bin/c++", "compilerVersion": "controlled compiler",
                     "bridgeSha256": old_bridge["sha256"], "cmakeCacheSha256": old_cache["sha256"],
                     "cmakeConfiguration": self.settings}
        old_record = self.file("identity-relink/original-build.json", json.dumps(old_build).encode())
        objects = ["CMakeFiles/GgmlOps.dir/ggml_ops_core.cpp.o", "libggml-metal.a"]
        self.inputs = [self.file("identity-relink/inputs/" + name, name.encode()) for name in objects]
        link = ["/usr/bin/c++", "-arch", "arm64", "-mmacosx-version-min=14.0", "-dynamiclib", "-o", "libGgmlOps.dylib",
                "CMakeFiles/GgmlOps.dir/ggml_ops_build_identity.cpp.o", *objects, "-framework", "Metal"]
        original_link = self.file("identity-relink/original-link.txt", (" ".join(link) + "\n").encode())
        replacement = self.file("identity-relink/replacement/identity.o", b"new identity object")
        source = self.file("identity-relink/identity-source.cpp", b"controlled identity source")
        self.compile = pack.identity_compile_arguments(self.directory / source["path"], self.directory / replacement["path"],
                                                       self.identity, "/usr/bin/c++", [])
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
                 "identitySource": source, "nativeSourceTree": "6" * 40, "compileIncludes": [],
                 "compileArgv": self.compile, "linkArgv": self.link}
        self.build = {"identityRelink": proof, "bridgeSha256": pack.sha256_bytes(self.binary.read_bytes())}

    def file(self, name, data):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": name, "size": len(data), "sha256": pack.sha256_bytes(data)}

    def validate(self):
        return pack.validate_identity_relink(self.build, self.identity, self.directory, self.binary)

    def test_complete_record_with_preserved_inputs_accepts(self):
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

    def test_absent_relink_preserves_ordinary_build(self):
        pack.validate_identity_relink({}, self.identity, self.directory, self.binary)


if __name__ == "__main__":
    unittest.main()
