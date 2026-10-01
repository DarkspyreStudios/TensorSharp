"""Build-record policy tests use mocked inspection, not target or GPU execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ggml_build_record", ROOT / "eng/record-ggml-native-build.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class BuildRecordTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="native-build-record-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ("TensorSharp.GGML.Native", "TensorSharp.Backends.GGML", "eng", "build"):
            (self.root / directory).mkdir()
        self.source, self.ggml = "1" * 40, "2" * 40
        (self.root / "eng/ggml-revision").write_text(self.ggml + "\n")
        (self.root / "Directory.Build.props").write_text("<Project><TensorSharpVersion>test</TensorSharpVersion></Project>")
        self.abi = record.pack.native_abi(self.root)
        self.binary = self.root / "libGgmlOps.so"
        self.identity = dict(format="1", tensorsharp="test", source=self.source, ggml=self.ggml,
                             rid="linux-arm64", variant="cuda13", cpu="armv8.2-a+dotprod", abi=self.abi)
        self.settings = {"TENSORSHARP_NATIVE_ABI": self.abi, "TENSORSHARP_NATIVE_RID": "linux-arm64",
                         "TENSORSHARP_NATIVE_VARIANT": "cuda13", "TENSORSHARP_GGML_NATIVE_PORTABLE": "ON",
                         "GGML_NATIVE": "OFF", "GGML_METAL": "OFF", "GGML_CUDA": "ON", "GGML_VULKAN": "OFF",
                         "CMAKE_CXX_COMPILER": '/compiler path/c++', "CMAKE_CUDA_COMPILER": "/cuda/nvcc",
                         "CMAKE_CUDA_ARCHITECTURES": record.CUDA_ARCHITECTURES,
                         "CMAKE_INSTALL_RPATH": "$ORIGIN", "CMAKE_BUILD_WITH_INSTALL_RPATH": "ON"}
        self.write_inputs()
        self.toolkit = "Cuda compilation tools, release 13.0, V13.0.88"
        self.command = patch.object(record, "command", side_effect=self.output).start()
        self.addCleanup(patch.stopall)
        self.inspection = patch.object(record.pack.INVENTORY, "describe", return_value={
            "format": "elf", "identity": {"arch": "arm64"}, "tsggmlBuildIdentityExport": True}).start()

    def output(self, args):
        if args[0] == "git":
            if "status" in args:
                return ""
            return self.ggml if str(self.root / "ExternalProjects/ggml") in args else self.source
        return self.toolkit if args[0] == "/cuda/nvcc" else "compiler version 1\nTarget: aarch64"

    def write_inputs(self):
        self.binary.write_bytes((";".join(key + "=" + value for key, value in self.identity.items()) + "\0").encode())
        self.cache = self.root / "build/CMakeCache.txt"
        self.cache.write_text("//Generated fixture\n" + "".join(key + ":STRING=" + value + "\n" for key, value in self.settings.items()))

    def create(self):
        return record.create_record(self.root, self.root / "build", self.binary, "linux-arm64", "cuda13", self.source)

    def test_records_observed_profile_hashes_compiler_and_full_cuda_architectures(self):
        result = self.create()
        self.assertEqual("armv8.2-a+dotprod", result["cpuProfile"])
        self.assertEqual(record.CUDA_ARCHITECTURES.split(";"), result["cuda"]["architectures"])
        self.assertEqual("13.0", result["cuda"]["toolkitVersion"])
        self.assertEqual(record.pack.sha256_bytes(self.binary.read_bytes()), result["bridgeSha256"])
        self.assertEqual(record.pack.sha256_bytes(self.cache.read_bytes()), result["cmakeCacheSha256"])
        self.assertEqual(self.settings, result["cmakeConfiguration"])
        self.assertEqual("not-recorded", result["qualification"]["targetExecution"])
        self.assertEqual("not-recorded", result["qualification"]["gpuExecution"])
        self.assertEqual('/compiler path/c++', result["compiler"])

    def test_wrong_binary_identity_and_cpu_floor_refuse(self):
        for key in ("source", "ggml", "abi", "rid", "variant", "cpu"):
            with self.subTest(key=key):
                original = self.identity[key]
                self.identity[key] = "wrong"
                self.write_inputs()
                with self.assertRaises(ValueError):
                    self.create()
                self.identity[key] = original

    def test_wrong_backend_portability_runpath_abi_or_cuda_profile_refuse(self):
        for key in ("GGML_NATIVE", "GGML_CUDA", "GGML_METAL", "GGML_VULKAN", "TENSORSHARP_NATIVE_ABI",
                    "TENSORSHARP_GGML_NATIVE_PORTABLE", "CMAKE_INSTALL_RPATH", "CMAKE_BUILD_WITH_INSTALL_RPATH",
                    "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER"):
            with self.subTest(key=key):
                original = self.settings[key]
                self.settings[key] = ""
                self.write_inputs()
                with self.assertRaises(ValueError):
                    self.create()
                self.settings[key] = original

    def test_wrong_or_unobserved_cuda_major_refuses(self):
        for toolkit in ("release 12.9", "unknown"):
            self.toolkit = toolkit
            with self.subTest(toolkit=toolkit), self.assertRaises(ValueError):
                self.create()

    def test_wrong_architecture_or_missing_defined_export_refuses(self):
        for facts in ({"format": "elf", "identity": {"arch": "x86_64"}, "tsggmlBuildIdentityExport": True},
                      {"format": "elf", "identity": {"arch": "arm64"}, "tsggmlBuildIdentityExport": False}):
            self.inspection.return_value = facts
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                self.create()

    def test_local_source_or_upstream_edits_refuse(self):
        for dirty_root in (str(self.root), str(self.root / "ExternalProjects/ggml")):
            self.command.side_effect = lambda args: " M file" if "status" in args and dirty_root in args else self.output(args)
            with self.subTest(dirty_root=dirty_root), self.assertRaises(ValueError):
                self.create()

    def test_linked_binary_refuses(self):
        self.binary.rename(self.root / "actual.so")
        self.binary.symlink_to(self.root / "actual.so")
        with self.assertRaises(ValueError):
            self.create()

    def test_malformed_or_duplicate_cache_refuses(self):
        for text in ("wrong", "GGML_NATIVE:BOOL=OFF\nGGML_NATIVE:BOOL=ON\n"):
            self.cache.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                record.read_cache(self.cache)


if __name__ == "__main__":
    unittest.main()
