"""Required-export policy checks do not execute a native backend."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ggml_export_packer", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class ExportInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("dotnet"):
            raise unittest.SkipTest("the inventory parser requires the installed .NET SDK")
        (ROOT / "tmp").mkdir(exist_ok=True)
        cls.env = dict(os.environ, TMPDIR=str(ROOT / "tmp"))
        subprocess.run(["dotnet", "build", str(ROOT / "eng/guard-ggml-interop/guard-ggml-interop.csproj"),
                        "--no-restore", "-v", "quiet"], check=True, capture_output=True, text=True, env=cls.env, timeout=30)
        cls.tool = ROOT / "eng/guard-ggml-interop/bin/Debug/net10.0/guard-ggml-interop.dll"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ggml-export-inventory-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ("eng", "TensorSharp.GGML.Native", "TensorSharp.Backends.GGML"):
            (self.root / directory).mkdir()
        (self.root / "eng/ggml-revision").write_text("1" * 40 + "\n")
        self.source = self.root / "TensorSharp.Backends.GGML/Example.cs"
        self.source.write_text('''using System.Runtime.InteropServices;
static partial class ExampleInterop {
    static void Example() { using var call = GgmlNativeLoader.EnterNativeCall(); Native_Example(); }
    [LibraryImport("GgmlOps", EntryPoint = "Example")]
    private static partial void Native_Example();
}
''')

    def generate(self, root=None):
        return subprocess.run(["dotnet", str(self.tool), "--inventory", str(root or self.root)],
                              capture_output=True, text=True, env=self.env, timeout=10)

    def test_committed_inventory_matches_every_guarded_import_and_exact_abi(self):
        result = self.generate(ROOT)
        self.assertEqual(0, result.returncode, result.stderr)
        generated = json.loads(result.stdout)
        committed = json.loads((ROOT / "eng/ggml-required-exports.json").read_text())
        self.assertEqual(committed, generated)
        self.assertEqual(pack.native_abi(ROOT), generated["nativeAbi"])
        self.assertEqual(296, len(generated["exports"]))
        self.assertTrue({"ggml_quantize_init", "ggml_quantize_requires_imatrix", "ggml_quantize_chunk"}
                        <= set(generated["exports"]))
        self.assertFalse(any(name.startswith("ggml_backend_cuda_") or name.startswith("ggml_backend_vk_")
                             for name in generated["exports"]))

    def test_generation_is_read_only_and_matches_the_shared_abi_algorithm(self):
        (self.root / "TensorSharp.GGML.Native/bridge.cpp").write_bytes(b"\xef\xbb\xbfinput\r\n\xff")
        original = self.source.read_bytes()
        result = self.generate()
        self.assertEqual(0, result.returncode, result.stderr)
        inventory = json.loads(result.stdout)
        self.assertEqual(["Example"], inventory["exports"])
        self.assertEqual(pack.native_abi(self.root), inventory["nativeAbi"])
        self.assertEqual(original, self.source.read_bytes())
        (self.root / "eng/ggml-required-exports.json").write_text(json.dumps(inventory))
        self.assertEqual(["Example"], pack.read_required_exports(self.root))

    def test_inventory_refuses_unguarded_or_nonliteral_imports(self):
        original = self.source.read_text()
        for source in (original.replace('EntryPoint = "Example"', 'EntryPoint = nameof(Example)'),
                       original.replace('GgmlNativeLoader.EnterNativeCall()', 'SomethingElse()')):
            self.source.write_text(source)
            with self.subTest(source=source):
                result = self.generate()
                self.assertNotEqual(0, result.returncode)
                self.assertEqual("", result.stdout)

    def test_stale_or_malformed_inventory_refuses(self):
        inventory = json.loads(self.generate().stdout)
        path = self.root / "eng/ggml-required-exports.json"
        for change in ({"nativeAbi": "0" * 64}, {"ggmlCommit": "wrong"}, {"schema": "wrong"},
                       {"exports": ["Example", "Example"]}, {"exports": []}, {"exports": ["bad-name"]},
                       {"exports": ["z", "a"]}, {"unexpected": True}):
            path.write_text(json.dumps(inventory | change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                pack.read_required_exports(self.root)
        path.write_text(json.dumps(inventory))
        self.source.write_text(self.source.read_text() + "// changed ABI input\n")
        with self.assertRaises(ValueError):
            pack.read_required_exports(self.root)

    def test_each_missing_symbol_refuses_and_additional_gpu_exports_do_not(self):
        required = pack.read_required_exports(ROOT)
        facts = {"path": "GgmlOps.dll", "format": "pe", "identity": {"arch": "arm64"},
                 "dependencies": [], "tsggmlBuildIdentityExport": True}
        files = [{"path": name} for name in ("GgmlOps.dll", *pack.REQUIRED_LICENSES)]
        check = lambda exports: pack.check_artifact("win-arm64", "cpu", None, {"files": files},
                                                    [facts | {"functionExports": exports}], required)
        self.assertEqual([], check(required + ["ggml_backend_cuda_init", "extra"]))
        for missing in required:
            with self.subTest(missing=missing):
                self.assertEqual([f"win-arm64/cpu: missing required managed entrypoint {missing}"],
                                 check([name for name in required if name != missing]))
        self.assertIn("bridge exports were not inspected", check(None)[0])

    def test_export_parsers_require_defined_function_symbols_not_counts_or_prefixes(self):
        cases = [(pack.INVENTORY.elf_dynamic,
                  "000 g DF .text 00010 Base ggml_quantize_chunk\n000 g DF *UND* 00010 Base missing\n000 g DO .data 00010 Base data\n",
                  " NEEDED libc.so.6\n"),
                 (pack.INVENTORY.macho_dynamic,
                  "000 T _ggml_quantize_chunk\n000 S _data\n U _missing\n", "fixture:\n /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n")]
        for reader, symbols, dependencies in cases:
            with self.subTest(reader=reader.__name__):
                def output(args):
                    if args[0] == "nm" or "-T" in args:
                        return symbols
                    return "fixture:\nfixture\n" if "-D" in args else dependencies
                with patch.object(pack.INVENTORY, "run", side_effect=output):
                    self.assertEqual(["ggml_quantize_chunk"], reader(Path("fixture"))["functionExports"])
                with patch.object(pack.INVENTORY, "run", side_effect=lambda args: None if args[0] == "nm" or "-T" in args else output(args)):
                    self.assertIsNone(reader(Path("fixture"))["functionExports"])
        for names in (" 1 0x1234 ggml_quantize_chunk\n", " [ 0] ggml_quantize_chunk\n"):
            with patch.object(pack.INVENTORY, "run", return_value="Export Table:\n" + names):
                self.assertEqual(["ggml_quantize_chunk"], pack.INVENTORY.pe_dynamic(Path("fixture"))["functionExports"])


if __name__ == "__main__":
    unittest.main()
