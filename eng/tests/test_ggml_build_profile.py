"""Recorded-profile validation uses mocked inspection, never target execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("profile_packer", ROOT / "eng/pack-ggml-natives.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
PROFILES = {"osx-arm64": "apple-m1", "linux-x64": "x86-64", "linux-arm64": "armv8.2-a+dotprod",
            "win-x64": "x86-64", "win-arm64": "armv8.2-a+dotprod"}
ARCHITECTURES = "75-real;80-real;86-real;89-real;120-real;120-virtual"


class RecordedProfileCollectionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="recorded-profile-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.stage = Path(temporary.name)
        self.rid, self.variant = "linux-arm64", "cuda13"
        self.abi = pack.native_abi(ROOT)
        self.ggml = (ROOT / "eng/ggml-revision").read_text().strip()
        self.configure()
        inspection = patch.object(pack.INVENTORY, "describe", side_effect=self.describe)
        self.addCleanup(inspection.stop)
        inspection.start()
        exports = patch.object(pack, "read_required_exports", return_value=["TSGgml_GetBuildIdentity"])
        self.addCleanup(exports.stop)
        exports.start()

    def configure(self):
        self.artifact = self.stage / "runtimes" / self.rid / "native" / self.variant
        self.artifact.mkdir(parents=True, exist_ok=True)
        self.record_dir = self.stage / "build" / f"{self.rid}-{self.variant}"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.identity = {"format": "1", "tensorsharp": "fixture", "source": "1" * 40, "ggml": self.ggml,
                         "rid": self.rid, "variant": self.variant, "cpu": PROFILES[self.rid], "abi": self.abi}
        self.bridge = self.artifact / pack.ENTRY[self.rid.split("-")[0]]
        self.bridge.write_bytes((";".join(f"{key}={value}" for key, value in self.identity.items()) + "\0").encode())
        self.settings = {"TENSORSHARP_NATIVE_ABI": self.abi, "TENSORSHARP_NATIVE_RID": self.rid,
                         "TENSORSHARP_NATIVE_VARIANT": self.variant, "TENSORSHARP_GGML_NATIVE_PORTABLE": "ON",
                         "GGML_NATIVE": "OFF", "CMAKE_CXX_COMPILER": "/fixture/compiler"}
        for backend in ("metal", "vulkan", "cuda13"):
            self.settings["GGML_" + ("CUDA" if backend == "cuda13" else backend.upper())] = "ON" if self.variant == backend else "OFF"
        if self.rid.startswith("linux-"):
            self.settings.update(CMAKE_INSTALL_RPATH="$ORIGIN", CMAKE_BUILD_WITH_INSTALL_RPATH="ON")
        if self.rid.startswith("osx-"):
            self.settings["CMAKE_OSX_DEPLOYMENT_TARGET"] = "14.0"
        self.build = {"tensorSharpBuild": "fixture", "sourceCommit": self.identity["source"], "ggmlCommit": self.ggml,
                      "nativeAbi": self.abi, "rid": self.rid, "variant": self.variant,
                      "cpuProfile": self.identity["cpu"], "cmakeConfiguration": self.settings,
                      "bridgeSha256": pack.sha256_bytes(self.bridge.read_bytes()), "cmakeCacheSha256": "2" * 64,
                      "compiler": self.settings["CMAKE_CXX_COMPILER"], "compilerVersion": "recorded fixture compiler",
                      "macosDeploymentTarget": self.settings.get("CMAKE_OSX_DEPLOYMENT_TARGET")}
        if self.variant == "cuda13":
            self.settings.update(CMAKE_CUDA_COMPILER="/fixture/nvcc", CMAKE_CUDA_ARCHITECTURES=ARCHITECTURES)
            self.build["cuda"] = {"compiler": "/fixture/nvcc", "compilerVersion": "Cuda compilation tools, release 13.0, V13.0.88",
                                  "toolkitVersion": "13.0", "architectures": ARCHITECTURES.split(";")}
        self.build["components"] = []
        for expected in pack.core_component_specs(self.build, self.bridge.name):
            evidence = self.artifact / expected["evidencePaths"][0]
            evidence.parent.mkdir(exist_ok=True)
            evidence.write_text("Synthetic source license inspection fixture\n")
            reference = lambda name: {key: value for key, value in pack.file_record(self.artifact, self.artifact / name).items()
                                      if key != "executable"}
            self.build["components"].append({key: value for key, value in expected.items() if key not in ("binaryPaths", "evidencePaths")} | {
                "binaryFiles": [reference(name) for name in expected["binaryPaths"]],
                "evidenceFiles": [reference(name) for name in expected["evidencePaths"]]})
        self.write_record()

    def write_record(self):
        (self.record_dir / "build-identity.json").write_text(json.dumps(self.build))
        self.settings_path = self.record_dir / "cmake-settings.txt"
        self.settings_path.write_text("".join(f"{key}={value}\n" for key, value in sorted(self.settings.items())))

    def describe(self, name, rid, variant, path):
        if name.startswith("licenses/"):
            return {"path": name, "format": "other"}
        fmt = pack.INVENTORY.RID_FORMAT[rid.split("-")[0]]
        return {"path": name, "format": fmt, "identity": {"arch": pack.INVENTORY.RID_ARCH[rid.split("-")[1]]},
                "functionExports": ["TSGgml_GetBuildIdentity"], "tsggmlBuildIdentityExport": True,
                "dependencies": [], "runpath": ["$ORIGIN"] if fmt == "elf" else []}

    def collect(self):
        return pack.collect_artifacts(self.stage, "fixture", self.ggml)

    def test_five_settled_cpu_profiles_accept_without_execution_claim(self):
        for rid in PROFILES:
            with self.subTest(rid=rid):
                self.rid, self.variant = rid, pack.BASELINE[rid]
                self.configure()
                _, errors = self.collect()
                self.assertEqual([], errors)

    def test_missing_empty_duplicate_malformed_or_mismatching_settings_refuse(self):
        valid = self.settings_path.read_text()
        for contents in (None, "", valid + "GGML_NATIVE=OFF\n", "not-an-entry\n", valid.replace("GGML_NATIVE=OFF", "GGML_NATIVE=ON")):
            with self.subTest(contents=contents):
                if contents is None:
                    self.settings_path.unlink()
                else:
                    self.settings_path.write_text(contents)
                self.assertTrue(self.collect()[1], "invalid normalized settings accepted")
                self.settings_path.write_text(valid)

    def test_missing_record_configuration_refuses(self):
        del self.build["cmakeConfiguration"]
        self.write_record()
        self.assertTrue(self.collect()[1])

    def test_portable_backend_target_and_linux_runpath_refuse(self):
        for key in ("TENSORSHARP_NATIVE_ABI", "TENSORSHARP_NATIVE_RID", "TENSORSHARP_NATIVE_VARIANT",
                    "TENSORSHARP_GGML_NATIVE_PORTABLE", "GGML_NATIVE", "GGML_CUDA", "GGML_METAL", "GGML_VULKAN",
                    "CMAKE_INSTALL_RPATH", "CMAKE_BUILD_WITH_INSTALL_RPATH", "CMAKE_CUDA_ARCHITECTURES"):
            with self.subTest(key=key):
                value = self.settings[key]
                self.settings[key] = "wrong"
                self.write_record()
                self.assertTrue(self.collect()[1], key + " was not validated")
                self.settings[key] = value
        self.write_record()

    def test_embedded_or_recorded_cpu_floor_refuses(self):
        self.build["cpuProfile"] = "native"
        self.write_record()
        self.assertTrue(self.collect()[1])
        self.build["cpuProfile"] = PROFILES[self.rid]
        self.write_record()
        self.bridge.write_bytes(self.bridge.read_bytes().replace(PROFILES[self.rid].encode(), b"native"))
        self.build["bridgeSha256"] = pack.sha256_bytes(self.bridge.read_bytes())
        for component in self.build["components"]:
            component["binaryFiles"][0].update(size=self.bridge.stat().st_size, sha256=self.build["bridgeSha256"])
        self.write_record()
        self.assertTrue(self.collect()[1])

    def test_actual_bridge_hash_differs_from_record_refuses(self):
        self.build["bridgeSha256"] = "0" * 64
        self.write_record()
        self.assertTrue(self.collect()[1])

    def test_cuda_evidence_missing_inconsistent_or_wrong_major_refuses(self):
        original = dict(self.build["cuda"])
        for change in ({"compilerVersion": "unknown"}, {"compilerVersion": "release 12.9"}, {"toolkitVersion": "13.1"},
                       {"compiler": "/wrong/nvcc"}, {"architectures": ["121a-real"]}, None):
            with self.subTest(change=change):
                if change is None:
                    del self.build["cuda"]
                else:
                    self.build["cuda"] = original | change
                self.write_record()
                self.assertTrue(self.collect()[1])
                self.build["cuda"] = original
        self.write_record()


if __name__ == "__main__":
    unittest.main()
