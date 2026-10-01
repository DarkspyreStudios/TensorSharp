import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReleaseBuilderTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="release-builder-", dir=ROOT / "tmp")
        self.root = Path(self.temporary.name)
        for directory in ("eng", "TensorSharp.GGML.Native", "ExternalProjects/ggml", "bin"):
            (self.root / directory).mkdir(parents=True)
        shutil.copyfile(ROOT / "eng/build-ggml-natives.sh", self.root / "eng/build-ggml-natives.sh")
        (self.root / "Directory.Build.props").write_text("<TensorSharpVersion>test</TensorSharpVersion>\n")
        (self.root / "eng/ggml-revision").write_text("a" * 40 + "\n")
        self.command("git", '#!/usr/bin/env bash\nif [[ "$*" == *status* ]]; then\n'
                     ' if [[ "$*" == *TensorSharp.Backends.GGML* && "$SIMULATE_DIRTY_ABI" == 1 ]]; then echo " M TensorSharp.Backends.GGML/GgmlContext.cs"; fi\n'
                     'else printf "%040d\\n" 0; fi\n')
        self.command("uname", '#!/usr/bin/env bash\ncase "$1" in -s) echo Linux;; -m) echo aarch64;; *) echo "Linux aarch64";; esac\n')
        script = self.root / "TensorSharp.GGML.Native/build-linux.sh"
        script.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_ROOT/build-arguments.txt"\nexit 37\n')
        self.environment = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                                TMPDIR=str(self.root), TEST_ROOT=str(self.root), SIMULATE_DIRTY_ABI="0")

    def command(self, name, content):
        path = self.root / "bin" / name
        path.write_text(content)
        path.chmod(0o755)

    def tearDown(self):
        self.temporary.cleanup()

    def run_builder(self, *extra, dirty=False):
        environment = dict(self.environment, SIMULATE_DIRTY_ABI="1" if dirty else "0")
        return subprocess.run(["bash", str(self.root / "eng/build-ggml-natives.sh"), "--rid", "linux-arm64",
                               "--variant", "cuda13", *extra], capture_output=True, text=True, env=environment, timeout=10)

    def test_linux_arm64_cuda_preserves_sass_and_ptx_without_local_gpu_detection(self):
        result = self.run_builder()
        self.assertEqual(37, result.returncode, result.stderr)
        arguments = (self.root / "build-arguments.txt").read_text().splitlines()
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;120-real;120-virtual", arguments)
        self.assertIn("-DTENSORSHARP_NATIVE_RID=linux-arm64", arguments)
        self.assertIn("-DTENSORSHARP_GGML_NATIVE_PORTABLE=ON", arguments)
        self.assertIn("-DCMAKE_INSTALL_RPATH=$ORIGIN", arguments)

    def test_release_architecture_profile_cannot_be_overridden(self):
        result = self.run_builder("--", "-DCMAKE_CUDA_ARCHITECTURES=120")
        self.assertEqual(2, result.returncode)
        self.assertIn("cannot be overridden", result.stderr)
        self.assertFalse((self.root / "build-arguments.txt").exists())

    def test_dirty_managed_abi_sources_refuse_false_commit_provenance(self):
        result = self.run_builder(dirty=True)
        self.assertEqual(1, result.returncode)
        self.assertIn("managed ABI", result.stderr)
        self.assertFalse((self.root / "build-arguments.txt").exists())


if __name__ == "__main__":
    unittest.main()
