"""Controlled MSBuild target policy only; no CUDA compiler or backend executes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj"
TARGETS = ("ResolveCudaPtxBuildSettings", "SelectCudaPtxContent", "CompileCudaPtx",
           "CopyCudaPtxToOutput", "UpdateCommittedCudaPtx")


@unittest.skipUnless(os.name == "posix", "The controlled compiler sentinels use POSIX shell scripts.")
class CudaNativeSkipTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="cuda-skip-", dir=ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.committed = self.root / "native/ptx/kernel.ptx"
        self.stale = self.root / "obj/cuda_ptx/ptx/kernel.ptx"
        self.trace = self.root / "commands.txt"
        for path in (self.committed, self.stale):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.committed.write_bytes(b"committed fixture PTX")
        self.stale.write_bytes(b"stale fixture PTX")
        (self.root / "native/kernels").mkdir()
        (self.root / "native/kernels/kernel.cu").write_text("controlled fixture input\n")
        self.sentinel(self.root / "bin/nvcc", """#!/usr/bin/env bash
printf 'nvcc %s\\n' "$*" >> "$CUDA_TEST_TRACE"
if [[ "$1" == --version ]]; then
    if [[ "$CUDA_TEST_NVCC_AVAILABLE" == true ]]; then echo 'controlled compiler sentinel'; exit 0; fi
    exit 1
fi
exit 97
""")
        self.sentinel(self.root / "native/resolve-cuda-arch.sh", """#!/usr/bin/env bash
printf 'resolver %s\\n' "$*" >> "$CUDA_TEST_TRACE"
printf 'compute_61\\n'
""")

        # Keep actual SDK target wiring and all five production CUDA targets intact.
        # Only project-local inputs are fixture bytes; no managed build is requested.
        tree = ET.parse(PROJECT)
        project = tree.getroot()
        properties = ET.SubElement(project, "PropertyGroup")
        ET.SubElement(properties, "OutDir").text = str(self.root / "output") + os.sep
        ET.SubElement(properties, "OutputPath").text = str(self.root / "output") + os.sep
        observe = ET.SubElement(project, "Target", Name="ObserveCudaContent", DependsOnTargets="AssignTargetPaths")
        ET.SubElement(observe, "WriteLinesToFile", File=str(self.root / "content.txt"),
                      Lines="@(Content->'%(Identity)|%(Link)|%(CopyToOutputDirectory)')", Overwrite="true")
        ET.SubElement(observe, "Copy", SourceFiles="@(Content)",
                      DestinationFiles="@(Content->'$(OutDir)%(Link)')")
        self.project = self.root / "cuda-policy.csproj"
        tree.write(self.project, encoding="utf-8", xml_declaration=True)
        self.environment = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                                TMPDIR=str(self.root), CUDA_TEST_TRACE=str(self.trace),
                                CUDA_TEST_NVCC_AVAILABLE="true")
        self.environment.pop("TensorSharpSkipCudaNative", None)
        self.configuration = os.environ.get("CUDA_SKIP_TEST_CONFIGURATION", "Debug")
        self.assertIn(self.configuration, ("Debug", "Release"))

    def sentinel(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def run_target(self, target, skip="true", available=True, properties=(), succeeds=True):
        arguments = ["dotnet", "msbuild", str(self.project), "-nologo", "-verbosity:minimal",
                     "-t:" + target, "-p:Configuration=" + self.configuration]
        if skip is not None:
            arguments.append("-p:TensorSharpSkipCudaNative=" + skip)
        arguments.extend("-p:" + value for value in properties)
        environment = dict(self.environment, CUDA_TEST_NVCC_AVAILABLE=str(available).lower())
        result = subprocess.run(arguments, capture_output=True, text=True, env=environment, timeout=20)
        self.assertEqual(succeeds, result.returncode == 0, result.stdout + result.stderr)
        return result

    def commands(self):
        return self.trace.read_text().splitlines() if self.trace.exists() else []

    def test_true_skips_every_direct_target_even_with_preexisting_compiler_success(self):
        for target in TARGETS:
            with self.subTest(target=target):
                if self.trace.exists():
                    self.trace.unlink()
                output = self.root / "output/cuda_kernels/kernel.ptx"
                if output.exists():
                    output.unlink()
                self.committed.write_bytes(b"committed fixture PTX")
                self.run_target(target, properties=("_NvccExitCode=0", "TensorSharpUpdateCommittedPtx=true"))
                self.assertEqual([], self.commands())
                self.assertEqual(b"committed fixture PTX", self.committed.read_bytes())
                self.assertFalse((self.root / "output/cuda_kernels/kernel.ptx").exists())
                self.assertEqual([], list((self.root / "obj/cuda_ptx").glob("*.stamp")))

    def test_true_preserves_sdk_hooks_without_native_work_or_stale_after_target_copy(self):
        self.run_target("BeforeBuild;ObserveCudaContent", properties=("TensorSharpUpdateCommittedPtx=true",))
        self.assertEqual([], self.commands())
        self.assertEqual(b"committed fixture PTX", self.committed.read_bytes())
        self.assertEqual(b"committed fixture PTX", (self.root / "output/cuda_kernels/kernel.ptx").read_bytes())

    def test_true_keeps_committed_content_and_copies_it_instead_of_stale_intermediates(self):
        self.run_target("ObserveCudaContent")
        self.assertEqual([], self.commands())
        content = (self.root / "content.txt").read_text()
        self.assertIn(str(self.committed), content)
        self.assertNotIn(str(self.stale), content)
        self.assertIn("PreserveNewest", content)
        self.assertEqual(b"committed fixture PTX", (self.root / "output/cuda_kernels/kernel.ptx").read_bytes())

    def test_true_suppresses_update_even_when_compiled_files_and_update_opt_in_exist(self):
        self.run_target("UpdateCommittedCudaPtx", properties=("_NvccExitCode=0", "TensorSharpUpdateCommittedPtx=true"))
        self.assertEqual(b"committed fixture PTX", self.committed.read_bytes())
        self.assertEqual([], self.commands())

    def test_unset_and_false_preserve_compiler_probe_and_architecture_resolution(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                if self.trace.exists():
                    self.trace.unlink()
                self.run_target("ResolveCudaPtxBuildSettings", skip=skip)
                self.assertEqual(["nvcc --version", "resolver compute_61"], self.commands())

    def test_unset_and_false_preserve_native_compile_attempt_without_running_a_compiler(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                if self.trace.exists():
                    self.trace.unlink()
                self.run_target("CompileCudaPtx", skip=skip, succeeds=False)
                commands = self.commands()
                self.assertEqual("nvcc --version", commands[0])
                self.assertEqual("resolver compute_61", commands[1])
                self.assertTrue(any(command.startswith("nvcc -ptx -arch=compute_61 ") for command in commands))
                self.assertEqual(b"committed fixture PTX", self.committed.read_bytes())

    def test_unset_and_false_select_existing_intermediate_content_when_compiler_is_available(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                self.run_target("ObserveCudaContent", skip=skip)
                content = (self.root / "content.txt").read_text()
                self.assertNotIn(str(self.committed), content)
                self.assertIn(str(self.stale), content)
                self.assertEqual(b"stale fixture PTX", (self.root / "output/cuda_kernels/kernel.ptx").read_bytes())

    def test_unset_and_false_keep_committed_content_when_compiler_is_unavailable(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                self.run_target("ObserveCudaContent", skip=skip, available=False)
                content = (self.root / "content.txt").read_text()
                self.assertIn(str(self.committed), content)
                self.assertNotIn(str(self.stale), content)
                self.assertEqual(b"committed fixture PTX", (self.root / "output/cuda_kernels/kernel.ptx").read_bytes())

    def test_unset_and_false_preserve_explicit_intermediate_copy(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                self.run_target("CopyCudaPtxToOutput", skip=skip)
                self.assertEqual(b"stale fixture PTX", (self.root / "output/cuda_kernels/kernel.ptx").read_bytes())

    def test_unset_and_false_preserve_explicit_update_of_fixture_only_baseline(self):
        for skip in (None, "false"):
            with self.subTest(skip=skip):
                self.committed.write_bytes(b"committed fixture PTX")
                self.run_target("UpdateCommittedCudaPtx", skip=skip,
                                properties=("_NvccExitCode=0", "TensorSharpUpdateCommittedPtx=true"))
                self.assertEqual(b"stale fixture PTX", self.committed.read_bytes())


if __name__ == "__main__":
    unittest.main()
