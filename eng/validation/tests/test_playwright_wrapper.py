"""Exercise wrapper argument forwarding using a fake npx; no browser or downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[3] / "TensorAgent/skills/playwright/scripts/playwright_cli.sh"


class PlaywrightWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="playwright-wrapper-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        npx = self.bin / "npx"
        npx.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        npx.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env.get("PATH", "")
        self.env.pop("PLAYWRIGHT_CLI_SESSION", None)

    def run_wrapper(self, *args):
        result = subprocess.run(["bash", str(WRAPPER), *args], cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual("", result.stderr)
        argv = json.loads(result.stdout)
        self.assertEqual(["--yes", "--prefer-offline", "--package", "@playwright/cli@0.1.21", "playwright-cli"], argv[:5])
        return argv[5:]

    def generated_file(self, argv):
        names = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--filename=")]
        self.assertEqual(1, len(names))
        path = Path(names[0])
        self.assertEqual(Path("output/playwright"), path.parent)
        self.assertTrue((self.root / path).is_file())
        return path

    def test_bare_snapshots_get_distinct_reserved_files(self):
        first = self.run_wrapper("snapshot")
        second = self.run_wrapper("snapshot")
        self.assertEqual("snapshot", first[-1])
        self.assertNotEqual(self.generated_file(first), self.generated_file(second))

    def test_target_and_snapshot_options_keep_boundaries(self):
        supplied = ["--json", "--session", "named session", "snapshot", "main > article",
                    "--depth", "3", "--boxes"]
        actual = self.run_wrapper(*supplied)
        self.generated_file(actual)
        self.assertEqual(supplied, actual[1:])

    def test_snapshot_flags_can_precede_command(self):
        for supplied in (["--depth", "3", "--boxes", "snapshot"],
                         ["--raw", "--session=snapshot", "snapshot"],
                         ["--raw", "false", "--json", "true", "snapshot"],
                         ["--boxes", "true", "--help", "false", "snapshot"],
                         ["-s", "snapshot", "snapshot"]):
            with self.subTest(args=supplied):
                actual = self.run_wrapper(*supplied)
                self.generated_file(actual)
                self.assertEqual(supplied, actual[1:])

    def test_explicit_filename_forms_preserved_without_generated_artifacts(self):
        for supplied in (["snapshot", "--filename", "chosen name.yml"],
                         ["snapshot", "--filename=chosen name.yml"],
                         ["snapshot", "--filename="],
                         ["--filename", "chosen.yml", "snapshot"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_help_and_version_do_not_create_files(self):
        for supplied in (["snapshot", "--help"], ["--help", "snapshot"],
                         ["snapshot", "-h"], ["--version", "snapshot"],
                         ["--help=true", "snapshot"], ["--version", "true", "snapshot"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_snapshot_as_other_command_argument_or_session_name_is_untouched(self):
        for supplied in (["fill", "e3", "snapshot"], ["open", "snapshot"],
                         ["--session", "snapshot", "goto", "https://example.test"],
                         ["-s=snapshot", "close"]):
            with self.subTest(args=supplied):
                self.assertEqual(supplied, self.run_wrapper(*supplied))
        self.assertFalse((self.root / "output").exists())

    def test_end_of_options_keeps_generated_filename_as_option(self):
        supplied = ["snapshot", "--", "--filename=literal-target"]
        actual = self.run_wrapper(*supplied)
        self.assertTrue(actual[0].startswith("--filename=output/playwright/"))
        self.assertEqual(supplied, actual[1:])

    def test_environment_session_is_applied_only_without_explicit_session(self):
        self.env["PLAYWRIGHT_CLI_SESSION"] = "from environment"
        actual = self.run_wrapper("snapshot")
        self.assertEqual(["--session", "from environment"], actual[:2])
        self.generated_file(actual)
        for flag in (["--session", "supplied"], ["--session=supplied"],
                     ["-s", "supplied"], ["-s=supplied"]):
            with self.subTest(flag=flag):
                supplied = flag + ["snapshot"]
                actual = self.run_wrapper(*supplied)
                self.generated_file(actual)
                self.assertEqual(supplied, actual[1:])


if __name__ == "__main__":
    unittest.main()
