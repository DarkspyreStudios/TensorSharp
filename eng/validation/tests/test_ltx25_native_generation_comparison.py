#!/usr/bin/env python3
"""Native RGB/PCM parity and numerically stable comparison metrics."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COMPARE = ROOT / "eng/validation/ltx25-compare-native-generations.py"


class NativeGenerationComparison(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ltx-native-comparison-test-", dir=ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def compare(self, before, after, *, require_bitwise=False, changed_audio=False):
        self.assertEqual(before.shape, after.shape)
        for name, pixels in (("before", before), ("after", after)):
            raw = np.asarray(pixels, dtype="<f4").tobytes()
            (self.root / f"{name}.json.0.rgb.f32").write_bytes(raw)
            for channel in range(2):
                audio = np.linspace(-.1, .1, 16, dtype="<f4")
                if changed_audio and name == "after" and channel == 1:
                    audio[0] = .2
                audio.tofile(self.root / f"{name}.json.0.audio-{channel}.f32")
            report = dict(completed=True, settings={"elements": len(pixels)}, prompt="test", mode=None,
                          sigmas=[1, 0], model="synthetic.gguf", reference_inputs=[], executions=[dict(
                              repeat=0, prompt="test", kind="benchmark", generation_ms=10 if name == "before" else 5,
                              video_sha256=hashlib.sha256(raw).hexdigest(), audio_samples=16)])
            (self.root / f"{name}.json").write_text(json.dumps(report))
        args = [sys.executable, "-W", "error::RuntimeWarning", str(COMPARE),
                "--before", str(self.root / "before.json"), "--after", str(self.root / "after.json"),
                "--output", str(self.root / "comparison.json")]
        if require_bitwise:
            args.append("--require-bitwise")
        return subprocess.run(args, capture_output=True, text=True)

    def row(self, result):
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stderr)
        return json.loads((self.root / "comparison.json").read_text())["executions"][0]

    def test_identical_bounded_rgb_crosses_chunk_boundary_without_warnings(self):
        pixels = np.linspace(0, 1, (1 << 20) + 17, dtype=np.float32)
        row = self.row(self.compare(pixels, pixels, require_bitwise=True))
        self.assertTrue(row["raw_rgb_bitwise_equal"])
        self.assertTrue(all(channel["bitwise_equal"] for channel in row["raw_audio"]))
        self.assertEqual((0, 0, 0, None), (row["max_absolute"], row["rmse"], row["relative_l2"], row["psnr_db"]))
        self.assertEqual(2, row["generation_speedup"])

    def test_known_nonzero_metrics_are_correct(self):
        before = np.full(105, .5, dtype=np.float32)
        row = self.row(self.compare(before, before + .25))
        self.assertEqual(.25, row["max_absolute"])
        self.assertEqual(.25, row["rmse"])
        self.assertEqual(.5, row["relative_l2"])
        self.assertAlmostEqual(-10 * math.log10(.25 ** 2), row["psnr_db"])

    def test_large_finite_f32_metrics_do_not_overflow(self):
        before = np.full(21, np.finfo(np.float32).max / 2, dtype=np.float32)
        row = self.row(self.compare(before, before / 2))
        self.assertTrue(all(math.isfinite(row[key]) for key in ("max_absolute", "rmse", "relative_l2", "psnr_db")))
        self.assertAlmostEqual(.5, row["relative_l2"])
        self.assertAlmostEqual(1, row["rmse"] / (float(before[0]) / 2))

    def test_nonfinite_rgb_is_rejected(self):
        result = self.compare(np.zeros(3, dtype=np.float32), np.array([0, np.nan, 1], dtype=np.float32))
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Nonfinite RGB", result.stderr)
        self.assertFalse((self.root / "comparison.json").exists())

    def test_bitwise_requirement_checks_both_audio_channels(self):
        pixels = np.zeros(3, dtype=np.float32)
        result = self.compare(pixels, pixels, require_bitwise=True, changed_audio=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("RGB/audio bitwise parity failed", result.stderr)
        self.assertFalse((self.root / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
