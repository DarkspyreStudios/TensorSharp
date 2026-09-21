import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "revalidate_inference_report", Path(__file__).parents[1] / "revalidate-inference-report.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class RevalidationEvidenceTests(unittest.TestCase):
    def capture(self):
        cases = []
        for repeat in range(2):
            for client in range(2):
                cases.append({"scenario": "decode", "tag": f"decode-c2-r{repeat}-i{client}",
                              "concurrency": 2, "repeat": repeat, "status": "ok", "total_wall_ms": 1000,
                              "turns": [{"request": {"max_tokens": 512}, "metrics": {
                                  "usage_present": True, "completion_tokens": 512, "finish_reason": "length",
                                  "output_text": "captured response", "decode_tps": 30.0,
                                  "assistant_message": {"role": "assistant", "content": "captured response"},
                                  "bad": repeat == 1 and client == 1}}]})
        return {"run_complete": True, "execution_plan": {"expected_cases": 4}, "cases": cases,
                "harness_sha256": {"validate_inference.py": "original-inference-hash"},
                "waves": [{"scenario": "decode", "concurrency": 2, "repeat": repeat,
                           "all_passed": True, "generated_tokens": 1024,
                           "end_to_end_tokens_per_second": 25.0 + repeat} for repeat in range(2)]}

    def run_replay(self, data):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "capture.json"
            original = json.dumps(data).encode()
            source.write_bytes(original)
            def gate(metrics, budget, text):
                if metrics["bad"]:
                    raise ValueError("saved response fails the current gate")
            with patch.object(replay.validator, "validate_decode", side_effect=gate):
                result = replay.revalidate(source)
            self.assertEqual(source.read_bytes(), original)
            return result

    def test_preserves_capture_and_excludes_whole_failed_wave_from_timing(self):
        result = self.run_replay(self.capture())
        self.assertFalse(result["new_inference_run"])
        self.assertEqual((result["passed"], result["failed"]), (3, 1))
        self.assertEqual(result["inference_harness_sha256"],
                         {"validate_inference.py": "original-inference-hash"})
        self.assertEqual([wave["all_passed"] for wave in result["waves"]], [True, False])
        summary = result["comparable_timing_summary"]["decode@c2"]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["complete_passing_waves"], 1)
        self.assertEqual(summary["aggregate_end_to_end_tps_median"], 25.0)

    def test_original_failure_cannot_be_promoted_by_new_gate(self):
        captured = self.capture()
        captured["cases"][0].update(status="fail", detail="original transport failure")
        result = self.run_replay(captured)
        self.assertEqual(result["cases"][0]["status"], "fail")
        self.assertIn("original transport failure", result["cases"][0]["detail"])
        self.assertEqual(result["comparable_timing_summary"], {})

    def test_incomplete_duplicate_or_missing_wave_evidence_is_rejected(self):
        for defect in ("incomplete", "count", "duplicate_tag", "missing_wave", "duplicate_wave"):
            with self.subTest(defect=defect):
                data = self.capture()
                if defect == "incomplete": data["run_complete"] = False
                elif defect == "count": data["execution_plan"]["expected_cases"] = 5
                elif defect == "duplicate_tag": data["cases"][1]["tag"] = data["cases"][0]["tag"]
                elif defect == "missing_wave": data["waves"].pop()
                elif defect == "duplicate_wave": data["waves"].append(data["waves"][0])
                with self.assertRaises(ValueError):
                    self.run_replay(data)


if __name__ == "__main__":
    unittest.main()
