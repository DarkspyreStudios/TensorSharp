#!/usr/bin/env python3
"""Exercise the Jev benchmark's HTTP validation without any model or GPU."""
import argparse
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

SPEC = importlib.util.spec_from_file_location("jev_benchmark", Path(__file__).parents[1] / "jev-benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FixtureServer(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "state" not in payload or not payload.get("questions") or any(
            question.get("type") not in ("choice", "noul", "score") or
            (question["type"] == "choice" and len(question.get("criteria", {})) < 2)
            for question in payload.get("questions", {}).values()
        ):
            code, content = 422, {"error": "invalid request"}
        else:
            answers = {}
            state = payload["state"]
            color = state.get("target", state)["color"]
            for key, question in payload["questions"].items():
                answers[key] = {"type": "choice", "choice": color, "probabilities": {label: float(label == color) for label in question["criteria"]}, "confidence": 1.0}
            code, content = 200, {"model": "mock-not-a-model", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 3}}
        encoded = json.dumps(content).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class JevBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1/systemone"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_score_is_zero_indexed_weighted_mean(self):
        question = {"type": "score", "criteria": ["small", "medium", "large"]}
        answer = {"type": "score", "score": 1.25, "legend": {"0": "small", "1": "medium", "2": "large"}, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.5}
        result = bench.validate_answer(question, answer, 2, "size")
        self.assertTrue(result["correct"])
        self.assertEqual(0.75, result["score_absolute_error"])
        answer["score"] = 2.25
        with self.assertRaisesRegex(ValueError, "zero-indexed"):
            bench.validate_answer(question, answer, 2, "size")

    def test_distribution_must_be_finite_and_normalized(self):
        question = {"type": "choice", "criteria": {"red": "red", "blue": "blue"}}
        for probabilities in ({"red": 0.9, "blue": 0.9}, {"red": float("nan"), "blue": 0}, {"red": True, "blue": 0}, {"red": 0.5, "unexpected": 0.5}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                bench.validate_answer(question, {"type": "choice", "choice": "red", "probabilities": probabilities, "confidence": 0.5}, "red", "color")

    def test_choice_must_match_distribution(self):
        question = {"type": "choice", "criteria": {"red": "red", "blue": "blue"}}
        with self.assertRaisesRegex(ValueError, "highest-probability"):
            bench.validate_answer(question, {"type": "choice", "choice": "red", "probabilities": {"red": 0.1, "blue": 0.9}, "confidence": 0.9}, "red", "color")

    def test_rejects_result_keys_from_another_request(self):
        case = bench.isolation_cases()[0]
        payload = bench.make_payload(case, case["questions"], "test", 1, 0)
        result = bench.post(self.url, payload, 5)
        result["body"]["answers"]["leaked_question"] = result["body"]["answers"].pop(case["id"])
        with self.assertRaisesRegex(ValueError, "contamination"):
            bench.validate_response(payload, result, case["expected"])

    def test_http_concurrent_isolation_and_invalid_requests(self):
        args = argparse.Namespace(timeout=5, samples=1, concurrency=[1, 4], isolation_tolerance=1e-8)
        evidence = []
        result = bench.run_isolation(self.url, None, args, "test", evidence.append)
        self.assertTrue(result["contract_passed"])
        self.assertTrue(result["known_labels_passed"])
        self.assertTrue(result["repeatability_within_tolerance"])
        self.assertEqual(8, len(evidence))
        invalid = bench.run_invalid(self.url, None, args, "test", evidence.append)
        self.assertTrue(invalid["passed"])
        self.assertEqual(12, len(evidence))

    def test_http_metrics_count_failures_as_wrong(self):
        case = bench.isolation_cases()[0]
        payload = bench.make_payload(case, case["questions"], "test", 1, 0)
        valid = bench.run_case(self.url, None, 5, case, payload, "measured")
        invalid = {**valid, "valid": False, "decisions": [], "validation_error": "malformed"}
        result = bench.summarize([valid, invalid], 2)
        self.assertEqual(0.5, result["accuracy_failures_count_as_wrong"])
        self.assertEqual(1, result["failures"])
        self.assertEqual(0.5, result["successful_requests_per_second"])

    def test_background_has_exact_word_count_and_retains_target(self):
        original = {"color": "red"}
        expanded = bench.state_for(original, 2048, "case-1")
        self.assertEqual(original, expanded["target"])
        self.assertEqual(2048, len(expanded["irrelevant_background"].split()))
        self.assertEqual("irrelevant_request_identifier", next(iter(expanded)))

    def test_performance_is_unmeasured_without_budgets_and_failures_cannot_pass(self):
        summary = {"successful_latency_ms": {"p95": 50}, "successful_requests_per_second": 20, "failures": 0}
        self.assertEqual("not_evaluated", bench.performance_checks(summary, None, None)["status"])
        self.assertEqual("passed", bench.performance_checks(summary, 100, 10)["status"])
        self.assertEqual("failed", bench.performance_checks(summary, 1, None)["status"])
        self.assertEqual("failed", bench.performance_checks(summary, None, 100)["status"])
        summary["failures"] = 1
        self.assertEqual("failed", bench.performance_checks(summary, 100, 10)["status"])

    def test_endpoint_normalization_and_secret_rejection(self):
        for url in ("http://localhost:8080", "http://localhost:8080/v1", "http://localhost:8080/v1/systemone"):
            self.assertEqual("http://localhost:8080/v1/systemone", bench.endpoint_url(url))
        with self.assertRaises(ValueError):
            bench.endpoint_url("http://secret@localhost:8080")

    def test_cli_writes_complete_evidence_for_mock_endpoint(self):
        artifacts = bench.ROOT / "artifacts/jev-harness-tests"
        artifacts.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="mock-only-", dir=artifacts))
        cases = bench.isolation_cases()[:2]
        fixture = {"schemas": {case["id"]: case["questions"] for case in cases}, "cases": [{**case, "schema": case["id"]} for case in cases]}
        path = run / "fixtures.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        result = subprocess.run([sys.executable, str(bench.ROOT / "eng/jev-benchmark.py"), "--endpoint", "mock=" + self.url, "--fixtures", str(path), "--concurrency", "1,2", "--out", str(run / "evidence"), "--timeout", "5"], capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        summary = json.loads((run / "evidence/summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["passed_requested_checks"])
        self.assertEqual(2, len(summary["endpoints"]["mock"]["groups"]))
        self.assertTrue((run / "evidence/manifest.json").exists())
        self.assertEqual(17, len((run / "evidence/results.jsonl").read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
