"""Prevent missing/partial HTTP streams from producing passing benchmark data."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "bonsai2-bench.py"
SPEC = importlib.util.spec_from_file_location("bonsai2_bench", MODULE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def event(delta=None, usage=None, finish=None):
    value = {"choices": [{"delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = usage
    return ("data: " + json.dumps(value) + "\n\n").encode()


class Bonsai2BenchTests(unittest.TestCase):
    def stream(self):
        return [event({"role": "assistant"}), event({"content": "Two complete words"}),
                event(finish="length"), event(usage={"completion_tokens": 7, "prompt_tokens": 12}),
                b"data: [DONE]\n"]

    def test_uses_usage_not_chunk_count_and_ignores_role_only_delta(self):
        calls = []
        result = BENCH.read_sse(self.stream(), lambda: calls.append(1) or .25)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["first_delta_seconds"], .25)
        self.assertEqual(calls, [1])

    def test_reasoning_is_part_of_first_delta_and_output_parity(self):
        stream = self.stream()
        stream.insert(1, event({"reasoning_content": "Thinking"}))
        result = BENCH.read_sse(stream, lambda: .1)
        self.assertEqual(result["reasoning"], "Thinking")
        changed = dict(result, reasoning="Different")
        self.assertFalse(BENCH.same_output(result, changed))

    def test_stream_errors_missing_usage_finish_or_done_fail(self):
        original = self.stream()
        for invalid in (original[:-1], [original[index] for index in [0, 1, 2, 4]],
                        [original[index] for index in [0, 1, 3, 4]],
                        [b'data: {"error":{"message":"failed"}}\n']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BENCH.read_sse(invalid, lambda: .1)

    def test_no_text_and_nonpositive_usage_fail(self):
        for count in (0, -1, "7", None):
            stream = [event({"content": "Answer"}), event(finish="stop"),
                      event(usage={"completion_tokens": count}), b"data: [DONE]\n"]
            with self.subTest(count=count), self.assertRaises(ValueError):
                BENCH.read_sse(stream, lambda: .1)
        with self.assertRaises(ValueError):
            BENCH.read_sse([event(finish="stop", usage={"completion_tokens": 7}), b"data: [DONE]\n"], lambda: .1)

    def test_same_text_with_different_token_count_or_stop_reason_is_not_parity(self):
        result = BENCH.read_sse(self.stream(), lambda: .1)
        self.assertTrue(BENCH.same_output(result, dict(result)))
        self.assertFalse(BENCH.same_output(result, dict(result, completion_tokens=8)))
        self.assertFalse(BENCH.same_output(result, dict(result, finish_reason="stop")))

    def test_reference_retains_eos_prefix_without_masking_raw_greedy_logits(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(url="http://unused", tokens=4, prompts=["Prompt"], timeout=1,
                                   golden=Path(temporary) / "golden.json")
            replies = [{"prompt": "Formatted"}, {"tokens": [4, 5]},
                       {"tokens": [6, 7], "stop_type": "eos", "content": "Answer"}]
            with patch.object(BENCH, "request_json", side_effect=replies) as request:
                report = {}
                self.assertTrue(BENCH.run_reference(args, report, lambda: None))
                body = request.call_args_list[-1].args[1]
                self.assertFalse(body["ignore_eos"])
                self.assertEqual(body["repeat_penalty"], 1.0)
                self.assertEqual(body["top_k"], 0)
                self.assertEqual(json.loads(args.golden.read_text())[0]["generated_tokens"], [6, 7])

    def test_reference_does_not_qualify_short_output_without_eos_or_empty_tokens(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(url="http://unused", tokens=4, prompts=["Prompt"], timeout=1,
                                   golden=Path(temporary) / "golden.json")
            for response in ({"tokens": [6, 7], "stop_type": "limit"},
                             {"tokens": [], "stop_type": "eos"}, {"tokens": [1, 2, 3, 4, 5]}):
                replies = [{"prompt": "Formatted"}, {"tokens": [4, 5]}, response]
                with self.subTest(response=response), patch.object(BENCH, "request_json", side_effect=replies):
                    with self.assertRaises(ValueError):
                        BENCH.run_reference(args, {}, lambda: None)
                self.assertFalse(args.golden.exists())


if __name__ == "__main__":
    unittest.main()
