import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engines
import validate_deepseek41_tools as tools


def call(city="Paris", identifier="call_actual_1"):
    return {"id": identifier, "type": "function", "function": {
        "name": "get_weather", "arguments": json.dumps({"city": city, "units": "celsius"})}}


def response(calls=None, content=None, reasoning=None, status=200, error=None, raw=None):
    result = engines.requests.Response()
    result.status_code = status
    result.url = "http://unused/v1/chat/completions"
    result.headers["Content-Type"] = "application/json" if status != 200 else "text/event-stream"
    if status != 200:
        data = json.dumps(error).encode()
    elif raw is not None:
        data = raw.encode()
    else:
        delta = {"content": content, "reasoning_content": reasoning}
        if calls is not None:
            delta["tool_calls"] = [{"index": index, **value} for index, value in enumerate(calls)]
        chunk = {"choices": [{"delta": delta, "finish_reason": "tool_calls" if calls else "stop"}],
                 "usage": {"prompt_tokens": 30, "completion_tokens": 20}}
        data = ("data: " + json.dumps(chunk) + "\n\ndata:[DONE]\n\n").encode()
    result._content = data
    result._content_consumed = True
    return result


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(model="fixture", url="http://unused", max_tokens=512,
                                       thinking_max_tokens=2048, timeout=10)

    def run_case(self, scenario, responses):
        with patch.object(engines.requests, "post", side_effect=responses) as post:
            result = tools.run_case(self.args, scenario, "test")
        return result, post

    def test_none_uses_generated_call_id_and_fixed_result_before_final_json(self):
        result, post = self.run_case("none_history", [response(calls=[call(identifier="call_from_model_987")]),
            response(content=json.dumps(tools.WEATHER_RESULT))])
        self.assertEqual(result["status"], "ok", result)
        first, last = [entry.kwargs["json"] for entry in post.call_args_list]
        self.assertEqual(len(last["tools"]), 2)
        self.assertEqual(last["tool_choice"], "none")
        self.assertEqual(last["response_format"], {"type": "json_object"})
        self.assertEqual(last["messages"][-2]["tool_call_id"], "call_from_model_987")
        self.assertEqual(json.loads(last["messages"][-2]["content"]), tools.WEATHER_RESULT)
        self.assertEqual(last["messages"][-3]["tool_calls"][0]["id"], "call_from_model_987")
        self.assertEqual(result["turns"][0]["request_sha256"], tools.digest(first))
        self.assertEqual(result["turns"][1]["request_sha256"], tools.digest(last))
        self.assertNotEqual(result["turns"][0]["request_sha256"], result["turns"][1]["request_sha256"])
        self.assertTrue(all(turn["response"]["sse_lines"] for turn in result["turns"]))

    def test_fragmented_parallel_calls_keep_full_names_arguments_ids_and_wire_output(self):
        chunks = []
        for index, city in enumerate(("London", "Paris")):
            chunks.extend([
                {"choices": [{"delta": {"tool_calls": [{"index": index, "id": "call_", "type": "function",
                    "function": {"name": "get_", "arguments": '{"city":'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": index, "id": str(index),
                    "function": {"name": "weather", "arguments": json.dumps(city)+',"units":"celsius"}'}}]}}]}])
        chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                       "usage": {"prompt_tokens": 30, "completion_tokens": 40}})
        raw = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        result, _ = self.run_case("two_calls", [response(raw=raw)])
        self.assertEqual(result["status"], "ok", result)
        details = result["turns"][0]["metrics"]["tool_call_details"]
        self.assertEqual([entry["id"] for entry in details], ["call_0", "call_1"])
        self.assertEqual(len([line for line in result["turns"][0]["response"]["sse_lines"] if line]), 6)

    def test_call_counts_and_duplicate_ids_cannot_pass(self):
        for scenario, calls in [("two_calls", [call()]),
                                ("single_call", [call(), call("London", "call_2")]),
                                ("two_calls", [call(), call("London")])]:
            with self.subTest(scenario=scenario, calls=calls):
                result, _ = self.run_case(scenario, [response(calls=calls)])
                self.assertEqual(result["status"], "fail")
                self.assertTrue(result["turns"][0]["response"]["sse_lines"])

    def test_malformed_ids_types_and_arguments_remain_failed(self):
        invalid = []
        for identifier in ("", " ", "call with spaces", "call\nnewline", 123):
            invalid.append(call(identifier=identifier))
        bad_type = call(); bad_type["type"] = "not-a-function"; invalid.append(bad_type)
        for arguments in ('{', '{"city":"Paris","units":"celsius","extra":1}',
                          '{"city":"London","city":"Paris","units":"celsius"}',
                          '{"city":NaN,"units":"celsius"}', {"city":"Paris","units":"celsius"}):
            bad = call(); bad["function"]["arguments"] = arguments; invalid.append(bad)
        wrong = call(); wrong["function"]["name"] = "read_invoice"; invalid.append(wrong)
        for bad in invalid:
            with self.subTest(call=bad):
                result, _ = self.run_case("required", [response(calls=[bad])])
                self.assertEqual(result["status"], "fail", result)
                self.assertTrue(result["turns"][0]["response"]["sse_lines"])

    def test_reasoning_only_never_satisfies_tool_or_final_answer_contract(self):
        result, _ = self.run_case("required_thinking", [response(reasoning="I will call get_weather for Paris.")])
        self.assertEqual(result["status"], "fail")
        result, _ = self.run_case("none_history", [response(calls=[call()]),
            response(reasoning=json.dumps(tools.WEATHER_RESULT))])
        self.assertEqual(result["status"], "fail")
        metrics = result["turns"][-1]["metrics"]
        self.assertEqual(metrics["output_text"], json.dumps(tools.WEATHER_RESULT))
        self.assertIsNone(metrics["assistant_message"]["content"])

    def test_none_rejects_calls_surrounding_prose_and_wrong_json_types(self):
        for final in [response(calls=[call("London", "call_2")]),
                      response(content='```json\n{"city":"Paris","temperature_c":19}\n```'),
                      response(content='{"city":"Paris","temperature_c":19.0}')]:
            result, _ = self.run_case("none_history", [response(calls=[call()]), final])
            self.assertEqual(result["status"], "fail")

    def test_partial_invalid_sse_is_retained_after_transport_parser_failure(self):
        raw = 'data: {"choices":[{"delta":{"reasoning_content":"partial"}}]}\n\ndata: malformed-json\n'
        result, _ = self.run_case("required", [response(raw=raw)])
        self.assertEqual(result["status"], "fail")
        lines = result["turns"][0]["response"]["sse_lines"]
        self.assertIn("partial", lines[0])
        self.assertEqual(lines[-1], "data: malformed-json")

    def test_actual_http400_must_identify_the_expected_defect(self):
        for scenario, message in [("invalid_required", "tool_choice requires at least one declared function"),
                                  ("invalid_named", "tool_choice names undeclared function undeclared_weather_tool"),
                                  ("invalid_schema", "schema keyword pattern cannot be enforced")]:
            error = {"error": {"message": message, "type": "invalid_request_error"}}
            result, post = self.run_case(scenario, [response(status=400, error=error)])
            self.assertEqual(result["status"], "ok", result)
            self.assertFalse(post.call_args.kwargs["json"]["stream"])
            if scenario == "invalid_required":
                self.assertNotIn("tools", post.call_args.kwargs["json"])
            self.assertEqual(json.loads(result["turns"][0]["response"]["body"]), error)
        for status, message in [(503, "pattern"), (400, "unknown model")]:
            result, _ = self.run_case("invalid_schema", [response(status=status,
                error={"error": {"message": message, "type": "invalid_request_error"}})])
            self.assertEqual(result["status"], "fail")

    def test_named_and_thinking_requests_keep_policy_and_both_declarations(self):
        for scenario in ("named", "named_thinking", "required_thinking"):
            result, post = self.run_case(scenario, [response(calls=[call()], reasoning="Checking the requested city.")])
            self.assertEqual(result["status"], "ok", result)
            body = post.call_args.kwargs["json"]
            self.assertEqual([t["function"]["name"] for t in body["tools"]], ["read_invoice", "get_weather"])
            self.assertEqual(body["think"], scenario.endswith("_thinking"))
            if scenario.startswith("named"):
                self.assertEqual(body["tool_choice"]["function"]["name"], "get_weather")

    def test_complete_run_retains_failed_two_call_cases_and_execution_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            def fixture_case(args, scenario, tag):
                return {"status": "fail" if scenario == "two_calls" else "ok", "scenario": scenario, "tag": tag}
            with patch.object(tools, "run_case", side_effect=fixture_case):
                code = tools.main(["--model", "fixture", "--weights-id", "fixed", "--profile", "test", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertTrue(report["run_complete"])
            self.assertEqual(report["execution_plan"]["expected_cases"], 30)
            self.assertEqual(len(report["cases"]), 30)
            self.assertEqual(sum(c["status"] == "fail" for c in report["cases"]), 5)
            self.assertTrue(all(c["concurrency"] == 1 for c in report["cases"] if c["scenario"].endswith("_thinking")))
            self.assertIn("engines.py", report["harness_sha256"])

    def test_interrupted_run_keeps_partial_results_and_never_marks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            def fixture_case(args, scenario, tag):
                if scenario == "named": raise RuntimeError("injected orchestration failure")
                return {"status":"ok", "scenario":scenario, "tag":tag}
            with patch.object(tools, "run_case", side_effect=fixture_case):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    tools.main(["--model", "fixture", "--weights-id", "fixed", "--profile", "test",
                                "--output", str(output), "--scenarios", "required,named", "--concurrency", "1"])
            report = json.loads(output.read_text())
            self.assertFalse(report["run_complete"])
            self.assertEqual(len(report["cases"]), 1)


if __name__ == "__main__":
    unittest.main()
