#!/usr/bin/env python3
"""Validate V4.1 OpenAI tool policies against an already running HTTP server.

All tool results are fixed fixtures. Model output never executes an external
tool, command, or network lookup. Raw responses and failures remain in reports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import engines
from validate_inference import SAMPLING, WEATHER_TOOL, INVOICE_TOOL, assistant_content, digest


SCENARIOS = ("required", "named", "none_history", "single_call", "two_calls",
             "required_thinking", "named_thinking", "invalid_required", "invalid_named", "invalid_schema")
WEATHER_RESULT = {"city": "Paris", "temperature_c": 19}
TOOLS = [INVOICE_TOOL, WEATHER_TOOL]  # Named weather is deliberately not the first declaration.


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def request_body(model, scenario, tag, max_tokens=512, thinking_max_tokens=2048):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown tool-policy scenario: {scenario}")
    thinking = scenario.endswith("_thinking")
    prompt = "Call get_weather exactly once for Paris using celsius. Do not invent weather or call another function."
    if scenario == "none_history":
        prompt = ('Use get_weather for Paris in celsius, then report its returned temperature '
                  'as JSON with exactly city and temperature_c.')
    elif scenario == "single_call":
        prompt = ("We need weather for Paris and London in celsius. Start with Paris. "
                  "This turn is serial: issue only the Paris get_weather call and wait for its result.")
    elif scenario == "two_calls":
        prompt = ("In this one assistant turn, issue exactly two get_weather calls: one for Paris "
                  "and one for London, both in celsius. Issue both now without waiting for the first result.")
    body = {"model": model, "messages": [{"role": "user", "content": f"[validation {tag}]\n{prompt}"}],
            "tools": copy.deepcopy(TOOLS), "tool_choice": "required",
            "parallel_tool_calls": scenario == "two_calls", "stream": True,
            "stream_options": {"include_usage": True}, "timings_per_token": True,
            "max_tokens": thinking_max_tokens if thinking else max_tokens,
            **SAMPLING, **engines.thinking_body("tensorsharp", thinking)}
    if scenario in ("named", "named_thinking", "invalid_named"):
        body["tool_choice"] = {"type": "function", "function": {
            "name": "undeclared_weather_tool" if scenario == "invalid_named" else "get_weather"}}
    if scenario == "invalid_required":
        body.pop("tools")
    elif scenario == "invalid_schema":
        body["tools"][1]["function"]["parameters"]["properties"]["city"]["pattern"] = "^[A-Z][a-z]+$"
    if scenario.startswith("invalid_"):
        body["stream"] = False
        body.pop("stream_options")
        body.pop("timings_per_token")
    return body


def check_calls(metrics, cities):
    if not metrics.get("usage_present") or metrics.get("finish_reason") != "tool_calls":
        raise ValueError("expected token usage and finish_reason=tool_calls")
    calls = metrics.get("tool_call_details")
    if not isinstance(calls, list) or len(calls) != len(cities):
        raise ValueError(f"expected exactly {len(cities)} structured tool calls")
    message = metrics.get("assistant_message")
    if not isinstance(message, dict) or message.get("tool_calls") != calls:
        raise ValueError("assistant tool calls do not match the parsed response")
    ids, actual = set(), []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("tool call must be an object")
        identifier = call.get("id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
                or identifier in ids or call.get("type") != "function"):
            raise ValueError("tool call needs a unique nonempty identifier and type=function")
        ids.add(identifier)
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "get_weather":
            raise ValueError("expected the declared get_weather function")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("tool arguments must be a complete JSON string")
        arguments = strict_json(function["arguments"])
        if (not isinstance(arguments, dict) or set(arguments) != {"city", "units"}
                or not isinstance(arguments["city"], str) or arguments["units"] != "celsius"):
            raise ValueError("weather arguments must contain only city and units=celsius")
        actual.append(arguments["city"])
    if sorted(actual) != sorted(cities):
        raise ValueError(f"expected weather cities {cities}, got {actual}")
    return calls


def check_final(metrics):
    if not metrics.get("usage_present") or metrics.get("finish_reason") != "stop":
        raise ValueError("final answer must have usage and finish_reason=stop")
    if (metrics.get("tool_call_details") or metrics.get("tool_calls")
            or (metrics.get("assistant_message") or {}).get("tool_calls")):
        raise ValueError("tool_choice=none emitted an actual tool call")
    content = assistant_content(metrics)
    answer = strict_json(content)
    if answer != WEATHER_RESULT or type(answer.get("temperature_c")) is not int:
        raise ValueError("final assistant content must exactly match the fixture weather JSON")
    return content


def check_rejection(scenario, response):
    if response["http_status"] != 400:
        raise ValueError(f"expected HTTP400, got HTTP{response['http_status']}")
    data = strict_json(response["body"])
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict) or error.get("type") != "invalid_request_error":
        raise ValueError("expected a structured invalid_request_error")
    marker = {"invalid_required": "at least one", "invalid_named": "undeclared_weather_tool",
              "invalid_schema": "pattern"}[scenario]
    message = error.get("message")
    if not isinstance(message, str) or marker not in message.lower():
        raise ValueError("HTTP400 did not identify the requested policy/schema defect")


def run_case(args, scenario, tag):
    body = request_body(args.model, scenario, tag, args.max_tokens, args.thinking_max_tokens)
    result = {"scenario": scenario, "tag": tag, "status": "fail", "thinking": body["think"],
              "input_sha256": digest(body), "turns": []}
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    start = time.monotonic()
    try:
        for step in range(2 if scenario == "none_history" else 1):
            turn = {"request": copy.deepcopy(body), "request_sha256": digest(body), "response": {}}
            result["turns"].append(turn)  # Record the request before any transport can fail.
            if scenario.startswith("invalid_"):
                with engines.requests.post(endpoint, json=body, timeout=(30, args.timeout)) as response:
                    turn["response"] = {"http_status": response.status_code,
                                        "content_type": response.headers.get("Content-Type"), "body": response.text}
                check_rejection(scenario, turn["response"])
                break
            metrics = engines._run_streaming(endpoint, body, args.timeout, response_log=turn["response"])
            turn["metrics"] = metrics
            if not any(line.startswith("data:") and line[5:].strip() == "[DONE]"
                       for line in turn["response"]["sse_lines"]):
                raise ValueError("completion SSE ended without [DONE]")
            if scenario == "none_history" and step == 1:
                result["validated_content"] = check_final(metrics)
            else:
                calls = check_calls(metrics, ["Paris", "London"] if scenario == "two_calls" else ["Paris"])
                if scenario == "none_history":
                    # Feed back the model's actual call ID and structured message.
                    # Only the returned weather data is synthetic; no tool executes.
                    tool_result = {"role": "tool", "tool_call_id": calls[0]["id"],
                                   "content": json.dumps(WEATHER_RESULT, sort_keys=True)}
                    body = {**body, "messages": [*body["messages"], copy.deepcopy(metrics["assistant_message"]),
                            tool_result, {"role": "user", "content":
                            "Use that returned weather result. Do not call a tool. Return only JSON with city and temperature_c."}],
                            "tool_choice": "none", "response_format": {"type": "json_object"}}
        result["status"] = "ok"
    except Exception as error:
        result["detail"] = f"{type(error).__name__}: {error}"
    result["wall_seconds"] = time.monotonic() - start
    return result


def execution_plan(scenarios, degrees, repetitions):
    return [{"scenario": scenario, "concurrency": concurrency, "repetition": repetition}
            for scenario in scenarios
            for concurrency in ([1] if scenario.endswith("_thinking") or scenario.startswith("invalid_") else degrees)
            for repetition in range(repetitions)]


def save(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--native-sha256")
    parser.add_argument("--server-build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    scenarios = args.scenarios.split(",")
    try:
        degrees = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("concurrency must contain positive integers")
    if (not scenarios or set(scenarios) - set(SCENARIOS) or len(scenarios) != len(set(scenarios))
            or not degrees or len(degrees) != len(set(degrees)) or any(value < 1 for value in degrees)
            or min(args.repetitions, args.max_tokens, args.thinking_max_tokens, args.timeout) <= 0):
        parser.error("select distinct known scenarios, distinct positive concurrency values and positive limits")
    plan = execution_plan(scenarios, degrees, args.repetitions)
    sources = [Path(__file__), Path(engines.__file__), Path(__file__).with_name("validate_inference.py")]
    report = {"model": args.model, "weights_id": args.weights_id, "profile": args.profile,
              "native_sha256": args.native_sha256, "server_build": args.server_build, "url": args.url,
              "started_utc": datetime.now(timezone.utc).isoformat(), "run_complete": False,
              "sampling": SAMPLING, "max_tokens": args.max_tokens, "thinking_max_tokens": args.thinking_max_tokens,
              "harness_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "fixtures": {"tools": TOOLS, "weather_result": WEATHER_RESULT},
              "fixtures_sha256": digest({"tools": TOOLS, "weather_result": WEATHER_RESULT}),
              "execution_plan": {"waves": plan, "expected_cases": sum(wave["concurrency"] for wave in plan)},
              "scope": "Tool-policy/schema transport and fixture arguments, not external tool execution or broad model quality/performance parity.",
              "cases": [], "waves": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    try:
        for wave in plan:
            start = time.monotonic()
            cases = []
            with ThreadPoolExecutor(max_workers=wave["concurrency"]) as pool:
                pending = {pool.submit(run_case, args, wave["scenario"],
                    f"{wave['scenario']}-c{wave['concurrency']}-r{wave['repetition']}-i{i}"): i
                    for i in range(wave["concurrency"])}
                for future in as_completed(pending):
                    case = future.result()
                    case.update(concurrency=wave["concurrency"], repetition=wave["repetition"], client=pending[future])
                    cases.append(case); report["cases"].append(case)
                    save(args.output, report)
            passed = sum(case["status"] == "ok" for case in cases)
            report["waves"].append({**wave, "passed": passed, "total": len(cases),
                                    "wall_seconds": time.monotonic() - start})
            print(f"{wave['scenario']} c{wave['concurrency']} r{wave['repetition']}: {passed}/{len(cases)} passed", flush=True)
        report["run_complete"] = len(report["cases"]) == report["execution_plan"]["expected_cases"]
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        save(args.output, report)
    return int(not report["run_complete"] or any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
