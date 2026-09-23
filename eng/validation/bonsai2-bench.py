#!/usr/bin/env python3
"""Record Bonsai2 HTTP workloads and raw-token llama.cpp reference outputs.

Start one server at a time, then run this tool against it. `http` measures
client-observed latency/aggregate throughput and verifies each concurrent greedy
response against its solo response. `reference` produces input for ParityHarness
--ref, using llama.cpp's own template/tokenizer and explicit raw prompt tokens.
Neither successful HTTP responses nor matching outputs establish model quality.
Generated evidence belongs in ignored docs/validation/ or artifacts/.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen
import zlib

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    "Explain how a rainbow forms in three clear sentences.",
    "Write a Python function that returns the sum of the even numbers in a list.",
    "请用三句话介绍中国的春节。",
    "A shop has 12 apples, sells 5, and receives 9 more. How many apples remain? Explain briefly.",
]


def capture(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def file_info(path, full_hash=False):
    path = Path(path).resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        stat = path.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                      sha256=sha256(path) if full_hash else None)
    return result


def revision(path):
    return {"path": str(Path(path).resolve()),
            "commit": capture(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "changes": capture(["git", "-C", str(path), "status", "--porcelain"])}


def inventory(args):
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu": capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": capture(["sysctl", "-n", "hw.memsize"]),
            "cpu_count": os.cpu_count(), "python": sys.version,
            "dotnet": capture(["dotnet", "--version"]),
            "python_packages": {name: importlib.util.find_spec(name) is not None
                                for name in ("torch", "vllm", "sglang")},
            "executables": {name: shutil.which(name) for name in ("nvidia-smi", "vllm", "sglang")},
            "repositories": [revision(path) for path in [ROOT] + args.repo],
            "files": [file_info(path, args.hash_files) for path in args.file],
            "limitations": ["Package/executable discovery does not establish engine or device compatibility.",
                            "Missing files, engines, and devices are unavailable scenarios, never passing tests."]}


def request_json(url, body, timeout):
    request = Request(url, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def read_sse(lines, elapsed):
    """Measure generated deltas, but count tokens only from server usage.

    A single delta may contain zero, one, or several tokens; counting events as
    tokens overstates throughput, especially with reasoning/tool buffering.
    """
    content, reasoning, events = [], [], []
    first, usage, finish, done = None, None, None, False
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise ValueError("Server stream error: " + json.dumps(event["error"]))
        events.append(event)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if text or thought:
                if first is None:
                    first = elapsed()
                content.append(text)
                reasoning.append(thought)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done:
        raise ValueError("Stream ended before [DONE].")
    if finish is None:
        raise ValueError("Stream has no finish reason.")
    if usage is None or not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] <= 0:
        raise ValueError("Stream lacks positive completion_tokens usage; token rate cannot be measured.")
    if first is None:
        raise ValueError("Stream contains no generated text or reasoning.")
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "first_delta_seconds": first, "completion_tokens": usage["completion_tokens"],
            "usage": usage, "finish_reason": finish, "events": events}


def completion(args, prompt, tokens, barrier=None):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
            "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0,
            "seed": 42, "max_tokens": tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "think": False, "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": False}
    body.update(args.request_extra)
    if barrier:
        barrier.wait(timeout=args.timeout)
    started = time.perf_counter()
    request = Request(args.url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    with urlopen(request, timeout=args.timeout) as response:
        result = read_sse(response, lambda: time.perf_counter() - started)
    result.update(wall_seconds=time.perf_counter() - started, request=body,
                  requested_token_budget_reached=result["completion_tokens"] == body["max_tokens"])
    return result


def same_output(first, second):
    return all(first[key] == second[key]
               for key in ("content", "reasoning", "completion_tokens", "finish_reason"))


def run_http(args, report, save):
    prompts = args.prompts or PROMPTS
    if max(args.concurrency) > len(prompts):
        raise ValueError("Provide at least as many prompts as the largest concurrency.")
    report.update(engine=args.engine, url=args.url, model=args.model,
                  request_extra=args.request_extra, concurrency=args.concurrency,
                  repeats=args.repeats, max_tokens=args.tokens, prompts=prompts,
                  limitations=[
                      "The endpoint is supplied by the operator; this client does not attest loaded binaries/weights.",
                      "Serialize server runs on the same device; competing GPU jobs invalidate comparisons.",
                      "cache_prompt:false is requested; verify each server's effective prefix-cache policy separately.",
                      "First-delta latency includes HTTP/queue/template/prefill time and may include reasoning buffering.",
                      "Throughput includes all client-observed request time; it is not isolated model decode throughput.",
                      "Exact solo/concurrent text parity is a regression check, not a semantic quality score.",
                      "Cross-engine templates and early stopping must be reviewed before claiming performance parity.",
                      "Request extras can override sampling/workload defaults; inspect each recorded payload.",
                  ], solo=[], runs=[])
    report["warmup"] = completion(args, prompts[0], min(8, args.tokens))
    save()
    for prompt in prompts[:max(args.concurrency)]:
        report["solo"].append(completion(args, prompt, args.tokens))
        save()
    for concurrency in args.concurrency:
        for repeat in range(args.repeats):
            barrier = threading.Barrier(concurrency + 1)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(completion, args, prompts[index], args.tokens, barrier)
                           for index in range(concurrency)]
                started = time.perf_counter()
                barrier.wait(timeout=args.timeout)
                requests = [future.result() for future in futures]
                seconds = time.perf_counter() - started
            for index, result in enumerate(requests):
                result["matches_solo"] = same_output(report["solo"][index], result)
            tokens = sum(result["completion_tokens"] for result in requests)
            report["runs"].append({"concurrency": concurrency, "repeat": repeat + 1,
                                   "wall_seconds": seconds, "completion_tokens": tokens,
                                   "aggregate_tokens_per_second": tokens / seconds,
                                   "requests": requests})
            save()
    report["summaries"] = []
    for concurrency in args.concurrency:
        runs = [run for run in report["runs"] if run["concurrency"] == concurrency]
        rates = [run["aggregate_tokens_per_second"] for run in runs]
        requests = [request for run in runs for request in run["requests"]]
        report["summaries"].append({"concurrency": concurrency,
                                   "aggregate_tps_median": statistics.median(rates),
                                   "aggregate_tps_min": min(rates), "aggregate_tps_max": max(rates),
                                   "first_delta_seconds_median": statistics.median(
                                       request["first_delta_seconds"] for request in requests),
                                   "matching_requests": sum(request["matches_solo"] for request in requests),
                                   "full_budget_requests": sum(request["requested_token_budget_reached"] for request in requests),
                                   "total_requests": len(requests)})
    return all(request["matches_solo"] for run in report["runs"] for request in run["requests"])


def run_reference(args, report, save):
    report.update(url=args.url, max_tokens=args.tokens, records=[],
                  limitations=["Generated tokens are reference observations, not semantic quality labels.",
                               "Use the exact same GGUF/KV dtype for the TensorSharp comparison."])
    for prompt in args.prompts or PROMPTS:
        formatted = request_json(args.url.rstrip("/") + "/apply-template",
                                 {"messages": [{"role": "user", "content": prompt}],
                                  "add_generation_prompt": True,
                                  "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
        tokens = request_json(args.url.rstrip("/") + "/tokenize",
                              {"content": formatted["prompt"], "add_special": True,
                               "parse_special": True}, args.timeout)["tokens"]
        if not tokens or not all(isinstance(token, int) for token in tokens):
            raise ValueError("Reference tokenizer did not return integer token IDs.")
        body = {"prompt": tokens, "temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0,
                "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "seed": 42,
                "n_predict": args.tokens, "ignore_eos": False, "return_tokens": True,
                "cache_prompt": False}
        response = request_json(args.url.rstrip("/") + "/completion", body, args.timeout)
        generated = response.get("tokens")
        if not generated or len(generated) > args.tokens or not all(isinstance(token, int) for token in generated):
            raise ValueError("Reference completion must return a nonempty raw token ID sequence within the requested budget.")
        if len(generated) < args.tokens and not (response.get("stopped_eos") or response.get("stop_type") == "eos"):
            raise ValueError("Reference completion ended early without EOS; partial output cannot qualify a golden.")
        report["records"].append({"prompt": prompt, "formatted_prompt": formatted["prompt"],
                                  "prompt_tokens": tokens, "generated_tokens": generated,
                                  "content": response.get("content"), "request": body,
                                  "response": response})
        save()
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.golden.write_text(json.dumps(report["records"], ensure_ascii=False, indent=2) + "\n")
    report["golden"] = file_info(args.golden, True)
    return True


def vision_fixture():
    """PNG with a red left square and blue right circle; no imaging dependencies."""
    width, height = 256, 128
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            color = (255, 255, 255)
            if 16 <= x < 112 and 16 <= y < 112:
                color = (255, 0, 0)
            if (x - 192) ** 2 + (y - 64) ** 2 < 48 ** 2:
                color = (0, 0, 255)
            rows.extend(color)
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def run_quality(args, report, save):
    common = {"model": args.model, "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
              "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "seed": 42,
              "max_tokens": max(128, args.tokens), "stream": False,
              "think": False, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    scenarios = [("arithmetic", {"messages": [{"role": "user", "content":
                 "Calculate 12 - 5 + 9. Reply with only the decimal integer."}]}),
                 ("structured_json", {"messages": [{"role": "user", "content":
                  'A shop has 12 apples, sells 5, and receives 9. Return {"answer":<remaining>,"unit":"apples"}.'}],
                  "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
                    "schema": {"type": "object", "properties": {"answer": {"type": "integer"},
                               "unit": {"type": "string"}}, "required": ["answer", "unit"],
                               "additionalProperties": False}}}}),
                 ("tool_call", {"messages": [{"role": "user", "content":
                  "Use get_weather to get the weather in Paris, France. Call the tool only."}],
                  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city.",
                  "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                 "required": ["location"], "additionalProperties": False}}}],
                  "tool_choice": "required"})]
    if args.vision:
        png = vision_fixture()
        fixture = args.output.with_name(args.output.stem + "-vision.png")
        fixture.write_bytes(png)
        report["vision_fixture"] = file_info(fixture, True)
        scenarios.append(("vision_colors", {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Name the color of the square on the left and the circle on the right. Reply only: <left color>, <right color>."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]}))
    report.update(engine=args.engine, url=args.url, model=args.model, scenarios=[],
                  limitations=["These are narrow deterministic smoke checks, not comprehensive reasoning/tool/vision evaluation.",
                               "Inspect server logs to confirm projector loading and actual vision execution."])
    for name, body in scenarios:
        request = dict(common, **body)
        request.update(args.request_extra)
        started = time.perf_counter()
        result = {"name": name, "request": request}
        try:
            response = request_json(args.url.rstrip("/") + "/v1/chat/completions", request, args.timeout)
            result["response"] = response
            message = response["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if name == "arithmetic":
                passed = content.rstrip(".") == "16"
            elif name == "structured_json":
                passed = json.loads(content) == {"answer": 16, "unit": "apples"}
            elif name == "tool_call":
                calls = message.get("tool_calls", [])
                passed = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and (
                    "paris" in json.loads(calls[0]["function"]["arguments"])["location"].lower())
            else:
                answer = content.lower()
                passed = "red" in answer and "blue" in answer and answer.index("red") < answer.index("blue")
            result["status"] = "passed" if passed else "failed"
        except Exception as error:
            result.update(status="failed", error=repr(error))
        result["wall_seconds"] = time.perf_counter() - started
        report["scenarios"].append(result)
        save()
    return all(result["status"] == "passed" for result in report["scenarios"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "reference", "http", "quality"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--engine", default="tensorsharp")
    parser.add_argument("--model", default="Bonsai2")
    parser.add_argument("--repo", type=Path, action="append", default=[])
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=lambda text: [int(value) for value in text.split(",")], default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--request-extra", type=json.loads, default={})
    parser.add_argument("--vision", action="store_true", help="quality: include generated red-square/blue-circle fixture")
    parser.add_argument("--golden", type=Path, default=ROOT / "docs/validation/bonsai2/golden.json")
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.timeout, *args.concurrency) <= 0:
        parser.error("tokens, repeats, timeout, and concurrency must be positive")
    if not isinstance(args.request_extra, dict):
        parser.error("request-extra must be a JSON object")
    if args.output.exists():
        parser.error("output already exists; choose a fresh evidence filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "performance_qualified": False, "mode": args.mode,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha256(__file__), "argv": sys.argv, "inventory": inventory(args)}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    try:
        save()
        passed = True
        if args.mode == "reference":
            passed = run_reference(args, report, save)
        elif args.mode == "http":
            passed = run_http(args, report, save)
        elif args.mode == "quality":
            passed = run_quality(args, report, save)
        report["status"] = "completed" if passed else "output_mismatch"
    except Exception as error:
        report.update(status="failed", error=repr(error))
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "summaries": report.get("summaries", [])}, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
