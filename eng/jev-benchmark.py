#!/usr/bin/env python3
"""Black-box Jev HTTP contract, quality-screening, latency and isolation checks.

Examples (servers must already be running):
  python eng/jev-benchmark.py --endpoint tensorsharp=http://127.0.0.1:8080
  python eng/jev-benchmark.py --endpoint ts=http://127.0.0.1:8080 \
      --endpoint localjev=http://127.0.0.1:8081 --concurrency 1,2,4 \
      --background-words 0,2048 --description "Same host/checkpoint; serialized runs"

Only Python's standard library is needed. Results go under ignored artifacts/ or
docs/validation/. The original fixtures are a small integration screen, not a
benchmark of general reasoning or proof of probability calibration. Probabilities
from LocalJev's generated JSON and seeded-logit engines have different semantics.
The benchmark does not start, stop, or install inference servers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "TensorSharp.TestMatrix/Inputs/jev/decisions.json"
LIMITATIONS = [
    "Original simple fixtures are an integration and quality screen, not representative task accuracy.",
    "Repeated requests are not independent quality samples; calibration metrics are descriptive only.",
    "Model load is excluded. Reported latency includes HTTP, inference and any server queueing.",
    "Endpoint hardware, weights, quantization and probability semantics must match before claiming performance parity.",
    "Endpoints run sequentially in supplied order; thermal and cache drift are not controlled.",
    "The client requests seed/samples extensions; reference engines may ignore them.",
    "Confidence definitions differ across engines and are not used as cross-engine calibration targets.",
    "Longer inputs contain artificial irrelevant background, not a natural long-document benchmark.",
    "A fixture PASS is not a performance verdict; performance budgets are evaluated only when explicitly supplied.",
]


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def hash_value(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def named_values(values, option):
    result = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or not name or not item or name in result:
            raise ValueError(f"{option} expects unique NAME=VALUE entries")
        result[name] = item
    return result


def csv_ints(value, minimum):
    try:
        items = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not items or any(item < minimum for item in items) or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError(f"Expected distinct integers >= {minimum}")
    return items


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def endpoint_url(value):
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Endpoints must be HTTP(S) URLs")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Keep credentials in --api-key-env, not in endpoint URLs")
    base = value.rstrip("/")
    return base if base.endswith("/v1/systemone") else base + ("/systemone" if base.endswith("/v1") else "/v1/systemone")


def post(url, body, timeout, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=dumps(body).encode("utf-8"), headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
        result = {"status": status, "body": json.loads(raw), "error": None}
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            content = json.loads(raw)
        except (ValueError, UnicodeError):
            content = {"unparsed_error": raw[:2000].decode("utf-8", errors="replace")}
        result = {"status": error.code, "body": content, "error": f"HTTP {error.code}"}
    except (OSError, ValueError, TimeoutError) as error:
        result = {"status": None, "body": None, "error": f"{type(error).__name__}: {error}"}
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def probability(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{path}: expected a finite number from 0 to 1")
    return float(value)


def validate_answer(question, answer, expected, qid):
    if not isinstance(answer, dict) or answer.get("type") != question["type"]:
        raise ValueError(f"{qid}: missing answer or incorrect type")
    kind = question["type"]
    score_error = None
    if kind == "noul":
        yes = probability(answer.get("noul"), qid + ".noul")
        probs, gold = [1 - yes, yes], int(expected)
        prediction = int(yes >= 0.5)
    else:
        labels = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
        distribution = answer.get("probabilities")
        if not isinstance(distribution, dict) or set(distribution) != set(labels):
            raise ValueError(f"{qid}: probability keys do not match criteria")
        probs = [probability(distribution[label], f"{qid}.{label}") for label in labels]
        if abs(sum(probs) - 1) > 1e-5:
            raise ValueError(f"{qid}: probabilities do not sum to one")
        probability(answer.get("confidence"), qid + ".confidence")
        prediction = max(range(len(probs)), key=probs.__getitem__)
        gold = labels.index(expected) if kind == "choice" else int(expected)
        if kind == "choice":
            selected = answer.get("choice")
            if selected not in labels or abs(distribution[selected] - max(probs)) > 1e-6:
                raise ValueError(f"{qid}: choice is not a highest-probability criterion")
        else:
            legend = {str(i): value for i, value in enumerate(question["criteria"])}
            if answer.get("legend") != legend:
                raise ValueError(f"{qid}: legend does not match ordered score criteria")
            score = answer.get("score")
            weighted = sum(i * p for i, p in enumerate(probs))
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or abs(score - weighted) > 1e-5:
                raise ValueError(f"{qid}: score must be the zero-indexed probability-weighted mean")
            score_error = abs(score - gold)
    return {
        "type": kind, "question": qid, "gold": gold, "prediction": prediction,
        "correct": prediction == gold, "probabilities": probs,
        "brier": sum((p - int(i == gold)) ** 2 for i, p in enumerate(probs)),
        "nll": -math.log(max(probs[gold], 1e-12)), "score_absolute_error": score_error,
    }


def validate_response(payload, result, expected):
    if result["status"] != 200:
        raise ValueError(result["error"] or f"HTTP {result['status']}")
    body = result["body"]
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise ValueError("Response must be an object containing the served model name")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(payload["questions"]):
        raise ValueError("Response question keys differ from this request (possible cross-request contamination)")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Missing usage")
    for name in ("input_tokens", "output_tokens"):
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"usage.{name} must be a nonnegative integer")
    return [validate_answer(q, answers[qid], expected[qid], qid) for qid, q in payload["questions"].items()]


def state_for(value, words, nonce):
    if not words and nonce is None:
        return value
    # Place the varying identifier before the target: a suffix-only nonce would
    # leave the entire target eligible for a prefix-cache hit.
    result = {"irrelevant_request_identifier": nonce} if nonce is not None else {}
    result["target"] = value
    if words:
        sentence = "The library catalog records shelf numbers book covers and publication dates."
        tokens = (sentence.split() * (words // len(sentence.split()) + 1))[:words]
        result["irrelevant_background"] = " ".join(tokens)
    return result


def make_payload(case, schema, model, samples, background, nonce=None):
    questions = copy.deepcopy(schema)
    if background or nonce is not None:
        for question in questions.values():
            question["instructions"] = "Evaluate only target; ignore irrelevant_background and irrelevant_request_identifier. " + question["instructions"]
    return {
        "model": model, "state": state_for(case["state"], background, nonce),
        "questions": questions, "seed": 42, "samples": samples,
    }


def run_case(url, key, timeout, case, payload, phase, concurrency=1, background=0, repetition=0):
    result = post(url, payload, timeout, key)
    row = {
        "phase": phase, "case": case["id"], "concurrency": concurrency,
        "background_words": background, "repetition": repetition,
        "request_sha256": hash_value(payload), "request": payload, **result,
        "valid": False, "decisions": [], "validation_error": None,
    }
    try:
        row["decisions"] = validate_response(payload, result, case["expected"])
        row["valid"] = True
    except (ValueError, TypeError, KeyError) as error:
        row["validation_error"] = str(error)
    return row


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def summarize(rows, elapsed_seconds):
    valid = [row for row in rows if row["valid"]]
    decisions = [decision for row in valid for decision in row["decisions"]]
    total_decisions = sum(len(row["request"]["questions"]) for row in rows)
    latency = [row["elapsed_ms"] for row in valid]
    score_errors = [d["score_absolute_error"] for d in decisions if d["score_absolute_error"] is not None]
    return {
        "requests": len(rows), "successful_requests": len(valid), "failures": len(rows) - len(valid),
        "unique_cases": len({row["case"] for row in rows}), "total_decisions": total_decisions,
        "correct_decisions": sum(d["correct"] for d in decisions),
        "accuracy_failures_count_as_wrong": sum(d["correct"] for d in decisions) / total_decisions if total_decisions else None,
        "mean_brier_valid_only": statistics.mean(d["brier"] for d in decisions) if decisions else None,
        "mean_nll_valid_only": statistics.mean(d["nll"] for d in decisions) if decisions else None,
        "score_mae_valid_only": statistics.mean(score_errors) if score_errors else None,
        "successful_latency_ms": {"p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95), "mean": statistics.mean(latency) if latency else None},
        "phase_elapsed_seconds": elapsed_seconds,
        "successful_requests_per_second": len(valid) / elapsed_seconds if elapsed_seconds else None,
        "successful_decisions_per_second": len(decisions) / elapsed_seconds if elapsed_seconds else None,
    }


def performance_checks(summary, max_p95_ms, minimum_rps):
    checks = []
    if max_p95_ms is not None:
        actual = summary["successful_latency_ms"]["p95"]
        checks.append({"metric": "successful_p95_ms", "maximum": max_p95_ms, "actual": actual, "passed": actual is not None and actual <= max_p95_ms and summary["failures"] == 0})
    if minimum_rps is not None:
        actual = summary["successful_requests_per_second"]
        checks.append({"metric": "successful_requests_per_second", "minimum": minimum_rps, "actual": actual, "passed": actual is not None and actual >= minimum_rps and summary["failures"] == 0})
    return {"status": "not_evaluated" if not checks else "passed" if all(check["passed"] for check in checks) else "failed", "checks": checks}


def command_output(command, cwd=ROOT):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def isolation_cases():
    cases = []
    for index, color in enumerate(("red", "blue", "green", "red")):
        key = f"isolated_request_{index}"
        cases.append({
            "id": key, "state": {"color": color}, "expected": {key: color},
            "questions": {key: {"type": "choice", "instructions": "Read the color field. Which color is explicitly specified?", "criteria": {"red": "red", "blue": "blue", "green": "green"}}},
        })
    return cases


def run_isolation(url, key, args, model, record):
    cases = isolation_cases()
    payloads = [make_payload(case, case["questions"], model, args.samples, 0) for case in cases]
    serial = [run_case(url, key, args.timeout, case, payload, "isolation_serial") for case, payload in zip(cases, payloads)]
    workers = max(2, max(args.concurrency))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        concurrent = list(executor.map(lambda pair: run_case(url, key, args.timeout, pair[0], pair[1], "isolation_concurrent", workers), zip(cases, payloads)))
    for row in serial + concurrent:
        record(row)
    comparisons = []
    for baseline, parallel in zip(serial, concurrent):
        comparable = baseline["valid"] and parallel["valid"]
        difference = max(abs(a - b) for a, b in zip(baseline["decisions"][0]["probabilities"], parallel["decisions"][0]["probabilities"])) if comparable else None
        comparisons.append({"case": baseline["case"], "comparable": comparable, "max_probability_difference": difference, "within_tolerance": difference <= args.isolation_tolerance if comparable else False})
    return {
        "requests": len(serial) + len(concurrent), "concurrency": workers,
        "contract_passed": all(row["valid"] for row in serial + concurrent),
        "known_labels_passed": all(row["valid"] and all(d["correct"] for d in row["decisions"]) for row in serial + concurrent),
        "repeatability_within_tolerance": all(c["within_tolerance"] for c in comparisons),
        "tolerance": args.isolation_tolerance, "comparisons": comparisons,
        "scope": "Unique question keys detect result leakage; contradictory states test semantic isolation. Probability drift may reflect nondeterminism rather than contamination.",
    }


def run_invalid(url, key, args, model, record):
    requests = [
        ("missing_state", {"model": model, "questions": {"x": {"type": "noul", "instructions": "yes?"}}}),
        ("empty_questions", {"model": model, "state": "test", "questions": {}}),
        ("unknown_question_type", {"model": model, "state": "test", "questions": {"x": {"type": "unknown"}}}),
        ("one_choice", {"model": model, "state": "test", "questions": {"x": {"type": "choice", "criteria": {"only": "only"}}}}),
    ]
    rows = []
    for name, body in requests:
        result = post(url, body, args.timeout, key)
        passed = result["status"] in (400, 422)
        row = {"phase": "invalid_request", "case": name, "request": body, **result, "rejected_correctly": passed}
        record(row)
        rows.append({"case": name, "status": result["status"], "passed": passed})
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", action="append", required=True, help="NAME=http://HOST:PORT (repeat to compare engines)")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--model-override", action="append", default=[], help="NAME=MODEL")
    parser.add_argument("--api-key-env", action="append", default=[], help="NAME=ENVIRONMENT_VARIABLE; values are never logged")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--concurrency", type=lambda v: csv_ints(v, 1), default=[1, 2])
    parser.add_argument("--background-words", type=lambda v: csv_ints(v, 0), default=[0])
    parser.add_argument("--repeats", type=positive, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=positive, help="Use only the first N cases; recorded as reduced coverage")
    parser.add_argument("--samples", default="1", help="Positive noise draw count or auto; reference servers may ignore it")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--state-bust", action="store_true", help="Vary an irrelevant state identifier on each measured request, preserving schema prefix")
    parser.add_argument("--skip-isolation", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--isolation-tolerance", type=float, default=1e-5)
    parser.add_argument("--strict-repeatability", action="store_true", help="Fail if serial/concurrent probabilities differ beyond tolerance")
    parser.add_argument("--min-accuracy", type=float, default=1.0, help="Exit unsuccessfully below this fixture accuracy; no general quality claim")
    parser.add_argument("--max-p95-ms", type=float, help="Optional p95 latency budget for every measured group; failure also requires zero request errors")
    parser.add_argument("--min-requests-per-second", type=float, help="Optional throughput budget for every measured group; request errors also fail")
    parser.add_argument("--description", default="", help="Hardware, quantization and server launch details for the comparison")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/jev-benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    if args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.min_accuracy <= 1 or not math.isfinite(args.isolation_tolerance) or args.isolation_tolerance < 0:
        parser.error("Invalid warmup, timeout, minimum accuracy, or isolation tolerance")
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in (args.max_p95_ms, args.min_requests_per_second)):
        parser.error("Performance budgets must be positive finite numbers")
    if args.samples != "auto":
        try:
            args.samples = positive(args.samples)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("--samples must be auto or a positive integer")
    return args


def main():
    args = parse_args()
    endpoints = named_values(args.endpoint, "--endpoint")
    overrides = named_values(args.model_override, "--model-override")
    key_envs = named_values(args.api_key_env, "--api-key-env")
    if (set(overrides) | set(key_envs)) - set(endpoints):
        raise ValueError("Model/key overrides must name a configured endpoint")
    urls = {name: endpoint_url(url) for name, url in endpoints.items()}
    keys = {name: os.environ[env] for name, env in key_envs.items()}
    if any(not key for key in keys.values()):
        raise ValueError("An API-key environment variable is empty")
    fixture_bytes = args.fixtures.read_bytes()
    suite = json.loads(fixture_bytes)
    all_cases = suite["cases"]
    cases = all_cases[:args.limit]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Fixtures must contain nonempty, unique cases")
    for case in cases:
        if set(case["expected"]) != set(suite["schemas"][case["schema"]]):
            raise ValueError(f"Fixture {case['id']} expected keys do not match its schema")
    output = args.out.resolve()
    allowed = [(ROOT / "artifacts").resolve(), (ROOT / "docs/validation").resolve()]
    if not any(root == output or root in output.parents for root in allowed):
        raise ValueError("Generated benchmark outputs must be under repository artifacts/ or docs/validation/")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "results.jsonl").exists():
        raise ValueError("Choose a new output directory; existing evidence will not be overwritten")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "description": args.description,
        "git_revision": command_output(["git", "rev-parse", "HEAD"]), "git_status": command_output(["git", "status", "--short"]),
        "ggml_revision": command_output(["git", "rev-parse", "HEAD"], ROOT / "ExternalProjects/ggml"),
        "ggml_status": command_output(["git", "status", "--short"], ROOT / "ExternalProjects/ggml"),
        "platform": platform.platform(), "python": sys.version,
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "endpoints": urls, "model": args.model, "model_overrides": overrides,
        "fixtures": str(args.fixtures.resolve()), "fixtures_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_ids": [case["id"] for case in cases], "available_cases": len(all_cases),
        "concurrency": args.concurrency, "background_words": args.background_words, "repeats": args.repeats,
        "samples_requested": args.samples, "seed_requested": 42, "warmup_requests": args.warmup,
        "warmup_state_nonce": True,
        "state_bust": args.state_bust, "timeout_seconds": args.timeout, "minimum_fixture_accuracy": args.min_accuracy,
        "maximum_p95_ms": args.max_p95_ms, "minimum_requests_per_second": args.min_requests_per_second,
        "limitations": LIMITATIONS,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summaries = {}
    passed = True
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for name, url in urls.items():
            model, key = overrides.get(name, args.model), keys.get(name)

            def record(row):
                stream.write(dumps({"endpoint": name, **row}) + "\n")
                stream.flush()

            print(f"[{name}] {url}", flush=True)
            summary = {"groups": [], "invalid_requests": None, "isolation": None, "warmup_requests": args.warmup, "warmup_failures": 0}
            for index in range(args.warmup):
                case = cases[index % len(cases)]
                payload = make_payload(case, suite["schemas"][case["schema"]], model, args.samples, 0)
                # Keep the measured question/schema prefix, but avoid warming
                # any measured exact prompt or the target's prefix-cache entry.
                payload["state"] = state_for(payload["state"], 0, f"excluded-warmup-{index}")
                row = run_case(url, key, args.timeout, case, payload, "warmup")
                record(row)
                if not row["valid"]:
                    summary["warmup_failures"] += 1
                    print(f"[{name}] warmup failed: {row['validation_error']}", flush=True)
            if not args.skip_invalid:
                summary["invalid_requests"] = run_invalid(url, key, args, model, record)
                passed &= summary["invalid_requests"]["passed"]
            for background in args.background_words:
                for concurrency in args.concurrency:
                    jobs = [(case, repetition) for repetition in range(args.repeats) for case in cases]

                    def perform(item):
                        case, repetition = item
                        nonce = f"case-{case['id']}-repeat-{repetition}-workers-{concurrency}" if args.state_bust else None
                        payload = make_payload(case, suite["schemas"][case["schema"]], model, args.samples, background, nonce)
                        return run_case(url, key, args.timeout, case, payload, "measured", concurrency, background, repetition)

                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        rows = []
                        for row in executor.map(perform, jobs):
                            record(row)
                            rows.append(row)
                            print(f"[{name}] c={concurrency} words={background} {row['case']}: {row['elapsed_ms']:.1f} ms {'valid' if row['valid'] else row['validation_error']}", flush=True)
                    stats = summarize(rows, time.perf_counter() - started)
                    stats["performance"] = performance_checks(stats, args.max_p95_ms, args.min_requests_per_second)
                    summary["groups"].append({"concurrency": concurrency, "background_words": background, **stats})
                    passed &= stats["failures"] == 0 and stats["accuracy_failures_count_as_wrong"] >= args.min_accuracy
                    passed &= stats["performance"]["status"] != "failed"
            if not args.skip_isolation:
                print(f"[{name}] running serial/concurrent isolation checks", flush=True)
                summary["isolation"] = run_isolation(url, key, args, model, record)
                passed &= summary["isolation"]["contract_passed"] and summary["isolation"]["known_labels_passed"]
                if args.strict_repeatability:
                    passed &= summary["isolation"]["repeatability_within_tolerance"]
            summaries[name] = summary
    report = {"passed_requested_checks": passed, "endpoints": summaries, "limitations": LIMITATIONS}
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    performance = "performance budgets checked" if args.max_p95_ms is not None or args.min_requests_per_second is not None else "performance parity not evaluated"
    print(f"{'PASS' if passed else 'FAIL'} requested checks; {performance}; evidence: {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"Benchmark setup failed: {error}", file=sys.stderr)
        sys.exit(2)
