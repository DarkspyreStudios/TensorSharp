#!/usr/bin/env python3
"""Recheck saved decode responses without sending requests or changing evidence.

This validates existing captures with the current benchmark gates. It is not a
new inference run. Only complete, passing waves qualify for comparable timing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


BENCHMARK_DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks/engine_comparison"
sys.path.insert(0, str(BENCHMARK_DIRECTORY))
import validate_inference as validator


def revalidate(source):
    payload = source.read_bytes()
    captured = json.loads(payload)
    cases = captured.get("cases")
    if captured.get("run_complete") is not True or not isinstance(cases, list) or not cases:
        raise ValueError("a completed, nonempty inference report is required")
    expected = captured.get("execution_plan", {}).get("expected_cases")
    if type(expected) is not int or expected != len(cases):
        raise ValueError("captured case count does not match its execution plan")
    tags = [case.get("tag") for case in cases]
    if any(not isinstance(tag, str) or not tag for tag in tags) or len(set(tags)) != len(tags):
        raise ValueError("capture tags must be present and unique")
    if any(case.get("scenario") not in ("decode", "decode_8k") for case in cases):
        raise ValueError("only decode and decode_8k captures are supported")

    revised = copy.deepcopy(cases)
    audit = []
    for case in revised:
        original_status = case["status"]
        detail = None
        metrics, request = {}, {}
        try:
            if len(case.get("turns", [])) != 1:
                raise ValueError("decode capture must contain exactly one turn")
            turn = case["turns"][0]
            metrics, request = turn["metrics"], turn["request"]
            if metrics.get("usage_present") is not True:
                raise ValueError("capture omitted token usage")
            budget = request.get("max_tokens")
            if type(budget) is not int or budget < 1:
                raise ValueError("capture omitted a positive integer generation budget")
            validator.validate_decode(metrics, budget, validator.assistant_content(metrics))
            if original_status != "ok":
                raise ValueError("original failed case remains failed: " + case.get("detail", original_status))
            case["status"] = "ok"
        except (KeyError, TypeError, ValueError) as error:
            case["status"] = "fail"
            detail = f"{type(error).__name__}: {error}"
            case["detail"] = detail
        audit.append({"tag": case["tag"], "scenario": case["scenario"],
                      "concurrency": case.get("concurrency", 1), "repeat": case.get("repeat", 0),
                      "original_status": original_status, "status": case["status"], "detail": detail,
                      "request_sha256": validator.digest(request), "metrics_sha256": validator.digest(metrics),
                      "completion_tokens": metrics.get("completion_tokens"),
                      "finish_reason": metrics.get("finish_reason")})

    waves, eligible = [], []
    seen = set()
    for wave in captured.get("waves", []):
        key = (wave["scenario"], wave["concurrency"], wave["repeat"])
        if key in seen:
            raise ValueError("capture contains duplicate waves")
        seen.add(key)
        members = [case for case in revised
                   if (case["scenario"], case.get("concurrency", 1), case.get("repeat", 0)) == key]
        if len(members) != wave["concurrency"]:
            raise ValueError("wave does not contain its declared number of clients")
        passing = all(case["status"] == "ok" for case in members)
        waves.append({**wave, "original_all_passed": wave["all_passed"],
                      "all_passed": passing, "eligible_for_comparable_throughput": passing})
        if passing:
            eligible.extend(members)
    case_keys = {(case["scenario"], case.get("concurrency", 1), case.get("repeat", 0)) for case in revised}
    if seen != case_keys:
        raise ValueError("wave coverage differs from the captured cases")

    summary = validator.summarize(eligible)
    for group, metrics in summary.items():
        selected = [wave for wave in waves if wave["eligible_for_comparable_throughput"]
                    and f"{wave['scenario']}@c{wave['concurrency']}" == group]
        metrics["complete_passing_waves"] = len(selected)
        metrics["aggregate_end_to_end_tps_median"] = statistics.median(
            wave["end_to_end_tokens_per_second"] for wave in selected)
    return {"mode": "captured_response_revalidation", "new_inference_run": False,
            "created_at_unix": time.time(), "source_report": str(source.resolve()),
            "source_report_sha256": hashlib.sha256(payload).hexdigest(),
            "inference_harness_sha256": captured.get("harness_sha256"),
            "validation_harness_sha256": hashlib.sha256(Path(validator.__file__).read_bytes()).hexdigest(),
            "revalidation_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "Budget and technical-coverage checks do not establish comprehensive answer quality. "
                      "Original captures are unchanged; only complete passing waves enter comparable timing.",
            "complete": True, "recorded_cases": len(audit),
            "passed": sum(case["status"] == "ok" for case in audit),
            "failed": sum(case["status"] != "ok" for case in audit),
            "cases": audit, "waves": waves, "comparable_timing_summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.output.resolve():
        parser.error("the output must not overwrite the original capture")
    report = revalidate(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Revalidated {report['passed']}/{report['recorded_cases']} captures; no inference was run")
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
