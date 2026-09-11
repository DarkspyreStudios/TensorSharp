#!/usr/bin/env python3
"""Check V4.1 Responses audio rejection on a running host containing the fix.

This records four HTTP rejection cases, independently of the image-input suite.
It does not measure audio inference or model performance.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_parts() -> tuple[tuple[str, dict], ...]:
    # A valid, tiny mono PCM WAV: rejecting it must precede media decoding.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(8000)
        clip.writeframes(b"\x00\x00" * 80)
    return (
        ("valid_shape_wav", {"type": "input_audio", "input_audio": {
            "format": "wav", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}}),
        ("missing_audio_payload", {"type": "input_audio"}),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--model", help="Defaults to the first /v1/models id.")
    parser.add_argument("--host-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    manifest = args.host_manifest.read_bytes()
    report = {
        "scope": "Four real HTTP Responses audio rejection checks; no inference or throughput claim.",
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__).read_bytes()),
        "host_manifest": str(args.host_manifest),
        "host_manifest_sha256": sha256(manifest),
        "host_provenance": json.loads(manifest),
        "base_url": args.base_url,
        "cases": [], "complete": False, "passed": False,
    }
    base = args.base_url.rstrip("/")
    try:
        model = args.model
        if not model:
            with urllib.request.urlopen(base + "/v1/models", timeout=args.timeout) as response:
                model = json.load(response)["data"][0]["id"]
        report["model"] = model
        for stream in (False, True):
            for kind, part in audio_parts():
                body = {"model": model, "stream": stream, "max_output_tokens": 1,
                        "input": [{"role": "user", "content": [part]}]}
                payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                item = {"endpoint": "/v1/responses", "stream": stream, "kind": kind,
                        "request": body, "request_sha256": sha256(payload), "passed": False}
                start = time.monotonic()
                try:
                    request = urllib.request.Request(base + "/v1/responses", data=payload,
                        headers={"Content-Type": "application/json"}, method="POST")
                    try:
                        response = urllib.request.urlopen(request, timeout=args.timeout)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        raw = response.read(65537)
                        item["status"] = response.status
                        item["content_type"] = response.headers.get("Content-Type", "")
                    item["response_sha256"] = sha256(raw)
                    item["response_body"] = raw.decode("utf-8")
                    parsed = json.loads(item["response_body"])
                    item["passed"] = (
                        len(raw) <= 65536 and item["status"] == 400
                        and item["content_type"].lower().startswith("application/json")
                        and parsed.get("error", {}).get("type") == "invalid_request_error"
                        and "DeepSeek V4.1 Flash does not support audio input"
                            in parsed.get("error", {}).get("message", "")
                    )
                except Exception as error:
                    item["exception"] = str(error)
                item["elapsed_seconds"] = time.monotonic() - start
                report["cases"].append(item)
        report["complete"] = len(report["cases"]) == 4
        report["passed"] = report["complete"] and all(case["passed"] for case in report["cases"])
    except Exception as error:
        report["fatal_error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "complete": report["complete"],
                      "cases": len(report["cases"]), "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
