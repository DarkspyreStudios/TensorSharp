#!/usr/bin/env python3
"""Check V4.1 image-input rejection through a running host's real HTTP adapters.

Run only after the host containing the image-validation fix has loaded.
The client sends image URLs as request data and never fetches those URLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        "scope": "Real HTTP rejection checks only; no valid-image inference or throughput claim.",
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__).read_bytes()),
        "host_manifest": str(args.host_manifest),
        "host_manifest_sha256": sha256(manifest),
        "host_provenance": json.loads(manifest),
        "base_url": args.base_url,
        "cases": [],
        "complete": False,
        "passed": False,
    }
    base = args.base_url.rstrip("/")
    try:
        model = args.model
        if not model:
            with urllib.request.urlopen(base + "/v1/models", timeout=args.timeout) as response:
                model = json.load(response)["data"][0]["id"]
        report["model"] = model
        for endpoint in ("/v1/chat/completions", "/v1/responses"):
            for stream in (False, True):
                for kind, image_url in (
                    ("remote_https", "https://example.invalid/never-fetch-image.png"),
                    ("invalid_base64", "data:image/png;base64,?"),
                ):
                    chat = endpoint.endswith("chat/completions")
                    part = (
                        {"type": "image_url", "image_url": {"url": image_url}}
                        if chat else {"type": "input_image", "image_url": image_url}
                    )
                    body = {
                        "model": model,
                        "stream": stream,
                        "max_tokens" if chat else "max_output_tokens": 1,
                        "messages" if chat else "input": [{"role": "user", "content": [part]}],
                    }
                    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    item = {
                        "endpoint": endpoint,
                        "stream": stream,
                        "kind": kind,
                        "request": body,
                        "request_sha256": sha256(payload),
                        "passed": False,
                    }
                    start = time.monotonic()
                    try:
                        request = urllib.request.Request(base + endpoint, data=payload,
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
                            len(raw) <= 65536
                            and item["status"] == 400
                            and item["content_type"].lower().startswith("application/json")
                            and parsed.get("error", {}).get("type") == "invalid_request_error"
                            and "image_url" in parsed.get("error", {}).get("message", "")
                        )
                    except Exception as error:
                        item["exception"] = str(error)
                    item["elapsed_seconds"] = time.monotonic() - start
                    report["cases"].append(item)
        report["complete"] = len(report["cases"]) == 8
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
