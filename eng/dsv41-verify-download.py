#!/usr/bin/env python3
"""Verify every published V4.1 Q2_K shard against pinned Hugging Face LFS SHA256.

This reads approximately 247 GiB. Run outside qualified inference benchmarks;
it verifies existing files and does not download or modify model weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository", default="vcruz305/DeepSeek-V4.1-Flash-GGUF")
    parser.add_argument("--revision", default="8e0c4de3cb6519bfc11ed69dc87184b457a57bb5")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or not 1 <= args.workers <= 8:
        raise ValueError("An immutable revision and 1–8 checksum workers are required")
    url = f"https://huggingface.co/api/models/{args.repository}/tree/{args.revision}?recursive=false&expand=false"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    shards = [entry for entry in entries if entry.get("type") == "file" and
              re.fullmatch(r"DeepSeek-V4\.1-Flash-Q2_K-\d{5}-of-00007\.gguf", entry["path"])]
    if {entry["path"] for entry in shards} != {
            f"DeepSeek-V4.1-Flash-Q2_K-{index:05}-of-00007.gguf" for index in range(1, 8)}:
        raise ValueError("Pinned model tree does not contain exactly the expected seven Q2_K shards")
    started = time.monotonic()
    print(f"Verifying {sum(entry['size'] for entry in shards) / 1024**3:.3f} GiB with {args.workers} workers", flush=True)
    def verify(entry):
        path = args.directory / entry["path"]
        expected = entry.get("lfs", {}).get("oid")
        if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing authoritative LFS SHA256")
        result = dict(file=entry["path"], expected_bytes=entry["size"], expected_sha256=expected)
        begin = time.monotonic()
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return dict(result, passed=False, error="Missing file or incorrect size")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        return dict(result, actual_sha256=actual, passed=actual == expected, elapsed_seconds=time.monotonic() - begin)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(verify, entry) for entry in shards]):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    report = dict(repository=args.repository, revision=args.revision, metadata_url=url,
                  workers=args.workers, elapsed_seconds=time.monotonic() - started,
                  passed=all(result["passed"] for result in results), shards=sorted(results, key=lambda row: row["file"]))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
