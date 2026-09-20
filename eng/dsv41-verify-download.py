#!/usr/bin/env python3
"""Verify every published V4.1 Q2_K shard against pinned Hugging Face LFS SHA256.

This reads approximately 247 GiB. Run outside qualified inference benchmarks;
it verifies existing files and does not download or modify model weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


READ_BYTES = 1024 * 1024
MAX_READ_RETRIES = 5
TRANSIENT_READ_ERRORS = {errno.EINTR, errno.EAGAIN, errno.ENOMEM}


def verify_shard(directory, entry):
    """Hash each returned byte once; retry transient reads at the verified offset."""
    path = directory / entry["path"]
    expected = entry.get("lfs", {}).get("oid")
    result = dict(file=entry["path"], expected_bytes=entry["size"], expected_sha256=expected,
                  bytes_read=0, retry_count=0, read_errors=[], short_read_count=0)
    begin = last_progress = time.monotonic()
    try:
        if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing authoritative LFS SHA256")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("Missing file or incorrect size")
        digest = hashlib.sha256()
        consecutive_retries = 0
        # Unbuffered IO prevents a failed buffered read from hiding consumed
        # bytes. Seek explicitly because an error may still advance the fd.
        with path.open("rb", buffering=0) as source:
            while result["bytes_read"] < entry["size"]:
                offset = result["bytes_read"]
                requested = min(READ_BYTES, entry["size"] - offset)
                try:
                    source.seek(offset)
                    block = source.read(requested)
                    if block is None:
                        # FileIO uses None for would-block; b"" alone is EOF.
                        raise BlockingIOError(errno.EAGAIN, "Unbuffered read returned None (would block)")
                except OSError as error:
                    retry = error.errno in TRANSIENT_READ_ERRORS and consecutive_retries < MAX_READ_RETRIES
                    delay = 0.05 * (consecutive_retries + 1) if retry else 0
                    event = dict(offset=offset, errno=error.errno, error=repr(error), retried=retry,
                                 consecutive_retry=consecutive_retries + 1 if retry else None, backoff_seconds=delay)
                    result["read_errors"].append(event)
                    print(json.dumps(dict(file=entry["path"], read_error=event)), flush=True)
                    if not retry:
                        raise
                    result["retry_count"] += 1
                    consecutive_retries += 1
                    time.sleep(delay)
                    continue
                if block == b"":
                    raise EOFError(f"Unexpected EOF at byte {offset} of {entry['size']}")
                if len(block) < requested:
                    result["short_read_count"] += 1
                digest.update(block)
                result["bytes_read"] += len(block)
                consecutive_retries = 0
                now = time.monotonic()
                if now - last_progress >= 10:
                    print(json.dumps(dict(file=entry["path"], bytes_read=result["bytes_read"],
                                          expected_bytes=entry["size"], retry_count=result["retry_count"],
                                          elapsed_seconds=now - begin)), flush=True)
                    last_progress = now
        # Reject a concurrent truncation/extension instead of certifying a
        # prefix of a file whose size changed while it was being verified.
        if path.stat().st_size != entry["size"]:
            raise ValueError("File size changed during verification")
        actual = digest.hexdigest()
        result.update(actual_sha256=actual, passed=actual == expected)
    except (OSError, EOFError, ValueError) as error:
        result.update(passed=False, error=repr(error))
    result["elapsed_seconds"] = time.monotonic() - begin
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository", default="vcruz305/DeepSeek-V4.1-Flash-GGUF")
    parser.add_argument("--revision", default="58d8ac86298fdf85a2440defee08b1abcad32e45")
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
    results = []
    report = dict(repository=args.repository, revision=args.revision, metadata_url=url, workers=args.workers,
                  read_bytes=READ_BYTES, maximum_consecutive_read_retries=MAX_READ_RETRIES,
                  retry_errno_names=["EINTR", "EAGAIN", "ENOMEM"])

    def save(status):
        report.update(status=status, elapsed_seconds=time.monotonic() - started,
                      passed=status == "complete" and all(result["passed"] for result in results),
                      shards=sorted(results, key=lambda row: row["file"]))
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    save("running")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(verify_shard, args.directory, entry) for entry in shards]):
            result = future.result()
            results.append(result)
            save("running")
            print(json.dumps(result), flush=True)
    save("complete")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
