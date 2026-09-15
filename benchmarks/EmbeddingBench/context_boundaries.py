#!/usr/bin/env python3
"""Verify full-context inputs, overflow and EOS-preserving Ollama truncation.

The two pinned test models tokenize the word 'a' as one token. Matching the
truncated text embedding to its explicit token input checks both truncation and
the terminal separator, without running an external tokenizer or backend.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import urllib.error
import urllib.request


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def check_vector(vector, dimensions):
    assert len(vector) == dimensions
    assert all(math.isfinite(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in vector))
    assert abs(norm - 1) < 1e-5, norm
    return {"dimensions": dimensions, "norm": norm,
            "float32_sha256": hashlib.sha256(struct.pack("<" + "f" * dimensions, *vector)).hexdigest()}


def verify_boundaries(base, name, dimensions, context):
    if name.startswith("all-MiniLM-L6-v2"):
        bos, content, eos, expected_context = 101, 1037, 102, 512
    elif name.startswith("snowflake-arctic-embed-l-v2.0"):
        bos, content, eos, expected_context = 0, 10, 2, 8192
    else:
        raise ValueError(f"No pinned tokenizer fixture for {name}")
    assert context == expected_context, (name, context)
    print(f"Checking {name}: {context}-token managed context and overflow handling.", flush=True)
    full_tokens = [bos] + [content] * (context - 2) + [eos]
    status, full = request(base, "/v1/embeddings", {"model": name, "input": full_tokens})
    assert status == 200, full
    assert full["usage"]["prompt_tokens"] == full["usage"]["total_tokens"] == context
    full_vector = full["data"][0]["embedding"]
    result = {"model": name, "context_tokens": context, "token_fixture": {"bos": bos, "word_a": content, "eos": eos},
              "full_context": dict(status=status, prompt_tokens=context, **check_vector(full_vector, dimensions))}
    overflow_text = "a " * (context - 1)  # context+1 tokens including BOS/EOS
    rejected = []
    for path, body in (
        ("/v1/embeddings", {"model": name, "input": full_tokens[:-1] + [content, eos]}),
        ("/v1/embeddings", {"model": name, "input": overflow_text}),
        ("/api/embed", {"model": name, "input": overflow_text, "truncate": False}),
    ):
        status, error = request(base, path, body)
        assert status == 400, (path, status, error)
        message = error["error"]["message"] if isinstance(error["error"], dict) else error["error"]
        assert str(context) in message, message
        rejected.append({"path": path, "input_type": "text" if isinstance(body["input"], str) else "tokens",
                         "status": status, "error": error})
    result["overflow_rejections"] = rejected
    status, truncated = request(base, "/api/embed", {"model": name, "input": overflow_text, "truncate": True})
    assert status == 200, truncated
    assert truncated["prompt_eval_count"] == context, truncated["prompt_eval_count"]
    truncated_vector = truncated["embeddings"][0]
    difference = max(abs(a - b) for a, b in zip(full_vector, truncated_vector))
    assert difference < 1e-6, difference
    result["ollama_truncation"] = dict(status=status, prompt_tokens=context,
        matches_explicit_eos_preserving_input=True, max_absolute_difference=difference,
        **check_vector(truncated_vector, dimensions))
    # A small request after both oversized errors and full-context inference
    # exercises scratch-buffer reuse and the request gate's recovery.
    status, recovery = request(base, "/v1/embeddings", {"model": name, "input": "a"})
    assert status == 200 and recovery["usage"]["prompt_tokens"] == 3, recovery
    result["subsequent_short_request"] = dict(status=status, prompt_tokens=3,
        **check_vector(recovery["data"][0]["embedding"], dimensions))
    result["passed"] = True
    print(f"Passed {name}: context={context}, overflow=400, truncation preserves EOS, subsequent request succeeds.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18383")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status, models = request(args.url, "/v1/models")
    assert status == 200 and len(models["data"]) == 1
    model = models["data"][0]
    result = verify_boundaries(args.url, model["id"], model["embedding_dimensions"], model["context_length"])
    result["verification_command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
