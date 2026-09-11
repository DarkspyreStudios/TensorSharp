#!/usr/bin/env python3
"""Prepare deterministic DeepSeek V4.1 Engram metadata without downloading weights.

Requires numpy and tokenizers. Inference reads the resulting binary directly;
Python and the Hugging Face libraries are not runtime dependencies.
"""
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while not d & 1:
        d >>= 1
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def token_map(tokenizer):
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    keys, result = {}, []
    fingerprint = 14695981039346656037
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        raw = tokenizer.id_to_token(token_id)
        if raw is None:
            raise ValueError(f"Tokenizer is missing token {token_id}")
        encoded = raw.encode("utf-8")
        for byte in struct.pack("<Q", len(encoded)) + encoded:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & ((1 << 64) - 1)
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        key = raw if "\ufffd" in text else (normalizer.normalize_str(text) or text)
        if key not in keys:
            keys[key] = len(keys)
        result.append(keys[key])
    return result, len(keys), fingerprint


def prepare(config, tokenizer):
    text = config.get("text_config", config)
    mapping, compressed_size, fingerprint = token_map(tokenizer)
    if len(mapping) != text["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size differs from config")
    if compressed_size != text["engram_compressed_vocab_size"]:
        raise ValueError(f"Compressed vocabulary is {compressed_size}, expected {text['engram_compressed_vocab_size']}")
    layer_ids = text["engram_layer_ids"]
    max_ngram = text["engram_max_ngram_size"]
    n_heads, head_dim = text["engram_n_heads"], text["engram_head_dim"]
    kv_sources, index_sources = text["kv_source_layer_ids"], text["index_source_layer_ids"]
    result = bytearray(b"TSD41E01")
    result += struct.pack("<7IQi4I", len(mapping), compressed_size, text["engram_pad_token_id"],
                          len(layer_ids), max_ngram, n_heads, head_dim, fingerprint,
                          text.get("candidate_source_layer_id", -1), text.get("candidate_topk_blocks", 0),
                          text.get("candidate_block_size", 0), len(kv_sources), len(index_sources))
    result += struct.pack(f"<{len(kv_sources)}i", *kv_sources)
    result += struct.pack(f"<{len(index_sources)}i", *index_sources)
    result += np.asarray(mapping, dtype="<i4").tobytes()
    seen, layouts = set(), []
    for index, layer_id in enumerate(layer_ids):
        primes = []
        for _ in range(max_ngram - 1):
            current = text["engram_vocab_size"] - 1
            for _ in range(n_heads):
                current += 1
                while current in seen or not is_prime(current):
                    current += 1
                seen.add(current)
                primes.append(current)
        if sum(primes) != text["engram_num_embeddings"][index]:
            raise ValueError(f"Engram bucket layout does not match table rows at layer {layer_id}")
        offsets = np.cumsum([0, *primes[:-1]], dtype=np.uint64)
        bound = max(1, ((2 ** 63 - 1) // compressed_size) // 2)
        rng = np.random.default_rng(10007 * layer_id)
        multipliers = rng.integers(low=0, high=bound, size=max_ngram, dtype=np.int64) * 2 + 1
        result += struct.pack("<iQ", layer_id, sum(primes))
        result += multipliers.astype("<u8").tobytes()
        result += np.asarray(primes, dtype="<u4").tobytes()
        result += offsets.astype("<u8").tobytes()
        layouts.append({"id": layer_id, "multipliers": multipliers.tolist(), "primes": primes, "offsets": offsets.tolist()})
    return result, {"tokenizer_fingerprint": fingerprint, "compressed_vocab_size": compressed_size,
                    "layers": layouts}, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing the model GGUF shards")
    parser.add_argument("--source-dir", type=Path, help="Read config.json and tokenizer.json locally")
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--revision", default="main", help="Pin a Hugging Face commit for reproducibility")
    args = parser.parse_args()
    files = {}
    for name in ("config.json", "tokenizer.json"):
        if args.source_dir:
            files[name] = (args.source_dir / name).read_bytes()
        else:
            url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                files[name] = response.read()
    config = json.loads(files["config.json"])
    data, metadata, _ = prepare(config, Tokenizer.from_str(files["tokenizer.json"].decode("utf-8")))
    metadata.update({"source_repo": args.repo, "source_revision": args.revision,
                     "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
                     "sidecar_sha256": hashlib.sha256(data).hexdigest(), "config": config})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("deepseek41.engram.bin", data),
                          ("deepseek41.config.json", (json.dumps(metadata, indent=2) + "\n").encode())):
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"Wrote {path} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
