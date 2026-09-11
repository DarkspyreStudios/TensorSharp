#!/usr/bin/env python3
"""Prepare a lossless V4.1 vision GGUF without downloading original text weights.

Downloads the isolated BF16 vision shard and HTTP byte ranges for three image
delimiters and forty VL router biases. Requires numpy and gguf; inference does
not depend on Python. The existing deepseek41.engram.bin identifies the parent
text tokenizer. Downloads and output tensors have SHA256 provenance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request
import time

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Download:
    def __init__(self, repository, revision, cache):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use an immutable 40-character Hugging Face commit SHA")
        self.repository, self.revision, self.cache = repository, revision, Path(cache) / revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.base = f"https://huggingface.co/{repository}/resolve/{revision}/"
        api = f"https://huggingface.co/api/models/{repository}/tree/{revision}?recursive=false&expand=false"
        with urllib.request.urlopen(api, timeout=60) as response:
            self.files = {entry["path"]: entry for entry in json.load(response) if entry["type"] == "file"}

    def json_file(self, name):
        with urllib.request.urlopen(self.base + urllib.parse.quote(name), timeout=60) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Unexpectedly large model metadata")
        (self.cache / name).write_bytes(data)
        return json.loads(data)

    def whole_file(self, name):
        info = self.files[name]
        expected = info.get("lfs", {}).get("oid")
        if not expected or info["size"] > 2 * 1024**3:
            raise ValueError("Vision preparer only downloads a SHA-identified isolated shard below 2 GiB")
        path = self.cache / name
        if path.exists() and path.stat().st_size == info["size"] and sha256(path) == expected:
            return path
        temporary = path.with_suffix(".partial")
        digest, count = hashlib.sha256(), 0
        with urllib.request.urlopen(self.base + name + "?download=true", timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > info["size"]:
                    raise ValueError("Downloaded vision shard exceeds its recorded size")
                digest.update(chunk)
                output.write(chunk)
        if count != info["size"] or digest.hexdigest() != expected:
            raise ValueError("Downloaded vision shard SHA256/size mismatch")
        temporary.replace(path)
        return path

    def byte_range(self, name, first, stop):
        if not 0 <= first < stop <= self.files[name]["size"]:
            raise ValueError("Invalid requested safetensors byte range")
        directory = self.cache / "ranges"
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}.{first}-{stop}.bin"
        if path.exists() and path.stat().st_size == stop - first:
            return path.read_bytes()
        # Include the range in the URL as well, avoiding CDN caches which key
        # a signed redirect without its Range request header.
        for attempt in range(4):
            request = urllib.request.Request(self.base + name + f"?tensorsharp_range={first}-{stop}&attempt={time.time_ns()}",
                headers={"Range": f"bytes={first}-{stop - 1}", "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{stop - 1}/{self.files[name]['size']}":
                        raise ValueError("Server did not honor the exact byte range; refusing a text-shard download")
                    data = response.read(stop - first + 1)
                break
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(0.25 * (2 ** attempt))
        if len(data) != stop - first:
            raise ValueError("Truncated safetensors range")
        path.write_bytes(data)
        return data

    def header(self, name):
        whole = self.cache / name
        if whole.exists():
            with whole.open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                if length > 16 * 1024 * 1024:
                    raise ValueError("Invalid safetensors header length")
                raw = source.read(length)
        else:
            length = struct.unpack("<Q", self.byte_range(name, 0, 8))[0]
            if length > 16 * 1024 * 1024:
                raise ValueError("Invalid safetensors header length")
            raw = self.byte_range(name, 8, length + 8)
        if len(raw) != length:
            raise ValueError("Truncated safetensors header")
        return length + 8, json.loads(raw), hashlib.sha256(raw).hexdigest()


def select_tensors(index, layers):
    selected = {name: shard for name, shard in index["weight_map"].items()
                if name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_end", "image_newline")}
    for layer in range(layers):
        name = f"layers.{layer}.ffn.gate.bias_vl"
        selected[name] = index["weight_map"][name]
    return selected


def tokenizer_fingerprint(tokenizer, vocabulary):
    """Same raw-token identity check as the native GGUF/Engram loader."""
    tokens = {int(index): token for token, index in tokenizer["model"]["vocab"].items()}
    tokens.update((int(token["id"]), token["content"]) for token in tokenizer["added_tokens"])
    if set(tokens) != set(range(vocabulary)):
        raise ValueError("Official tokenizer does not contain the expected contiguous vocabulary")
    fingerprint = 14695981039346656037
    for index in range(vocabulary):
        encoded = tokens[index].encode("utf-8")
        for byte in struct.pack("<Q", len(encoded)) + encoded:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return fingerprint


def add_metadata(writer, config, tokenizer_hash, repository, revision):
    writer.add_string("general.source.huggingface.repository", repository)
    writer.add_string("general.source.huggingface.revision", revision)
    writer.add_uint64("deepseek41.tokenizer_hash", tokenizer_hash)
    for name in ("hidden_size", "num_hidden_layers"):
        writer.add_uint32("deepseek41." + name, config["text_config"][name])
    writer.add_uint32("deepseek41.image_token_id", config["image_token_id"])
    for name, value in config["vision_config"].items():
        if name == "model_type":
            continue
        key = "deepseek41.vision." + name
        if name == "rope_theta":
            writer.add_float32(key, float(value))
        else:
            writer.add_uint32(key, 0 if value is None else int(value))
    writer.add_float32("deepseek41.vision.rms_norm_eps", 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--parent-engram", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true", help="Fetch only headers and small delimiter/router ranges; defer large shard verification/GGUF writing")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Metadata download workers must be in [1, 8]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent = args.parent_engram or args.output_dir / "deepseek41.engram.bin"
    with parent.open("rb") as source:
        metadata = source.read(44)
    if len(metadata) != 44 or metadata[:8] != b"TSD41E01":
        raise ValueError("Invalid parent Engram sidecar")
    parent_vocab = struct.unpack_from("<I", metadata, 8)[0]
    tokenizer_hash = struct.unpack_from("<Q", metadata, 36)[0]
    download = Download(args.repository, args.revision, args.cache_dir or args.output_dir / ".deepseek41-vision-cache")
    config, index = download.json_file("config.json"), download.json_file("model.safetensors.index.json")
    if parent_vocab != config["text_config"]["vocab_size"]:
        raise ValueError("Vision configuration does not match parent vocabulary")
    tokenizer = download.json_file("tokenizer.json")
    if tokenizer_fingerprint(tokenizer, parent_vocab) != tokenizer_hash:
        raise ValueError("Vision checkpoint tokenizer does not match the parent Engram fingerprint")
    selected = select_tensors(index, config["text_config"]["num_hidden_layers"])
    vision_shards = {shard for name, shard in selected.items() if name.startswith(("vision.", "aligner."))}
    for shard in sorted(vision_shards):
        if any(name not in selected for name, file in index["weight_map"].items() if file == shard):
            raise ValueError("Vision is not isolated from text weights in this checkpoint revision")
        print(f"Preparing isolated vision shard {shard}", flush=True)
        if not args.metadata_only:
            download.whole_file(shard)
    shards = sorted(set(selected.values()))
    print(f"Reading safetensors metadata from {len(shards)} shards with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        headers = dict(zip(shards, pool.map(download.header, shards)))
    def small_tensor(item):
        name, shard = item
        header_bytes, header, _ = headers[shard]
        begin, end = header[name]["data_offsets"]
        return name, download.byte_range(shard, header_bytes + begin, header_bytes + end)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        small_tensors = dict(pool.map(small_tensor, ((name, shard) for name, shard in selected.items() if shard not in vision_shards)))
    if args.metadata_only:
        print("Vision metadata and small tensor ranges cached; full GGUF preparation deferred", flush=True)
        return
    output = args.output_dir / "deepseek41.vision.gguf"
    temporary = output.with_suffix(".gguf.partial")
    writer = GGUFWriter(str(temporary), "deepseek41_vision")
    add_metadata(writer, config, tokenizer_hash, args.repository, args.revision)
    tensors = []
    for name, shard in sorted(selected.items()):
        header_bytes, header, header_hash = headers[shard]
        entry = header[name]
        begin, end = entry["data_offsets"]
        shape, dtype = entry["shape"], entry["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"Unexpected vision tensor type {dtype}: {name}")
        numpy_dtype = np.dtype("<u2" if dtype == "BF16" else "<f4")
        expected_bytes = int(np.prod(shape)) * numpy_dtype.itemsize
        if end - begin != expected_bytes:
            raise ValueError(f"Invalid vision tensor shape: {name}")
        path = download.cache / shard
        if shard in vision_shards:
            array = np.memmap(path, mode="r", dtype=numpy_dtype, offset=header_bytes + begin, shape=tuple(shape))
        else:
            raw = small_tensors[name]
            array = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)
        digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        writer.add_tensor(name, array, raw_dtype=GGMLQuantizationType.BF16 if dtype == "BF16" else GGMLQuantizationType.F32)
        tensors.append(dict(name=name, dtype=dtype, shape=shape, source=shard,
                            source_lfs_sha256=download.files[shard]["lfs"]["oid"],
                            source_header_sha256=header_hash, range=[header_bytes + begin, header_bytes + end],
                            tensor_sha256=digest, entire_source_verified=shard in vision_shards))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    temporary.replace(output)
    manifest = dict(repository=args.repository, revision=args.revision, tokenizer_hash=tokenizer_hash,
                    config=config, source_metadata_sha256={name: sha256(download.cache / name) for name in
                        ("config.json", "tokenizer.json", "model.safetensors.index.json")},
                    output=output.name, output_sha256=sha256(output), tensors=tensors)
    (args.output_dir / "deepseek41.vision.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size, tensors=len(tensors), sha256=manifest["output_sha256"])), flush=True)


if __name__ == "__main__":
    main()
