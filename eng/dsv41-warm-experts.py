#!/usr/bin/env python3
"""Read host-offloaded V4.1 expert weights before a warm benchmark.

This deliberately performs real model I/O and must run outside timing windows.
It neither pins pages nor changes inference. Requires the gguf Python package.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="First GGUF shard")
    parser.add_argument("--layers", type=int, required=True, help="Leading CPU-MoE layers")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layers <= 40 or not 1 <= args.workers <= 8:
        parser.error("layers must be 1..40 and workers 1..8")

    from gguf import GGUFReader
    match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", args.model.name)
    paths = ([args.model.with_name(f"{match[1]}-{i:05}-of-{int(match[2]):05}.gguf")
              for i in range(1, int(match[2]) + 1)] if match else [args.model])
    wanted = {f"blk.{layer}.ffn_{projection}_exps.weight"
              for layer in range(args.layers) for projection in ("gate", "up", "down")}
    ranges = []
    for shard_index, path in enumerate(paths):
        reader = GGUFReader(str(path), mode="r")
        architecture = reader.get_field("general.architecture")
        # Split writers may put the architecture only in shard zero. Require
        # it there, and reject a conflicting value in any subsequent shard.
        if (architecture is None and shard_index == 0) or (
                architecture is not None and architecture.contents() != "deepseek41"):
            raise ValueError(f"{path}: expected deepseek41 architecture")
        for tensor in reader.tensors:
            if tensor.name not in wanted:
                continue
            offset, size = int(tensor.data_offset), int(tensor.n_bytes)
            if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
                raise ValueError(f"Invalid tensor range: {tensor.name}")
            ranges.append({"file": str(path), "tensor": tensor.name,
                           "offset": offset, "bytes": size})
        del reader
    names = [entry["tensor"] for entry in ranges]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates or set(names) != wanted:
        raise ValueError(f"Missing CPU expert tensors: {sorted(wanted - set(names))}; duplicates: {duplicates}")

    def read_range(entry):
        started = time.monotonic()
        remaining = entry["bytes"]
        with open(entry["file"], "rb", buffering=0) as source:
            source.seek(entry["offset"])
            while remaining:
                block = source.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError(entry["tensor"])
                remaining -= len(block)
        return {**entry, "wall_seconds": time.monotonic() - started}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(read_range, ranges))
    report = {"purpose": "CPU expert page warming; excluded from inference timings",
              "model": str(args.model), "layers": args.layers, "workers": args.workers,
              "bytes_read": sum(row["bytes"] for row in results),
              "wall_seconds": time.monotonic() - started, "tensors": results,
              "limitations": "Pages remain evictable; this is not a residency or checksum guarantee."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}))


if __name__ == "__main__":
    main()
