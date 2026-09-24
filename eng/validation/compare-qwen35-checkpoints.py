#!/usr/bin/env python3
"""Compare production Qwen35 Q5KC-v2 snapshots without loading a model or device.

Reports per-layer K/V prefix and newly written last-row errors, canonical conv
ring errors and GDN delta errors. This is diagnostic evidence, not a pass gate.
Snapshots taken by Qwen35BatchedDecodeProbe flush arena ownership, so subsequent
decode/performance from those runs is not an unmodified production measurement.
Requires numpy. Generated reports belong in artifacts/ or docs/validation/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


class Reader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read(self, size: int) -> bytes:
        if size < 0 or size > 2**31:
            raise ValueError(f"Invalid field length {size}")
        value = self.stream.read(size)
        if len(value) != size:
            raise ValueError("Truncated checkpoint")
        return value

    def integer(self, fmt="<i"):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def header(self):
        if self.integer("<I") != 0x51354B43 or self.integer() != 2:
            raise ValueError("Requires the production Q5KC version-2 format")
        length = 0
        for shift in range(0, 35, 7):
            part = self.integer("<B")
            length |= (part & 127) << shift
            if not part & 128:
                break
        else:
            raise ValueError("Invalid BinaryWriter string length")
        fingerprint = self.read(length).decode("utf-8")
        names = ("layers", "rows", "rope_delta", "kv_heads", "head_dim", "conv_kernel", "qkv_dim")
        result = {"fingerprint": fingerprint, **{name: self.integer() for name in names}}
        if result["layers"] < 1 or result["rows"] < 1 or result["head_dim"] < 1:
            raise ValueError("Invalid checkpoint dimensions")
        return result

    def kv(self, header):
        heads, row_bytes = self.integer(), self.integer("<q")
        if heads != header["kv_heads"]:
            raise ValueError("K/V head count disagrees with header")
        raw = self.read(heads * header["rows"] * row_bytes)
        dim = header["head_dim"]
        shape = (heads, header["rows"], dim)
        if row_bytes == dim * 2:
            return np.frombuffer(raw, dtype="<f2").reshape(shape), "F16"
        if row_bytes == dim * 4:
            return np.frombuffer(raw, dtype="<f4").reshape(shape), "F32"
        if dim % 32 == 0 and row_bytes == dim // 32 * 34:
            blocks = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]))
            values = blocks["d"].astype(np.float32)[:, None] * blocks["q"]
            return values.reshape(shape), "Q8_0"
        if dim % 32 == 0 and row_bytes == dim // 32 * 18:
            blocks = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "u1", (16,))]))
            q = np.concatenate((blocks["q"] & 15, blocks["q"] >> 4), axis=1).astype(np.int16) - 8
            return (blocks["d"].astype(np.float32)[:, None] * q).reshape(shape), "Q4_0"
        raise ValueError(f"Unknown cache row layout: {row_bytes} bytes, head dimension {dim}")

    def layer(self, header):
        recurrent = self.integer("<?")
        if not recurrent:
            k, kd = self.kv(header)
            v, vd = self.kv(header)
            return {"recurrent": False, "k": k, "v": v, "dtype": [kd, vd]}
        conv_len = self.integer()
        conv = np.frombuffer(self.read(conv_len * 4), dtype="<f4")
        write_index = self.integer()
        delta_bytes = self.integer("<q")
        if delta_bytes % 4:
            raise ValueError("Delta state byte count is not float-aligned")
        delta = np.frombuffer(self.read(delta_bytes), dtype="<f4")
        ring_size = header["conv_kernel"] - 1
        if conv_len != ring_size * header["qkv_dim"] or not 0 <= write_index < ring_size:
            raise ValueError("Invalid convolution ring geometry")
        # Host mirrors may use different circular offsets for the same history.
        conv = np.roll(conv.reshape(ring_size, header["qkv_dim"]), -write_index, axis=0)
        return {"recurrent": True, "conv": conv, "delta": delta, "write_index": write_index}


def metrics(reference, observed):
    if reference.shape != observed.shape:
        raise ValueError(f"Tensor shapes differ: {reference.shape} / {observed.shape}")
    a, b = reference.astype(np.float64), observed.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Snapshot contains non-finite state")
    if not a.size:
        return {"elements": 0}
    error = b - a
    norm_a, norm_b = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    norm_error = float(np.linalg.norm(error.ravel()))
    rmse = float(np.sqrt(np.mean(error * error)))
    cosine = float(np.sum(a * b) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
    return {"elements": int(a.size), "cosine": cosine, "max_absolute": float(np.max(np.abs(error))),
            "rmse": rmse, "reference_rms": float(np.sqrt(np.mean(a * a))),
            "relative_l2": norm_error / max(norm_a, 1e-30),
            "normalized_rmse": rmse / max(float(np.std(a)), 1e-30)}


def compare(reference: Path, observed: Path):
    layers = []
    with reference.open("rb") as af, observed.open("rb") as bf:
        a, b = Reader(af), Reader(bf)
        header, observed_header = a.header(), b.header()
        if header != observed_header:
            raise ValueError(f"Checkpoint headers differ: {header} / {observed_header}")
        for index in range(header["layers"]):
            left, right = a.layer(header), b.layer(header)
            if left["recurrent"] != right["recurrent"]:
                raise ValueError(f"Layer {index} types differ")
            if left["recurrent"]:
                components = {name: metrics(left[name], right[name]) for name in ("conv", "delta")}
                details = {"reference_write_index": left["write_index"], "observed_write_index": right["write_index"]}
            else:
                if left["dtype"] != right["dtype"]:
                    raise ValueError(f"Layer {index} cache types differ")
                components = {}
                for name in ("k", "v"):
                    components[name + "_prefix"] = metrics(left[name][:, :-1], right[name][:, :-1])
                    components[name + "_last_row"] = metrics(left[name][:, -1], right[name][:, -1])
                details = {"dtype": left["dtype"]}
            layers.append({"layer": index, "recurrent": left["recurrent"], **details, "components": components})
        if af.read(1) or bf.read(1):
            raise ValueError("Trailing checkpoint bytes")
    return {"reference": str(reference.resolve()), "observed": str(observed.resolve()),
            "diagnostic_only": True, "header": header, "layers": layers,
            "limitations": "Checkpoint/export flushes ownership and observes synchronized host state. This localizes differences; it does not establish kernel correctness, bitwise equivalence, or performance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.observed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("layer component             relative_l2    normalized_rmse    max_absolute")
    for layer in report["layers"]:
        for name, values in layer["components"].items():
            if values["elements"]:
                print(f"{layer['layer']:5} {name:21} {values['relative_l2']:12.6g} {values['normalized_rmse']:18.6g} {values['max_absolute']:15.6g}")


if __name__ == "__main__":
    main()
