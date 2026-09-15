#!/usr/bin/env python3
"""Compare concurrent single-input embedding requests using a recorded setup.

Example (run from the repository root with both Release binaries built):
  python3 benchmarks/EmbeddingBench/concurrency_bench.py \
    --base-results docs/validation/embeddings-2026-09/minilm-cpu/results.json \
    --output /tmp/minilm-cpu-concurrency --require-performance

The default workload has eight persistent HTTP clients, each sending four
requests per round. Engines run sequentially with the exact base commands.
Quality calculations and artifact writes happen outside measured rounds.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.parse

import embedding_bench as bench


PERFORMANCE_LIMIT = 1.05
CROSS_ENGINE_MIN_COSINE = .999


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_results(destination, result):
    """Retain every parsed vector exactly, storing repeated payloads only once."""
    vectors = {}

    def compact(value):
        if isinstance(value, dict):
            packed = {}
            for key, item in value.items():
                if key == "vector" and isinstance(item, list):
                    payload = json.dumps(item, separators=(",", ":")).encode("utf-8")
                    identity = hashlib.sha256(payload).hexdigest()
                    if identity in vectors and vectors[identity] != item:
                        raise AssertionError("Vector payload hash collision")
                    vectors[identity] = item
                    packed["vector_ref"] = identity
                else:
                    packed[key] = compact(item)
            return packed
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    packed = compact(result)
    packed["vector_storage"] = {
        "format": "vector_ref points to the vectors table; payloads are exact parsed HTTP values",
        "key": "SHA-256 of UTF-8 json.dumps(vector, separators=(',', ':'))",
        "unique_vectors": len(vectors),
    }
    packed["vectors"] = vectors
    destination.write_text(json.dumps(packed, indent=2) + "\n")


def close_thread_connections():
    for connection in getattr(bench.CONNECTIONS, "connections", {}).values():
        connection.close()
    bench.CONNECTIONS.connections = {}


@contextmanager
def server(name, command, url, output):
    # Reuse the established HTTP helpers while keeping each owned process's
    # lifecycle explicit. Never send shutdown signals to pre-existing servers.
    address = urllib.parse.urlparse(url)
    try:
        existing = socket.create_connection((address.hostname, address.port or 80), timeout=.25)
    except OSError:
        pass
    else:
        existing.close()
        raise RuntimeError(f"Refusing to benchmark against an occupied server address: {url}")
    with (output / (name + ".log")).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        started = time.monotonic()
        try:
            while time.monotonic() - started < 180:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with {process.returncode}; see {name}.log")
                try:
                    status, _ = bench.request(url, "/v1/models")
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError(f"{name} startup")
            print(name, "ready", flush=True)
            yield
        finally:
            close_thread_connections()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def concurrent_round(pool, clients, inputs, url, model):
    barrier = threading.Barrier(clients, timeout=30)
    round_start = time.perf_counter()

    def client(client_index):
        # All client workers reach this barrier before any starts inference.
        # Later calls from the same client are sequential and reuse its socket.
        barrier.wait()
        rows = []
        for request_index in range(client_index, len(inputs), clients):
            started = time.perf_counter()
            vectors, usage = bench.embedding(url, model, inputs[request_index])
            completed = time.perf_counter()
            if len(vectors) != 1:
                raise AssertionError((request_index, "Expected one embedding", len(vectors)))
            rows.append(dict(request_index=request_index, client_index=client_index,
                             started_ms=(started - round_start) * 1000,
                             completed_ms=(completed - round_start) * 1000,
                             latency_ms=(completed - started) * 1000,
                             usage=usage, vector=vectors[0]))
        return rows

    futures = [pool.submit(client, index) for index in range(clients)]
    concurrent.futures.wait(futures)
    wall_ms = (time.perf_counter() - round_start) * 1000
    rows = sorted((row for future in futures for row in future.result()),
                  key=lambda row: row["request_index"])
    assert [row["request_index"] for row in rows] == list(range(len(inputs)))
    return dict(wall_ms=wall_ms, requests_per_second=len(inputs) * 1000 / wall_ms,
                requests=rows)


def validate_round(result, expected, dimensions, min_cosine):
    # Do this only after all requests have completed, so Python's vector work
    # cannot contend with other client threads during the timing interval.
    for row, reference in zip(result["requests"], expected):
        vector = row["vector"]
        bench.assert_vectors([vector], dimensions)
        assert row["usage"] == reference["usage"], (row["request_index"], "Token accounting changed")
        similarity = bench.cosine(vector, reference["vector"])
        assert similarity >= min_cosine, (row["request_index"], similarity, min_cosine)
        row["norm"] = bench.norm(vector)
        row["single_request_cosine"] = similarity
        row["float32_sha256"] = hashlib.sha256(struct.pack("<" + "f" * dimensions, *vector)).hexdigest()
    result["prompt_tokens"] = sum(row["usage"]["prompt_tokens"] for row in result["requests"])
    result["tokens_per_second"] = result["prompt_tokens"] * 1000 / result["wall_ms"]


def run_engine(name, setup, inputs, model, args, base, result):
    dimensions = setup["correctness"]["dimensions"]
    consistency_gate = .9999 if name == "tensorsharp" else setup["correctness"].get("batch_consistency_gate", .9999)
    result.update(command=setup["command"], url=setup["url"], dimensions=dimensions,
                  single_request_consistency_gate=consistency_gate, warmup_rounds=[], rounds=[])
    with server(name, setup["command"], setup["url"], args.output):
        expected = []
        for text in inputs:
            vectors, usage = bench.embedding(setup["url"], model, text)
            bench.assert_vectors(vectors, dimensions)
            assert len(vectors) == 1
            expected.append(dict(vector=vectors[0], usage=usage))
        result["single_request_references"] = expected
        prewarm_start = time.perf_counter()
        calls = 0
        prewarm_text = base.get("benchmark_inputs", {}).get("single_short", [inputs[0]])[0]
        while time.perf_counter() - prewarm_start < args.prewarm_seconds:
            bench.embedding(setup["url"], model, prewarm_text)
            calls += 1
        result["prewarm"] = dict(calls=calls, elapsed_seconds=time.perf_counter() - prewarm_start)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as pool:
            for _ in range(args.warmup):
                warmup = concurrent_round(pool, args.clients, inputs, setup["url"], model)
                result["warmup_rounds"].append(warmup)
                validate_round(warmup, expected, dimensions, consistency_gate)
            for round_index in range(args.rounds):
                measured = concurrent_round(pool, args.clients, inputs, setup["url"], model)
                measured["round_index"] = round_index
                result["rounds"].append(measured)
                validate_round(measured, expected, dimensions, consistency_gate)
                print(name, "round", round_index + 1, round(measured["wall_ms"], 3), "ms", flush=True)
            # Submit one barrier-synchronized cleanup per worker so all eight
            # thread-local sockets close before the owned server is stopped.
            cleanup_barrier = threading.Barrier(args.clients, timeout=30)
            def close_worker(_):
                cleanup_barrier.wait()
                close_thread_connections()
            list(pool.map(close_worker, range(args.clients)))
    walls = [round_["wall_ms"] for round_ in result["rounds"]]
    latencies = [row["latency_ms"] for round_ in result["rounds"] for row in round_["requests"]]
    percentile95 = lambda values: sorted(values)[min(len(values) - 1, math.ceil(len(values) * .95) - 1)]
    result["summary"] = dict(round_median_ms=statistics.median(walls), round_p95_ms=percentile95(walls),
                             request_median_ms=statistics.median(latencies), request_p95_ms=percentile95(latencies),
                             requests_per_second=len(inputs) * 1000 / statistics.median(walls),
                             total_measured_requests=len(latencies),
                             minimum_single_request_cosine=min(row["single_request_cosine"]
                                 for round_ in result["rounds"] for row in round_["requests"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--input-case", default="batch32_short")
    parser.add_argument("--model", help="Defaults to llama's recorded --alias, or the GGUF filename stem")
    parser.add_argument("--prewarm-seconds", type=float, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tensorsharp-first", action="store_true")
    parser.add_argument("--require-performance", action="store_true")
    args = parser.parse_args()
    if args.clients < 1 or args.requests < args.clients or args.requests % args.clients:
        parser.error("requests must be a positive multiple of clients")
    if args.warmup < 0 or args.rounds < 1 or not math.isfinite(args.prewarm_seconds) or args.prewarm_seconds < 0:
        parser.error("rounds must be positive; warmup and finite prewarm-seconds must be nonnegative")
    destination = args.output / "results.json"
    if destination.exists():
        parser.error("output results already exist; use a new directory to retain earlier evidence")
    base = json.loads(args.base_results.read_text())
    inputs = base.get("benchmark_inputs", {}).get(args.input_case)
    if not isinstance(inputs, list) or len(inputs) != args.requests or any(not isinstance(text, str) or not text for text in inputs):
        parser.error("base-results input-case must contain exactly requests nonempty strings")
    for name in ("tensorsharp", "llama"):
        setup = base["engines"][name]
        if not isinstance(setup["command"], list) or not setup["command"] or any(not isinstance(arg, str) for arg in setup["command"]):
            parser.error("base-results commands must be argument arrays")
    model = args.model or Path(base["model_file"]).stem
    llama_command = base["engines"]["llama"]["command"]
    if not args.model and "--alias" in llama_command:
        model = llama_command[llama_command.index("--alias") + 1]
    model_hash = sha256(base["model_file"])
    if model_hash != base["model_sha256"]:
        parser.error("current model hash differs from base-results")
    args.output.mkdir(parents=True, exist_ok=True)
    bench.KEEP_ALIVE = True
    names = ["tensorsharp", "llama"] if args.tensorsharp_first else ["llama", "tensorsharp"]
    result = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), platform=platform.platform(),
                  machine=platform.machine(), command=[sys.executable, *sys.argv],
                  base_results=str(args.base_results), base_results_sha256=sha256(args.base_results),
                  model=model, model_file=base["model_file"], model_sha256=model_hash,
                  clients=args.clients, requests_per_round=args.requests, warmup_rounds=args.warmup,
                  measured_rounds=args.rounds, prewarm_seconds=args.prewarm_seconds,
                  connection_mode="persistent-per-thread", input_case=args.input_case, inputs=inputs,
                  latency_definition="Per-request wall time from HTTP call start through response JSON parsing; round wall includes client dispatch and completion.",
                  engine_order=names, performance_limit=PERFORMANCE_LIMIT,
                  cross_engine_min_cosine=CROSS_ENGINE_MIN_COSINE,
                  base_binary_artifacts=base.get("binary_artifacts", {}),
                  binary_artifacts={name: bench.command_artifacts(base["engines"][name]["command"]) for name in names},
                  engines={})
    try:
        for name in names:
            engine_result = result["engines"][name] = {}
            run_engine(name, base["engines"][name], inputs, model, args, base, engine_result)
            write_results(destination, result)
        ts, llama = result["engines"]["tensorsharp"], result["engines"]["llama"]
        cosines, errors = [], []
        for ts_round, llama_round in zip(ts["rounds"], llama["rounds"]):
            current = []
            for a, b in zip(ts_round["requests"], llama_round["requests"]):
                assert a["usage"] == b["usage"], (a["request_index"], "Cross-engine token accounting differs")
                current.append(bench.cosine(a["vector"], b["vector"]))
                errors.append(max(abs(x - y) for x, y in zip(a["vector"], b["vector"])))
            cosines.append(current)
        ratio = ts["summary"]["round_median_ms"] / llama["summary"]["round_median_ms"]
        result["comparison"] = dict(round_median_ratio_tensorsharp_over_llama=ratio,
                                    cross_engine_cosines_by_round=cosines,
                                    min_cosine=min(min(row) for row in cosines), max_absolute_error=max(errors),
                                    token_accounting_equal=True, performance_pass=ratio <= PERFORMANCE_LIMIT)
        assert result["comparison"]["min_cosine"] >= CROSS_ENGINE_MIN_COSINE, {
            "min_cosine": result["comparison"]["min_cosine"], "required": CROSS_ENGINE_MIN_COSINE}
        print(json.dumps({key: value for key, value in result["comparison"].items()
                          if key != "cross_engine_cosines_by_round"}, indent=2), flush=True)
        if args.require_performance:
            assert result["comparison"]["performance_pass"], {
                "round_median_ratio_tensorsharp_over_llama": ratio, "limit": PERFORMANCE_LIMIT}
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_results(destination, result)


if __name__ == "__main__":
    main()
