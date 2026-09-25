# Gemma 4 heterogeneous cache probe

Reproduces a 16,384-token root cache alongside retained-prefix clones compacted
to 512/1,024 tokens. Children grow independently during decode. Uses production
checkpoint, clone, holder, and forward APIs; reflection observes capacities only.

Each comparison uses fixed teacher-forced tokens, rotates request order, checks
full vocabulary logits for finite values, and checks a final solo continuation
after the batched episode. Numerical correctness compares compact caches with
full-capacity batched caches using identical prefill and ready-row schedules:
every logit must agree within `1e-4 + 1e-4 * abs(reference)`, including the solo
continuation. Eligible batches must execute the native fused path, with positive
model counters. A row needing cache growth gets one serial step, as in the runtime
scheduler. Other ready rows remain batched.

Serial-vs-batched logit differences are reported separately. The existing native
quantized batched GEMM differs numerically from single-token GEMV; the reference
keeps those matrix shapes identical to isolate cache correctness. Do not report
these tests as bit-exact serial-vs-batched parity.
In JSON, `MinCosine`, `MaxRmse`, `MaxAbsoluteError`, and `TopTokenDifferences`
describe serial-vs-batched differences. `UniformBatchedReference` contains the
separately asserted cache-correctness comparison.

```powershell
dotnet build eng/validation/Gemma4BatchedCacheProbe -c Release `
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
$env:TS_TEST_GEMMA4_MODEL = 'C:\path\gemma-4-12B-it.gguf'
$env:TS_TEST_GGML_BACKEND = 'cuda' # or metal, vulkan, cpu
$env:TS_GEMMA4_PROBE_OUT = 'artifacts/gemma4-batched-cache/report.json'
dotnet eng/validation/Gemma4BatchedCacheProbe/bin/Release/net10.0/Gemma4BatchedCacheProbe.dll
```

The benchmark forces `MAX_CONTEXT=16384` and `TS_KV_INITIAL_TOKENS=0` to retain
the full initial root capacity. Defaults: widths 2,3,4; 12 decode steps; one warmup
and three measured serial/batched pairs per width, alternating pair order.
Override `TS_GEMMA4_PROBE_WIDTHS`, `TS_GEMMA4_PROBE_STEPS`, or
`TS_GEMMA4_PROBE_PAIRS` to bound a run. `--allow-fallback` records a pre-fix baseline;
its successful completion does **not** prove batching. Reports distinguish native
fused steps, fallback steps, and individual growth rows.

`--uniform-control` uses full-size holders throughout. `--diagnostic-only` records
numerical differences without asserting reference parity and explicitly marks
the report as not passing validation. These switches support comparison with a
preserved previous native binary; they are not correctness-test substitutes.

The xUnit regression shares the same implementation and is gated by the explicit
`TS_TEST_GEMMA4_MODEL` file path, so both dense 12B and E4B PLE/shared-KV models can
be tested. Missing models are skipped. Run with native build flags above and
`--filter FullyQualifiedName~Gemma4HeterogeneousCacheTests`.

These are direct decode timings, including graph setup and holder growth but
excluding model load, prefill, cache allocation, and solo continuation. They do
not measure full request or multi-agent task latency. Preserve benchmark JSON,
logs, binary hashes, model identity and upstream revision under ignored
`artifacts/` or `docs/validation/`.
