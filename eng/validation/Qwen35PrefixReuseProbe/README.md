# Qwen35 concurrent reviewer prefix reuse

This opt-in CUDA probe loads real weights once and runs two independent reviewer
requests through `InferenceEngine`. `MultiAgentSession` constructs their messages
and tools; the model's tokenizer and template determine the verified public prefix.
The fixed fixture includes every number from proposals A and B. It uses a short,
documented governing prompt, rather than reproducing the entire HTTP host catalog.

```powershell
dotnet build eng/validation/Qwen35PrefixReuseProbe -c Release
dotnet eng/validation/Qwen35PrefixReuseProbe/bin/Release/net10.0/Qwen35PrefixReuseProbe.dll --model C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ3_XXS.gguf --out artifacts/qwen35-agent-prefix-reuse/direct-probe.json --steps 32 --pairs 3
```

The probe sets an 8192-token context and enables checkpoint support. Cache dtype
comes from `KV_CACHE_DTYPE`. Optional `TS_VALIDATION_GGML_REVISION` and
`TS_VALIDATION_DEVICE` identify the unchanged dependency checkout and actual device;
the report also hashes the loaded native and managed binaries.

Every measured pair has distinct request scopes and is submitted while a
`ComputeGate` holds the engine. Neither completion is awaited until both requests
are queued. The cold case requires reuse `[0, shared-prefix-length]`; seeded warm
cases require full public-prefix reuse by both siblings. Disabled cases require
zero reuse. All paired enabled/disabled generated token arrays, actual counts,
completion statuses, and finish reasons must match exactly. `--steps` sets the
maximum new tokens per request; a normal EOS may complete a short answer or
tool-call turn sooner. Every actual count and completion row is retained before
validation. Safely completed failures do not prevent collecting the remaining cases.
Total generated counts include the final EOS when present; a separate non-EOS
count must be positive so an EOS-only empty response cannot pass validation.
Failure produces a nonzero exit status and an unqualified report, preserving the
measured rows and validation failures.

Warm performance samples alternate enabled/disabled order with fresh engines.
The three warmup seed costs are reported separately. Elapsed time includes prefill,
adoption, decoding, and completion after the gate opens; loading, rendering, and
engine setup are excluded. Time to first token uses the engine's first-token
timestamp relative to gate release. Empty, aborted, erroneous, or mismatched
completions disqualify validation and performance, including when all pairs finish.

This verifies token-prefix reuse and short greedy continuations. It does not prove
full reviewer reasoning quality, HTTP latency, physical page sharing, or VRAM
savings: Qwen35 checkpoint adoption still creates model-owned writable copies.
`CachedPayloadMemoryAfterCompletion` samples the model's public diagnostics after
both requests finish. It excludes active peak memory, weights, allocator reserves,
and primary/private cache bytes; it is not a GPU VRAM measurement.
The HTTP multi-agent probe supplies separate end-to-end evidence.
