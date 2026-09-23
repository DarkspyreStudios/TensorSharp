# Bonsai2 validation

`bonsai2-bench.py` uses the Python standard library. It records full requests,
responses, generated token references, timing samples, source revisions, and
optional file hashes. Evidence defaults in examples to ignored
`docs/validation/bonsai2/`; do not add generated evidence to Git.

The two Bonsai2 GGUFs use publisher-specific PQ2_0/PTQ1_0 storage. The baseline
must understand those formats. The reference used during implementation was
`PrismML-Eng/llama.cpp`, branch `prism`, commit
`bdc23b56b4458b9f1655aec5287f3ab56ee8daaa`. Keep its checkout separate from
TensorSharp's unchanged upstream ggml dependency. An ordinary upstream
llama.cpp build rejecting the files is an unavailable baseline, not a
TensorSharp performance win.

## Native and metadata checks

Build the TensorSharp-owned tests against the unmodified dependency checkout:

```sh
cmake -S TensorSharp.GGML.Native -B TensorSharp.GGML.Native/build \
  -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON
cmake --build TensorSharp.GGML.Native/build \
  --target GgmlOps GgmlOpsBonsaiHadamardTest GgmlOpsBonsaiQuantTest GgmlOpsGraphOptimizeTest
ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure \
  -R 'bonsai2-|graph-optimizer-allocation-dependencies'

TENSORSHARP_BONSAI2_MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter FullyQualifiedName~Bonsai2MetadataTests

TS_TEST_GGML_BACKEND=cpu \
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter FullyQualifiedName~Bonsai2ConstructionFailureTests
```

Repeat the metadata test with PTQ1_0. Without the environment variable, its
real-model-header check is skipped; synthetic metadata tests do not replace
it. Native quantization tests exercise both formats and compare backend
matmul results, while Hadamard tests use an independent dense oracle for
forward/inverse, grouped, strided, and F16 cases. Device tests are registered
according to the configured backend; an absent or skipped device test does
not establish device support. The graph-allocation test runs on CPU.
The construction-failure tests require the native bridge and create tiny
malformed GGUFs to verify buffer release, native registration rollback, repeated
failed loads, and preservation of the original error when cleanup also fails.

## Inventory

Record every source checkout, exact model, projector, and binary being tested.
The optional full hashes read each entire file and should run before timing.

```sh
python3 eng/validation/bonsai2-bench.py inventory \
  --repo artifacts/bonsai2/llama.cpp-prism \
  --repo /path/to/vllm --repo /path/to/sglang \
  --file /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --file /path/to/libGgmlOps.dylib --hash-files \
  --output docs/validation/bonsai2/inventory.json
```

The inventory checks packages in the Python interpreter actually running the
script. Discovery alone does not prove that an engine supports the model or
the available hardware. Record untested vLLM/SGLang hardware and formats as
unavailable; do not count them as passing comparisons.

## Exact reference outputs

Start the publisher's server with the exact model under test, F16 K/V caches,
and a bounded context. The example uses four 2,048-token slots:

```sh
artifacts/bonsai2/llama.cpp-prism/build/bin/llama-server \
  -m /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 \
  -c 8192 -np 4 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 \
  --host 127.0.0.1 --port 8083 --jinja

python3 eng/validation/bonsai2-bench.py reference \
  --url http://127.0.0.1:8083 --tokens 32 \
  --golden docs/validation/bonsai2/pq2-golden.json \
  --output docs/validation/bonsai2/prism-pq2-reference.json
```

This calls the reference server's template and tokenizer, then sends explicit
token IDs to its completion endpoint. Generation stops at EOS or the token
budget; EOS is not masked because the raw TensorSharp probe uses unmodified
greedy logits. Early completion requires an explicit EOS stop. English, code, Chinese, and arithmetic prompts are
included; repeat `--prompt` to replace the suite. The output is directly
compatible with the existing managed probe:

```sh
MAX_CONTEXT=2048 KV_CACHE_DTYPE=f16 dotnet \
  benchmarks/ParityHarness/bin/Release/net10.0/ParityHarness.dll \
  /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf --ref \
  docs/validation/bonsai2/pq2-golden.json ggml_metal 32
```

Stop the reference server before running TensorSharp inference. Compare both
formats independently. Inspect numerical differences rather than claiming
general quality solely from a short matching greedy sequence.

## HTTP single and parallel requests

Run each server separately on the same device with the same model, cache
dtype, context budget, thinking policy, and output limit. Disable prefix
caching consistently in the server configuration; the client also requests
`cache_prompt:false`, but servers may ignore unsupported request extensions.
Retain launch commands and server logs to audit actual settings.

```sh
python3 eng/validation/bonsai2-bench.py http \
  --url http://127.0.0.1:8083 --engine llama-prism \
  --model Ternary-Bonsai-2-27B-PQ2_0 \
  --tokens 64 --repeats 3 --concurrency 1,2,4 \
  --output docs/validation/bonsai2/prism-pq2-http.json
```

Repeat against the TensorSharp server with `--engine tensorsharp`. The client
first warms the server, collects each prompt alone, then submits concurrent
requests from a barrier. It compares text, reasoning, reported token counts,
and stop reasons against the solo run. Partial streams, missing completion
usage, and server errors fail the run. A streaming chunk is **never** treated
as a token. Per-repeat aggregate throughput includes HTTP, queuing, prefill,
and generation; first-delta time may include buffering by the server.
Sampling is explicit on both engines: temperature 0, top-k 0, top-p 1,
min-p 0, repeat penalty 1, and zero presence/frequency penalties. The summary
also counts requests reaching the full output-token budget so early stopping
cannot silently change the workload. `--request-extra` can override these
defaults; review the retained request payloads when using it.

These results are not isolated decode-kernel measurements. Review actual
prompt token counts, templates, generated counts, early stopping, resource
contention, and thermal drift before comparing engines. Alternate or bracket
engine runs when measuring small differences. All raw samples are retained;
the summary uses their median and reports minimum/maximum.

For model-only measurements, use `llama-bench` JSON (including `samples_ts`)
and `ParityHarness --bench` with matched prompt/decode lengths, F16 KV,
physical batch size, and attention depth. The managed harness preserves its
historic best-rate `[bench]` lines and additionally emits JSON
`[bench-sample]` and `[bench-statistics]` records with all samples, mean, sample
standard deviation, median, minimum, and maximum. Compare the same statistic
on both engines. Startup and warmup are excluded. The final optional argument
sets context depth; for example:

```sh
MAX_CONTEXT=2048 KV_CACHE_DTYPE=f16 TS_PREFILL_CHUNK=512 dotnet \
  benchmarks/ParityHarness/bin/Release/net10.0/ParityHarness.dll \
  /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf --bench ggml_metal 128,512 64 3 128
```

TensorSharp recomputes the context-depth prefix outside each measured interval;
llama-bench may restore a saved prefix state. This difference can affect
thermal history even though neither includes prefix setup in the samples.

## Arithmetic, structured output, tools, and vision

```sh
python3 eng/validation/bonsai2-bench.py quality \
  --url http://127.0.0.1:8083 --engine llama-prism \
  --model Ternary-Bonsai-2-27B-PQ2_0 \
  --output docs/validation/bonsai2/prism-pq2-quality.json
```

The checks expect arithmetic `16`, a constrained JSON answer, and a
`get_weather` tool call for Paris. Add `--vision` after loading a projector to
test a generated red-square/blue-circle PNG; the fixture is saved alongside
the evidence. Run BF16 and Q8_0 projectors separately, retaining server logs
that confirm vision execution. These are narrow functional smokes, not a
comprehensive model evaluation. A successful text-only run never qualifies
either projector.

Validate the evidence parser without weights:

```sh
python3 -m unittest discover -s eng/validation/tests -p 'test_bonsai2_bench.py' -v
```
