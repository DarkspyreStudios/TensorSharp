# Model Lifetime Fixture

The fixture privately loads the real TensorSharp assemblies in a collectible generation.
Each mode/backend runs in a separate process against an explicit matching native bridge.
The existing model modes generate a tiny F32 HunyuanDense GGUF. The ownership-only quantized
modes generate one valid Q4_0 tensor and do not execute a model forward.

## Quantized Ownership Gates

`raw-quantized-read-refusal` and `stacked-quantized-read-refusal` capture the destination
of the production fallback read before a test FileStream returns EOF. Ordinary storage has
a local disposable owner before the read. Stacked storage transfers to the model before
the read. `stacked-owner-insertion-refusal` injects a dictionary-comparer exception before
the stack owner transfers. The production catch explicitly frees that untransferred buffer.
`stacked-partial-view-refusal` instead refuses insertion of the second expert view after
the stack and first view transfer. That case is a positive explicit rollback gate.

Every defined backend selects mmap in current production code. These four modes temporarily
force only the private backend classifier to an undefined sentinel. They preserve the actual
CPU/Metal context and ExecutionPlan and restore the classifier before cleanup. They do not
qualify a supported fallback configuration. Captured allocation and owner-transfer evidence
does not measure raw-buffer liveness. The tests do not read or free a captured pointer after
rollback. Zero GGML lease counts do not prove that BCL NativeMemory allocations were freed.
The source cleanup paths cover the untransferred 144/288-byte buffers.

`bonsai-unregister-refusal` first registers a real Q2_0 weight at its host and GCHandle keys.
It invokes the existing terminal owner poison hook. The guarded native unregister then
refuses. Disposal preserves both managed keys, host ownership and the GCHandle after
that refusal. The mode observes managed fields only after poisoning. It does not reset the
owner, bypass the guard, dereference freed memory or qualify an actual GPU/synchronization fault.
Shared model disposal retains its context/Model leases and all foreign roots after refusal.

`local-quantized-transfer` verifies explicit disposal after a failed local ownership transfer.
`local-bonsai-transfer-refusal` checks preservation of the original work error and cleanup
error, then checks the real unregistered weight remains strongly owned after finalizer drainage.
`bonsai-registration-refusal` uses a real native parameter rejection. The reserved identity
survives the rejected registration until a successful guarded unregister confirms retirement.

`quantized-fusion-ownership` checks contiguous mapped views retain the common mapping owner.
It also checks an owned fusion copy remains readable after explicit source retirement.
`quantized-fusion-source-refusal` refuses a real source unregister after the fused view transfers.
The original source entries and fused backing owner remain strongly owned. Its terminal owner
poisoning is a controlled managed refusal, not an actual GPU fault. All three poisoned modes
check actual model/weight ownership and eight foreign roots after finalizer drainage; numeric
leases alone are not their ownership assertion. Clean modes explicitly dispose and shut down
before checking collection.

The fixture does not qualify pretrained or complete quantized models,
whole-native executors, multiple devices, captured CUDA teardown or other operating systems.

## DeepSeek41 Vision Lifetime

`vision-normal`, `vision-mismatch`, `vision-load-cleanup-refusal` and
`vision-dispose-cleanup-refusal` generate a complete
one-layer F32 DeepSeek41 text GGUF with 129265 tokenizer entries and the exact image token at
129264. The text file is 36105440 bytes. Its 38 tensors include the real Engram, compressor,
indexer, hyper-connection and expert weights required by the native loader. The matching real
vision companion has one four-dimensional layer, 19 F32 tensors and 7840 bytes. The mismatch
companion declares text dimension 64. Neither generator downloads a model or requires Python
packages. Files exist only under the supplied repository `TMPDIR` and are deleted in `finally`.

The four modes qualify lifetime behavior on the preserved bridge's CPU backend. `vision-normal`
normally constructs the actual DeepSeek41Model, loads and attaches its real vision companion,
then explicitly disposes all text/vision native handles before guarded shutdown and foreign
generation collection. `vision-mismatch` receives a real native vision handle and its actual
VisionInfo before validation refuses the dimension mismatch. Successful rollback preserves
the original validation error, frees that handle and leaves native text reset and the original
tokenizer usable. No text forward, image encode, mixed-logit or pretrained accuracy is tested.

`vision-load-cleanup-refusal` installs a test-only delegating ITokenizer wrapper through the
existing protected setter. Only image-placeholder lookup poisons the existing GGML owner and
throws an original validation error. Every other operation delegates unchanged. The original
tokenizer restores in `finally`. Refused guarded Free preserves the original validation error
and its stack together with the cleanup error in an aggregate. The model reserves the actual
returned vision field before validation. Existing generation-local failed ownership retains
the actual model and exact vision/text native identities after diagnostic finalizer drainage.
Two managed context/Model leases and both native handles remain owned; guarded shutdown refuses
and the complete foreign generation stays rooted. This mode distinguishes actual instance
ownership from native-table accounting alone.

`vision-dispose-cleanup-refusal` loads and attaches the actual companion before controlled
owner poisoning. The shared disposal pipeline refuses and retains the loaded model and both
exact native identities. Both refusal modes call Dispose twice more and verify the shared
terminal guard preserves the first cleanup diagnostic and all still-owned fields/identities.
No mode performs a native call or pointer read after poisoning. The disposal source routes
vision/text release through one shared graph callback after HostReadBarrier. These controlled
refusals do not prove actual captured graph teardown, failed GPU synchronization or image encode.

Metal class execution is unqualified. DeepSeek4Model normalizes its base constructor to
GgmlCpu. A runtime already initialized as Metal refuses that CPU request before the text
executor loads. The fixture does not bypass that owner policy. These modes do not qualify
physical multi-device execution, CUDA/MLX, vision encode or release.

## Terminal Execution Admission

`execution-cleanup-failure`, `execution-worker-cleanup-failure` and
`execution-dispatch-cleanup-failure` normally construct the complete synthetic HunyuanDense
model and run real prefill/decode before the failure. A real GGML tensor uses a test storage
whose managed Destroy refuses before native freeing. The existing model ownership helper
receives that actual refusal. The fixture does not seed the model flag or poison the runtime.
The failed model and exact tensor/storage owners survive diagnostic finalizer drainage.
Guarded shutdown refuses and the complete foreign generation remains rooted.

Shared and family entrypoints reject operations with the exact first cleanup failure as
InnerException. The shared case observes no control broadcast or accepted guarded native call.
It preserves the live KV cache references, sequence length and caller snapshot bytes.
The null batched context is only an admission-order observation, not paged execution.
Worker cases use a logical recording group on one actual context. They check refusal before
receive and after a controlled failure during receive, before direct Core dispatch.
These cases do not qualify network collectives or physical multi-device execution.

`vision-execution-cleanup-failure` normally constructs DeepSeek41 and loads its actual
vision companion before the same real-storage refusal. Media operations reject admission
before native handles or prepared prompt state change. Rejected input embeddings remain
caller-owned. The runtime remains Ready in every admission mode. These controlled managed
cleanup failures do not qualify GPU faults, forward cancellation or concurrent disposal.

The source verifier in `eng/guard-model-execution` inventories all mapped admission methods,
including direct family, speculative, prefix-cache, media, encoder and image/video child paths.
It verifies unconditional first-statement checks and worker checks outside provider catches.
Source coverage is not execution qualification for families without an actual model fixture.

The terminal model check changes neither ordinary disposed-model policy nor backend failure
handling. Metadata, diagnostic getters and independent managed sampling remain available.
Model-bound resource and media operations refuse before accepting new argument ownership.
Associated encoders check their host model; standalone encoders retain their original behavior.

CPU admission-refusal processes complete successfully. On the matching preserved Metal bridge,
`execution-cleanup-failure metal` completes its admission and retention assertions, then exits
with code 134. The unchanged upstream `ggml_metal_rsets_free` asserts that its resident-set
collection is empty during native process-exit destruction. The intentionally retained failed
model still has Metal resident resources after its real prefill/decode. This process is a failed
terminal-exit qualification, not a passing Metal refusal case. Clean Metal normal, phase-order
and fusion processes explicitly retire ownership and complete successfully. No fixture resets
terminal ownership or frees its retained resources to hide the exit failure.
