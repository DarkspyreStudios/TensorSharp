# Model Lifetime Fixture

The fixture privately loads the real TensorSharp assemblies in a collectible generation.
Each mode/backend runs in a separate process against an explicit matching native bridge.
The existing model modes generate a tiny F32 HunyuanDense GGUF. The ownership-only quantized
modes generate one valid Q4_0 tensor and do not execute a model forward.

The model-lease census reads each runtime resource record's `Kind` field. A retained
resource record is not an enum value. The base-construction refusal supplies all
four native-admission reflection parameters and uses the optional parameter defaults.
These checks observe actual retained owners without changing runtime ownership.

`base-cleanup-failure` holds a real native call while a base constructor fails.
The pre-effect Busy refusal leaves the actual partial model on its release-only
cleanup handle. It does not poison the model. Once the call drains, explicit handle
disposal releases the context and model lease without replaying construction.
The clean generation then collects. Terminal-refusal modes retain their unsafe owners.

`local-cleanup-failure` checks the construction cleanup exception's actual nested
read/rollback errors. Its terminal cleanup cause is the same exception instance as
the local rollback cause. The cleanup handle remains unreleased and the failed
model retains its actual unregistered storage.

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

## Tensor-Parallel Ownership Observations

These modes use two logical ranks on one real CPU or Metal context. They do not execute a
collective or qualify physical multi-device placement. The retained `observe-*` mode names
now check explicit production ownership and cleanup; their original red assertions remain in git.

`tp-broadcast-ownership` verifies that every returned GGML rank owns an independent real
storage, including rank zero. Mutating one rank leaves the other rank and borrowed source
unchanged. `tp-broadcast-source-disposal` uses the owning caller's retirement helper and
observes that only the source storage retires. `tp-broadcast-temporary-retirement` explicitly
retires all temporary copies and their owned original source. The broadcast helper itself
borrows its input; it never implicitly disposes it.

`tp-broadcast-partial-copy` and `tp-broadcast-second-copy` refuse the first or second copy
dispatch after actual destination allocation. The hook restores before cleanup. Production
rollback explicitly destroys every partial destination while preserving the borrowed source
and exact original error. Clean modes drain guarded shutdown before foreign collection.

`tp-broadcast-source-cleanup-refusal` refuses source storage destruction after two successful
copies. `tp-broadcast-rollback-refusal` refuses the second copy and then refuses destruction of
the first destination during rollback. The local rollback owner retains the distinct borrowed
source dependency without disposing it. The fixture supplies no extra source dictionary owner.
`tp-broadcast-temporary-cleanup-refusal` refuses the first temporary copy's destruction. The
joint retirement owner also retains its original source without a separate dictionary owner. These
modes preserve actual source and all returned/partial copy owners after outer-frame finalizer
drainage; repeated model disposal refuses and guarded shutdown remains busy. Original work
and cleanup errors are preserved together where both exist. The exact cleanup-failing storage
subclass has a finite fixture-only copy registration delegating to unchanged `GgmlBasicOps.Copy`.
The named static handler has no model/source closure. Registry, handler, storage and actual model belong to the same private generation; no handler
is registered in the default context. This is controlled managed cleanup refusal, not native
free failure, physical GPU fault or captured asynchronous teardown qualification.

`observe-tp-column-owner` creates an actual owned Q4_0 source and two column views. The source
leaves the ordinary dictionary. The views hold its exact wrapper until model disposal. Model
disposal retires the views before explicitly disposing their removed backing owner. Its actual
host ownership fields clear before finalizer drainage and foreign generation collection.
The mode does not read or reclaim a captured pointer after disposal.

Six F32 modes refuse the second real storage allocation after a first real shard allocates:

| Mode | Source Allocation Route |
| --- | --- |
| `observe-tp-generic-column` | Generic output-dimension split |
| `observe-tp-generic-row` | Generic input-dimension split |
| `observe-tp-concatenated` | Concatenated column segments |
| `observe-tp-separate` | Separate source projections, F32 destination branch |
| `observe-tp-concatenated-bias` | Concatenated bias segments |
| `observe-tp-separate-bias` | Separate source biases |

Every first shard has an actual native storage weak reference and a model-owned array entry
before the second allocation refuses. Diagnostic finalizer drainage preserves those owned
storages. `observe-tp-generic-copy` refuses the second copy after both destination storages
allocate. Its hook restores in `finally` before cleanup. Temporary source `Narrow` views unwind
through the existing original-plus-cleanup rollback helper. Explicit model disposal destroys all
actual shard/source storages before finalizer drainage, guarded shutdown and foreign collection.
No mode resets terminal ownership or manually changes reference/storage counts.

The source inventory also includes raw quantized row copies in the generic split, raw
quantized concatenated copies, and separate-source Q8 requantization. Those loops allocate
owned wrappers directly into reserved arrays before writing. The six allocation modes exercise
their F32 routes, not a simulated native malloc failure. `tp-quantized-copy-row`,
`tp-quantized-copy-concatenated` and `tp-quantized-requantize-separate` execute those actual
quantized copy/requantization routes on generated Q4_0 sources. They verify source and destination
ownership fields clear through explicit disposal. Host ownership observations are not a raw-heap
measurement or pretrained quantized-model qualification. Source retirement precedes dictionary
removal; a refused cleanup keeps the model's actual source and destination owners.

`tp-view-unregister-refusal` registers a real mapped view with Bonsai, then poisons the existing
managed owner. Guarded unregister refuses before model cleanup and the backing mapping stays
owned. `observe-tp-view-cleanup-order` poisons in the existing derived-resource callback after
generic GGML caches retire. TP view unregister still precedes backing/mapping retirement, so
its refusal also keeps the actual mapping open. Both modes
check managed ownership and the real failed registration after finalizer drainage. Neither
dereferences a pointer after poisoning or claims an actual GPU fault.

`observe-tp-sync-retirement` supplies a group with a controlled synchronization refusal. GGML
model disposal consults that group before graphs/caches/storage retire. The actual model/source
storage and complete foreign generation remain retained. Repeated disposal preserves the first
failure and does not retry synchronization. `tp-sync-preflight-refusal` separately refuses a
caller preflight, then explicitly cleans up through the successful production synchronization.
Neither mode proves CUDA/MLX drainage. Allocator-wide CUDA arena release skips a borrowed group.
Direct CUDA/MLX retains its existing late TP-view/cache sequence and does not newly release
column backing owners. Its synchronization, per-shard device cache closure and terminal strong
retention remain unqualified; a GGML process-exit root does not prove them.

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
