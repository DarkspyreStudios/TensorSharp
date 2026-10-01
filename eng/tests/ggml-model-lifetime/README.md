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

## Tensor-Parallel Ownership Observations

These modes use two logical ranks on one real CPU or Metal context. They do not execute a
collective or qualify physical multi-device placement. Their `observe-*` assertions reproduce
defects in the tested source; they are not release success gates.

`observe-tp-column-owner` creates an actual owned Q4_0 source and two column views. The source
leaves the ordinary dictionary. The views hold its exact wrapper until model disposal. Model
disposal retires the views without explicitly disposing the source. The source still has host
ownership metadata before its weak reference collects. This observes loss of a disposal owner,
not raw heap liveness. The mode does not read or reclaim a captured pointer after disposal.

Six F32 modes refuse the second real storage allocation after a first real shard allocates:

| Mode | Source Allocation Route |
| --- | --- |
| `observe-tp-generic-column` | Generic output-dimension split |
| `observe-tp-generic-row` | Generic input-dimension split |
| `observe-tp-concatenated` | Concatenated column segments |
| `observe-tp-separate` | Separate source projections, F32 destination branch |
| `observe-tp-concatenated-bias` | Concatenated bias segments |
| `observe-tp-separate-bias` | Separate source biases |

The first shard has an actual native storage weak reference but no model-owned array entry.
Finalizer drainage loses that storage while the source model remains live. This is defect
evidence, not explicit shard cleanup. `observe-tp-generic-copy` instead refuses the second
copy through the existing dispatch hook after both destination storages allocate. Its hook
restores in `finally` before cleanup. Generic column/row/copy modes also expose an undisposed
source `Narrow`: guarded context disposal refuses after the registered source tensor retires.
The retained model/context and eight foreign roots survive finalizer drainage. No mode resets
terminal ownership or manually changes reference/storage counts.

The source inventory also includes raw quantized row copies in the generic split, raw
quantized concatenated copies, and separate-source Q8 requantization. Those loops allocate
unregistered buffers before wrapper/array transfer. The six controlled allocation modes above
exercise their F32 routes, not a simulated native malloc failure. The column-owner mode covers
real quantized borrowing. Bias source removal and all separate/concatenated source retirement
occur outside a failed-destination ownership boundary in the tested source.

`tp-view-unregister-refusal` registers a real mapped view with Bonsai, then poisons the existing
managed owner. Guarded unregister refuses before model cleanup and the backing mapping stays
owned. `observe-tp-view-cleanup-order` poisons in the existing derived-resource callback after
generic GGML caches retire. Model disposal closes the mapping before the TP view unregister
refuses. The actual failed view stays retained, but its mapping is already closed. Both modes
check managed ownership and the real failed registration after finalizer drainage. Neither
dereferences a pointer after poisoning or claims an actual GPU fault.

`observe-tp-sync-retirement` supplies a group with a controlled synchronization refusal. Current
model disposal does not call it and runs a controlled graph-phase marker and real weight
retirement. `tp-sync-preflight-refusal` checks that an explicit caller preflight refuses before
any retirement, then disables only the
test group's refusal for explicit cleanup. The positive preflight does not prove production
CUDA/MLX drainage. Direct CUDA/MLX terminal retention and shared arena/cache policy are not
qualified by a GGML process-exit root or these logical-rank fixtures.
