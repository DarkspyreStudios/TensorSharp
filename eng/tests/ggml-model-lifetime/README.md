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
