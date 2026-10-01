# Model Lifetime Fixture

The fixture privately loads the real TensorSharp assemblies in a collectible generation.
Each mode/backend runs in a separate process against an explicit matching native bridge.
The existing model modes generate a tiny F32 HunyuanDense GGUF. The ownership-only quantized
modes generate one valid Q4_0 tensor and do not execute a model forward.

## Quantized Ownership Observations

`observe-raw-quantized-read` and `observe-stacked-quantized-read` capture the destination
of the production fallback read before a test FileStream returns EOF. The raw allocation
has no dictionary owner when loading fails. `observe-stacked-owner-insertion` completes
the raw read and injects a dictionary-comparer exception before the stack owner transfers.
`stacked-partial-view-refusal` instead refuses insertion of the second expert view after
the stack and first view transfer. That case is a positive explicit rollback gate.

Every defined backend selects mmap in current production code. These four modes temporarily
force only the private backend classifier to an undefined sentinel. They preserve the actual
CPU/Metal context and ExecutionPlan and restore the classifier before cleanup. They do not
qualify a supported fallback configuration. Captured allocation and owner-transfer evidence
does not measure raw-buffer liveness. The tests do not read or free a captured pointer after
rollback. Zero GGML lease counts do not prove that BCL NativeMemory allocations were freed.
The current source has no cleanup owner for the untransferred 144/288-byte buffers.

`observe-bonsai-unregister` first registers a real Q2_0 weight at its host and GCHandle keys.
It invokes the existing terminal owner poison hook. The guarded native unregister then
refuses. Current disposal discards both managed keys, host ownership and the GCHandle despite
that refusal. The mode observes managed fields only after poisoning. It does not reset the
owner, bypass the guard, dereference freed memory or qualify an actual GPU/synchronization fault.
Shared model disposal retains its context/Model leases and all foreign roots after refusal.

The `observe-*` modes reproduce defects in their source checkpoint. They are not release
success gates. The fixture does not qualify pretrained or complete quantized models,
whole-native executors, multiple devices, captured CUDA teardown or other operating systems.
