# Controlled CUDA Ownership

This nonpackable executable exercises actual managed CUDA classes with an instance-local
finite driver implementation. Unconfigured driver calls throw. It never loads CUDA,
cuBLAS, GGML, models or a native bridge. Opaque fixture handles are never dereferenced.

Build each configuration with all native hooks disabled:

```bash
env TENSORSHARP_GGML_NATIVE_SKIP=true TENSORSHARP_MLX_NATIVE_SKIP=true \
dotnet build eng/tests/cuda-quarantine/cuda-quarantine.csproj -c Debug \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  -p:TensorSharpSkipCudaNative=true -p:GeneratePackageOnBuild=false -p:PublishAot=false
dotnet eng/tests/cuda-quarantine/bin/Debug/net10.0/TensorSharp.CudaQuarantineFixture.dll context-clean
```

Each mode runs in a fresh process. `context-clean` proves explicit primary-reference
release, same-thread current-context rebinding, idempotent disposal and actual managed
context/API collection. It preserves an unrelated current context using a resource-free
wildcard restoration after the known-device lease exits. `context-drain-refusal` proves
checked context drain precedes unbind and primary release: refusal retains the context
and makes zero primary-release calls. `context-release-refusal` preserves the exact original failure,
the actual context handle and owner/API graph after finalizer drainage. Later context
entry and repeated disposal refuse before injected effects. `context-construction-rollback`
preserves the original factory error after successful release. Its `-refusal` counterpart
preserves both original and cleanup errors and records unsafe constructor cleanup.

`foreign-context-clean` and `foreign-context-release-refusal` execute the same actual
context in a private collectible fixture/CUDA/Core generation. The outer frame verifies
the real CUDA MVID and uses weak references to the actual owners, API, assemblies and
ALC. Clean explicit release permits collection. Unsafe release retains those roots.

The stream modes use actual `CudaStream` with either a known owned context or verified
ambient device discovery. Each stream owner reserves its known scope before acquisition.
The ambient association borrows the original context handle; its caller must keep that
external context alive. No primary reference is retained or implicitly transferred.
`stream-clean` and `stream-ambient-clean` prove explicit drain/destroy and rebinding.
The sync/destroy refusal modes retain the handle and actual dependent context, prevent
later native entry, and leave an independently registered known device usable. Stream
construction modes preserve original acquisition and cleanup failures. Foreign stream
modes check actual private CUDA/Core/fixture roots outside the execution frame.

This proves managed ownership routing only. It does not qualify primary-context driver
semantics, GPU execution, physical multi-device cleanup or complete backend integration.
The allocator, storage, graph, model and GGML dependency callers are not covered by these
context and stream modes.
