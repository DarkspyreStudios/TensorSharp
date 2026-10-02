# Controlled CUDA Ownership

This nonpackable executable exercises actual managed CUDA classes with an instance-local
finite driver implementation. Unconfigured driver calls throw. It never loads CUDA,
cuBLAS, GGML, models or a native bridge. Opaque fixture handles are never dereferenced.
Core and CUDA grant this exact fixture friend access for typed actual-parent effect
composition. No public native driver or callback injection API is added.

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

Module modes exercise actual managed module acquisition, cached function lookup,
context drain before unload, cleanup refusal and constructor rollback. Functions and
module ownership remain retained after unsafe cleanup; later lookup refuses before
publishing even a cached function. Foreign module modes prove managed owner/assembly
collection or retention after actual explicit cleanup. Their PTX bytes and opaque
function handles are controlled inputs, not compiled GPU kernels.

The composed-context modes hold a real parent effect across checked child release.
One same-thread token captures the original current opaque handle before the outer
cleanup gates. Child cleanup does not restore intermediate contexts. The token skips
only original handles actually released by that cleanup and restores unrelated borrowed
handles once after all effects exit. Post-release restoration errors do not quarantine
retired owners. Drain refusal stops later child release and restoration, retaining the
actual parent/context/API graph. The token rejects cross-thread use and repeated restore.
Foreign modes prove actual owner/generation collection or retention outside the frame.
This is context composition, not an allocator, group or model cleanup qualification.
`context-independent-postfault-release` proves known independent cleanup succeeds despite
another recorded device failure. A subsequent arbitrary borrowed-context restoration
refuses under the wildcard policy without changing the completed release or poisoning
the independent device. Its actual released context owner collects.

The `blas-*` modes use the internal known-context `CudaCublasHandle` factory, not the
public standalone factory. They prove owner reservation, context rebinding, checked drain
before destroy, original-plus-cleanup constructor errors, actual retained owner graphs,
and foreign generation collection or retention. Supplied stream pointers are borrowed;
their actual external owners must remain alive through use. These tests do not execute
cuBLAS or qualify the public standalone context-acquisition behavior. That public factory
retains its existing raw path. Allocator construction still calls that path.

The kernel modes exercise actual `CudaKernels` construction and its transferred module,
central launch path, scratch ownership and shared-memory attributes. Failed required-symbol
lookup unloads the transferred module; refused cleanup preserves original and cleanup causes.
Scratch resize drains before freeing the old block. Failed allocation rolls back any returned
OUT pointer. Kernel cleanup drains before scratch free and module unload. Unsafe refusal
retains actual kernel/module/context/API owners and prevents later cache or launch publication.
Ordinary launch and attribute errors remain ordinary and do not publish unsafe cleanup.
Foreign cases check actual owner and collectible-generation roots after successful explicit
cleanup or controlled refusal. Private scratch and attribute boundary observations use fixed
reflection calls inside the fixture; they do not qualify model execution or GPU kernels.
All passed pointers and streams remain borrowed and are never dereferenced by this fixture.

The pool modes use actual small and large `CudaDeviceMemoryPool` caches with controlled
backing callbacks. A refused callback leaves its exact block and byte accounting owned;
successful callbacks remove only completed blocks. Callback replay observes bookkeeping
only. It does not permit repeated native cleanup after recorded CUDA quarantine.

After a focused build, `python3 eng/tests/run-cuda-quarantine.py --configuration Debug
--output tmp/cuda-quarantine/debug` runs the finite controlled modes without building.
Build and run Debug completely before building and running Release because supplier
assemblies use shared output paths. Every mode has a 20-second process bound.

This proves managed ownership routing only. It does not qualify primary-context driver
semantics, GPU execution, physical multi-device cleanup or complete backend integration.
The allocator, storage, captured graph, model and GGML dependency callers are not covered
by these leaf-owner modes.
