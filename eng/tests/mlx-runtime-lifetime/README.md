# MLX Runtime Lifetime Probe

This executable uses a selected prebuilt MLX bridge on macOS ARM64. Each invocation
starts a fresh process. It checks real compiled GELU creation and cached reuse against
the CPU formula, explicit tensor and allocator disposal, native active/cache bytes,
checked terminal worker shutdown, repeated shutdown, rejected post-retirement work
and collection of the actual storage owners. The ten finalizer passes bound the
weak-reference observation; the probe never resets ownership or retries native frees.

The executable preloads the explicit library and sets the existing
`TENSORSHARP_MLX_LIBRARY` selector. It records the bridge hash and actual Core/MLX MVIDs.
It loads no model weights, modifies no shared driver and builds no native library.
The selected library's dependencies and `mlx.metallib` must already be present.

Build the managed executable with native hooks disabled:

```bash
dotnet build eng/tests/mlx-runtime-lifetime/mlx-runtime-lifetime.csproj -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:TensorSharpSkipCudaNative=true
```

Run against the absolute shared bridge path:

```bash
dotnet eng/tests/mlx-runtime-lifetime/bin/Debug/net10.0/mlx-runtime-lifetime.dll /absolute/shared/driver/libmlxc.dylib
```

An unsupported platform or missing runtime fails visibly. A successful probe does not
qualify actual synchronization/free failures, concurrent callbacks, quantized model
caches, multi-device execution, foreign-generation collection, AOT, archive provenance
or package delivery. The quarantine snapshot reports recorded cleanup failures only;
it does not establish native readiness. Memory observations do not establish every
native reference's lifetime.
