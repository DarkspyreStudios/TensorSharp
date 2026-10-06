# MLX Runtime Lifetime Probe

This executable uses a selected prebuilt MLX bridge on macOS ARM64. Each invocation
starts a fresh process. It checks real compiled GELU creation and cached reuse against
the CPU formula, tiny Q8 weight preload and cached matmul reuse against CPU dot
products, explicit weight/tensor/allocator disposal, native active/cache bytes,
checked terminal worker shutdown, repeated shutdown, rejected post-retirement work
and collection of the actual storage owners. The ten finalizer passes bound the
weak-reference observation; the probe never resets ownership or retries native frees.

The executable preloads the explicit library and sets the existing
`TENSORSHARP_MLX_LIBRARY` selector. It records the bridge hash and actual Core/MLX MVIDs.
It loads no model weights, modifies no shared driver and builds no native library.
The selected library's dependencies and `mlx.metallib` must already be present.

The explicit `--collectible` mode loads the actual probe, Core and MLX assemblies
into one private collectible generation. It executes the same native checks,
unloads after checked worker retirement, and observes the actual assembly and
load-context weak roots. Its 30 collection passes and 10 ms waits match the
existing GGML fixture's bounded observation. It never resets native ownership.
The nested native-work result does not establish generation collection. The final
collectible result and process exit code report that separate check. Retained roots
produce a failed final result and exit code 1 after the native checks complete.

The explicit `--collectible-dump` mode performs the same root observation, prints
the process ID and retained names, then waits for one input line before reporting
the result. This pause permits a dump of the actual retired generation. It does
not retain the generation through a strong managed reference or alter cleanup.

JSON output contains only BCL dictionary, array and primitive values. Reflection-based
serialization of a private anonymous type or `MlxMemorySnapshot` retains its loader
through the shared serializer's member-accessor cache and invalidates unload checks.

Build the managed executable with native hooks disabled:

```bash
dotnet build eng/tests/mlx-runtime-lifetime/mlx-runtime-lifetime.csproj -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:TensorSharpSkipCudaNative=true
```

Run against the absolute shared bridge path:

```bash
dotnet eng/tests/mlx-runtime-lifetime/bin/Debug/net10.0/mlx-runtime-lifetime.dll /absolute/shared/driver/libmlxc.dylib
```

Run the same native checks in a collectible generation:

```bash
dotnet eng/tests/mlx-runtime-lifetime/bin/Debug/net10.0/mlx-runtime-lifetime.dll --collectible /absolute/shared/driver/libmlxc.dylib
```

An unsupported platform or missing runtime fails visibly. A successful probe does not
qualify actual synchronization/free failures, concurrent callbacks, full quantized
model caches, multi-device execution, AOT, archive provenance
or package delivery. The quarantine snapshot reports recorded cleanup failures only;
it does not establish native readiness. Memory observations do not establish every
native reference's lifetime. The default mode does not prove foreign-generation
collection. The collectible mode qualifies only this tiny normal native lifetime,
not callbacks during failed retirement, raw external supplier use or full models.
