# Qwen CUDA generation proof

This Windows validation tool loads the arbiter's existing TensorSharp assemblies
and runs the hash-table benchmark prompt for 16 tokens with the Cuda backend,
16,384-token context, temperature zero and reasoning disabled. The model keeps
its default KV cache configuration. The report reads the actual cache type from
the loaded model.

Build the diagnostic CUDA backend from this branch with the GPU's compatible
CUDA toolkit and architecture. Build this tool against the arbiter's assemblies:

```powershell
dotnet build eng/QwenCudaProof/QwenCudaProof.csproj -c Release `
  -p:ApplicationDirectory=G:/AI/gpu-arbiter-tensor/smoke `
  -p:CudaBuildDirectory=G:/AI/proof/TensorSharp.Backends.Cuda/bin
```

Run with the application directory, GGUF path and output JSON path as arguments.
Use a runtime compatible with all the application's dependencies. For a
self-contained arbiter, copy the application into an isolated validation folder,
replace its entry DLL with QwenCudaProof.dll, and invoke its existing executable.
Replace both TensorSharp.Backends.Cuda.dll and **all cuda_kernels/*.ptx** with the
diagnostic build's matching files. Preserve the other managed DLLs. Unload any
other model before the test and restore it afterward.

TS_CUDA_PROFILE=1 enables named counters after successful cuLaunchKernel calls
through CudaKernels.Launch. CUDA graph capture can record these calls; subsequent
graph replays are not counted, so this is not a count of every device execution.
Generation counts subtract the snapshot taken after
model loading. These counts include first-request preparation and prefill.
CPU fallback counts cover CudaCpuFallback.Invoke operations; model-specific
managed paths and CPU tokenization/sampling are outside that counter. Inspect
the console for other fallback warnings. A nonzero successful launch count and
generated tokens prove GPU work; loading a CUDA DLL alone does not.

The JSON preserves kernel names, counts, counted CPU fallbacks, generated output,
terminal token metrics and the diagnostic DLL path. The process exits naturally
and prints the full CUDA profile. This run verifies backend execution. Its short
output and enabled profiling make it unsuitable as a throughput comparison.
