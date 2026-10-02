# Windows GGML CUDA compatibility smoke

This tool checks a Windows GgmlOps.dll against the application's existing
TensorSharp managed assemblies. It requires native CUDA backend initialization
and confirms that the backend identifies itself as CUDA. It checks fill, addition
and matrix multiplication against known results. An optional GGUF argument runs
the hash-table prompt for 16 tokens through ModelService with ggml_cuda, Q8_0 KV,
a 16,384-token context, temperature zero and reasoning disabled.

```powershell
dotnet build eng/GgmlCudaCompatibilitySmoke/GgmlCudaCompatibilitySmoke.csproj `
  -c Release -p:ApplicationDirectory=G:/AI/gpu-arbiter-tensor/smoke -warnaserror
```

Run with the application directory and optional GGUF path. All application
runtime dependencies must be compatible with the host runtime. To reuse a
self-contained arbiter's runtime, copy the application into an isolated test
folder, replace its entry DLL with GgmlCudaCompatibilitySmoke.dll, and replace
GgmlOps.dll with the bridge under test. Invoke the copied executable with the
application directory and optional model path. Preserve the remaining managed
DLLs. Make CUDA runtime dependencies available on PATH.

The bridge's Windows build uses the TensorSharp.GGML.Native CMake project:

```powershell
cmake -S TensorSharp.GGML.Native -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON `
  -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF `
  -DTENSORSHARP_GGML_NATIVE_ENABLE_NCCL=OFF `
  -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON
cmake --build build --target GgmlOps GgmlOpsFlashAttnGuardTest --parallel 8
```

The reference 3090 Ti build uses CUDA 12.8, MSVC 14.44 x64 and unchanged upstream
GGML revision 456172ec733a135778adcd32d00e576a58232e45. Import that MSVC toolset's
full environment before CMake configuration. The repository's eng/fetch-ggml.ps1
accepts the revision through TENSORSHARP_GGML_GIT_REF.

GgmlOps.dll embeds GGML's CUDA backend. It is a separate native artifact from
TensorSharp.Backends.Cuda.dll and its PTX modules. Consumers select ggml_cuda.
Replacing a bridge under test does not update a published NuGet package.

Unload other models before the optional model test. Restore them afterward.
The native GgmlOpsFlashAttnGuardTest covers CPU reference attention and CUDA
attention, including supported shapes and deliberate guarded shape fallbacks.
The compatibility smoke establishes loading and correctness. It does not measure
throughput, prove compatibility with every GPU or qualify every model family.
