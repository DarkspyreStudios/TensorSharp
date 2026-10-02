# Windows RTX 3090 Ti CUDA build

The primary Windows CUDA target is the RTX 3090 Ti on `gpu-box`, with compute
capability 8.6 and NVIDIA driver 595.97. Separate builds may support other baselines.

## Build

The source baseline is tag `v2.8.6.7` (`9e36cdd3`), with DSV4 infinity constants
changed to CUDA's `CUDART_INF_F`. Windows CRT `INFINITY` expressions fail with
CUDA 12.8. The CUDA constant preserves float infinity without that CRT expression.

The verified build uses CUDA 12.8.61, MSVC 14.44.35207 and .NET SDK 10.0.201.
Compiler and include paths must select the same MSVC toolset. The installed MSVC
14.51 headers require CUDA 13.2. Selecting only `nvcc --compiler-bindir` does not
select matching headers. Run from an isolated source tree in a Windows command prompt:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
dotnet build TensorSharp.Backends.Cuda\TensorSharp.Backends.Cuda.csproj -c Release -p:CudaArch=compute_86 -p:Version=2.8.6.7-sm86.local -warnaserror
dotnet build eng\CudaCompatibilitySmoke -c Release -warnaserror
dotnet eng\CudaCompatibilitySmoke\bin\Release\net10.0\CudaCompatibilitySmoke.dll
```

The local version identifies an unpublished build. It does not replace a published
NuGet version. The DLL and compiled PTX live under `TensorSharp.Backends.Cuda/bin/`.
Both PTX files contain `.version 8.7` and `.target sm_86`. Their contents come from
compilation; editing a PTX target header does not create a compatible build.
This build branch also carries those compiled files in `native/ptx/` so its
compiler-free builds use the same verified 3090 Ti baseline. Other architectures
require a separate native build variant with an explicit compatible target.

## Deployment GPU verification

The 2026-10-02 build and smoke run on `gpu-box` have zero warnings and errors.
The smoke checks both PTX module loads, GPU fill, GPU elementwise addition and
cuBLAS matrix multiplication. DSV4 argmax checks exercise negative infinity,
positive infinity and tied maxima. Missing main kernels and recorded CPU fallback
operations fail the smoke. The tool uses tiny tensors without loading models or
changing services. Full Qwen inference and throughput are separate checks.

The runtime resolves `cublas64_12.dll` and `cublasLt64_12.dll` from the CUDA 12.8
toolkit `bin` directory. Deployment must provide compatible runtime libraries.
The backend output is not a self-contained application or a compiler toolkit.
The backend dependency audit reports no vulnerable packages.

The output is on `gpu-box` at
`G:\AI\builds\tensorsharp-2.8.6.7-sm86\TensorSharp.Backends.Cuda\bin`.
Build and smoke logs remain in the parent directory. PTX SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `tensorsharp_kernels.ptx` | `0db193c6a51ce1ecd1efaa1e8c162c535554e64759f7765fbd43095fd7405680` |
| `tensorsharp_dsv4_kernels.ptx` | `4c94146456e0ef905bd30395a5fcb5c6d81b0bd6b025de8e38290efa24ee3502` |

The published 2.8.6.7 package still carries incompatible `sm_120` PTX. Applications
must consume a corrected immutable package version to certify package deployment.
This local build does not alter the running arbiter or publish a package.
