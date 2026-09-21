Qwen-Image-2.1 conditioning parity probe
======================================

This diagnostic compares the entire **pre-final-normalization** Qwen3-VL-8B
hidden tensor between TensorSharp's managed operator chain and its fused native
trunk. It requires the real model and an available backend. It does not download
weights or substitute missing device/model scenarios with a passing result.

Build the current TensorSharp native library first, then:

```sh
dotnet build eng/QwenImage21CompanionProbe -c Release
TS_QWEN_TE_FUSED=0 dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-managed.f32 GgmlMetal
TS_QWEN_TE_FUSED=1 dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-fused.f32 GgmlMetal
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  compare docs/validation/qwen-image21/text-managed.f32 \
  docs/validation/qwen-image21/text-fused.f32
```

Use `GgmlCpu` or `GgmlCuda` to test a different available backend. A fused-kernel
fallback is explicitly logged; such a run measures the fallback and cannot
establish fused-kernel parity. The comparison reports relative L2 error and
maximum absolute error, rejects non-finite outputs, and exits unsuccessfully at
relative L2 error of 1% or above. This is a diagnostic tolerance, not a claim of
pixel-level end-to-end equivalence. Generated evidence belongs in ignored
`docs/validation/` or `artifacts/`.
