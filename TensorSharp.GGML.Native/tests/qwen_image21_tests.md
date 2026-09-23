Qwen-Image-2.1 native regression tests
====================================

Configure with `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`, build
`GgmlOpsQwenImage21Test` and `GgmlOpsQwenImage21VaeShortcutTest`, then run
`ctest --test-dir <build-directory> -R qwen-image21 --output-on-failure`.

The CPU whole-graph test writes a synthetic reference fixture under the build
directory. The CUDA test compares 171 forwards with that reference, including
changed inputs and shapes, graph reuse, weight invalidation, and forced
graph-owned weights. VAE tests check independent shortcut/convolution oracles
and finite activations above the FP16 range. These are numerical regressions;
they do not establish real-model image quality or performance parity.

On a cuDNN-enabled Windows build, `qwen-image21-vae-missing-cudnn-cuda` runs in a
fresh process with an isolated stub `cudnn64_9.dll` that exports no cuDNN
functions. Its sibling DLL names are fixture copies generated in the build
directory; no installed runtime is modified. This reproduces an incomplete
runtime export table, which must fall back to full-F32 ggml convolution rather
than call a null function pointer. The test verifies the stub was actually
loaded and the complete VAE numerical checks passed. A wholly absent runtime
returns the same unavailable API state, but is not separately simulated.

Run VAE checks with `TS_VAE_CUDNN_CONV=0` for the ggml path and `=1` for vendor
offload when available. Unset selects vendor offload automatically for Qwen21.
`TS_GGML_NODE_PROFILE=1` shows actual vendor calls and fallbacks. CUDA absence
returns skip code 77 and must not be counted as passing device coverage.
