# TensorSharp project status

This page keeps repository-level status and longer explanations that do not belong in the README.

## Current direction

TensorSharp is a native .NET 10 inference engine for GGUF models. The current source includes CLI, server/Web UI, compatible HTTP APIs, AgentHost, and the TensorAgent iOS/iPadOS application. AgentHost and TensorAgent are source-first capabilities: the latest tagged release may not contain them yet.

### Newest architectures

Two families landed after the last release tag, and both carry limits worth
knowing before you plan around them.

- **DeepSeek V4.1 Flash (`deepseek41`)** — a dedicated native V4.1 graph with an
  optional vision companion. `ggml_cuda` is the serving backend; `ggml_cpu`
  runs the same graph on scalar fallbacks as a correctness path, `ggml_vulkan`
  and `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`, and every other backend
  refuses the checkpoint rather than loading V4.1 weights into the V4 graph. The
  Q2_K release needs a prepared Engram sidecar before it will run at all.
  Multi-GPU means a layer split; routed-MoE tensor parallelism exists behind
  `TS_DSV41_TP` and has measured slower than the split. Concurrent requests get
  isolated slots but fall back to per-slot forwards, so concurrency is not
  batched GPU throughput yet, and there is no V4.1 DSpark. What is measured, and
  what is explicitly not, is tracked in the
  [validation report](deepseek41_validation.md) beside the
  [model card](models/deepseek41.md).
- **Hunyuan Dense (`hunyuan-dense`)** — Tencent's dense Hunyuan decoders, added
  so the official Hy-MT2 GGUFs load instead of failing on an unregistered
  architecture. First cut: text only, single device, generic per-op path, no
  tools and no thinking. See the [model card](models/hunyuan-dense.md).

### TensorAgent and iOS

TensorAgent is a .NET MAUI iOS/iPadOS application that runs the TensorSharp engine locally. It links the native GGML library as an iOS `.xcframework`, uses `ggml_metal` on physical devices, and shares the host-neutral chat pipeline (`TensorSharp.Chat`) with the CLI and server. The iOS target is enabled with `TensorSharpIosTargets=true`; it is not a separate numerical backend or a remote inference service.

The app includes on-device model downloads, saved conversations, attachments, dictation, Agent Skills, and bounded in-process agent tools. iOS does not support ASP.NET Core runtime hosting or child processes, so TensorAgent uses an in-process loopback server and runtime-backed shell/Python/JavaScript integrations. What it shares with the desktop is the API, not the page: the app ships its own phone-shaped UI bound to the same `WebUiChatService` and `SkillsService` routes.

Three phone-specific constraints shape the current implementation:

- **The screen goes away mid-answer.** Generation is owned by a host-side manager rather than the WebView, because WebKit suspends a page whose view has left the window. A turn continues across an app switch and the page re-attaches to it.
- **The first message of a launch would otherwise pay for the whole shared prefix.** The model state at the end of the prompt every chat shares is checkpointed to disk per model (`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`) and restored at admission, so a launch's first message costs a restore rather than a full prefill.
- **iOS jetsam kills without a stack or a message.** `EngineMemoryPolicy` sizes what the engine holds against what jetsam actually charges — wired file pages count against the device, not the process — and a memory warning releases what only serves the next request's speed.

See the [TensorAgent README](../TensorAgent/README.md) for build, simulator, device, packaging, and test instructions, including the measured numbers behind those three points.

## Make It Fast

The short version is:

1. Pick a step-distilled checkpoint for Wan, or the Lightning LoRA for Qwen-Image-Edit.
2. Match the backend to the hardware: `ggml_cuda` for NVIDIA, `ggml_metal` for Apple Silicon and iOS, and `ggml_cpu` for native CPU.
3. Reduce resolution, frame count, or diffusion steps before changing advanced flags. MiniMax-H3's practical point is `--cfg 1.0` with 4–8 steps.
4. For text workloads, try speculative decoding, CPU MoE offload, or `--tp N` only when the model and workload benefit.

For measurements and caveats, read the [engine comparison report](engine_comparison_report.md), [ggml_metal vs llama.cpp on Apple Silicon](perf/metal-vs-llama-cpp.md), [model cards](models/README.md), [feature guide](../FEATURES.md), and [environment-variable matrix](env_var_feature_matrix.md).

## Where details live

- [Getting started](../README.md#quick-start) — first run and backend selection.
- [Compute backends](../USAGE.md#compute-backends) — capabilities, build requirements, and fallbacks.
- [Agent Skills and agentic work](agent_skills.md) — skills, tools, workspaces, and security.
- [TensorAgent](../TensorAgent/README.md) — iOS application architecture and verification.
- [Development guide](../DEVELOPMENT.md) — project layering and native builds.
