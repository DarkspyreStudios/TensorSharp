# Qwen-Image-2.1

Qwen-Image-2.1 uses the `QwenImageModel` image pipeline for text-to-image generation
and image editing. It requires a Qwen3-VL-8B text encoder and the dedicated 2.1
VAE. Native RGBA input and PNG output preserve transparency; the Qwen3-VL
conditioning branch composites the reference over white while the VAE keeps alpha.
The Qwen2.5-VL encoder, earlier Qwen-Image VAE, and 2511 Lightning LoRAs
are not interchangeable with these components.

The download configuration is [`config/qwen-image-2.1.json`](../../config/qwen-image-2.1.json).
It pins repository revisions and SHA-256 checksums for new downloads; existing
cached files are reused. The four files total
11,051,668,216 bytes (about 10.29 GiB); this is download size, not peak inference
memory. Runtime memory also includes activations, decoding buffers and working
weights. Larger images and multiple references increase that requirement.

| Component | File | Source |
|---|---|---|
| Diffusion transformer | `qwen_image_2.1_Q4_K_M.gguf` | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF) |
| Dedicated VAE | `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) |
| Text encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| Vision encoder for editing | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | Same Qwen3-VL repository |

## Launch the CLI

Run these commands from the TensorSharp repository root. The configuration
selects `ggml_metal` for Apple Silicon. On an NVIDIA machine with the CUDA backend
built, append `--backend ggml_cuda`; the native CPU backend is `ggml_cpu`.
Missing models download automatically at startup. Set `TENSORSHARP_MODELS` to an
absolute directory to choose where they are stored; files go in its
`qwen-image-2.1` subdirectory. Without this variable, the configuration resolves
`../models` relative to `config/`, giving `<repository>/models/qwen-image-2.1/`.

For the models downloaded in this workspace, set:

```bash
export TENSORSHARP_MODELS="$PWD/../models"
```

Build:

```bash
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
```

Text-to-image:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 1024 --height 1024 --diffusion-steps 40 --cfg 6 \
  --diffusion-seed 42 --output generated.png
```

Image editing:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 1024 --height 1024 --diffusion-steps 40 --cfg 6 \
  --diffusion-seed 42 --output edited.png
```

No `--image` selects generation; one or more `--image` arguments select editing.
Repeat `--image first.png --image second.png` for multiple references in that order.
`--input prompt.txt` can supply the prompt instead. Both modes accept
`--negative-prompt 'blur, low detail'`. Omitted sampling settings select 40 Euler
steps and CFG 6.0 for 2.1. Set width and height together, in multiples of 32.
TensorSharp chooses approximately 1024×1024 when dimensions are omitted, following
the first reference aspect ratio for editing. This is a conservative local default;
the [official model example](https://huggingface.co/Qwen/Qwen-Image-2.1) uses
2048×2048 and 40 steps.

For a quick executable smoke test use 256×256 and one step. Such a run verifies
loading and the end-to-end data path; it does not demonstrate image quality or
performance at the normal 1024×1024/40-step settings.

## Launch TensorSharp.Server.Host

```bash
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. A prompt without an attachment generates an image;
attach one or more images to edit. The page displays denoising progress and the
output download link. Image operations are serialized because the diffusion
pipeline shares mutable working state.

Text-to-image JSON API:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An orange cat beside a blue ceramic vase, soft daylight","width":1024,"height":1024,"steps":40,"cfg":6,"seed":42}'
```

The response is `{ "ok": true, "url": "...", "width": 1024, "height": 1024,
"elapsedSeconds": ... }`. Download the returned URL from the same server.

Multipart image editing:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@generated.png' \
  -F 'prompt=Change the blue vase to red. Preserve the cat and composition.' \
  -F 'width=1024' -F 'height=1024' -F 'steps=40' -F 'cfg=6' -F 'seed=42'
```

Repeat the `image` part for multiple references. Alternatively upload files to
`/api/upload` and send JSON containing `imagePaths` (or legacy `imagePath`) to
`/api/image-edit`. Both image endpoints accept `negativePrompt`, `targetArea`,
`width`, `height`, `steps`, `cfg` and `seed`. `targetArea` controls automatic
geometry; explicit dimensions take precedence.

For progress, use the JSON routes `/api/image-generate/stream` and
`/api/image-edit/stream` with `curl -N`. They emit SSE `data:` frames with
`imageGenerate: true` or `imageEdit: true`, `step` and `total`, optionally a
preview `image` data URL. The terminal frame contains `done: true` and the final
`url`, dimensions and elapsed seconds, or `error`. Chat-completion routes do not
run this diffusion model. Existing `/api/image-edit` requests still require
at least one reference; generation has its own endpoint.

The 2.1 diffusion transformer runs a complete GGML graph with resident quantized
weights. The earlier image-edit `--offload-cpu` streaming path and Lightning LoRAs
do not apply to this implementation. Start with smaller dimensions if available
memory is insufficient. CUDA and Vulkan are selectable GGML backends but have not
been exercised on this local Apple Silicon validation machine.

## Validation and comparison

The final focused suite passed **317 tests**, with **one explicit skip** for a
legacy Qwen-Image DiT full-weight test because its older checkpoint was unavailable.
The skipped scenario is not counted as validation. Coverage includes request and
configuration handling, sampling/layout math, companion compatibility checks
against the downloaded files, RGBA round trips, and regressions in the shared
text/vision code. Evidence: ignored
`docs/validation/qwen-image-2.1/final-focused.trx`.

The real HTTP harness passed **16/16 cases** at 128×128: JSON generation, JSON and
multipart edits, two-reference editing, SSE progress/previews, RGBA uploads and
outputs, repeated-seed determinism, request refusals, and cancellation followed
by a successful generation. `http/report.json` records the results. These short
runs cover endpoint execution, not visual quality or full-resolution performance.

Run the 2.1 request/math/media regressions and downloaded companion metadata checks:

```bash
TENSORSHARP_QWEN21_DIT="$TENSORSHARP_MODELS/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf" \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --filter 'FullyQualifiedName~QwenImage21|FullyQualifiedName~QwenImageAlphaTests|FullyQualifiedName~QwenImageRequestTests|FullyQualifiedName~WebUiChatServiceTests|FullyQualifiedName~UploadRootConfinementTests'
```

The real-companion metadata test reports a skip if its model environment variable
is absent. The tests above do not replace the image-generation benchmarks.

The independent NumPy/native transformer oracle passed **12 cases on CPU and
12 on Metal**, covering text-only, one-reference and multiple-reference layouts,
fused/unfused MLP projections, and flash-attention on/off. This checks small
synthetic graphs, not full-model CPU inference. See
[`eng/tests/qwen-image21-dit.py`](../../eng/tests/qwen-image21-dit.py).

Matched generation and editing runs on an Apple M5 Pro with 48 GiB unified
memory used the configuration's Q4_K_M diffusion/text weights, 512×512 pixels,
20 Euler steps, CFG 6 and seed 42. Both engines ran on Metal. Each row is a
single fresh process; OS file caches were warm and shader compilation was
excluded. These are observations for these workloads.

| Task | Engine | Inference | Process wall time | Process peak RSS |
|---|---|---:|---:|---:|
| Text-to-image | TensorSharp | 84.714 s | 85.22 s | 6.17 GiB |
| Text-to-image | stable-diffusion.cpp | 113.43 s | 113.72 s | 9.81 GiB |
| Image edit | TensorSharp | 175.645 s | 176.00 s | 7.53 GiB |
| Image edit | stable-diffusion.cpp | 212.80 s | 213.06 s | 10.71 GiB |

TensorSharp inference was 1.34× as fast for generation and 1.21× as fast for
editing in these runs. The generated teapot images were visually close;
comparing RGB after compositing over white gave MAE 0.2606 and RMSE 0.5838 on
the 0–255 scale.

The editing run used the same reference file, `sd-t2i-512.png`, in both engines
and this instruction: “Change the red teapot to cobalt blue. Keep its shape,
lighting, the wooden table and background unchanged.” Both outputs showed a
blue teapot with the scene preserved. Raw RGBA comparison gave MAE 0.620672
and RMSE 1.12041; white-composited RGB gave MAE 0.825444 and RMSE 1.292416,
on the 0–255 scale. TensorSharp spent 1.664 s encoding the reference VAE,
6.599 s encoding text/vision, 162.564 s denoising and 4.817 s decoding.

Pixel agreement is not a general prompt-adherence score. Logs, PNGs and comparison
images are under ignored `docs/validation/qwen-image-2.1/` with the stems
`ts-t2i-512`, `sd-t2i-512`, `ts-edit-512` and `sd-edit-512`. These 512-pixel runs
do not establish quality or speed at the official 2048-pixel example, across
arbitrary prompts, or on other devices.

A separate TensorSharp quality run completed at **1024×1024, 40 Euler steps,
CFG 6, seed 42**, producing an RGBA PNG. Its prompt described a rainy-evening
bookstore with warm windows and a sign reading “TENSORSHARP.” Visual inspection
confirmed the main sign's lettering, detailed shelves and architecture, warm
lighting and reflections on wet pavement. Inference took **751.156 s** (751.94 s
process wall time) with **17.864 GiB** peak RSS; text encoding took 0.443 s,
denoising 715.220 s and VAE decoding 35.493 s. The output and log are
`docs/validation/qwen-image-2.1/ts-t2i-1024-40.png` and
`docs/validation/qwen-image-2.1/ts-t2i-1024-40.log`. This is one prompt/run;
no matched 1024×1024 stable-diffusion.cpp benchmark was performed.

Dependency revisions for this local comparison:

- TensorSharp's unchanged ggml: `456172ec733a135778adcd32d00e576a58232e45`.
- stable-diffusion.cpp: `c678dfe704a2230342376b46add9c8ca736a653d`.
- llama.cpp source studied: `ce8caa6e60a03093351d6016a818720e0d46f0fb`.
- ComfyUI-GGUF source studied: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.

The stable-diffusion.cpp reference used its supported `SD_USE_UPSTREAM_GGML=ON`
configuration with TensorSharp's unchanged ggml checkout. Its default local ggml
submodule did not match the current sd.cpp sources and failed to build; no
reference-tree files were changed. Upstream mode disables tensorwise INT8 and
convrot paths. The benchmark uses Q4_K_M/BF16 weights, so it does not compare
every available stable-diffusion.cpp build or optimization.

ComfyUI-GGUF's loading code was inspected; full ComfyUI was unavailable and no
ComfyUI end-to-end benchmark was run. llama.cpp supplied conditioning/GGUF
reference behavior, not a standalone image-generation benchmark.

For reproducible performance comparisons, record hardware, backend and dependency
revisions; model hashes; output dimensions; reference images; seed; sampler;
step count; CFG; cache settings; and whether weight loading and first-run shader
compilation are included. Compare both cold end-to-end latency and repeated
inference with the same loaded process. Identical seeds across engines do not
guarantee identical initial noise; compare images as well as timing.

The reusable [benchmark runner](../../eng/validation/qwen-image21-bench.py)
launches both engines serially with matched settings and records commands,
model hashes, revisions, phase timings, peak RSS and image diagnostics:

```bash
python3 eng/validation/qwen-image21-bench.py \
  --models-dir "$TENSORSHARP_MODELS/qwen-image-2.1" \
  --width 512 --height 512 --steps 20 --cfg 6 --seed 42 \
  --prompt 'A red ceramic teapot on a wooden table, soft daylight, product photograph.'
```

Add `--mode edit --image reference.png` and an editing prompt for a matched
edit, `--repeat 3` for repeated fresh-process measurements, or `--dry-run` to
inspect commands. The default reference binary is
`artifacts/qwen-image-2.1/sd-build/bin/sd-cli`; override `--sd-cli` when needed.
Image diagnostics use NumPy and Pillow.

The reference implementations are
[stable-diffusion.cpp's 2.1 guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md),
[llama.cpp](https://github.com/ggml-org/llama.cpp) for Qwen3-VL/GGUF, and
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) for quantized tensor loading.
The model card's ComfyUI workflows also provide generation and editing recipes.
Generated logs, benchmark records and output images belong under ignored
`docs/validation/` or `artifacts/`. A unit test, low-resolution smoke image or
unavailable device scenario is not evidence of full-resolution quality or speed
parity; no such parity claim follows from the commands above.
