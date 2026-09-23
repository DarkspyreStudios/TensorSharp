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
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
```

Image editing:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
```

No `--image` selects generation; one or more `--image` arguments select editing.
Repeat `--image first.png --image second.png` for multiple references in that order.
`--input prompt.txt` can supply the prompt instead. Omitted sampling settings
select **40 Euler steps and CFG 1.0**, following
[Qwen's recommended unguided sampling](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage21.md).
CFG 1 runs one transformer prediction per step; the previous CFG 6 default ran
both positive and negative predictions. Explicit CFG above 1 still enables the
second prediction and applies `--negative-prompt 'blur, low detail'`. Negative
prompts have no effect at CFG 1.

Omitting dimensions selects **2048×2048 for generation**, or approximately the
same pixel area with the first reference's aspect ratio for editing. Set width
and height together, in multiples of 32, to override this. The model supports
[native 2K aspect ratios](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios).
Reference images are conditioned at approximately 1 megapixel each, or the
output area if smaller; increasing the output to 2K does not also quadruple each
reference's VAE, vision-encoder and transformer workload.

The schedule now follows the
[official scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json):
exponential dynamic shifting with the 256/0.5 and 8192/0.9 sequence-length/shift
anchors, followed by terminal stretching to 0.02 and a final Euler step to zero.
This replaces the earlier Flux-derived 4096/1.15 schedule, so existing seeds can
produce different images after this correction.

For faster drafts, specify `--width 1024 --height 1024`. A 2K square has four
times the latent image tokens and more attention work than a 1K square.
`--diffusion-steps 25 --cfg 1` is an optional faster profile used by the official
ComfyUI [generation](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
and [editing](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)
workflows. Forty steps remains the default; fewer steps are a quality/speed
tradeoff, not a claim of equivalent image quality.

No compatible Qwen-Image-2.1 acceleration LoRA was verified in the September 20,
2026 research. The published
[Qwen-Image Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
and [Edit-2511 Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
adapters target earlier architectures and are rejected by this pipeline. The
CFG 1 speed improvement uses the released 2.1 checkpoint directly.

For a quick executable smoke test use 256×256 and one step. Such a run verifies
loading and the end-to-end data path; it does not demonstrate image quality or
performance at the default 2048×2048/40-step settings.

## Launch TensorSharp.Server.Host

```bash
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. A prompt without an attachment generates an image;
attach one or more images to edit. The page displays denoising progress and the
output download link. Image operations are serialized because the diffusion
pipeline shares mutable working state.

Existing Unsloth downloads can be passed directly after rebuilding the host:

```bash
TensorSharp.Server.Host/bin/TensorSharp.Server.Host \
  --model ~/work/models/qwen-image-2.1-unsloth/qwen-image-2.1-Q8_0.gguf \
  --qwen-image-vae ~/work/models/qwen-image-2.1-unsloth/qwen_image_2.1_vae_bf16.safetensors \
  --qwen-image-vl ~/work/models/qwen-image-2.1-unsloth/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --qwen-image-mmproj ~/work/models/qwen-image-2.1-unsloth/mmproj-BF16.gguf \
  --backend ggml_metal
```

The loader recognizes metadata-free diffusion GGUFs by their tensor layout,
including the `model.diffusion_model.` prefix. The dedicated 2.1 VAE accepts
both original Wan names and Diffusers names with spatial convolution kernels;
the adapter preserves the stored weights. No model-file conversion is required.

Text-to-image JSON API:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An orange cat beside a blue ceramic vase, soft daylight","width":2048,"height":2048,"steps":40,"cfg":1,"seed":42}'
```

The response is `{ "ok": true, "url": "...", "width": 2048, "height": 2048,
"elapsedSeconds": ... }`. Download the returned URL from the same server.

Multipart image editing:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@generated.png' \
  -F 'prompt=Change the blue vase to red. Preserve the cat and composition.' \
  -F 'width=2048' -F 'height=2048' -F 'steps=40' -F 'cfg=1' -F 'seed=42'
```

Repeat the `image` part for multiple references. Alternatively upload files to
`/api/upload` and send JSON containing `imagePaths` (or legacy `imagePath`) to
`/api/image-edit`. Both image endpoints accept `negativePrompt`, `targetArea`,
`width`, `height`, `steps`, `cfg` and `seed`. `targetArea` controls automatic
geometry; explicit dimensions take precedence. Omitting `width`, `height`,
`targetArea`, `steps` and `cfg` selects the model defaults above. `targetArea: 1048576`
selects approximately 1K output while retaining automatic aspect-ratio selection.

For progress, use the JSON routes `/api/image-generate/stream` and
`/api/image-edit/stream` with `curl -N`. They emit SSE `data:` frames with
`imageGenerate: true` or `imageEdit: true`, `step` and `total`, optionally a
preview `image` data URL. The terminal frame contains `done: true` and the final
`url`, dimensions and elapsed seconds, or `error`. Chat-completion routes do not
run this diffusion model. Existing `/api/image-edit` requests still require
at least one reference; generation has its own endpoint.
Previews decode the estimated clean latent from the current flow prediction.

The 2.1 diffusion transformer runs a complete GGML graph with resident quantized
weights. The earlier image-edit `--offload-cpu` streaming path and Lightning LoRAs
do not apply to this implementation. Start with smaller dimensions if available
memory is insufficient. CUDA and Vulkan are selectable GGML backends but have not
been exercised on this local Apple Silicon validation machine.

## Current Unsloth Q8_0 validation

The metadata-free Unsloth Q8_0 diffusion model and the three companion files in
the direct launch command above passed generation and editing on Apple M5 Pro,
48 GiB unified memory, macOS 27.0. Both engines used Metal, Euler, CFG 1, seed 42,
matching F32 sigma vectors and Philox noise, in serial fresh processes.

| Workload | TensorSharp wall time | stable-diffusion.cpp wall time |
|---|---:|---:|
| 1024×1024 generation, 40 steps | 337.654 s | 383.080 s |
| 512×512 color-change edit, 25 steps | 101.664 s | 114.186 s |

TensorSharp used 11.9% and 11.0% less wall time respectively in these single-run
comparisons. The generation images were visually close (48.38 dB RGB PSNR after
white compositing); both editing outputs changed the teapot to blue. This does
not establish broad quality or performance superiority. TensorSharp's 1K VAE
decode remained slower (13.319 s versus 7.270 s), and peak process RSS was higher
(16.50 GiB versus 13.29 GiB). File-cache and thermal state were uncontrolled.

The exact files passed 17/17 real-server HTTP cases, including previews,
multi-reference editing, cancellation and recovery. The managed suite passed
265 tests with two explicit skips for other unavailable model fixtures; the
native suite passed all five CTests, with zero skips. A 2048×2048 single-step
execution check completed in 136.092 s with 34.83 GiB peak process RSS and
48.45 GiB macOS peak memory footprint. It is not a full-quality 2K measurement.
Neither memory metric is dedicated GPU allocation.

Both benchmark builds used unchanged ggml
`179b60f27b1019d42da01ac532cabdb8f73ba8b7`; stable-diffusion.cpp was
`c92d73c408515c94beef32161bb5960764fde7a0`. Commands, model/binary hashes,
images, numerical checks and limitations are recorded in ignored
`docs/validation/qwen-image21-unsloth/REPORT.md`. CUDA and Vulkan were not tested.

## Earlier Q4_K_M full-model performance

On this Apple M5 Pro/48 GiB machine, the Q4_K_M model generated the same bookstore
prompt at 1024×1024, 40 steps and seed 42 in **353.120 seconds** (353.734 seconds
process wall time), compared with the historical **751.156 seconds** below.
This is a **2.13× inference speedup for this workload**. Denoising took 334.695
seconds, VAE decoding 18.117 seconds, and peak process RSS was 16.69 GiB.

Both runs used the same model files, resolution, prompt and step count on Metal,
but the current run uses the recommended CFG 1 and corrected Qwen scheduler;
the historical run used CFG 6 and the Flux-derived schedule. This measures the
combined change in defaults and implementation, not isolated kernel speed or
equivalent output pixels. Each result is one fresh process with warm OS file
caches; thermal state was uncontrolled. Visual inspection of the new PNG found
correct “TENSORSHARP” lettering, detailed shelves, warm lighting and wet-pavement
reflections. One image is not a general quality score.

That run's commands, model hashes, dependency revisions, phase timings,
memory measurement and RGBA PNG are in ignored
`docs/validation/qwen-image-2.1/performance/quality-1024-40/`.

A real **2048×2048, 25-step, CFG 1** run with the same prompt and seed also
completed, producing a native-resolution RGBA PNG. It took **1548.719 seconds**
for inference (**1551.161 seconds / 25m 51s process wall time**): 1438.921 seconds
denoising and 109.483 seconds decoding. Peak process RSS was **32.30 GiB**;
macOS also reported a 57.33 GiB peak memory footprint. Neither measure is a
dedicated GPU-allocation measurement. This run demonstrates that native 2K
works here, while also showing its substantial time and memory cost.

Visual inspection at full resolution found correct main-sign lettering,
detailed brickwork and window frames, warm interior lighting and wet-pavement
reflections. Small interior details remain synthesized; this single prompt does
not establish general quality superiority over the 1K/40-step image. The PNG,
log and benchmark manifest are in
`docs/validation/qwen-image-2.1/performance/quality-2048-25/`.
The default 2K/40-step run, full-resolution editing, and CUDA/Vulkan generation
were not run in this validation and are not counted as passing scenarios.

That earlier Release build and focused suite passed **169 tests, zero skipped**,
including real companion-file metadata, automatic/explicit output geometry,
reference geometry, official 1K/2K sigma golden vectors, CPU VAE primitives,
RGBA handling, request parsing, Web UI service and upload-confinement regressions.
The legacy `QwenImageDiTWeightDtypeTests` GPU-forward test was outside this focused
run. Evidence is `docs/validation/qwen-image-2.1/performance/final-managed.trx`;
native operator coverage is detailed below.

## Native optimization validation

CUDA and Metal retain up to two complete DiT graphs for the positive/negative
CFG layouts. `TS_QWEN21_GRAPH_REUSE=1` is the default on both backends; `0`
rebuilds the graph for each prediction. Reuse retains graph metadata and scratch
allocations while refreshing all dynamic inputs. Weight invalidation, cache
clearing and scratch release retire the graphs. `TS_QWEN21_GRAPH_TRACE=1` logs
builds and replay counts. Scratch is released before final VAE decoding.

The Metal VAE uses F32 MPS convolutions, with direct F32 ggml convolution for
unsupported vendor shapes. This preserves activations above the FP16 range:
upstream Metal matrix multiplication can narrow F32 operands internally even
when accumulation is F32. `TS_VAE_MPS_CONV=0` selects the direct convolution
baseline. TensorSharp releases its MPS graphs and staging buffers on explicit
scratch release and backend shutdown. Upstream ggml sources remain unchanged.

Metal reuse passed **159 native forwards** against the CPU explicit-attention
reference and **16 independent NumPy cases / 80 forwards**, including multiple
reference images and flash-attention tile boundaries. The maximum normalized
native error was 0.00054533; the maximum NumPy absolute error was 0.00140832,
within the existing 0.002 tile-boundary tolerance. These are synthetic numerical
checks, not downloaded-model quality or performance measurements. The unchanged
ggml revision for these checks was `179b60f27b1019d42da01ac532cabdb8f73ba8b7`;
evidence is in ignored `docs/validation/qwen21-native-metal-report.md`.

CPU and Metal image attention now uses each segment's exact key/value length
without a dense padding mask. Causal text masks are retained. Metal casts the
strided key/value tensors directly to F16, and supported backends use upstream
fused SwiGLU to avoid intermediate feed-forward copies. These operations preserve
the mathematical computation, with possible floating-point rounding differences;
other backends retain the padded attention path pending device validation.
The earlier attention optimization measurements below used unchanged ggml at
`456172ec733a135778adcd32d00e576a58232e45`.

The independent NumPy transformer oracle passed **16 cases on CPU and 16 on
Metal**, each with five forwards. It covers generation, editing, multiple
references, fused/separate MLP weights, flash/explicit attention, changed latent
inputs, shape shrink/restore, and attention tile boundaries. Maximum absolute
error was 0.000005282 on CPU and 0.001409 on Metal. The larger boundary fixture
uses a 0.002 Metal tolerance because the unchanged baseline already has 0.001333
error from its half-precision matmuls; the existing small-case tolerances remain
0.001 on Metal and 0.0001 on CPU.

On Apple M5 Pro/Metal, a synthetic two-layer F32 transformer with 16,384 target
tokens, 64 prefix tokens, hidden size 256 and two 128-wide attention heads had
median warm forward latency **194.34 ms before → 167.81 ms after** (13.65% lower).
Each version ran in a separate process with five measurements after the first
forward. This geometry eliminates a 520 MiB image-mask allocation per prediction;
that is the calculated allocation size, not a measured reduction in peak RSS.
These timings exclude the full model, conditioning, scheduler and VAE and do not
establish full-resolution generation speed or visual quality. CUDA and Vulkan
were unavailable and are not counted as passing validation.

Run the oracle and optional synthetic benchmark with:

```bash
python3 eng/tests/qwen-image21-dit.py --backend cpu
python3 eng/tests/qwen-image21-dit.py --backend metal
python3 eng/tests/qwen-image21-dit.py --backend metal --benchmark-target-tokens 16384
```

The benchmark mode reports timings separately and does not count as an oracle
test. Logs, the before/after measurements and dependency metadata are under
ignored `docs/validation/qwen-image-2.1/performance-native-20260920/`.

The 2.1 VAE also routes spatial attention through native matrix operations,
tiling queries while every query still attends to every key. Each score tile is
bounded to 16 MiB, supporting the VAE's 768/1152-channel attention heads without
requiring a flash-attention kernel for those widths. The managed implementation
remains the fallback; `TS_QWEN21_VAE_ATTN=0` selects it for comparison.

Large VAE CPU normalization passes now visit contiguous spatial tiles, and SiLU
uses parallel ranges. Twelve scalar-oracle tests pass with bit-exact outputs,
including real channel counts, tile/chunk boundaries and extreme inputs. Resident
DiT weights are released before final VAE decoding to reduce memory pressure at
2K. These changes also retain the small-array path for previews.

[`eng/tests/qwen-image21-vae-attention.py`](../../eng/tests/qwen-image21-vae-attention.py)
passed **16 numerical cases and seven invalid-argument cases on each of CPU and
Metal**, including actual head widths, partial query tiles, uniform attention,
large logits, input/shape reuse and output guard regions. Maximum absolute error
was 0.000006323 on CPU and 0.002377 on Metal; maximum relative L2 error was
0.000000705 and 0.000891 respectively. Metal uses an explicit 0.003 absolute and
0.002 relative-L2 tolerance for upstream half-precision matrix operands with F32
accumulation. These are operator checks, not a VAE image-quality or speed
benchmark. Evidence is under ignored
`docs/validation/qwen-image-2.1/vae-attention-{cpu,metal}.{json,log}`.

## Historical validation and comparison

The results below predate the current 2K/CFG 1 defaults, official scheduler
correction and native optimizations. Image-generation measurements used CFG 6
and the earlier Flux-derived schedule. They describe those historical outputs
and workloads, not current-default performance or quality.

The earlier focused suite passed **317 tests**, with **one explicit skip** for a
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

## Reproduce comparisons with the current implementation

The reusable [benchmark runner](../../eng/validation/qwen-image21-bench.py)
launches both engines serially with matched request settings and records commands,
model hashes, revisions, phase timings, peak RSS and image diagnostics. The
command below uses the current sampling settings; reproducing the historical
images above requires their earlier implementation and schedule. Older sd.cpp
revisions use the Flux-derived schedule, so matching command-line settings alone
does not establish matching sigma schedules or image quality:

```bash
python3 eng/validation/qwen-image21-bench.py \
  --models-dir "$TENSORSHARP_MODELS/qwen-image-2.1" \
  --width 1024 --height 1024 --steps 40 --cfg 1 --seed 42 \
  --prompt 'A red ceramic teapot on a wooden table, soft daylight, product photograph.'
```

Add `--mode edit --image reference.png` and an editing prompt for an edit,
`--engine tensorsharp` to run only TensorSharp without a reference binary,
`--repeat 3` for repeated fresh-process measurements, or `--dry-run` to
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
