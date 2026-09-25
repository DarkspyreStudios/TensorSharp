# Qwen-Image-2.1

[← 返回模型索引](README_zh-cn.md) | [English](qwenimage21.md)

Qwen-Image-2.1 通过 `QwenImageModel` 图像流水线完成文生图与图像编辑。它需要
Qwen3-VL-8B 文本编码器和专用的 2.1 VAE。原生 RGBA 输入与 PNG 输出会保留透明度；
Qwen3-VL 条件分支把参考图合成到白色背景上，而 VAE 保留 alpha 通道。更早的
Qwen-Image / Qwen-Image-Edit 检查点（例如 Qwen-Image-Edit-2511）已不再受支持，
加载时会被拒绝（退出码 2）；`--qwen-image-lora` 与 `--offload-cpu` 选项已移除。

下载配置为 [`config/qwen-image-2.1.json`](../../config/qwen-image-2.1.json)。
它固定了仓库修订版本，并为新下载的文件校验 SHA-256；已缓存的文件会直接复用。
四个文件合计 11,051,668,216 字节（约 10.29 GiB）；这是下载大小，而不是推理时的
峰值内存。运行时内存还包括激活值、解码缓冲区和工作权重。更大的图像和多张参考图
都会提高内存需求。

| 组件 | 文件 | 来源 |
|---|---|---|
| 扩散 Transformer | `qwen_image_2.1_Q4_K_M.gguf` | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF) |
| 专用 VAE | `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) |
| 文本编码器 | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| 编辑用视觉编码器 | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 同一个 Qwen3-VL 仓库 |

## 启动 CLI

以下命令都在 TensorSharp 仓库根目录下运行。该配置在 Apple Silicon 上选择
`ggml_metal`。在已构建 CUDA 后端的 NVIDIA 机器上，追加 `--backend ggml_cuda`；
原生 CPU 后端为 `ggml_cpu`。缺失的模型会在启动时自动下载。把 `TENSORSHARP_MODELS`
设为一个绝对路径即可选择存放位置，文件会放在它的 `qwen-image-2.1` 子目录中。
不设置该变量时，配置会相对 `config/` 解析 `../models`，即
`<仓库>/models/qwen-image-2.1/`。

对于下载到本工作区的模型，设置：

```bash
export TENSORSHARP_MODELS="$PWD/../models"
```

构建：

```bash
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
```

文生图：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
```

图像编辑：

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
```

不带 `--image` 时执行生成；带一个或多个 `--image` 参数时执行编辑。重复
`--image first.png --image second.png` 可按该顺序传入多张参考图。也可以用
`--input prompt.txt` 提供提示词。省略采样设置时使用 **40 步 Euler、CFG 1.0**，
遵循 [Qwen 推荐的无引导采样](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage21.md)。
CFG 1 每一步只运行一次 Transformer 预测；此前的默认值 CFG 6 会同时运行正向和
负向两次预测。显式设置大于 1 的 CFG 仍会启用第二次预测，并应用
`--negative-prompt 'blur, low detail'`。在 CFG 1 下，负向提示词不起作用。

省略尺寸时，**生成默认为 2048×2048**；编辑则使用与第一张参考图宽高比一致、
像素面积大致相同的尺寸。要覆盖此行为，请同时设置宽度和高度，且都取 32 的倍数。
该模型支持[原生 2K 宽高比](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios)。
每张参考图以约 1 百万像素（若输出面积更小，则以输出面积）作为条件输入；把输出
提高到 2K 并不会同时把每张参考图的 VAE、视觉编码器和 Transformer 工作量翻四倍。

调度现在遵循
[官方调度器配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json)：
使用指数动态偏移（序列长度/偏移锚点为 256/0.5 与 8192/0.9），随后做终端拉伸到
0.02，并以最后一个 Euler 步走到零。它取代了此前源自 Flux 的 4096/1.15 调度，
因此在这一修正之后，相同的种子可能生成不同的图像。

需要更快的草图时，指定 `--width 1024 --height 1024`。2K 正方形的潜在图像 token
数是 1K 正方形的四倍，注意力计算也更多。`--diffusion-steps 25 --cfg 1` 是一个
可选的更快配置，官方 ComfyUI
[生成](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
与[编辑](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)
工作流使用的就是它。默认仍为 40 步；更少的步数是质量与速度之间的权衡，并不意味着
图像质量相当。

Qwen-Image-2.1 不加载 LoRA 适配器。CFG 1 带来的提速直接来自已发布的 2.1 检查点。

快速冒烟测试可以用 256×256、一步。这样的运行只验证加载和端到端数据通路，
并不能说明默认 2048×2048/40 步设置下的图像质量或性能。

## 启动 TensorSharp.Server.Host

```bash
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

打开 `http://127.0.0.1:5000`。不带附件的提示词会生成图像；附加一张或多张图像即可
编辑。页面会显示去噪进度和输出文件的下载链接。由于扩散流水线共享可变的工作状态，
图像操作会串行执行。

重新构建主机后，可以直接传入已有的 Unsloth 下载文件：

```bash
TensorSharp.Server.Host/bin/TensorSharp.Server.Host \
  --model ~/work/models/qwen-image-2.1-unsloth/qwen-image-2.1-Q8_0.gguf \
  --qwen-image-vae ~/work/models/qwen-image-2.1-unsloth/qwen_image_2.1_vae_bf16.safetensors \
  --qwen-image-vl ~/work/models/qwen-image-2.1-unsloth/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --qwen-image-mmproj ~/work/models/qwen-image-2.1-unsloth/mmproj-BF16.gguf \
  --backend ggml_metal
```

加载器通过张量布局识别不带元数据的扩散 GGUF，包括带 `model.diffusion_model.`
前缀的文件。专用 2.1 VAE 同时接受原始 Wan 命名和带空间卷积核的 Diffusers 命名；
适配器保留存储的权重，无需转换模型文件。

### HTTP API

文生图 JSON API：

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An orange cat beside a blue ceramic vase, soft daylight","width":2048,"height":2048,"steps":40,"cfg":1,"seed":42}'
```

响应为 `{ "ok": true, "url": "...", "width": 2048, "height": 2048,
"elapsedSeconds": ... }`。请从同一台服务器下载返回的 URL。

Multipart 图像编辑：

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@generated.png' \
  -F 'prompt=Change the blue vase to red. Preserve the cat and composition.' \
  -F 'width=2048' -F 'height=2048' -F 'steps=40' -F 'cfg=1' -F 'seed=42'
```

重复 `image` 部分即可传入多张参考图。也可以先把文件上传到 `/api/upload`，再向
`/api/image-edit` 发送包含 `imagePaths`（或旧的 `imagePath`）的 JSON。两个图像端点都
接受 `negativePrompt`、`targetArea`、`width`、`height`、`steps`、`cfg` 和 `seed`。
`targetArea` 控制自动尺寸选择；显式尺寸优先。省略 `width`、`height`、`targetArea`、
`steps` 和 `cfg` 时使用上文的模型默认值。`targetArea: 1048576` 选择约 1K 的输出，
同时保留自动宽高比选择。

需要进度时，配合 `curl -N` 使用 JSON 路由 `/api/image-generate/stream` 和
`/api/image-edit/stream`。它们发出 SSE `data:` 帧，包含 `imageGenerate: true` 或
`imageEdit: true`、`step` 与 `total`，并可能附带预览 `image` data URL。终止帧包含
`done: true` 以及最终的 `url`、尺寸和耗时秒数，或者包含 `error`。聊天补全路由不会
运行这个扩散模型。现有的 `/api/image-edit` 请求仍至少需要一张参考图；生成有自己的
端点。预览由当前流预测估计出的干净潜变量解码而来。

## 加速与内存

2.1 扩散 Transformer 以常驻的量化权重运行完整的 GGML 图；没有 CPU 权重流式加载
模式。可用内存不足时，请先使用较小的尺寸。CUDA 与 Vulkan 已在 NVIDIA A40 上验证，
各项测量见[英文版模型卡](qwenimage21.md#prefix-kv-cache)。

### 前缀 KV 缓存

Qwen-Image-2.1 用 `t = 0` 那一行调制文本与参考图 token，并且其块因果注意力从不
让它们关注正在生成的图像（检查点的 `causal_condition`）。因此它们的隐藏状态，
以及每个块的 K 和 V，在每个去噪步都完全相同。TensorSharp 实现了官方的
[前缀 KV 缓存](https://github.com/QwenLM/Qwen-Image-2.1#prefix-kv-cache)：
第一步运行整个序列，把每个块前缀部分经过 RoPE 之后的 K 和 V 存到设备上；之后
每一步只计算目标图像的 token，并对"已存前缀 + 目标"做注意力。CFG 运行为每个
分支各保留一份缓存。去噪结束、VAE 解码之前释放缓存。每一步的日志行以
`prefix=extract` 或 `prefix=cached` 结尾。

缓存默认开启；`TS_QWEN21_PREFIX_CACHE=0` 关闭它。默认情况下它存的正是注意力内核
读取的类型（Metal 与 CUDA flash attention 为 F16，其他为 F32），所以缓存步复现
不用缓存时的计算：在 Metal 上，同一个种子在开启缓存、关闭缓存以及加入缓存之前的
构建中得到逐字节相同的 PNG。已对生成、单参考图与双参考图编辑、以及带两份缓存的
CFG 4 做过验证。diffusers 的 `use_kv_cache` 文档指出其 PyTorch 实现在两种设置下
不能逐位复现图像；TensorSharp 的计算图可以。

代价是内存：F16 下每个前缀 token、每个 CFG 分支 512 KiB（32 个块、K 与 V、
4096 个 2 字节的值）。提示词只有几十到几百个 token；每张约 1 百万像素的参考图
增加 4,096 个 token，约 2 GiB。一份缓存最多使用设备报告的空闲内存的一半，
`TS_QWEN21_PREFIX_CACHE_MAX_MIB` 可以进一步限制。放不下的缓存会被拒绝并在
stderr 上给出警告，该请求像以前一样每一步都重算前缀。

`TS_QWEN21_PREFIX_CACHE_TYPE` 选择存储类型：`auto`（默认，与不用缓存完全一致）、
`f16`、`f32`、`q8_0`（K 与 V 均为 Q8_0，每个 token 约 272 KiB）和 `q8_0_v`
（仅 V 为 Q8_0，约 392 KiB）。两种 8 位设置对应 vLLM-Omni 的 `fp8` 与 `fp8_v`
前缀缓存，但使用每 32 个值一个缩放的 ggml Q8_0 块，而不是每个 token 与头一个
缩放的 FP8 E4M3；每一步都会把前缀转换回注意力类型，因此目标自身的 K/V 保持精确。
它们对输出质量的实测影响见[英文版模型卡](qwenimage21.md#prefix-kv-cache)。

实测（默认存储，输出 PNG 与不用缓存时逐字节相同）：在 Apple M5 Pro（`ggml_metal`）上，
1024² 单参考图编辑每步从 17.97 秒降至 9.65–10.38 秒，双参考图从 33.32 秒降至 11.46 秒；
在 NVIDIA A40（`ggml_cuda`）上分别从 2.547 秒降至 1.328 秒、从 4.106 秒降至 1.538 秒。
文生图的前缀只有提示词，只快 1–3%；2048² 时目标 token 占主导，编辑只快 13–17%。

### CUDA Graph 与张量并行

上游 ggml-cuda 在同一张图连续两次执行不变后把它捕获为 CUDA Graph 并重放。缓存步
的计算图在各步之间保留（`TS_QWEN21_GRAPH_REUSE=1`，默认），输入与缓存缓冲区
固定，因此从一次请求的第三步起，每个去噪步都是一次图重放——这就是 vLLM-Omni
CUDA Graph 解码的效果，无需另写捕获代码。与 vLLM-Omni 一样，存储前缀的第一步
不被捕获。在 A40 上统计 CUDA 运行时调用确认了这一点：10 步请求只捕获 1 次、
重放 8 次，双卡请求捕获 130 次（每卡 65 段）、重放 1,040 次；开启与关闭 CUDA Graph
的输出 PNG 相同。由于每步的内核都很大，收益很小：256×256 单卡每步快 1.5%，1024²
无可测差异。

在 `ggml_cuda` 或 `ggml_vulkan` 上使用 `--tp N` 时，扩散 Transformer 按
Megatron 方式切分到 N 张 GPU：每张卡持有 32/N 个完整注意力头与 12,288/N 个 MLP
列，其余投影与所有归一化权重复制；每个块的两个行并行乘积在 GPU 之间求和
（ggml-cuda 有集合通信时在设备上完成，否则经由主机内存）。每张卡缓存自己那些头
的前缀。N 必须整除 32 且不拆开量化块：对已发布的文件为 2、4 或 8。文本编码器、
视觉编码器与 VAE 留在第一张卡上；不支持多节点组。在两张 A40（NCCL 经共享内存传输）上，1024² 文生图每步
从 1.107 秒降至 0.823 秒（1.34×），1024² 编辑从 1.326 秒降至 0.934 秒（1.42×），2048²
文生图与编辑分别为 1.54× 与 1.57×。`ggml_vulkan` 经主机内存归约，双卡反而慢 14%。
TP 改变了部分和的相加顺序，输出与单卡不逐位相同；这种舍入差异会沿去噪轨迹累积，
构图与质量相同但细节可能不同（与单卡相比 PSNR 34–52 dB）。

Vulkan VAE：ggml-vulkan 通过 F16 协作矩阵运算 F32 矩阵，而 VAE 的部分激活超过
65,504，因此此前在 `ggml_vulkan` 下 VAE 卷积一直回退到 CPU（512×512 解码超过 8 分钟）。
现在原生 F32 卷积在 Vulkan 上先把输入按 2 的整数次幂缩放到 32,768 以内，再把 F32 结果
还原，VAE 因而在 Vulkan 设备上运行：512×512 端到端从 683.5 秒降至 48.7 秒，与 F32 CPU
参考相比相对 L2 误差为 0.14%。

FP8 权重：TensorSharp 加载块量化的 GGUF 权重；ggml 没有 FP8 E4M3 张量类型，8 位
权重配置就是 Q8_0 GGUF。vLLM-Omni 的 FP8 工作中适用于这里的是 8 位前缀存储，
见上文。

## 验证记录

Unsloth Q8_0 验证、完整模型性能测量、原生优化验证、与 stable-diffusion.cpp 的
历史对比以及复现命令，请参阅[英文版模型卡](qwenimage21.md)。
