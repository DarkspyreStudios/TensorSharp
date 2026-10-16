# MiniMax-H3

[English](minimax-h3.md) | [中文](minimax-h3_zh-cn.md)

MiniMax-H3 在同一条打包扩散序列中生成视频与原生 **32 kHz 立体声音频**。
TensorSharp 通过 `MiniMaxH3Model.GenerateVideo`、CLI 和视频 HTTP API 提供该模型。
FL2VA 用于文本和关键帧；Ref2VA 用于文本和参考媒体，两者是独立检查点。

`ggml_metal`、`ggml_cuda`、`ggml_cpu` 使用 TensorSharp 自有的原生图和未修改的
上游 ggml；`mlx`、`cuda`、`cpu` 使用直接张量实现。CUDA 与 MLX 的 DiT、
文本/视觉编码器、音频 VAE 和因果 3-D 视频编码器均在所选设备上执行；
`cpu` 前向使用托管 C# 算术。实现某条路径不表示已验证每种输入或长片段。

```sh
TensorSharp.Cli --config config/minimax-h3-fl2va.json --backend ggml_metal \
  --prompt "A red fox walks through a quiet forest. Forest birds chirp." \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --diffusion-seed 42 --output fox.mp4
```

这会生成 MP4 与立体声 WAV 旁挂文件。小尺寸、少步数请求只适合作为执行冒烟，
不能由文件成功导出推断感知质量。

## 当前验证范围

最新记录为 **2026-10-15**：本地 Apple M5 Pro、48 GiB 统一内存，远端 NVIDIA A40、
46,068 MiB 显存，使用未修改的上游 ggml
`3612c0af93f2cfc8f048295300a61187723b12ae`。
最终 candidate 7 在本地通过 36 项 CPU/MLX 专项测试，在远端通过 29 项专项测试。
Direct MLX、原生 Metal、Direct CUDA 与原生 CUDA 各自的九项组件 oracle 全部通过，
没有跳过。最终完整生成保真对比在这四条 GPU 路径上复现相同的解码 RGB 与立体声音频，
包括 MLX 的图像加标准化音频 Ref2VA，以及原生 CUDA 的首尾帧条件。

| 路径 | 已记录覆盖 | 未建立的结论 |
|---|---|---|
| `ggml_metal` / `mlx` | 组件 oracle、完整生成保真、首尾帧或混合参考的专项检查 | 任意提示词、种子、时长和自然媒体质量 |
| `ggml_cuda` / `cuda` | 组件 oracle、完整生成保真；Direct CUDA 独立 50 层 F32 预测门禁 | 多步独立真值、普遍优于其他引擎 |
| `ggml_cpu` / `cpu` | 修正后的 DiT 组件门禁；托管路径另有延迟加载编/解码门禁 | 修正后的完整配对生成仍待运行 |

组件夹具分别覆盖两层文本片段、两块 DiT 片段、完整视觉塔、视频/音频 VAE 编解码，
以及五帧和十七帧因果视频编码。它们不等同于完整文本编码器或任意形状的验证。
缺失模型、不可用设备和跳过场景均不算通过。

此前文本细化错误地使用 RoPE，采样器缺少音频载体漂移项，终点还缺少
`audioShift/videoShift` 缩放。这三项已修正；更早的 Metal **2.4× / 1.7×**
生成速度和质量对比已被取代，不再作为当前性能结论。

### 独立的完整深度 F32 检查

独立 PyTorch oracle 从同一份 GGUF 解量化，禁用 TF32，使用 IEEE F32 和 math SDPA，
重算全部 50 块。最新 19-token、sigma-one 合成输入相对 L2 为：
音频 `0.0001258906`、视频 `0.0000847537`，均通过 `0.001` 严格门禁。
BF16 权重投影现默认使用 pedantic F32；旧默认分别为 `0.0070538`、`0.0046911`。
这不是自然媒体或多步生成的独立质量证明。

另一个较早的 118-token 配对输入来自未修改 SD CLI 的公开 eval callback，
覆盖 256×160×5、22 个文本 token、sigma 1。Direct CUDA 对独立 F32 的音频/视频
相对 L2 为 `0.00007429` / `0.00031502`。单步物理终点经真实立体声 VAE 解码后，
波形余弦为 `0.999999745`；两边 PCM16 化后最大差为一个 LSB。
两组输入不同，不能混成一条精度曲线，也不能据此声称八步成片相同。

## 性能与精度

最新完整进程 A/B 使用 FL2VA Q4_K、640×384×22、24 fps、20 Euler 步、
seed 42、CFG 1、flow shift 12。两轮交替顺序包含加载、生成和编码，
文件/JIT 缓存未受控，不代表常驻模型服务延迟。

| 设备 / 后端 | 基线中位数 | 候选中位数 | 结果 |
|---|---:|---:|---|
| M5 Pro `ggml_metal` | 151.246 秒 | 143.622 秒 | RGB/音频精确相同，延迟降低 5.0% |
| M5 Pro `mlx` | 143.264 秒 | 125.426 秒 | 精确相同，延迟降低 12.5% |
| A40 `ggml_cuda` | 68.029 秒 | 67.174 秒 | 精确相同；1.3% 不构成有意义提速 |
| A40 `cuda`，同精度 | 220.456 秒 | 178.667 秒 | 精确相同，延迟降低 19.0% |

Direct CUDA 对比给两个版本都显式启用 pedantic BF16 权重投影，避免把精度变化
计作速度提升。最终构建另做的保真对比不算新的一轮速度基准。
256×160×5、八步、图像加标准化音频的 Ref2VA 两轮对比为
82.037 → 43.630 秒，RGB/音频精确相同；合成短片不证明自然参考质量或长片性能。

这些改动复用连续 token 行视图与精确 RoPE 表，融合保持算术顺序的 MLX
调制、门控残差和 SwiGLU，并缓存原生 CUDA 图存储与权重绑定。
MLX 视频 VAE 的 F32 权重直接上传到自有设备数组，避免保留 9.02 GiB 重复主机权重。
固定潜变量检查中，采样到的暖进程 RSS 中位数从 23.048 降至 14.024 GiB，输出字节相同；
这是进程驻留采样，不能当作统一内存或显存峰值。

CUDA 量化投影默认以最多 64 MiB 转换暂存区运行 pedantic F32 GEMM，保留 F32
激活与输出范围。`TS_H3_CUDA_F32_BANDS=0` 选择旧 F32 内核；`1` / `tf32`
显式选择近似 TF32。BF16 稠密权重投影也默认 pedantic F32，
`TS_H3_CUDA_BF16_PEDANTIC=0` 才恢复历史近似。

实验性批量注意力通过合成 full50 检查，却未通过完整生成 RGB/音频保真门禁，
因此模型入口已移除；探针拒绝过时的 `TS_H3_CUDA_BATCHED_ATTENTION` 覆盖。
独立后端 helper 和微基准只保留作研究工具。

较早与干净 stable-diffusion.cpp 的比较使用其
`d637cfb935feb2254898a8f1524046c958f3df80`，以及它固定的 ggml
`d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`。
640×384×22、八步原生 Metal 为 70.019 对 102.262 秒；此前原生 CUDA
20 步两轮为 71.899 / 64.904 秒，对照为 145.020 / 50.238 秒。
缓存波动很大，暖对照仍更快，因此没有普遍 CUDA 整进程持平结论。
当前 A/B 和这些历史跨引擎数据不能合并。完整旧数据与限制见[英文卡片](minimax-h3.md)。

## 需要的文件

| 文件 | 大小 | 用途 / 来源 |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | 文本 + 关键帧，[unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | 文本 + 参考，同仓库，选择其中一个去噪器 |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | 共用 Qwen3-VL-32B 文本/视觉编码器，同仓库 |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 视频 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | 同仓库，音频 VAE；静音输出可省略 |

一套约 35.5 GB；第二种去噪器共用编码器、VAEs 和分词器。
伴随文件可放在去噪器旁，或用 `--video-text-encoder`、`--video-vae`、`--audio-vae`
指定。`config/minimax-h3-fl2va.json` 与 `config/minimax-h3-ref2va.json`
下载网络文件，默认 `ggml_cuda`、640×384×22、24 fps；命令行后端优先。
服务端步数使用 `--video-steps`，CLI 使用 `--diffusion-steps`；两份配置不写步数与 CFG，
由模型采用默认 20 步和 CFG 1。

**手动安装还需要分词器。** 文本 GGUF 不带分词器，配置也不会下载它。
将 [MiniMaxAI/MiniMax-H3 的 processor 文件](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
`vocab.json`、`merges.txt`、`tokenizer_config.json` 放在编码器旁，
或以 `TS_VIDEO_TOKENIZER` 指向其目录。TensorAgent 目录会自动下载这三份文件。
视觉输入需要 `tokenizer_config.json` 中的特殊标记；缺失时图片或片段参考在视觉塔前
就会被拒绝，避免默默忽略输入。纯文本或纯音频参考不需要这些视觉标记。
已发布 GGUF 没有元数据，TensorSharp 按张量表识别；声明时也接受
`minimax-h3` / `minimax_h3`。重新命名请保留 `fl2va` / `ref2va` 分区信息。

## 条件模式

| 输入 | 模式 | 检查点 |
|---|---|---|
| 文本 | `t2v` | FL2VA 或 Ref2VA |
| 首帧图像 | `i2v` | FL2VA |
| 首帧和/或尾帧 | `fl2v` | FL2VA |
| 参考图、片段、音轨 | `ref` | Ref2VA |

`--video-mode` 可显式选择，否则按输入推断。关键帧不能与命名参考混用；
分区不符时会拒绝。关键帧同时编码为视频潜变量和 Qwen3-VL 特征，按输出比例中心裁剪；
未指定尺寸时使用图像比例。Ref2VA 保持参考自身比例，不以参考改变输出画布。
`--ref-image` 最多九份，每份增加同一条注意力序列的 token 数、内存和耗时。

```sh
TensorSharp.Cli --config config/minimax-h3-fl2va.json --backend ggml_metal \
  --image start.png --end-image end.png --video-mode fl2v \
  --prompt "A slow cinematic push-in" --diffusion-seed 42 --output morph.mp4

TensorSharp.Cli --config config/minimax-h3-ref2va.json --backend ggml_metal \
  --ref-image person.jpg --ref-audio song.wav --video-mode ref \
  --prompt "She performs on a small club stage" --diffusion-seed 42 --output stage.mp4
```

`--ref-video` 接受文件或帧目录，`--ref-video-audio` 按索引配对其音轨；
独立音轨用 `--ref-audio`。视频按 24 fps 取样、对齐 `17k+5` 帧且不超过输出长度。
独立音轨重采样为 32 kHz 立体声，按输出时长裁剪或补零，再向下对齐 800-sample VAE hop。
片段配对音轨按片段自身时长处理。被检查的 SD 版本保留完整独立音轨，因此相同文件
也可能得到不同参考上下文。配对基准必须明确标准化，不能把上下文差异当成算术错误。
已有合成首尾帧及标准化图像加音频测试；自然参考、视频片段及广泛提示词质量仍未建立。

## 尺寸与设置

- 宽高向上对齐 32；默认 640×384，或在关键帧比例下保持相同面积。
- 帧数向上对齐 5、22、39、56……（`17k+5`），默认 22。
- 固定 24 fps、CFG 1、自有 Euler flow schedule，默认 20 步；其他 CFG 会拒绝。
- Wan 的负面提示词、采样器与 guidance-cache 选项不改变 H3。
- `--diffusion-seed` 决定视频噪声；`--seed` 只控制文本采样。
- 视频 VAE 按 256 px 空间分块、按时间分段；长片的有限性与感知质量仍需单独验证。

## HTTP API

`POST /api/video-generate`、`POST /api/video-generate/stream`（SSE）与
`POST /v1/videos/generations` 共用解析器。

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "A red fox walks through a quiet forest. Forest birds chirp.",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v", "seed": 42
}'
```

主要字段有 `prompt`、`width`、`height`、`frames`、`steps`、`cfg`、`fps`、`seed`、
`flowShift`、`imagePath` 或 base64 `image`、`videoMode`、`generateAudio`、`endImage`、
`referenceImages`、`referenceVideos`、`referenceAudios`、`referenceVideoAudios`。
优先使用 camelCase；只有模式、音频开关、尾图和四个参考列表也接受 snake_case，
`image_path` / `flow_shift` 会被忽略。参考视频与音频按索引配对。
图片路径必须来自 `/api/upload`；上传目录外路径会拒绝。模型模式错误返回带具体原因的 400。

响应包含 `url`、`audioUrl`、`audioMuxed`、尺寸、帧数、fps、种子、codec、耗时等。
OpenAI 形式返回 `audio_url`，没有 mux 标记。`GET /api/models` 的 `video` 对象声明
音频、首帧、尾帧及参考能力，Web UI 据此显示输入控件。

### 音频输出

生成音轨同时保存为 WAV。桌面提供器有 FFmpeg 时会把音轨混入 MP4；
OpenCV/OS 回退只写视频，并返回 `audioMuxed: false`。
Apple AVAssetWriter（TensorAgent Mac 应用）写入 32 kHz、双声道、128 kbit/s AAC，
并返回 `audioMuxed: true`。编码格式差异也应记录在音频对比中。

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

`--no-audio` / `"generateAudio": false` 跳过音频解码，减少音频 VAE 时间与内存。

## 在 TensorAgent Mac 应用里

32 GB 档位提供 **MiniMax-H3**（`minimax-h3-fl2va-q4k`）和
**MiniMax-H3 References**（`minimax-h3-ref2va-q4k`），自动下载带哈希校验的网络和三份
分词器文件。首个条目把一张上传图作为首帧，两张作为首尾帧；参考条目接受参考媒体。
目录可用性不等于已完成每个模型/设备的验证；当前 Windows 与 iOS 的媒体生成未验证。
完整应用与构建说明见 [TensorAgent](../../TensorAgent/README.md)。

## 架构与内存

主要实现位于 [`MiniMaxH3Model`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Model.cs)、
[`MiniMaxH3Pipeline`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Pipeline.cs)、
[`MiniMaxH3DirectDiT`](../../TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3DirectDiT.cs)
和 TensorSharp 自有的原生图
[`ggml_ops_minimax_h3.cpp`](../../TensorSharp.GGML.Native/ggml_ops_minimax_h3.cpp)。

去噪器为约 193 亿参数、50 块单流 Transformer，hidden 5376，56×128 维注意力，
SwiGLU；文本、条件、目标音频和视频共同做双向注意力。没有 cross-attention。
视频 patch 为 `1×2×2`，24 通道展平为 96 值，次序为 `c*4 + ky*2 + kx`。
AdaLN 来自学习的 `[8,1025]` 曲线表，三模态分别取调制参数。
主干使用连续浮点三轴 RoPE，音频和视频共享时间轴；两个文本 refiner 块不使用 RoPE。

视频 shift 12、音频 shift 3，采样器保存
`carrier = (sigma_video / sigma_audio) * physical_audio`；送入 DiT 前还原物理音频，
积分时包含漂移项，终点乘 `audioShift/videoShift`，再做 VAE 通道反归一化。
文本编码器是 Qwen3-VL-32B 前 50 层，去掉 final norm，直接读取原始提示词，不套聊天模板。
视频 VAE 为 16× 空间/4× 时间压缩，36 层 Transformer 解码器；256 px 分块是坐标正确性要求。
因果 CNN 编码器处理单图时有两帧前导零填充。音频 DAC/BigVGAN 使用 40 Hz、32 通道
潜变量，800 倍上采样至 32 kHz；解码前需恢复通道均值/标准差。

模型分阶段交还设备权重。Metal 默认在下一段文本编码前释放上段去噪器和 VAE，
`TS_H3_KEEP_RESIDENT=1` 保留旧行为。原生 CUDA 预读不可变待上传权重，
自动预算为物理/cgroup 内存较小值的一半，未知容量时关闭；它不同于托管去噪器文件预读。

## 环境变量

| 变量 | 作用 |
|---|---|
| `TS_H3_CUDA_F32_BANDS` | 默认 pedantic F32；`0` 旧内核，`1` / `tf32` 近似模式 |
| `TS_H3_CUDA_BF16_PEDANTIC` | BF16 稠密投影默认 pedantic F32；`0` 恢复历史近似 |
| `TS_H3_CUDA_MAPPED_VAE_WEIGHTS=1` | 可选映射 F16 VAE 权重，存在冷/暖权衡，默认仍展开 F32 |
| `TS_H3_UPLOAD_PREFAULT` | 原生 CUDA 自动半内存预算；`0` 禁用，`1` 强制 |
| `TS_H3_UPLOAD_WORKERS` | 原生 CUDA 预读 1–8 个 worker，默认 8 |
| `TS_H3_UPLOAD_PROFILE=1` | 预读/上传/图诊断，不能混入正式计时 |
| `TS_H3_PREFAULT` | 托管去噪器预读：0 关、1 串行、2 提前重叠、3 默认流水线 |
| `TS_H3_PREFAULT_THREADS` | 托管预读流，默认 1 |
| `TS_H3_TE_GROUP` | 可选原生文本主干分组，默认关闭 |
| `TS_H3_KEEP_RESIDENT=1` | Metal 片段间保留网络 |
| `TS_H3_PHASE=1` / `TS_H3_TRACE=1` | 阶段耗时 / 潜变量与速度幅度 |
| `TS_H3_DIT_TRACE_DIR` / `TS_H3_LINEAR_TRACE_FILE` | 会同步下载的边界诊断，排除在正式性能数据外 |

## 复现验证

[eng/validation/README-minimax-h3.md](../../eng/validation/README-minimax-h3.md)
提供冻结二进制 A/B、独立 full50、组件与完整媒体检查的命令及证据索引。
`minimax-h3-oracles.py` 拒绝缺失输入、跳过和不完整用例数；
`minimax-h3-generation.py` 记录模型/二进制哈希、依赖、实际硬件和输入，结构通过不代表质量持平。
`minimax-h3-ab.py` 对冻结前后版本要求解码 RGB/音频精确相同。
独立夹具位于 `InferenceWeb.Tests/Fixtures/`，可复用工具位于 `eng/`；
日志、报告与快照留在被忽略的 `docs/validation/` 或 `artifacts/`。
本页整理已记录覆盖，没有把文档更新当作新模型运行。
