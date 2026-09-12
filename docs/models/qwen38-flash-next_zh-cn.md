# Qwen 3.8 Flash Next（`qwen4exp`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen38-flash-next.md)

Qwen3.8-Flash-Next 是一个混合型 MoE：GatedDeltaNet 递归层与全注意力层交错
（其中一部分全注意力层挂在 Qwen Sparse Attention 的 indexer 后面），再加上一个
PLE n-gram 嵌入块、×4 hyper-connection 流以及 512 专家的 MoE。GGUF 架构 id 是
`qwen4exp`。权重：
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片；把
`mmproj-BF16.gguf` 放在模型旁边即可启用图像输入）。

## TensorSharp 如何运行它

在 GGML 后端上，整个 token（几乎）只跑一张图——嵌入、PLE（在图内）、全部 48 层、
最后的 mixer 以及 LM head——并配一个按形状索引的已捕获图缓存
（`TS_Q4E_TOKEN_GRAPH=0` 回退到逐层融合 kernel，后者再逐算子回退）。视觉沿用
Qwen3.5-VL 塔，位置用 (T,H,W) IMRoPE；支持多图与多轮图像会话，并在轮次之间复用
KV（GDN 递归无法回退，因此只有当新 prompt **恰好扩展**已缓存前缀时才复用）。

## 连续批处理

并发请求通过**逐序列状态持有者**（per-sequence state holders）来服务：每个在飞请求
各自拥有自己的注意力 KV 与 QSA indexer 缓存、GDN 卷积与 delta-net 状态、PLE 卷积
历史与 n-gram 窗口，以及固定下来的 kernel 描述符。原生 kernel 用持有者的 host 种子
指针作为设备驻留递归状态的键，用描述符地址作为已缓存图的键，所以切换请求只是一次
引用交换——不需要状态下载 / 上传，也不需要重建图——每个序列都在自己那张已捕获的
单图融合 decode 上解码。引擎按步在各序列间轮询（`SupportsPerSequenceFusedForward`）；
融合的 N 路批量 decode 属于后续优化。

## 多 GPU

`qwen4exp` 上的 `--tp N` 跑的是**按层切分**：每张 GPU 持有一段连续的完整层。它不是
张量并行——`qwen4exp` 不切分任何权重——而且这也正是 llama.cpp 为该架构提供的
（唯一）多 GPU 模式（`-sm row` 直接拒绝加载）。它是**容量**特性，不是速度特性：
单卡装不下时靠它把模型装下。

实测：2× A100-80GB，Qwen3.8-Flash-Next-UD-Q2_K_XL（73.4 GiB）：

- 1 卡与 2 卡运行的贪心输出**逐字节一致**（SHA-256 相同）。
- 显存 24.2 GB + 26.2 GB——大约每张卡各放半个模型，而不是一张卡放下全部。
- 吞吐不变：两种情况下 prefill 都在 ~1520–1550 t/s，decode 都在 ~56 t/s。
  作为参照，同一台机器上的 llama.cpp：1 张 GPU pp1536 1094 / tg128 61.2；
  2 张 GPU `-sm layer` 1200 / 61.5——也就是说 llama.cpp 从第二张卡上同样只拿到
  约 10% 的 prefill 提升、decode 基本为 0。

启动时会打印实际走的是哪种模式，以及每张 GPU 分到的层数 / 字节数。
`TS_Q4E_LAYER_SPLIT=20,28` 可以用显式的每卡层数覆盖自动均衡（精神上等同于
llama.cpp 的 `--tensor-split`），并且在无法满足给定值时直接抛异常，而不是悄悄忽略
——这很有用，因为自动均衡只按权重计价，看不见视觉塔，而视觉塔加载得更晚、会落在
GPU 0 上。

## 基准矩阵

[`benchmark_config_glm53_qwen38.json`](../../benchmarks/engine_comparison/benchmark_config_glm53_qwen38.json)
以 `qwen38-flash-next` 的名字把本模型注册到固定的 Hugging Face revision 上，并挂上
它的 `mmproj-BF16.gguf`，好让 `image` 场景能跑。其中两条事实值得在这里重复。

一是已发布的 Q8_0 分片里**完全没有** `nextn` / `mtp` 张量，因此 `mtp_supported`
为 false，`--mtp on` 的格子会带着理由被跳过，而不是悄悄按普通解码跑掉。

二是**本模型只能跑在会传 `--tp N` 的那一列上**。原因就在上一节：切分度来自 `--tp`，
所以在不传 `--tp` 的后端列上，TensorSharp 只会建单设备上下文，175.3 GiB 会全部压到
一张卡上。因此配置里给了它 `min_tp`（4，仅按权重算出的下限——8 才是这台 8×A40 机器
应当使用的度数），在不传 `--tp` 的那一列上，这些格子会被记为
`needs --tp 4 (does not fit 1 GPU(s))` 的跳过，而不是留给它去 OOM。跑法：

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_tp
```

那一列会让 llama.cpp 用 `--split-mode layer` 切在同样这些 GPU 上，于是参照列两边是
同一种放置方式——对 `qwen4exp` 而言，这也是两个引擎各自唯一的多卡模式。
