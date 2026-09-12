# Hunyuan Dense（`hunyuan-dense`）

[← 返回模型索引](README_zh-cn.md) | [English](hunyuan-dense.md)

TensorSharp 支持腾讯的稠密 Hunyuan 解码器——即 GGUF 中 `general.architecture`
为 `hunyuan-dense` 的检查点，例如 Hy-MT2 翻译系列。在该架构被注册之前，这些官方
Q4 文件会在加载阶段直接失败，而不会退回到某个相近的系列：加载器遇到未知架构时
选择失败关闭，而不是猜测一套计算图。

这是第一版实现。它是走通用 per-op 执行器的单设备、纯文本路径：没有融合的整模型
计算图，没有张量并行，也没有按层切分。

| 属性 | 取值 |
|---|---|
| GGUF 架构 | `hunyuan-dense` |
| 源码类 | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) |
| 架构插件 | [`HunyuanDenseArchitecture`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseArchitecture.cs) |
| 参考模板 | `tencent/Hy-MT2-1.8B` 的 `chat_template.jinja` |
| 模态 | 仅文本 |
| 思考模式 | 否 |
| 工具调用 | 否——该协议既不渲染工具声明，也不渲染 `role: "tool"` 结果 |
| 投机解码 | 不支持 |
| 多 GPU | 单设备。`MultiGpuLimitation` 会在 stderr 上明说，而不是让多余的 GPU 静默闲置 |
| 后端 | 通过通用 per-op 路径支持全部后端：`cpu`、`ggml_cpu`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`cuda`、`mlx` |

## 运行

```bash
dotnet run --project TensorSharp.Cli -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_metal \
    --input "Translate to French: the harbour was quiet before dawn."
```

服务端使用同样的模型参数：

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_cuda --port 5000
```

## 架构

文本计算图与 llama.cpp 的 `hunyuan-vl` 文本塔一致：

```text
tokens -> embedding
  -> N x [ RMSNorm
            -> 融合 QKV -> NeoX RoPE -> per-head Q/K RMSNorm
            -> 因果 GQA
            -> 输出投影 + 残差
            -> RMSNorm -> SwiGLU(gate, up) -> down + 残差 ]
  -> 最终 RMSNorm -> LM head
```

**QK-norm 在 RoPE 之后**，与 Qwen 3.5 的顺序正好相反。搞反这个顺序不会导致加载
失败，只会产出流畅但错误的文本，因此加载器把这两个 norm 视为必需：任何一层缺少
`attn_q_norm.weight` 或 `attn_k_norm.weight` 都会在加载时抛异常。key 与 value 的
头宽必须相等；不相等时抛出 `NotSupportedException`，而不是悄悄 reshape。

### RoPE base 与 NTK alpha

当 GGUF 带有 `hunyuan-dense.rope.scaling.alpha` 时，加载器会在计算任何位置编码
之前套用 llama.cpp 的 Hunyuan 公式：

```text
base = rope_theta * alpha^(dim / (dim - 2))
```

Hy-MT2 的 Q4 文件写的是 `scaling.type = none` 且没有 alpha，因此直接使用发布的
base。启动日志会打印生效的 base、scale 和 RoPE 维度数，走了哪条分支可以直接看到，
不用猜。

### 权重融合与 KV cache

加载时，模型把每层的 Q/K/V 融合为一个 `attn_qkv.weight`，把每层的 gate/up 融合为
一个 `ffn_gate_up.weight`；量化路径与 F32 路径都会做，且仅当三个张量的 GGML 类型
与输入宽度一致时才融合。不满足条件的层保留独立投影；每层的融合标志在加载时预计算
一次，而不是每个 token 重新判断。

每层的 K/V cache 按模型对齐的 KV dtype 分配，从初始分配长度开始，按需翻倍直到配置
的最大上下文。每次扩容都会打印。

### 分词器

`hunyuan-dense` 使用与 DeepSeek V3/V4、JoyAI 词表相同的三遍 Unicode 预分词：先切
最长 3 位的数字串，再切 CJK 连续段，最后套用通用模式。把这几遍合并成一个 alternation
会改变中英混排处的切分边界，所以它们保持分开。

## 对话模板

Hy-MT2 的框架始终以 BOS 开头，助手标记只在需要生成提示时追加——它不会被粘在用户
轮次后面。这与 llama.cpp 的 `LLM_CHAT_TEMPLATE_HUNYUAN_DENSE`（Hunyuan-4B-Instruct）
不同，后者不带 BOS，且确实会把标记粘上去。

| 轮次 | 渲染结果 |
|---|---|
| 只有用户 | `<｜hy_begin▁of▁sentence｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| 系统 + 用户 | `<｜hy_begin▁of▁sentence｜>SYSTEM<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| 带助手历史 | 每条历史回答以 `<｜hy_place▁holder▁no▁2｜>` 收尾 |
| 不加生成提示 | 渲染结果以 `<｜hy_place▁holder▁no▁8｜>` 结束 |

该架构优先使用内置渲染器而非 GGUF 中的 Jinja 模板，并且不输出工具声明。因此
Agent Skills 在这里退回到内联指令，与所有没有工具解析器的系列一致。

## 当前限制

- 仅文本。没有接入投影器，`--image`、`--video`、`--audio` 均不适用。
- 没有思考通道，没有工具调用解析器。
- 单设备：没有张量并行、没有按层切分、没有融合整模型计算图、不支持投机解码。
