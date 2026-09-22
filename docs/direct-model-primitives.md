# Direct model primitives

`TensorSharp.Models.Direct` is the public, framework-free surface for model
graphs implemented directly on TensorSharp's CPU and CUDA allocators. It is
intended for model libraries that own their architecture while reusing the
backend dispatch, tensor storage, and optimized kernels in TensorSharp.

The public surface contains:

- `DirectContext`, which owns an allocator and long-lived tensors;
- `DirectLinear`, for row-major linear weights from GGUF, safetensors, or F32
  arrays;
- `DirectEmbedding`, for device-resident token embedding lookup;
- `DirectConv2D`, for channels-last image convolution with PyTorch-layout
  weights and a bounded im2col workspace;
- `DirectOps`, for normalization, attention, RoPE, activation, row-wise, and
  matrix operations; and
- `DirectImageOps`, for channels-last group normalization and nearest-neighbor
  spatial upsampling.

Direct image activations consistently use
`[batch, height, width, channels]`. Convolution weights use
`[out, in, kernelHeight, kernelWidth]`, matching the native safetensors order
written by PyTorch. Direct sequence activations consistently use
`[tokens, hidden]`; attention projections use `[tokens, heads * headDim]`.

All direct primitives operate on contiguous F32 activations. Weight readers may
load F16 or BF16 safetensors because `IFloatTensorStore` converts those values
to F32 during construction. Model-specific graph structure, tokenization,
sampling, and request contracts remain in the consuming model library.
