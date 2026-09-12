The native CUDA build applies these upstream correctness fixes before compiling
ggml. Each patch is checked before application, and already applied patches are
recognized. A conflicting upstream revision fails configuration so the change
can be reviewed instead of silently omitted.

`0001-cuda-honor-f32-matmul-precision.patch` makes the explicit
`ggml_prec_set_src(node, GGML_PREC_F32, 1)` input request avoid the custom TF32
matrix kernel, half conversion, and cuBLAS TF32 mode. It preserves
the request through indexed expert matrix slices and the corresponding CUDA
graph synchronization check. Accumulator precision alone does not prohibit
lower input precision; those existing fast paths remain available. Small indexed
F32 requests use the accurate vector kernel. Quantized weights retain their
fast paths unless full input precision is explicitly requested.

DeepSeek V4.1 requests this precision for its F32 projections because truncated
mantissas can change FP4 cache bins and sparse expert or attention selection.
`GgmlOpsCudaMatmulPrecisionTest` compares both ordinary and indexed matrix
products against independent double arithmetic at several batch sizes. Its
wide Engram-shaped projection also covers cuBLAS dispatch: explicit F32 sources
must use the pedantic `GemmEx` path, since `Sgemm` inherits the handle's TF32 math
mode. Default precision runs also report timing and error for comparison.
