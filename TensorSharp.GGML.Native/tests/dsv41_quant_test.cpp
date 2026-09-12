// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "dsv41_quant.h"
#include "ggml_ops_dsv4_fused.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#ifdef TSG_GGML_USE_CUDA
#include "ggml-alloc.h"
#include "ggml-cuda.h"
static ggml_backend_t cuda_backend = nullptr;
static ggml_backend_t fused_backend = nullptr;
#endif

static void require(bool condition, const char * message)
{
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static void compute(ggml_tensor * reference, tsg_dsv4_fused_desc * desc)
{
    for (int worker = 0; worker < 4; ++worker)
        tsg_dsv4_fused_cpu(reference, worker, 4, desc);
#ifdef TSG_GGML_USE_CUDA
    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto * ctx = ggml_init(params);
    ggml_tensor * inputs[GGML_MAX_SRC];
    int count = 0;
    while (count < GGML_MAX_SRC && reference->src[count])
    {
        inputs[count] = ggml_dup_tensor(ctx, reference->src[count]);
        ++count;
    }
    auto * dst = ggml_custom_4d(ctx, reference->type, reference->ne[0], reference->ne[1],
        reference->ne[2], reference->ne[3], inputs, count, tsg_dsv4_fused_cpu, 1, desc);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    require(buffer != nullptr, "CUDA tensor allocation failed");
    for (int i = 0; i < count; ++i)
        ggml_backend_tensor_set(inputs[i], reference->src[i]->data, 0, ggml_nbytes(inputs[i]));
    require(ggml_backend_graph_compute(fused_backend, graph) == GGML_STATUS_SUCCESS, "CUDA fused graph failed");
    ggml_backend_synchronize(fused_backend);
    std::vector<unsigned char> actual(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    require(std::memcmp(actual.data(), reference->data, actual.size()) == 0, "CUDA/CPU fused op outputs differ");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
#endif
}

// Independently enumerate E4M3 values from their exponent/mantissa fields and
// choose nearest with ties to even. Tests include every midpoint and its two
// adjacent F32 values, not only values exactly representable by the format.
static std::vector<float> fp8_values()
{
    std::vector<float> values;
    for (int code = 0; code <= 126; ++code)
    {
        const int exponent = code >> 3, mantissa = code & 7;
        values.push_back(exponent == 0 ? mantissa / 512.0f : std::ldexp(1 + mantissa / 8.0f, exponent - 7));
    }
    return values;
}

static float nearest(float value, const std::vector<float> & values)
{
    const float magnitude = std::fabs(value);
    int best = 0;
    float distance = INFINITY;
    for (int code = 0; code < (int) values.size(); ++code)
    {
        const float next = std::fabs(values[code] - magnitude);
        if (next < distance || (next == distance && (code & 1) == 0))
        { best = code; distance = next; }
    }
    return std::copysign(values[best], value);
}

static void check_formats()
{
    const auto fp8 = fp8_values();
    const std::vector<float> fp4 = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int kind = 0; kind != 2; ++kind)
    {
        const auto & values = kind == 0 ? fp8 : fp4;
        for (size_t i = 1; i < values.size(); ++i)
        {
            const float mid = (values[i-1] + values[i]) / 2;
            for (float x : {std::nextafter(mid, -INFINITY), mid, std::nextafter(mid, INFINITY)})
                for (float sign : {-1.0f, 1.0f})
                {
                    const float actual = kind == 0 ? tsg_dsv41_e4m3(sign*x) : tsg_dsv41_e2m1(sign*x);
                    require(actual == nearest(sign*x, values), "FP8/FP4 midpoint rounding mismatch");
                }
        }
    }
    require(tsg_dsv41_bf16(1.00390625f) == 1.0f, "BF16 even midpoint mismatch");
    require(tsg_dsv41_bf16(1.01171875f) == 1.015625f, "BF16 odd midpoint mismatch");
    require(tsg_dsv41_quant_scale(0, 0) == 0x1p-22f, "FP8 zero-block scale mismatch");
    require(tsg_dsv41_quant_scale(0, 1) == 0x1p-126f, "MXFP4 zero-block scale mismatch");
    require(tsg_dsv41_quant_scale(0, 2) == 0x1p-9f, "NVFP4 zero-block scale mismatch");
    require(tsg_dsv41_e4m3(1000) == 448, "FP8 saturation mismatch");
}

static void check_quant(ggml_context * ctx)
{
    auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 17);
    auto * dst = ggml_dup_tensor(ctx, src);
    dst->src[0] = src;
    std::mt19937 random(471);
    std::uniform_real_distribution<float> uniform(-100, 100);
    auto * input = (float *) src->data;
    auto * output = (float *) dst->data;
    for (int i = 0; i < 64*17; ++i) input[i] = i < 64 ? 0 : uniform(random);
    tsg_dsv4_fused_desc desc;
    desc.kind = TSG_DSV41_FUSED_QUANT;
    for (int mode = 0; mode < 3; ++mode)
    {
        desc.i0 = mode;
        compute(dst, &desc);
        const int block = mode == 2 ? 16 : 32;
        const auto values = mode == 0 ? fp8_values() : std::vector<float>{0, .5f, 1, 1.5f, 2, 3, 4, 6};
        for (int first = 0; first < 64*17; first += block)
        {
            float maximum = 0;
            for (int i = 0; i < block; ++i) maximum = std::max(maximum, std::fabs(tsg_dsv41_bf16(input[first+i])));
            float scale;
            if (mode == 0) scale = std::exp2(std::ceil(std::log2(std::max(maximum, 1e-4f) / 448)));
            else if (mode == 1) scale = std::exp2(std::ceil(std::log2(std::max(maximum, 6*0x1p-126f) / 6)));
            else scale = nearest(std::max(maximum, 6*0x1p-9f) / 6, fp8_values());
            for (int i = 0; i < block; ++i)
            {
                const float expected = tsg_dsv41_bf16(nearest(tsg_dsv41_bf16(input[first+i])/scale, values)*scale);
                require(output[first+i] == expected, "block quant/dequant reference mismatch");
            }
        }
    }
}

static void check_candidates(ggml_context * ctx)
{
    auto * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 3);
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    auto * pooled = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
    pooled->src[0] = scores; pooled->src[1] = positions;
    auto * p = (int32_t *) positions->data; p[0] = 9; p[1] = 2; p[2] = -1;
    auto * s = (float *) scores->data;
    std::fill(s, s+30, -INFINITY);
    for (int i = 0; i < 10; ++i) s[i] = 100-i;
    s[10] = 1; s[11] = 2; s[12] = 3;
    tsg_dsv4_fused_desc desc;
    desc.kind = TSG_DSV41_CANDIDATE_SCORES; desc.i0 = 4;
    compute(pooled, &desc);
    auto * result = (float *) pooled->data;
    require(result[0] == 100 && result[1] == 96 && result[2] == INFINITY, "newest partial block must be pinned");
    require(result[3] == INFINITY && result[4] == -INFINITY && result[5] == -INFINITY, "causal block pooling mismatch");
    require(result[6] == -INFINITY && result[7] == -INFINITY && result[8] == -INFINITY, "unreachable query must not pin a block");
    auto * topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
    auto * selected = (int32_t *) topk->data;
    selected[0] = 2; selected[1] = 0; selected[2] = 0; selected[3] = 1; selected[4] = 0; selected[5] = 2;
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 10, 3);
    mask->src[0] = pooled; mask->src[1] = topk;
    desc.kind = TSG_DSV41_CANDIDATE_MASK;
    compute(mask, &desc);
    auto * out = (ggml_fp16_t *) mask->data;
    for (int i = 0; i < 30; ++i)
    {
        const bool kept = (i < 10 && (i < 4 || i >= 8)) || (i >= 10 && i < 14);
        require(ggml_fp16_to_fp32(out[i]) == (kept ? 0 : -INFINITY), "candidate expansion selected an unreachable block");
    }
}

int main()
{
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    cuda_backend = ggml_backend_cuda_init(0);
    fused_backend = tsg_dsv4_fused_backend_init(cuda_backend);
    require(cuda_backend && fused_backend, "CUDA fused backend initialization failed");
#endif
    check_formats();
    ggml_init_params params = {4*1024*1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml_init failed");
    check_quant(ctx);
    check_candidates(ctx);
    ggml_free(ctx);
#ifdef TSG_GGML_USE_CUDA
    ggml_backend_free(fused_backend);
    ggml_backend_free(cuda_backend);
#endif
    std::puts("DeepSeek V4.1 FP8/FP4/BF16 quantization and candidate-mask tests passed");
}
