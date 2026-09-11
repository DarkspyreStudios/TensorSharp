// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void require(bool value, const char * message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static void check(ggml_backend_t backend, int tokens, bool indexed, bool precise, bool half_weights,
                  int inner = 256, int rows = 64)
{
    const int experts = indexed ? 4 : 1, used = indexed ? 2 : 1;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * weights = ggml_new_tensor_3d(ctx, half_weights ? GGML_TYPE_F16 : GGML_TYPE_F32, inner, rows, experts);
    auto * input = indexed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, inner, 1, tokens)
                           : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, tokens);
    auto * ids = indexed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens) : nullptr;
    auto * output = indexed ? ggml_mul_mat_id(ctx, weights, input, ids) : ggml_mul_mat(ctx, weights, input);
    if (precise)
    {
        require(ggml_prec_set_acc(output, GGML_PREC_F32), "Cannot set accumulator precision");
        require(ggml_prec_set_src(output, GGML_PREC_F32, 1), "Cannot set input precision");
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "CUDA allocation failed");
    std::mt19937 random(41091);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> a(inner * rows * experts), b(inner * tokens), result(rows * used * tokens);
    std::vector<int32_t> selections(used * tokens);
    for (float & value : a) value = uniform(random);
    for (float & value : b) value = uniform(random);
    for (int t = 0; t < tokens; ++t)
        for (int e = 0; e < used; ++e) selections[t * used + e] = (t + e) % experts;
    if (half_weights)
    {
        std::vector<ggml_fp16_t> half(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            half[i] = ggml_fp32_to_fp16(a[i]);
            a[i] = ggml_fp16_to_fp32(half[i]);
        }
        ggml_backend_tensor_set(weights, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }
    else ggml_backend_tensor_set(weights, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(input, b.data(), 0, b.size() * sizeof(float));
    if (ids) ggml_backend_tensor_set(ids, selections.data(), 0, selections.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA matmul failed");
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    double squared_error = 0.0, squared_reference = 0.0;
    float maximum = 0.0f;
    bool within_tolerance = true;
    for (int t = 0; t < tokens; ++t) for (int e = 0; e < used; ++e) for (int row = 0; row < rows; ++row)
    {
        double reference = 0.0;
        const int expert = selections[t * used + e];
        for (int k = 0; k < inner; ++k)
            reference += double(a[(expert * rows + row) * inner + k]) * b[t * inner + k];
        const double error = std::abs(result[(t * used + e) * rows + row] - reference);
        maximum = std::max(maximum, float(error));
        squared_error += error * error;
        squared_reference += reference * reference;
        if (precise && error > 3e-5 + 3e-6 * std::abs(reference)) within_tolerance = false;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 5; ++repeat)
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Repeated CUDA matmul failed");
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5;
    std::printf("%s nt=%d precision=%s weights=%s inner=%d rows=%d max_abs=%.8g rel_l2=%.8g mean_us=%.1f\n",
        indexed ? "MUL_MAT_ID" : "MUL_MAT", tokens, precise ? "F32" : "default",
        half_weights ? "F16" : "F32", inner, rows, maximum, std::sqrt(squared_error / squared_reference), us);
    require(within_tolerance, "Explicit F32 matmul lost precision");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main()
{
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto * backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    for (bool indexed : {false, true}) for (int tokens : {1, 5, 16, 31})
        for (bool half_weights : {false, true}) for (bool precise : {false, true})
            check(backend, tokens, indexed, precise, half_weights);
    // Wide Engram projections can select a TF32 cuBLAS kernel even when a
    // small GEMM happens to use CUDA cores. Cover both sides of that dispatch.
    for (int tokens : {16, 31}) for (bool precise : {false, true})
        check(backend, tokens, false, precise, false, 128, 1280);
    ggml_backend_free(backend);
    return 0;
}
