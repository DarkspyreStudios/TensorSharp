// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int head = 512, heads = 64, raw_ring = 512, window = 128, top_k = 512;
constexpr int finite_bound = window + top_k;
constexpr float scale = 1.0f / 22.627416997969522f;
// F32 accumulator precision does not remove F16 MMA operand/probability
// rounding. The absolute envelope is below one F16 ULP at the largest
// generated value (0.7); relative L2 is bounded independently below.
constexpr double oracle_atol = 3e-4, oracle_rtol = 1.5e-3;

void require(bool value, const char * message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct error_stats {
    double squared = 0.0, reference_squared = 0.0, maximum = 0.0;
    size_t count = 0;
    bool close = true;
    void add(float actual, double reference, double tolerance_scale = 1.0) {
        const double error = std::abs(actual - reference);
        close &= std::isfinite(actual) && std::isfinite(reference) &&
                 error <= tolerance_scale * (oracle_atol + oracle_rtol * std::abs(reference));
        maximum = std::max(maximum, error);
        squared += error * error;
        reference_squared += reference * reference;
        ++count;
    }
    double relative() const { return std::sqrt(squared / std::max(reference_squared, 1e-30)); }
    void merge(const error_stats & other) {
        squared += other.squared;
        reference_squared += other.reference_squared;
        maximum = std::max(maximum, other.maximum);
        count += other.count;
        close &= other.close;
    }
};

double timed_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    const auto begin = std::chrono::steady_clock::now();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA attention failed");
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool check(ggml_backend_t backend, int nt, int nkv, int iterations) {
    // The dimensions, MQA ratio, F16 shared K/V and attention sinks match the
    // released checkpoint. Only the weights/activations here are synthetic.
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create attention context");
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head, nt, heads);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head, nkv, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nt);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * dense = ggml_flash_attn_ext(ctx, q, k, k, mask, scale, 0.0f, 0.0f);
    auto * sparse = ggml_flash_attn_ext(ctx, q, k, k, mask, scale, 0.0f, 0.0f);
    for (auto * node : {dense, sparse}) {
        ggml_flash_attn_ext_add_sinks(node, sinks);
        ggml_prec_set_acc(node, GGML_PREC_F32);
        require(ggml_backend_supports_op(backend, node), "CUDA does not support model-shaped attention");
    }
    ggml_flash_attn_ext_set_n_kv_max(sparse, finite_bound);
    auto * dense_graph = ggml_new_graph(ctx);
    auto * sparse_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(dense_graph, dense);
    ggml_build_forward_expand(sparse_graph, sparse);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate CUDA attention tensors");

    std::mt19937 random(410510 + nt + nkv);
    std::uniform_real_distribution<float> uniform(-0.7f, 0.7f);
    std::vector<float> queries(size_t(head) * heads * nt), keys(size_t(head) * nkv), sink_values(heads);
    std::vector<ggml_fp16_t> half_keys(keys.size());
    for (float & value : queries) value = uniform(random);
    for (size_t i = 0; i < keys.size(); ++i) {
        half_keys[i] = ggml_fp32_to_fp16(uniform(random));
        keys[i] = ggml_fp16_to_fp32(half_keys[i]);
    }
    for (int h = 0; h < heads; ++h) sink_values[h] = (h % 9) - 3.0f;
    const auto negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    std::vector<ggml_fp16_t> half_mask(size_t(nkv) * nt, negative_infinity);
    std::vector<std::vector<int>> selected(nt);
    const int compressed = nkv - raw_ring;
    const int p0 = compressed - nt, last = p0 + nt - 1;
    for (int t = 0; t < nt; ++t) {
        // Include an entirely masked query (sink alone), a one-key query,
        // maximum occupancy, rotating ring slots, arbitrary candidate holes,
        // causal exclusion, and finite biases. Sparse compaction must preserve
        // the original mask values, not just membership.
        if (nt > 1 && t == 0) continue;
        const int position = p0 + t;
        auto select = [&](int row) {
            half_mask[size_t(t) * nkv + row] = ggml_fp32_to_fp16(-0.125f * ((row + t) % 7));
            selected[t].push_back(row);
        };
        if (nt > 1 && t == 1) { select(position % raw_ring); continue; }
        for (int slot = 0; slot < raw_ring; ++slot) {
            const int token = last - ((last - slot) % raw_ring + raw_ring) % raw_ring;
            if (token <= position && token > position - window) select(slot);
        }
        const int visible = position + 1;
        int step = 97;
        while (std::gcd(step, visible) != 1) ++step;
        const int count = t <= 2 ? top_k : top_k - (t % 7) * 31;
        for (int j = 0; j < count; ++j) select(raw_ring + (j * step + t * 13) % visible);
        require(selected[t].size() <= finite_bound, "Mask exceeds declared sparse bound");
        std::sort(selected[t].begin(), selected[t].end());
        require(std::adjacent_find(selected[t].begin(), selected[t].end()) == selected[t].end(),
                "Reference mask contains duplicate keys");
    }
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_set(k, half_keys.data(), 0, half_keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, half_mask.data(), 0, half_mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(sinks, sink_values.data(), 0, sink_values.size() * sizeof(float));
    timed_compute(backend, dense_graph);
    timed_compute(backend, sparse_graph);
    std::vector<float> dense_result(queries.size()), sparse_result(queries.size());
    ggml_backend_tensor_get(dense, dense_result.data(), 0, dense_result.size() * sizeof(float));
    ggml_backend_tensor_get(sparse, sparse_result.data(), 0, sparse_result.size() * sizeof(float));
    error_stats difference, dense_reference, sparse_reference;
    // Both implementations round internal F16 MMA products independently.
    // Their pairwise absolute envelope is the sum of the two oracle bounds;
    // each implementation still has to pass the tighter independent bound.
    for (size_t i = 0; i < dense_result.size(); ++i) difference.add(sparse_result[i], dense_result[i], 2.0);

    // Compare every output element between CUDA paths and independently
    // evaluate every query/head against original F32 Q and decoded F16 K
    // with double-precision dot products and softmax; do not derive the oracle
    // from either CUDA path. A bounded CPU worker group evaluates all outputs
    // before timing starts, including every nt=256 prefill query.
    auto reference_query_head = [&](int t, int h, error_stats & dense_error, error_stats & sparse_error) {
        const float * query = queries.data() + size_t(h * nt + t) * head;
        const auto & ids = selected[t];
        std::vector<double> scores(ids.size()), output(head, 0.0);
        double maximum = sink_values[h];
        for (size_t i = 0; i < ids.size(); ++i) {
            const float * key = keys.data() + size_t(ids[i]) * head;
            double dot = 0.0;
            for (int c = 0; c < head; ++c) dot += double(query[c]) * key[c];
            scores[i] = dot * scale + ggml_fp16_to_fp32(half_mask[size_t(t) * nkv + ids[i]]);
            maximum = std::max(maximum, scores[i]);
        }
        double denominator = std::exp(sink_values[h] - maximum);
        for (size_t i = 0; i < ids.size(); ++i) {
            const double weight = std::exp(scores[i] - maximum);
            denominator += weight;
            const float * key = keys.data() + size_t(ids[i]) * head;
            for (int c = 0; c < head; ++c) output[c] += weight * key[c];
        }
        for (int c = 0; c < head; ++c) {
            const size_t index = size_t(t * heads + h) * head + c;
            const double reference = output[c] / denominator;
            // The baseline dense prefill kernel repeatedly rounds/rescales
            // across KV tiles. Retain its wider F16 envelope explicitly;
            // sparse attention must satisfy the tighter envelope above.
            dense_error.add(dense_result[index], reference, 2.0);
            sparse_error.add(sparse_result[index], reference);
        }
    };
    constexpr int workers = 4;
    std::array<error_stats, workers> dense_errors, sparse_errors;
    std::array<std::thread, workers> threads;
    std::atomic<int> next{0};
    for (int worker = 0; worker < workers; ++worker) threads[worker] = std::thread([&, worker] {
        for (;;) {
            const int job = next.fetch_add(1, std::memory_order_relaxed);
            if (job >= nt * heads) break;
            reference_query_head(job / heads, job % heads, dense_errors[worker], sparse_errors[worker]);
        }
    });
    for (int worker = 0; worker < workers; ++worker) {
        threads[worker].join();
        dense_reference.merge(dense_errors[worker]);
        sparse_reference.merge(sparse_errors[worker]);
    }
    const bool correct = difference.close && difference.relative() < 3e-3 &&
        dense_reference.close && dense_reference.relative() < 3e-3 &&
        sparse_reference.close && sparse_reference.relative() < 1.5e-3;
    std::printf("head=%d heads=%d nt=%d nkv=%d bound=%d elements=%zu reference_elements=%zu "
                "dense_sparse_max_abs=%.8g dense_sparse_rel_l2=%.8g "
                "dense_reference_max_abs=%.8g dense_reference_rel_l2=%.8g "
                "sparse_reference_max_abs=%.8g sparse_reference_rel_l2=%.8g correctness=%s\n",
                head, heads, nt, nkv, finite_bound, difference.count, dense_reference.count,
                difference.maximum, difference.relative(), dense_reference.maximum, dense_reference.relative(),
                sparse_reference.maximum, sparse_reference.relative(), correct ? "pass" : "fail");
    std::fflush(stdout);

    if (iterations > 0) {
        for (int warm = 0; warm < 3; ++warm) {
            timed_compute(backend, dense_graph);
            timed_compute(backend, sparse_graph);
        }
        std::vector<double> dense_us, sparse_us;
        for (int repeat = 0; repeat < iterations; ++repeat) {
            // Alternate order to reduce bias from GPU clocks and temperature.
            if (repeat % 2 == 0) {
                dense_us.push_back(timed_compute(backend, dense_graph));
                sparse_us.push_back(timed_compute(backend, sparse_graph));
            } else {
                sparse_us.push_back(timed_compute(backend, sparse_graph));
                dense_us.push_back(timed_compute(backend, dense_graph));
            }
        }
        std::printf("benchmark nt=%d nkv=%d iterations=%d dense_median_us=%.3f sparse_median_us=%.3f speedup=%.4f correctness=%s\n",
                    nt, nkv, iterations, median(dense_us), median(sparse_us), median(dense_us) / median(sparse_us),
                    correct ? "pass" : "fail");
        std::fflush(stdout);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return correct;
}
} // namespace

int main(int argc, char ** argv) {
    int iterations = 0, only_tokens = 0, only_kv = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") iterations = 9;
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "--tokens" && i + 1 < argc) only_tokens = std::atoi(argv[++i]);
        else if (arg == "--nkv" && i + 1 < argc) only_kv = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "Usage: %s [--bench] [--iterations N] [--tokens 1|17|256] [--nkv 4096|8192|16384|32768]\n", argv[0]); return 1; }
    }
    require(iterations >= 0 && iterations <= 1000, "Invalid benchmark iteration count");
    require(only_tokens == 0 || only_tokens == 1 || only_tokens == 17 || only_tokens == 256, "Invalid token count");
    require(only_kv == 0 || only_kv == 4096 || only_kv == 8192 || only_kv == 16384 || only_kv == 32768, "Invalid KV count");
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto * backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    bool correct = true;
    for (int nt : {1, 17, 256}) for (int nkv : {4096, 8192, 16384, 32768}) {
        if (only_tokens && only_tokens != nt) continue;
        if (only_kv && only_kv != nkv) continue;
        correct = check(backend, nt, nkv, iterations) && correct;
    }
    ggml_backend_free(backend);
    return correct ? 0 : 1;
}
