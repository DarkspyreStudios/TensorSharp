// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Run explicitly on two idle CUDA GPUs; this is not an end-to-end benchmark.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int head = 512, heads = 64, window = 128, top_k = 512, compressed_rows = 32768;
// The head512 GQA flash kernel requires a KV length divisible by256.
constexpr int raw_storage = ((window + top_k + 255) / 256) * 256 - top_k;
constexpr float scale = 1.0f / 22.627416997969522f;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct devices {
    ggml_backend_t gpu[2] = {}, fused[2] = {}, cpu = nullptr;
    ggml_backend_t ordered[5] = {};
    ggml_backend_buffer_type_t types[5] = {};
    devices() {
        for (int i = 0; i < 2; ++i) {
            gpu[i] = ggml_backend_cuda_init(i);
            require(gpu[i] != nullptr, "CUDA initialization failed");
            fused[i] = tsg_dsv4_fused_backend_init(gpu[i]);
            require(fused[i] != nullptr, "Fused CUDA initialization failed");
        }
        cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend initialization failed");
        ordered[0] = gpu[0]; ordered[1] = gpu[1];
        ordered[2] = fused[0]; ordered[3] = fused[1]; ordered[4] = cpu;
        for (int i = 0; i < 5; ++i) types[i] = ggml_backend_get_default_buffer_type(ordered[i]);
    }
    ~devices() {
        if (cpu) ggml_backend_free(cpu);
        for (auto value : fused) if (value) ggml_backend_free(value);
        for (auto value : gpu) if (value) ggml_backend_free(value);
    }
};

struct input_fixture {
    ggml_context * ctx[2] = {};
    ggml_backend_buffer_t buffer[2] = {};
    ggml_tensor * raw = nullptr, * compressed = nullptr, * top = nullptr;
    ggml_tensor * raw_ids = nullptr, * q = nullptr, * sinks = nullptr;
    int ring, position;
    std::vector<ggml_fp16_t> raw_values, compressed_values;
    std::vector<float> query_values, sink_values;
    std::vector<int32_t> selected_raw, selected_compressed;

    input_fixture(devices & dev, int ring_size, int query_position) : ring(ring_size), position(query_position) {
        for (auto & context : ctx) {
            context = ggml_init({1024 * 1024, nullptr, true});
            require(context != nullptr, "Input metadata allocation failed");
        }
        raw = ggml_new_tensor_2d(ctx[0], GGML_TYPE_F16, head, ring);
        compressed = ggml_new_tensor_2d(ctx[1], GGML_TYPE_F16, head, compressed_rows);
        q = ggml_new_tensor_3d(ctx[0], GGML_TYPE_F32, head, 1, heads);
        sinks = ggml_new_tensor_1d(ctx[0], GGML_TYPE_F32, heads);
        top = ggml_new_tensor_1d(ctx[0], GGML_TYPE_I32, top_k);
        raw_ids = ggml_new_tensor_1d(ctx[0], GGML_TYPE_I32, raw_storage);
        for (auto * tensor : {raw, compressed, q, sinks, top, raw_ids}) ggml_set_input(tensor);
        ggml_set_name(raw, "consumer.raw_ring");
        ggml_set_name(compressed, "source.compressed_cache");
        for (int i = 0; i < 2; ++i) {
            buffer[i] = ggml_backend_alloc_ctx_tensors(ctx[i], dev.gpu[i]);
            require(buffer[i] != nullptr, "Input device allocation failed");
            ggml_backend_buffer_set_usage(buffer[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        std::mt19937 random(410510 + ring + position);
        std::uniform_real_distribution<float> uniform(-0.7f, 0.7f);
        raw_values.resize(size_t(head) * ring);
        compressed_values.resize(size_t(head) * compressed_rows);
        query_values.resize(head * heads); sink_values.resize(heads);
        for (auto & value : raw_values) value = ggml_fp32_to_fp16(uniform(random));
        for (auto & value : compressed_values) value = ggml_fp32_to_fp16(uniform(random));
        for (auto & value : query_values) value = uniform(random);
        for (int i = 0; i < heads; ++i) sink_values[i] = float(i % 9 - 3);
        for (int i = 0; i < window; ++i) selected_raw.push_back((position - window + 1 + i) % ring);
        std::sort(selected_raw.begin(), selected_raw.end());
        for (int i = 0; i < top_k; ++i) selected_compressed.push_back((i * 61 + 17) % compressed_rows);
        auto upload = [](ggml_tensor * tensor, const auto & values) {
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(values[0]));
        };
        upload(raw, raw_values); upload(compressed, compressed_values);
        upload(q, query_values); upload(sinks, sink_values);
        upload(top, selected_compressed);
        auto padded_raw = selected_raw;
        padded_raw.resize(raw_storage, selected_raw.front());
        upload(raw_ids, padded_raw);
    }
    ~input_fixture() {
        for (auto value : buffer) if (value) ggml_backend_buffer_free(value);
        for (auto value : ctx) if (value) ggml_free(value);
    }
};

struct variant {
    ggml_context * ctx = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * packed = nullptr, * output = nullptr, * mask = nullptr;
    std::array<tsg_dsv4_fused_desc, 2> descriptions;
    bool compact;
    int nraw;
    variant(devices & dev, input_fixture & inputs, bool compact_raw) : compact(compact_raw), nraw(compact ? raw_storage : inputs.ring) {
        ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Graph metadata allocation failed");
        scheduler = ggml_backend_sched_new(dev.ordered, dev.types, 5, 1024, false, true);
        require(scheduler != nullptr, "Scheduler creation failed");
        graph = ggml_new_graph_custom(ctx, 1024, false);
        auto gather = [&](int descriptor, ggml_tensor * raw, ggml_tensor * cache, ggml_tensor * ids,
                          int raw_rows, int rows, int device) {
            auto & description = descriptions[descriptor];
            description.kind = TSG_DSV4_FUSED_KGATHER;
            description.i0 = raw_rows;
            ggml_tensor * args[] = {raw, cache, ids};
            auto * result = ggml_custom_4d(ctx, GGML_TYPE_F16, head, 1, rows, 1,
                args, 3, tsg_dsv4_fused_cpu, 1, &description);
            ggml_backend_sched_set_tensor_backend(scheduler, result, dev.fused[device]);
            return result;
        };
        auto * raw = inputs.raw;
        if (compact) {
            // First gather stays on the consumer: otherwise its scheduler
            // would copy the full input ring before running the small kernel.
            raw = gather(0, raw, raw, inputs.raw_ids, 0, raw_storage, 0);
            ggml_set_name(raw, "consumer.raw_window");
        }
        packed = gather(1, raw, inputs.compressed, inputs.top, nraw, nraw + top_k, 1);
        ggml_set_name(packed, "source.selected_keys");
        ggml_set_output(packed); // Preserve for independent exact-copy checks.
        auto * keys = ggml_reshape_3d(ctx, packed, head, nraw + top_k, 1);
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nraw + top_k, 1);
        ggml_set_input(mask);
        ggml_backend_sched_set_tensor_backend(scheduler, mask, dev.gpu[0]);
        output = ggml_flash_attn_ext(ctx, inputs.q, keys, keys, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(output, inputs.sinks);
        ggml_prec_set_acc(output, GGML_PREC_F32);
        require(ggml_backend_supports_op(dev.gpu[0], output), "CUDA cannot run this attention shape; a CPU fallback would invalidate the benchmark");
        ggml_backend_sched_set_tensor_backend(scheduler, output, dev.gpu[0]);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "Cross-device graph allocation failed");
        require(ggml_backend_sched_get_tensor_backend(scheduler, packed) == dev.fused[1], "Selected keys moved off source device");
        require(ggml_backend_sched_get_tensor_backend(scheduler, output) == dev.gpu[0], "Attention moved off query device");
        std::vector<ggml_fp16_t> values(nraw + top_k, ggml_fp32_to_fp16(0.0f));
        if (compact) {
            // Duplicated rows are valid memory, but never attention members.
            std::fill(values.begin() + window, values.begin() + raw_storage, ggml_fp32_to_fp16(-INFINITY));
        } else {
            std::fill_n(values.begin(), nraw, ggml_fp32_to_fp16(-INFINITY));
            for (int index : inputs.selected_raw) values[index] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(mask, values.data(), 0, values.size() * sizeof(values[0]));
    }
    ~variant() { if (scheduler) ggml_backend_sched_free(scheduler); if (ctx) ggml_free(ctx); }
    double compute() {
        const auto start = std::chrono::steady_clock::now();
        require(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS, "Cross-device attention failed");
        ggml_backend_sched_synchronize(scheduler);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    }
    std::vector<float> result() const {
        std::vector<float> result(head * heads);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
    void check_keys(const input_fixture & inputs) const {
        std::vector<ggml_fp16_t> result(size_t(head) * (nraw + top_k));
        ggml_backend_tensor_get(packed, result.data(), 0, result.size() * sizeof(result[0]));
        for (int row = 0; row < nraw + top_k; ++row) {
            const auto * expected = row < nraw
                ? inputs.raw_values.data() + size_t(compact ? inputs.selected_raw[row < window ? row : 0] : row) * head
                : inputs.compressed_values.data() + size_t(inputs.selected_compressed[row - nraw]) * head;
            require(std::memcmp(result.data() + size_t(row) * head, expected, head * sizeof(result[0])) == 0,
                "Cross-device gather did not preserve exact selected rows");
        }
    }
};

struct error_stats {
    double maximum = 0, squared = 0, reference_squared = 0;
    bool close = true;
    void add(double value, double reference) {
        const double error = std::abs(value - reference);
        close &= std::isfinite(value) && error <= 3e-4 + 1.5e-3 * std::abs(reference);
        maximum = std::max(maximum, error);
        squared += error * error; reference_squared += reference * reference;
    }
    double relative() const { return std::sqrt(squared / std::max(reference_squared, 1e-30)); }
};

std::vector<double> oracle(const input_fixture & inputs) {
    std::vector<const ggml_fp16_t *> keys;
    for (int row : inputs.selected_raw) keys.push_back(inputs.raw_values.data() + size_t(row) * head);
    for (int row : inputs.selected_compressed) keys.push_back(inputs.compressed_values.data() + size_t(row) * head);
    std::vector<double> result(head * heads), scores(keys.size());
    for (int h = 0; h < heads; ++h) {
        double maximum = inputs.sink_values[h];
        for (size_t row = 0; row < keys.size(); ++row) {
            double dot = 0;
            for (int c = 0; c < head; ++c) dot += double(inputs.query_values[h * head + c]) * ggml_fp16_to_fp32(keys[row][c]);
            scores[row] = dot * scale; maximum = std::max(maximum, scores[row]);
        }
        double denominator = std::exp(inputs.sink_values[h] - maximum);
        for (size_t row = 0; row < keys.size(); ++row) {
            const double weight = std::exp(scores[row] - maximum);
            denominator += weight;
            for (int c = 0; c < head; ++c) result[h * head + c] += weight * ggml_fp16_to_fp32(keys[row][c]);
        }
        for (int c = 0; c < head; ++c) result[h * head + c] /= denominator;
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
} // namespace

int main(int argc, char ** argv) {
    try {
        // Graph-reuse DEBUG messages occur on every compute. Retain warnings
        // and errors without including log writes in the measured operation.
        ggml_log_set([](ggml_log_level level, const char * message, void *) {
            if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(message, stderr);
        }, nullptr);
        const int repeats = argc == 2 ? std::atoi(argv[1]) : 21;
        require(repeats >= 3 && repeats <= 1001 && argc <= 2, "Usage: GgmlOpsDsv41CompactGatherBench [3..1001 repetitions]");
        if (ggml_backend_cuda_get_device_count() < 2) {
            std::fprintf(stderr, "Requires two visible CUDA GPUs; benchmark not run\n"); return 77;
        }
        devices dev;
        for (int ring : {512, 1280}) for (int offset : {31, 255}) {
            input_fixture inputs(dev, ring, 8 * ring + offset);
            variant full(dev, inputs, false), compact(dev, inputs, true);
            full.compute(); compact.compute(); full.compute(); compact.compute();
            full.check_keys(inputs); compact.check_keys(inputs);
            const auto expected = oracle(inputs);
            const auto original = full.result(), optimized = compact.result();
            error_stats baseline_error, compact_error, difference;
            for (size_t i = 0; i < expected.size(); ++i) {
                baseline_error.add(original[i], expected[i]); compact_error.add(optimized[i], expected[i]);
                difference.add(optimized[i], original[i]);
            }
            if (!(baseline_error.close && compact_error.close && difference.close &&
                baseline_error.relative() < .0015 && compact_error.relative() < .0015)) {
                std::fprintf(stderr, "ring=%d position=%d baseline maxabs=%.9g relL2=%.9g compact maxabs=%.9g relL2=%.9g pair maxabs=%.9g\n",
                    ring, inputs.position, baseline_error.maximum, baseline_error.relative(),
                    compact_error.maximum, compact_error.relative(), difference.maximum);
                throw std::runtime_error("Attention exceeded independent F16 numerical envelope");
            }
            // The CPU oracle can let the GPUs downclock. Rewarm both paths
            // immediately before the alternating measurements.
            for (int repeat = 0; repeat < 10; ++repeat) { full.compute(); compact.compute(); }
            std::vector<double> original_times, compact_times;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                if (repeat % 2) { compact_times.push_back(compact.compute()); original_times.push_back(full.compute()); }
                else { original_times.push_back(full.compute()); compact_times.push_back(compact.compute()); }
            }
            std::printf("{\"ring\":%d,\"position\":%d,\"head\":512,\"heads\":64,\"top_k\":512,\"source_gpu\":1,\"consumer_gpu\":0,"
                "\"repeats\":%d,\"baseline_median_us\":%.3f,\"compact_median_us\":%.3f,\"speedup\":%.4f,"
                "\"baseline_oracle_max_abs\":%.9g,\"compact_oracle_max_abs\":%.9g,\"pair_max_abs\":%.9g,"
                "\"baseline_relative_l2\":%.9g,\"compact_relative_l2\":%.9g,\"exact_gather_rows\":true,"
                "\"baseline_logical_transfer_bytes\":%zu,\"compact_logical_transfer_bytes\":%zu,"
                "\"baseline_splits\":%d,\"compact_splits\":%d,\"baseline_samples_us\":[",
                ring, inputs.position, repeats, median(original_times), median(compact_times), median(original_times) / median(compact_times),
                baseline_error.maximum, compact_error.maximum, difference.maximum, baseline_error.relative(), compact_error.relative(),
                size_t(2 * ring + top_k) * head * 2 + top_k * sizeof(int32_t),
                size_t(2 * raw_storage + top_k) * head * 2 + top_k * sizeof(int32_t),
                ggml_backend_sched_get_n_splits(full.scheduler), ggml_backend_sched_get_n_splits(compact.scheduler));
            for (size_t i = 0; i < original_times.size(); ++i) std::printf("%s%.3f", i ? "," : "", original_times[i]);
            std::printf("],\"compact_samples_us\":[");
            for (size_t i = 0; i < compact_times.size(); ++i) std::printf("%s%.3f", i ? "," : "", compact_times[i]);
            std::printf("]}\n");
        }
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
