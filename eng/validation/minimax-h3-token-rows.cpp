// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Compare the old copied-token modulation graph with borrowed contiguous rows.
// --test covers allocator reuse, zero/unequal segments and strided fallback.
// Default dimensions model H3 modulation traffic, not transformer inference;
// no attention, projections, model loading, sampling or VAE is measured here.
#include "ggml_h3_token_rows.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * mods = nullptr, * result = nullptr;
    int copies = 0;

    Graph(ggml_backend_t backend, int hidden, const std::vector<int> & lengths, bool copy, int layers) {
        int tokens = 0;
        for (int n : lengths) tokens += n;
        ctx = ggml_init({8u << 20, nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        mods = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, lengths.size() * 6);
        ggml_set_input(input); ggml_set_input(mods);
        auto rows = [&](ggml_tensor * source, int start, int count) {
            if (!copy) return tsg::h3_token_rows(ctx, source, start, count);
            return ggml_cont(ctx, ggml_view_2d(ctx, source, hidden, count, source->nb[1],
                                             size_t(start) * source->nb[1]));
        };
        auto column = [&](int index) {
            return ggml_view_2d(ctx, mods, hidden, 1, mods->nb[1], size_t(index) * mods->nb[1]);
        };
        auto modulate = [&](ggml_tensor * source, int shift, int scale) {
            ggml_tensor * output = nullptr;
            int start = 0;
            for (size_t s = 0; s < lengths.size(); ++s) {
                const int count = lengths[s];
                if (!count) continue;
                auto * part = rows(source, start, count);
                part = ggml_add(ctx, ggml_add(ctx, part, ggml_mul(ctx, part, column(int(s) * 6 + scale))),
                                column(int(s) * 6 + shift));
                output = output ? ggml_concat(ctx, output, part, 1) : part;
                start += count;
            }
            return output;
        };
        auto residual = [&](ggml_tensor * base, ggml_tensor * update, int gate) {
            ggml_tensor * output = nullptr;
            int start = 0;
            for (size_t s = 0; s < lengths.size(); ++s) {
                const int count = lengths[s];
                if (!count) continue;
                auto * part = ggml_add(ctx, rows(base, start, count),
                    ggml_mul(ctx, rows(update, start, count), column(int(s) * 6 + gate)));
                output = output ? ggml_concat(ctx, output, part, 1) : part;
                start += count;
            }
            return output;
        };
        result = input;
        for (int layer = 0; layer < layers; ++layer) {
            auto * h = modulate(ggml_rms_norm(ctx, result, 1e-6f), 0, 1);
            result = residual(result, ggml_scale(ctx, h, .125f), 2);
            h = modulate(ggml_rms_norm(ctx, result, 1e-6f), 3, 4);
            result = residual(result, ggml_scale(ctx, h, -.25f), 5);
        }
        ggml_set_output(result);
        graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, result);
        for (int n = 0; n < ggml_graph_n_nodes(graph); ++n) {
            auto * node = ggml_graph_node(graph, n);
            require(ggml_backend_supports_op(backend, node), "backend does not support graph");
            copies += node->op == GGML_OP_CONT;
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    }
    ~Graph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    void fill(int round) {
        for (auto * t : {input, mods}) {
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = std::sin(float((i * 17 + round * 19) % 257) * .021f) * (t == input ? 3.0f : .2f);
            // Include signed zero and a large finite activation in every reuse.
            data[0] = round % 2 ? -0.0f : 0.0f;
            if (t == input && data.size() > 7) data[7] = 10000.0f;
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    }
    double run(ggml_backend_t backend) {
        const auto begin = std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        ggml_backend_synchronize(backend);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }
    std::vector<float> read() {
        std::vector<float> output(ggml_nelements(result));
        ggml_backend_tensor_get(result, output.data(), 0, ggml_nbytes(result));
        return output;
    }
};

void fallback_test(ggml_backend_t backend) {
    auto * ctx = ggml_init({1u << 20, nullptr, true});
    require(ctx != nullptr, "fallback context allocation failed");
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 7);
    ggml_set_input(input);
    auto * narrowed = ggml_view_2d(ctx, input, 7, 7, input->nb[1], 3 * sizeof(float));
    auto * output = tsg::h3_token_rows(ctx, narrowed, 2, 4);
    require(output->op == GGML_OP_CONT, "strided rows failed to materialize");
    auto * contiguous = tsg::h3_token_rows(ctx, input, 1, 5);
    require(contiguous->op == GGML_OP_VIEW && contiguous->view_src == input,
            "contiguous rows lost their source dependency");
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "fallback graph allocation failed");
    std::vector<float> values(16 * 7), actual(7 * 4);
    for (int i = 0; i < int(values.size()); ++i) values[i] = float(i) * .25f;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "fallback compute failed");
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (int row = 0; row < 4; ++row)
        require(std::memcmp(&actual[row * 7], &values[(row + 2) * 16 + 3], 7 * sizeof(float)) == 0,
                "strided fallback selected the wrong values");
    ggml_gallocr_free(allocator); ggml_free(ctx);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size()-1)/2] + values[values.size()/2]) * .5;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--test"))) {
        std::fprintf(stderr, "usage: %s cpu|metal|cuda [--test]\n", argv[0]); return 2;
    }
    ggml_backend_t backend = nullptr;
    if (!std::strcmp(argv[1], "cpu")) {
        backend = ggml_backend_cpu_init();
        if (backend) ggml_backend_cpu_set_n_threads(backend, 4);
    }
#ifdef TSG_GGML_USE_METAL
    if (!std::strcmp(argv[1], "metal")) backend = ggml_backend_metal_init();
#endif
#ifdef TSG_GGML_USE_CUDA
    if (!std::strcmp(argv[1], "cuda")) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { std::fprintf(stderr, "requested backend unavailable\n"); return 77; }
    try {
        fallback_test(backend);
        const bool test = argc == 3;
        const std::vector<int> lengths = test ? std::vector<int>{1, 0, 3, 17, 2, 41} :
            std::vector<int>{23, 1, 40, 760, 0, 1540};
        Graph copied(backend, test ? 24 : 5376, lengths, true, 2);
        Graph borrowed(backend, test ? 24 : 5376, lengths, false, 2);
        require(copied.copies == 60 && borrowed.copies == 0, "unexpected graph copy count");
        std::vector<double> before, after;
        for (int round = 0; round < (test ? 4 : 10); ++round) {
            copied.fill(round); borrowed.fill(round);
            double a, b;
            if (round % 2) { b = borrowed.run(backend); a = copied.run(backend); }
            else { a = copied.run(backend); b = borrowed.run(backend); }
            const auto expected = copied.read(), actual = borrowed.read();
            require(expected.size() == actual.size(), "result size mismatch");
            require(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }),
                    "nonfinite result");
            require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0,
                    "borrowed rows changed output bits");
            if (round >= 2) { before.push_back(a); after.push_back(b); }
        }
        std::printf("{\"backend\":\"%s\",\"mode\":\"%s\",\"exact\":true,\"copiedNodes\":%d,\"borrowedNodes\":%d,"
                    "\"removedCopies\":%d,\"beforeMedianMs\":%.6f,\"afterMedianMs\":%.6f,\"speedup\":%.6f,"
                    "\"scope\":\"two modulation blocks, no transformer projections or attention\"}\n",
                    ggml_backend_name(backend), test ? "regression" : "benchmark", ggml_graph_n_nodes(copied.graph),
                    ggml_graph_n_nodes(borrowed.graph), copied.copies, median(before), median(after), median(before)/median(after));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what()); ggml_backend_free(backend); return 1;
    }
    ggml_backend_free(backend); return 0;
}
