// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Check split-graph scale arithmetic and the complete node properties used by
// upstream CUDA graph reuse while rebuilding a graph in stable host metadata.
#include "ggml_h3_graph_inputs.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct NodeProperties {
    ggml_tensor node;
    void* source_data[GGML_MAX_SRC];
    int64_t source_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t source_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
};
std::vector<NodeProperties> properties(ggml_cgraph* graph) {
    std::vector<NodeProperties> result(ggml_graph_n_nodes(graph));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        auto& prop = result[i];
        std::memcpy(&prop.node, node, sizeof(*node));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j]) continue;
            prop.source_data[j] = node->src[j]->data;
            std::memcpy(prop.source_ne[j], node->src[j]->ne, sizeof(prop.source_ne[j]));
            std::memcpy(prop.source_nb[j], node->src[j]->nb, sizeof(prop.source_nb[j]));
        }
    }
    return result;
}
bool same(const std::vector<NodeProperties>& a, const std::vector<NodeProperties>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(NodeProperties)) == 0;
}
struct Allocation {
    ggml_gallocr_t value;
    explicit Allocation(ggml_backend_t backend) : value(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
        require(value != nullptr, "allocator creation failed");
    }
    ~Allocation() { ggml_gallocr_free(value); }
};
struct Context {
    ggml_context* value;
    explicit Context(std::vector<unsigned char>& arena) : value(ggml_init({arena.size(), arena.data(), true})) {
        require(value != nullptr, "context creation failed");
    }
    ~Context() { ggml_free(value); }
};
void test(ggml_backend_t backend, bool cuda) {
    std::vector<unsigned char> candidate_arena(1u << 20), reference_arena(1u << 20);
    Allocation candidate_alloc(backend), reference_alloc(backend);
    std::vector<NodeProperties> previous, previous_reference;
    const void* previous_key = nullptr;
    // Realistic audio slopes, signed zero, overflow and underflow. Different
    // values on every call must affect output data but never graph properties.
    const float scales[] = {-.1f, -.370257f, -1.0f, -3.161277f, -10.0f, -.0f, -1.0e-20f, -1.75f};
    int stable_rebuilds = 0, property_resets = 0;
    for (int iteration = 0; iteration < 12; ++iteration) {
        // Grow the allocator and change topology once, then shrink it again.
        // Reuse must notice these changes, without a forced graph UID.
        const int count = iteration == 8 ? 65537 : 257;
        const float scale = scales[iteration % 8];
        Context context(candidate_arena), reference_context(reference_arena);
        auto* ctx = context.value;
        auto* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        auto* output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        ggml_set_input(input);
        auto* result = ggml_cpy(ctx, input, output);
        ggml_set_output(result);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        require(ggml_gallocr_alloc_graph(candidate_alloc.value, graph), "candidate graph allocation failed");
        auto* scale_graph = tsg::h3_output_scale_graph(ctx, output, scale);
        require(ggml_graph_n_nodes(scale_graph) <= 2, "output scale traversed the main graph");
        const auto current = properties(graph);
        if (iteration) {
            require(previous_key == ggml_graph_node(graph, 0), "stable arena changed the CUDA graph key");
            const bool topology_changed = iteration == 8 || iteration == 9;
            require(same(previous, current) != topology_changed, "graph property stability/invalidation failed");
            if (topology_changed) ++property_resets; else ++stable_rebuilds;
        }
        previous = current;
        previous_key = ggml_graph_node(graph, 0);

        auto* rctx = reference_context.value;
        auto* reference_input = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, count);
        ggml_set_input(reference_input);
        auto* reference = ggml_scale(rctx, reference_input, scale);
        ggml_set_output(reference);
        auto* reference_graph = ggml_new_graph(rctx);
        ggml_build_forward_expand(reference_graph, reference);
        require(ggml_gallocr_alloc_graph(reference_alloc.value, reference_graph), "reference graph allocation failed");
        const auto reference_properties = properties(reference_graph);
        if (iteration) require(!same(previous_reference, reference_properties), "constant SCALE failed to invalidate old graph properties");
        previous_reference = reference_properties;

        std::vector<float> values(count), actual(count), expected(count);
        for (int i = 0; i < count; ++i) values[i] = std::sin(float(i * 7 + iteration) * .03125f) * 12000.0f;
        values[0] = 0.0f; values[1] = -0.0f;
        values[2] = std::numeric_limits<float>::denorm_min(); values[3] = -values[2];
        values[4] = std::numeric_limits<float>::min(); values[5] = -values[4];
        values[6] = std::numeric_limits<float>::max(); values[7] = -values[6];
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(reference_input, values.data(), 0, ggml_nbytes(reference_input));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "candidate compute failed");
        require(ggml_backend_graph_compute(backend, scale_graph) == GGML_STATUS_SUCCESS, "output scale compute failed");
        require(ggml_backend_graph_compute(backend, reference_graph) == GGML_STATUS_SUCCESS, "reference compute failed");
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        ggml_backend_tensor_get(reference, expected.data(), 0, ggml_nbytes(reference));
        for (int i = 0; i < count; ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
                uint32_t actual_bits, expected_bits;
                std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
                std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
                std::fprintf(stderr, "scale mismatch: iteration=%d index=%d input=%g scale=%g actual=%08x expected=%08x\n",
                             iteration, i, values[i], scale, actual_bits, expected_bits);
                throw std::runtime_error("split scale changed F32 output bits");
            }
        }
    }
    std::printf("{\"backend\":\"%s\",\"exact\":true,\"stableRebuilds\":%d,\"propertyResets\":%d,"
                "\"cudaScaleCompared\":%s,\"scope\":\"scalar arithmetic and graph metadata, no model inference\"}\n",
                ggml_backend_name(backend), stable_rebuilds, property_resets, cuda ? "true" : "false");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "cpu") && std::strcmp(argv[1], "cuda"))) {
        std::fprintf(stderr, "usage: %s cpu|cuda\n", argv[0]); return 2;
    }
    const bool cuda = !std::strcmp(argv[1], "cuda");
    ggml_backend_t backend = cuda ? nullptr : ggml_backend_cpu_init();
#ifdef TSG_GGML_USE_CUDA
    if (cuda) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { std::fprintf(stderr, "requested backend unavailable\n"); return 77; }
    int status = 0;
    try { test(backend, cuda); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAILED: %s\n", e.what()); status = 1; }
    ggml_backend_free(backend);
    return status;
}
