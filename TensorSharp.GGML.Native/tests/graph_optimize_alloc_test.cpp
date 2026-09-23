// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_graph_optimize.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {
struct request { std::vector<ggml_tensor *> keep; ggml_tensor * until; };
request * active = nullptr;
void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void optimizer(ggml_backend_t, ggml_cgraph *, ggml_backend_graph_optimize_params * params) {
    for (auto * kept : active->keep) {
        params->add_alloc_dep(params->user_data, kept, active->until);
        params->add_alloc_dep(params->user_data, kept, active->until); // duplicate must not create another edge
    }
}
void fused_read(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * data) {
    const auto * kept = static_cast<const ggml_tensor *>(data);
    const auto * a = static_cast<const float *>(kept->data);
    const auto * x = static_cast<const float *>(src->data);
    auto * y = static_cast<float *>(dst->data);
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) y[i] = x[i] + a[i];
}

void lifetime_test(ggml_backend_t backend, int mode, bool view) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    auto * a = ggml_sqr(ctx, input);
    auto * kept = view ? ggml_view_1d(ctx, a, 1024, 0) : a;
    auto * b = ggml_scale(ctx, kept, 2.0f);
    auto * c = ggml_scale(ctx, b, 3.0f);
    // A fused backend still reads a after its last formal graph consumer b.
    // Its extra allocator dependency is the sole protection against aliasing.
    auto * output = ggml_map_custom1(ctx, c, fused_read, 4, kept);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    const int original_count = graph->n_nodes;
    request req{{kept}, output}; active = &req;
    const int original_size = graph->size;
    if (mode == 1) graph->size = graph->n_nodes; // force graph-capacity fallback
    tsg_optimize_graph_with_alloc_dependencies(backend, mode == 2 ? nullptr : ctx, graph);
    graph->size = original_size;
    if (mode == 0) {
        require(graph->n_nodes == original_count + 1, "dependency was not inserted or duplicate was retained");
        auto * dep = graph->nodes[graph->n_nodes - 1];
        require(dep->op == GGML_OP_NONE && dep->src[0] == kept && dep->src[1] == nullptr,
                "allocation dependency is not a no-op source view");
        require((a->flags & GGML_TENSOR_FLAG_OUTPUT) == 0, "ordinary dependency pinned storage for entire graph");
    } else {
        require(graph->n_nodes == original_count, "fallback grew graph past capacity");
        require((a->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "fallback failed to retain storage base");
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation");
    require(a->data != b->data && a->data != c->data, "fused input reused before its final read");
    std::vector<float> values(1024), actual(1024);
    for (int i = 0; i < 1024; ++i) values[i] = float(i % 37 - 18) / 16.0f;
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute with allocation dependencies");
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (int i = 0; i < 1024; ++i)
        require(std::abs(actual[i] - 7 * values[i] * values[i]) < 1e-6, "fused read observed overwritten storage");
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
}

void grouped_dependencies_test(ggml_backend_t backend) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    auto * output = ggml_sqr(ctx, input);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    request req{{}, output};
    for (int i = 0; i < GGML_MAX_SRC + 3; ++i) req.keep.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1));
    active = &req;
    tsg_optimize_graph_with_alloc_dependencies(backend, ctx, graph);
    require(graph->n_nodes == 3, "dependencies exceeding source capacity were not split");
    for (int i = 0; i < GGML_MAX_SRC + 3; ++i)
        require(graph->nodes[1 + i / GGML_MAX_SRC]->src[i % GGML_MAX_SRC] == req.keep[i], "lost dependency source");
    req.until = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    bool rejected = false;
    try { tsg_optimize_graph_with_alloc_dependencies(backend, ctx, graph); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "out-of-graph dependency silently ignored");
    ggml_free(ctx);
}
} // namespace

int main() {
    auto backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend");
    ggml_backend_cpu_set_n_threads(backend, 4);
    auto original = backend->iface.graph_optimize;
    backend->iface.graph_optimize = optimizer;
    for (int mode = 0; mode < 3; ++mode)
        for (bool view : {false, true}) lifetime_test(backend, mode, view);
    grouped_dependencies_test(backend);
    backend->iface.graph_optimize = original;
    ggml_backend_free(backend);
    std::puts("PASS: direct graph optimizer retains fused inputs and views, including metadata-capacity fallback");
}
