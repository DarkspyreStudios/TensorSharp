// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message)
{
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main()
{
    // Eight accelerator backends + eight fused-op backends + one CPU need
    // seventeen scheduler entries. Distinct CPU handles exercise that exact
    // capacity without requiring an eight-GPU machine in CI.
    std::array<ggml_backend_t, 17> backends{};
    for (auto & backend : backends)
    {
        backend = ggml_backend_cpu_init();
        require(backend != nullptr, "CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(backend, 1);
    }
    // Distinguish the last allocator: otherwise ggml legitimately promotes
    // the pinned operation to the first CPU handle sharing the same allocator.
    // CPU_Mapped still allocates ordinary host buffers and needs no fake backend.
    alignas(64) std::array<unsigned char, 64> mapped_storage{};
    auto * mapped = ggml_backend_cpu_buffer_from_ptr(mapped_storage.data(), mapped_storage.size());
    std::array<ggml_backend_buffer_type_t, 17> buffer_types{};
    for (size_t i = 0; i < backends.size(); ++i)
        buffer_types[i] = ggml_backend_get_default_buffer_type(backends[i]);
    buffer_types.back() = ggml_backend_buffer_get_type(mapped);
    auto * scheduler = ggml_backend_sched_new(backends.data(), buffer_types.data(),
        (int) backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(scheduler != nullptr, "17-backend scheduler creation failed");
    require(ggml_backend_sched_get_n_backends(scheduler) == 17, "scheduler lost backend entries");
    ggml_init_params params = {1024*1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "graph context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto * output = ggml_scale(ctx, input, 3.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(scheduler, output, backends.back());
    require(ggml_backend_sched_alloc_graph(scheduler, graph), "17-backend graph allocation failed");
    const float values[4] = {2, 3, -4, 0};
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    require(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS,
            "graph execution on backend seventeen failed");
    require(ggml_backend_sched_get_tensor_backend(scheduler, output) == backends.back(),
            "output was not executed on backend seventeen");
    float actual[4]{};
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    for (int i = 0; i < 4; ++i) require(actual[i] == 3*values[i], "incorrect backend-seventeen output");
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(mapped);
    ggml_free(ctx);
    for (auto backend : backends) ggml_backend_free(backend);
    std::puts("17-backend scheduler creation, allocation and computation passed");
}
