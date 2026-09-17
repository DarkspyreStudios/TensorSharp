// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_alloc.h"
#include "ggml-cpu.h"
#if defined(TSG_GGML_USE_METAL)
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    struct Graph
    {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_buffer_t cache_buffer = nullptr;
        bool owns_buffer = true;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* q = nullptr;
        ggml_tensor* k = nullptr;
        ggml_tensor* v = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* output = nullptr;
        ggml_tensor* retained_view = nullptr;
        ggml_tensor* fork_output = nullptr;
        std::vector<ggml_tensor*> attention;

        ~Graph()
        {
            if (owns_buffer)
                ggml_backend_buffer_free(buffer);
            ggml_backend_buffer_free(cache_buffer);
            ggml_free(ctx);
        }

        void build(ggml_backend_t backend, ggml_type kv_type, int queries, int kv_length,
                   int layers, bool reuse, bool retain_first, bool late_view,
                   const std::function<ggml_backend_buffer_t(std::size_t)>& allocate = {},
                   bool independent = false, bool fork_first = false)
        {
            constexpr int head_dim = 64;
            constexpr int heads = 4;
            constexpr int kv_heads = 2;
            ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
            ctx = ggml_init(params);
            require(ctx != nullptr, "Could not create graph context.");
            q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, queries, heads);
            // Decode uses a truncated view of a larger, externally bound cache.
            // Exercise both its head stride and view initialization here.
            k = ggml_new_tensor_3d(ctx, kv_type, head_dim, kv_length + 17, kv_heads);
            v = ggml_new_tensor_3d(ctx, kv_type, head_dim, kv_length + 17, kv_heads);
            ggml_tensor* k_window = ggml_view_3d(ctx, k, head_dim, kv_length, kv_heads, k->nb[1], k->nb[2], 0);
            ggml_tensor* v_window = ggml_view_3d(ctx, v, head_dim, kv_length, kv_heads, v->nb[1], v->nb[2], 0);
            const std::size_t alignment = ggml_backend_get_alignment(backend);
            const std::size_t cache_bytes = 2 * ((ggml_nbytes(k) + alignment - 1) / alignment) * alignment + alignment;
            cache_buffer = ggml_backend_alloc_buffer(backend, cache_bytes);
            require(cache_buffer != nullptr, "Could not allocate external KV cache.");
            ggml_tallocr cache_allocator = ggml_tallocr_new(cache_buffer);
            require(ggml_tallocr_alloc(&cache_allocator, k) == GGML_STATUS_SUCCESS &&
                    ggml_tallocr_alloc(&cache_allocator, v) == GGML_STATUS_SUCCESS,
                    "Could not bind external KV cache.");
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_length, queries);
            ggml_set_input(q);
            ggml_set_input(k);
            ggml_set_input(v);
            ggml_set_input(mask);
            ggml_tensor* query = q;
            std::vector<ggml_tensor*> branch_results;
            for (int i = 0; i < layers; ++i)
            {
                ggml_tensor* fa = ggml_flash_attn_ext(ctx, query, k_window, v_window, mask,
                    1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
                ggml_prec_set_acc(fa, GGML_PREC_F32);
                ggml_format_name(fa, "attention_%d", i);
                require(ggml_backend_supports_op(backend, fa), "Backend does not support test attention shape.");
                attention.push_back(fa);
                if (i == 0 && (retain_first || late_view))
                {
                    retained_view = ggml_reshape_1d(ctx, ggml_view_tensor(ctx, fa), ggml_nelements(fa));
                    if (retain_first)
                        ggml_set_output(retained_view);
                }
                // A regular intermediate consumes attention before the next
                // layer, as output projection/gating does in model decode.
                ggml_tensor* projected = ggml_scale(ctx, fa, 0.75f);
                query = independent ? q : ggml_permute(ctx, projected, 0, 2, 1, 3);
                output = projected;
                branch_results.push_back(projected);
            }
            if (independent)
                for (int i = 0; i < layers - 1; ++i)
                    output = ggml_add(ctx, output, branch_results[i]);
            if (late_view)
                output = ggml_add(ctx, ggml_reshape_1d(ctx, output, ggml_nelements(output)), retained_view);
            ggml_set_output(output);
            graph = ggml_new_graph(ctx);
            if (fork_first)
            {
                // The main chain consumes A, then an independent reader also
                // consumes A, then attention B depends only on the main chain.
                // The final reader precedes B in node order but is not B's
                // ancestor, so A/B must not share a workspace.
                fork_output = ggml_scale(ctx, attention[0], 0.5f);
                ggml_set_output(fork_output);
                ggml_build_forward_expand(graph, branch_results[0]);
                ggml_build_forward_expand(graph, fork_output);
            }
            ggml_build_forward_expand(graph, output);
            // A retained view need not be a graph node: OUTPUT alone must keep
            // its source alive for the caller's read after the final graph node.
            owns_buffer = !allocate;
            buffer = reuse || allocate
                ? tsg::alloc_ctx_tensors_with_attention_reuse(ctx, graph, backend, allocate)
                : ggml_backend_alloc_ctx_tensors(ctx, backend);
            require(buffer != nullptr, "Could not allocate graph buffer.");
            ggml_backend_buffer_clear(buffer, 0);
        }

        void upload(int seed)
        {
            auto fill = [seed](ggml_tensor* tensor, float phase)
            {
                std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(static_cast<float>(i) * 0.021f + phase + seed * 0.31f) * 0.7f;
                if (tensor->type == GGML_TYPE_F32)
                    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
                else
                {
                    std::vector<std::uint8_t> encoded(ggml_nbytes(tensor));
                    const std::size_t written = ggml_quantize_chunk(tensor->type, values.data(), encoded.data(),
                        0, values.size() / tensor->ne[0], tensor->ne[0], nullptr);
                    require(written == encoded.size(), "Unexpected quantized input byte count.");
                    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
                }
            };
            fill(q, 0.2f);
            fill(k, 0.9f);
            fill(v, 1.7f);
            std::vector<ggml_fp16_t> mask_values(static_cast<std::size_t>(ggml_nelements(mask)));
            for (int row = 0; row < mask->ne[1]; ++row)
                for (int col = 0; col < mask->ne[0]; ++col)
                    mask_values[static_cast<std::size_t>(row) * mask->ne[0] + col] =
                        ggml_fp32_to_fp16(col >= mask->ne[0] - (row % 3 + seed % 2 + 1) ? -INFINITY : 0.0f);
            ggml_backend_tensor_set(mask, mask_values.data(), 0, ggml_nbytes(mask));
        }
    };

    void compare(ggml_tensor* actual, ggml_tensor* expected)
    {
        std::vector<float> a(static_cast<std::size_t>(ggml_nelements(actual)));
        std::vector<float> b(a.size());
        ggml_backend_tensor_get(actual, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(expected, b.data(), 0, b.size() * sizeof(float));
        for (std::size_t i = 0; i < a.size(); ++i)
            require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i] - b[i]) <= 5e-5f,
                    "Reused attention storage changed numerical output.");
    }

    void verify_unique_intermediates(const Graph& test)
    {
        std::set<void*> pointers;
        std::vector<ggml_tensor*> ordinary;
        for (ggml_tensor* tensor = ggml_get_first_tensor(test.ctx); tensor != nullptr;
             tensor = ggml_get_next_tensor(test.ctx, tensor))
        {
            if (tensor->view_src == nullptr && tensor->op != GGML_OP_FLASH_ATTN_EXT && !ggml_is_empty(tensor))
            {
                require(pointers.insert(tensor->data).second, "An ordinary tensor lost its unique storage.");
                ordinary.push_back(tensor);
            }
        }
        const auto overlap = [](ggml_tensor* a, ggml_tensor* b)
        {
            if (a->buffer != b->buffer)
                return false;
            const auto first_a = reinterpret_cast<std::uintptr_t>(a->data);
            const auto first_b = reinterpret_cast<std::uintptr_t>(b->data);
            const auto last_a = first_a + ggml_backend_buffer_get_alloc_size(a->buffer, a);
            const auto last_b = first_b + ggml_backend_buffer_get_alloc_size(b->buffer, b);
            return first_a < last_b && first_b < last_a;
        };
        for (std::size_t i = 0; i < ordinary.size(); ++i)
            for (std::size_t j = i + 1; j < ordinary.size(); ++j)
                require(!overlap(ordinary[i], ordinary[j]), "Ordinary tensor allocation ranges overlap.");
        for (ggml_tensor* tensor : test.attention)
            for (ggml_tensor* other : ordinary)
                require(!overlap(tensor, other), "Attention scratch overlaps an ordinary tensor.");
    }

    void run_case(ggml_backend_t backend, ggml_type kv_type, int queries, int kv_length,
                  bool retain_first, bool late_view, int layers = 5)
    {
        Graph reference;
        Graph reused;
        reference.build(backend, kv_type, queries, kv_length, layers, false, retain_first, late_view);
        reused.build(backend, kv_type, queries, kv_length, layers, true, retain_first, late_view);
        verify_unique_intermediates(reused);
        const std::size_t old_size = ggml_backend_buffer_get_size(reference.buffer);
        const std::size_t new_size = ggml_backend_buffer_get_size(reused.buffer);
        require(new_size < old_size, "Attention workspace reuse did not reduce allocation size.");
        std::set<void*> slots;
        for (ggml_tensor* tensor : reused.attention)
            slots.insert(tensor->data);
        require(slots.size() == (retain_first || late_view ? 2u : 1u), "Unexpected attention lifetime slot count.");
        if (retain_first || late_view)
            require(reused.attention[0]->data != reused.attention[1]->data, "A live view was overwritten.");
        for (int replay = 0; replay < 4; ++replay)
        {
            reference.upload(replay);
            reused.upload(replay);
            require(ggml_backend_graph_compute(backend, reference.graph) == GGML_STATUS_SUCCESS,
                    "Reference graph execution failed.");
            require(ggml_backend_graph_compute(backend, reused.graph) == GGML_STATUS_SUCCESS,
                    "Reused graph execution failed.");
            compare(reused.output, reference.output);
            if (retain_first)
                compare(reused.retained_view, reference.retained_view);
        }
        std::cout << ggml_backend_name(backend) << " " << ggml_type_name(kv_type)
                  << " q=" << queries << " kv=" << kv_length << " retained=" << retain_first
                  << " late_view=" << late_view << " slots=" << slots.size()
                  << " bytes=" << old_size << " -> " << new_size << '\n';
    }

    void run_suite(ggml_backend_t backend)
    {
        for (ggml_type kv_type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0})
        {
            run_case(backend, kv_type, 1, 65, false, false);
            run_case(backend, kv_type, 1, 128, true, false);
            run_case(backend, kv_type, 1, 129, false, true);
            run_case(backend, kv_type, 32, 129, false, false);
            run_case(backend, kv_type, 1, 4096, false, false, 16);
        }
        run_case(backend, GGML_TYPE_F16, 1, 65, false, false, 70);
        {
            Graph reference;
            Graph independent;
            reference.build(backend, GGML_TYPE_F16, 1, 129, 5, false, false, false, {}, true);
            independent.build(backend, GGML_TYPE_F16, 1, 129, 5, true, false, false, {}, true);
            std::set<void*> slots;
            for (ggml_tensor* tensor : independent.attention)
                slots.insert(tensor->data);
            require(slots.size() == independent.attention.size(),
                    "Independent attention branches were forced to share storage.");
            reference.upload(5);
            independent.upload(5);
            require(ggml_backend_graph_compute(backend, reference.graph) == GGML_STATUS_SUCCESS &&
                    ggml_backend_graph_compute(backend, independent.graph) == GGML_STATUS_SUCCESS,
                    "Independent attention graph execution failed.");
            compare(independent.output, reference.output);
        }
        {
            Graph reference;
            Graph forked;
            reference.build(backend, GGML_TYPE_F16, 1, 129, 5, false, false, false, {}, false, true);
            forked.build(backend, GGML_TYPE_F16, 1, 129, 5, true, false, false, {}, false, true);
            std::set<void*> slots;
            for (ggml_tensor* tensor : forked.attention)
                slots.insert(tensor->data);
            require(slots.size() == 2 && forked.attention[0]->data != forked.attention[1]->data,
                    "An independent final consumer gained an attention storage dependency.");
            reference.upload(7);
            forked.upload(7);
            require(ggml_backend_graph_compute(backend, reference.graph) == GGML_STATUS_SUCCESS &&
                    ggml_backend_graph_compute(backend, forked.graph) == GGML_STATUS_SUCCESS,
                    "Forked-consumer attention graph execution failed.");
            compare(forked.output, reference.output);
            compare(forked.fork_output, reference.fork_output);
        }
        // Replan into a retained buffer across both growth and shrink, including
        // crossing the quantized prefill/decode boundary. The callback owns the
        // buffer; dropping any graph must leave it usable by the next graph.
        struct RetainedBuffer
        {
            ggml_backend_buffer_t value = nullptr;
            ~RetainedBuffer() { ggml_backend_buffer_free(value); }
        } retained;
        const auto acquire = [&](std::size_t bytes)
        {
            if (retained.value == nullptr || ggml_backend_buffer_get_size(retained.value) < bytes)
            {
                ggml_backend_buffer_free(retained.value);
                retained.value = ggml_backend_alloc_buffer(backend, bytes);
            }
            return retained.value;
        };
        for (int queries : {1, 32, 8, 1})
        {
            Graph reference;
            Graph reused;
            reference.build(backend, GGML_TYPE_Q8_0, queries, 129, 5, false, false, false);
            reused.build(backend, GGML_TYPE_Q8_0, queries, 129, 5, true, false, false, acquire);
            require(reused.buffer == retained.value, "Retained buffer ownership was not preserved.");
            verify_unique_intermediates(reused);
            reference.upload(queries);
            reused.upload(queries);
            require(ggml_backend_graph_compute(backend, reference.graph) == GGML_STATUS_SUCCESS &&
                    ggml_backend_graph_compute(backend, reused.graph) == GGML_STATUS_SUCCESS,
                    "Retained buffer graph execution failed.");
            compare(reused.output, reference.output);
        }

        // Allocation rejection must leave the graph unbound so a caller can
        // safely fall back to ggml's allocator.
        Graph rejected;
        bool failed = false;
        try
        {
            rejected.build(backend, GGML_TYPE_F16, 1, 65, 5, true, false, false,
                [](std::size_t) -> ggml_backend_buffer_t { return nullptr; });
        }
        catch (const std::runtime_error&)
        {
            failed = true;
        }
        require(failed, "Allocation rejection was not propagated.");
        for (ggml_tensor* tensor : rejected.attention)
            require(tensor->buffer == nullptr && tensor->data == nullptr,
                    "Failed allocation left dangling tensor bindings.");
        rejected.owns_buffer = true;
        rejected.buffer = ggml_backend_alloc_ctx_tensors(rejected.ctx, backend);
        require(rejected.buffer != nullptr, "Could not allocate the rejected graph through the fallback.");
        rejected.upload(3);
        require(ggml_backend_graph_compute(backend, rejected.graph) == GGML_STATUS_SUCCESS,
                "Fallback after allocation rejection failed.");
    }
}

int main()
{
    try
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend initialization failed.");
        ggml_backend_cpu_set_n_threads(cpu, 2);
        run_suite(cpu);
        ggml_backend_free(cpu);
#if defined(TSG_GGML_USE_METAL)
        ggml_backend_t metal = ggml_backend_metal_init();
        require(metal != nullptr, "Metal backend initialization failed.");
        run_suite(metal);
        ggml_backend_free(metal);
#endif
        std::cout << "Attention allocation regression tests passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
