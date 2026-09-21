// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Qwen-Image-2.1: single-stream, segmented causal/image attention and shared AdaLN.
// All operations use unchanged upstream ggml. One complete graph per velocity
// prediction keeps intermediate activations on the device and weights resident.
// The t=0 prefix is currently recomputed. Caching its post-RoPE K/V is exact,
// but F32 storage costs 1 MiB per prefix token per CFG branch at 32x4096, so a
// future implementation needs explicit request lifetimes and a device budget.
#include "ggml_ops_internal.h"
#include "ggml_ops_qwen_image21.h"

using namespace tsg;
namespace {
struct Upload { ggml_tensor* tensor; const void* data; size_t bytes; };
struct Builder {
    ggml_context* ctx;
    std::vector<Upload> uploads;
    std::vector<std::vector<ggml_fp16_t>> masks;

    void bind(ggml_tensor* t, const void* data, size_t bytes) {
        if (!data || bytes != ggml_nbytes(t)) throw std::invalid_argument("QwenImage21: invalid weight descriptor");
        ggml_backend_buffer_t buffer = nullptr;
        void* address = nullptr;
        bool needs_upload = false;
        void* key = const_cast<void*>(data);
        if (try_get_cacheable_tensor_buffer(g_backend, ggml_backend_get_device(g_backend), t,
                key, bytes, buffer, address, needs_upload) &&
            ggml_backend_tensor_alloc(buffer, t, address) == GGML_STATUS_SUCCESS) {
            // Publish initialized resident buffers immediately. A failed graph allocation
            // must never leave a cache hit pointing to an uninitialized weight.
            if (needs_upload) { host_read_barrier(); ggml_backend_tensor_set(t, data, 0, bytes); }
            return;
        }
        invalidate_cached_buffer(key);
        ggml_set_input(t);
        uploads.push_back({t, data, bytes});
    }
    ggml_tensor* weight(const TSGQi21Weight& w) {
        if (w.ne0 <= 0 || w.ne1 <= 0 || w.type < 0 || w.type >= GGML_TYPE_COUNT)
            throw std::invalid_argument("QwenImage21: invalid weight shape/type");
        auto t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(w.type), w.ne0, w.ne1);
        bind(t, w.data, static_cast<size_t>(w.bytes));
        return t;
    }
    ggml_tensor* gain(void* p, int n) {
        auto t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind(t, p, n * sizeof(float));
        return t;
    }
    ggml_tensor* input(const float* p, int n0, int n1) {
        auto t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_input(t);
        uploads.push_back({t, p, ggml_nbytes(t)});
        return t;
    }
    ggml_tensor* linear(const TSGQi21Weight& w, ggml_tensor* x) {
        if (x->ne[0] != w.ne0) throw std::invalid_argument("QwenImage21: projection shape mismatch");
        return ggml_mul_mat(ctx, weight(w), x);
    }
    ggml_tensor* slice(ggml_tensor* x, int start, int length) {
        return ggml_view_2d(ctx, x, x->ne[0], length, x->nb[1], start * x->nb[1]);
    }
    ggml_tensor* modulate(ggml_tensor* x, ggml_tensor* m, int prefix, bool gate) {
        auto target = slice(m, 0, 1);
        auto preceding = slice(m, 1, 1);
        target = gate ? ggml_tanh(ctx, target) : ggml_scale_bias(ctx, target, 1.f, 1.f);
        preceding = gate ? ggml_tanh(ctx, preceding) : ggml_scale_bias(ctx, preceding, 1.f, 1.f);
        auto result = ggml_mul(ctx, slice(x, prefix, x->ne[1] - prefix), target);
        if (prefix) result = ggml_concat(ctx, ggml_mul(ctx, slice(x, 0, prefix), preceding), result, 1);
        return result;
    }
    ggml_tensor* rope(ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin, int hd, int heads, int seq) {
        const int half = hd / 2;
        auto x4 = ggml_reshape_4d(ctx, x, 2, half, heads, seq);
        auto e = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], 0);
        auto o = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], sizeof(float));
        auto c = ggml_reshape_4d(ctx, cos, 1, half, 1, seq);
        auto s = ggml_reshape_4d(ctx, sin, 1, half, 1, seq);
        auto ep = ggml_reshape_3d(ctx, ggml_sub(ctx, ggml_mul(ctx, e, c), ggml_mul(ctx, o, s)), half, heads, seq);
        auto op = ggml_reshape_3d(ctx, ggml_add(ctx, ggml_mul(ctx, o, c), ggml_mul(ctx, e, s)), half, heads, seq);
        // Q and K share the same half-split permutation; their dot product is
        // unchanged, and this avoids a slow four-dimensional interleave concat.
        return ggml_concat(ctx, ep, op, 0);
    }
    ggml_tensor* attention(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                          const TSGQi21Desc& d, const std::vector<ggml_tensor*>& mask_tensors) {
        auto qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        auto kp = ggml_permute(ctx, k, 0, 2, 1, 3);
        auto vp = ggml_permute(ctx, v, 0, 2, 1, 3);
        ggml_tensor* joined = nullptr;
        const float scale = 1.f / std::sqrt(static_cast<float>(d.head_dim));
        const char* flash_env = std::getenv("TS_QWEN21_FLASH");
        for (int i = 0; i < d.num_segments; ++i) {
            const auto& seg = d.segments[i];
            int nq = seg.end - seg.start, nk = seg.end;
            auto qs = ggml_view_3d(ctx, qp, d.head_dim, nq, d.heads, qp->nb[1], qp->nb[2], seg.start * qp->nb[1]);
            auto ks = ggml_view_3d(ctx, kp, d.head_dim, nk, d.heads, kp->nb[1], kp->nb[2], 0);
            auto vs = ggml_view_3d(ctx, vp, d.head_dim, nk, d.heads, vp->nb[1], vp->nb[2], 0);
            ggml_tensor* out = nullptr;
            if (!flash_env || flash_env[0] != '0') {
                int pad = mask_tensors[i]->ne[0] - nk;
                auto kpad = pad ? ggml_pad(ctx, ks, 0, pad, 0, 0) : ggml_cont(ctx, ks);
                auto vpad = pad ? ggml_pad(ctx, vs, 0, pad, 0, 0) : ggml_cont(ctx, vs);
                if (g_backend_type == BACKEND_TYPE_METAL) {
                    kpad = ggml_cast(ctx, kpad, GGML_TYPE_F16);
                    vpad = ggml_cast(ctx, vpad, GGML_TYPE_F16);
                }
                auto fa = ggml_flash_attn_ext(ctx, qs, kpad, vpad, mask_tensors[i], scale, 0.f, 0.f);
                ggml_prec_set_acc(fa, GGML_PREC_F32);
                if (backend_supports_op(fa)) out = ggml_reshape_2d(ctx, fa, d.dim, nq);
            }
            if (!out) {
                auto scores = ggml_mul_mat(ctx, ks, qs);
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                ggml_tensor* causal = nullptr;
                if (!seg.is_image) {
                    // DIAG_MASK_INF is unavailable in unchanged upstream Metal.
                    // Supply the exact same causal mask to the softmax instead.
                    causal = ggml_view_2d(ctx, mask_tensors[i], nk, nq, mask_tensors[i]->nb[1], 0);
                    causal = ggml_cast(ctx, ggml_cont(ctx, causal), GGML_TYPE_F32);
                }
                auto probs = ggml_soft_max_ext(ctx, scores, causal, scale, 0.f);
                auto vt = ggml_cont(ctx, ggml_permute(ctx, vs, 1, 0, 2, 3));
                auto av = ggml_mul_mat(ctx, vt, probs);
                out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, av, 0, 2, 1, 3)), d.dim, nq);
            }
            joined = joined ? ggml_concat(ctx, joined, out, 1) : out;
        }
        return joined;
    }
};
}

TSG_EXPORT int TSGgml_QwenImage21Forward(const TSGQi21Desc* d) {
    try {
        if (!d || d->struct_bytes != sizeof(TSGQi21Desc) || !d->images || !d->text || !d->output ||
            !d->time_embedding || !d->cos || !d->sin || !d->blocks || !d->segments ||
            d->dim <= 0 || d->head_dim <= 0 || d->head_dim % 2 || d->heads <= 0 ||
            d->dim != d->heads * d->head_dim || d->channels <= 0 || d->text_dim <= 0 ||
            d->num_layers <= 0 || d->num_segments <= 0 || d->image_seq <= 0 || d->text_seq <= 0 ||
            d->prefix_seq < 0 || d->total_seq <= d->prefix_seq)
            throw std::invalid_argument("QwenImage21: invalid forward descriptor");
        int end = 0;
        for (int i = 0; i < d->num_segments; ++i) {
            const auto& s = d->segments[i];
            if (s.start != end || s.end <= s.start || s.end > d->total_seq || s.source_start < 0 ||
                s.source_start + s.end - s.start > (s.is_image ? d->image_seq : d->text_seq))
                throw std::invalid_argument("QwenImage21: invalid sequence segment");
            end = s.end;
        }
        if (end != d->total_seq || d->segments[d->num_segments - 1].start != d->prefix_seq ||
            !d->segments[d->num_segments - 1].is_image)
            throw std::invalid_argument("QwenImage21: missing target image segment");
        if (!ensure_backend()) return 0;
        size_t nodes = static_cast<size_t>(d->num_layers) * (180 + d->num_segments * 45) + 1024;
        ggml_init_params init{ggml_tensor_overhead() * (nodes + 1024) + ggml_graph_overhead_custom(nodes, false), nullptr, true};
        ContextHandle context(ggml_init(init));
        if (!context.value) throw std::runtime_error("QwenImage21: graph context allocation failed");
        Builder b{context.value, {}, {}};
        auto ctx = context.value;
        auto images = b.input(d->images, d->channels, d->image_seq);
        auto text = b.input(d->text, d->text_dim, d->text_seq);
        auto time = b.input(d->time_embedding, 256, 2);
        auto cos = b.input(d->cos, d->head_dim / 2, d->total_seq);
        auto sin = b.input(d->sin, d->head_dim / 2, d->total_seq);
        std::vector<ggml_tensor*> mask_tensors;
        b.masks.reserve(d->num_segments);
        for (int i = 0; i < d->num_segments; ++i) {
            const auto& s = d->segments[i];
            int nk = (s.end + 255) / 256 * 256, nq = (s.end - s.start + 63) / 64 * 64;
            b.masks.emplace_back(static_cast<size_t>(nk) * nq, ggml_fp32_to_fp16(-INFINITY));
            auto& data = b.masks.back();
            for (int q = 0; q < s.end - s.start; ++q)
                for (int k = 0; k < (s.is_image ? s.end : s.start + q + 1); ++k) data[static_cast<size_t>(q) * nk + k] = 0;
            auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, nq);
            ggml_set_input(mask);
            b.uploads.push_back({mask, data.data(), data.size() * sizeof(ggml_fp16_t)});
            mask_tensors.push_back(mask);
        }
        time = ggml_silu(ctx, b.linear(d->time_out, ggml_silu(ctx, b.linear(d->time_in, time))));
        auto mod = b.linear(d->modulation, time);
        std::vector<ggml_tensor*> modulation;
        for (int i = 0; i < 4; ++i)
            modulation.push_back(ggml_cont(ctx, ggml_view_2d(ctx, mod, d->dim, 2, mod->nb[1], i * d->dim * sizeof(float))));
        auto norm = ggml_scale_bias(ctx, b.gain(d->text_norm, d->text_dim), 1.f, 1.f);
        text = ggml_mul(ctx, ggml_rms_norm(ctx, text, d->eps), norm);
        text = b.linear(d->text_out, ggml_gelu(ctx, b.linear(d->text_in, text)));
        images = b.linear(d->image_in, images);
        ggml_tensor* joint = nullptr;
        for (int i = 0; i < d->num_segments; ++i) {
            const auto& s = d->segments[i];
            auto h = b.slice(s.is_image ? images : text, s.source_start, s.end - s.start);
            joint = joint ? ggml_concat(ctx, joint, h, 1) : h;
        }
        for (int i = 0; i < d->num_layers; ++i) {
            const auto& w = d->blocks[i];
            auto h = b.modulate(ggml_norm(ctx, joint, d->eps), modulation[0], d->prefix_seq, false);
            auto q = ggml_reshape_3d(ctx, b.linear(w.q, h), d->head_dim, d->heads, d->total_seq);
            auto k = ggml_reshape_3d(ctx, b.linear(w.k, h), d->head_dim, d->heads, d->total_seq);
            auto v = ggml_reshape_3d(ctx, b.linear(w.v, h), d->head_dim, d->heads, d->total_seq);
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, d->eps), b.gain(w.norm_q, d->head_dim));
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, d->eps), b.gain(w.norm_k, d->head_dim));
            q = b.rope(q, cos, sin, d->head_dim, d->heads, d->total_seq);
            k = b.rope(k, cos, sin, d->head_dim, d->heads, d->total_seq);
            h = b.linear(w.out, b.attention(q, k, v, *d, mask_tensors));
            joint = ggml_add(ctx, joint, b.modulate(h, modulation[1], d->prefix_seq, true));
            h = b.modulate(ggml_norm(ctx, joint, d->eps), modulation[2], d->prefix_seq, false);
            ggml_tensor *gate, *up;
            if (!w.up.data) {
                auto gu = b.linear(w.gate, h);
                int ff = gu->ne[0] / 2;
                gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, ff, d->total_seq, gu->nb[1], 0));
                up = ggml_cont(ctx, ggml_view_2d(ctx, gu, ff, d->total_seq, gu->nb[1], ff * sizeof(float)));
            } else { gate = b.linear(w.gate, h); up = b.linear(w.up, h); }
            h = b.linear(w.down, ggml_mul(ctx, up, ggml_silu(ctx, gate)));
            joint = ggml_add(ctx, joint, b.modulate(h, modulation[3], d->prefix_seq, true));
        }
        auto target = b.slice(joint, d->prefix_seq, d->total_seq - d->prefix_seq);
        auto scale = ggml_scale_bias(ctx, b.linear(d->norm_out, b.slice(time, 0, 1)), 1.f, 1.f);
        auto output = b.linear(d->proj_out, ggml_mul(ctx, ggml_norm(ctx, target, d->eps), scale));
        ggml_set_output(output);
        auto graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(graph, output);
        if (!alloc_graph_reuse_gallocr(graph)) throw std::runtime_error("QwenImage21: graph allocation failed");
        host_read_barrier();
        // Masks of a declined/disabled flash operation are not reachable from
        // the final graph and therefore have no allocator slot.
        for (const auto& u : b.uploads)
            if (u.tensor->buffer) ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        if (compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("QwenImage21: graph compute failed");
        sync_backend(g_backend);
        ggml_backend_tensor_get(output, d->output, 0, ggml_nbytes(output));
        clear_last_error();
        return 1;
    } catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("QwenImage21: unknown native error"); return 0; }
}
