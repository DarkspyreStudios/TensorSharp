// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstddef>
#include <cstdint>

// ABI mirrored by QwenImage21Native.cs. Weights retain their original GGUF type.
struct TSGQi21Weight {
    void* data;
    std::int32_t type, reserved;
    std::int64_t ne0, ne1, bytes;
};
struct TSGQi21Block {
    TSGQi21Weight q, k, v, out, gate, up, down;
    void* norm_q;
    void* norm_k;
};
struct TSGQi21Segment {
    std::int32_t start, end, source_start, is_image;
};

// Storage of a prefix KV cache. AUTO stores what the attention kernel consumes
// (F16 for Metal and CUDA flash attention, F32 otherwise), so reading the cache
// reproduces the uncached computation. Q8_0 stores K and V in 8 bits and Q8_0_V
// only V, the counterparts of vLLM-Omni's "fp8" and "fp8_v" prefix caches; both
// are dequantized to the attention type each step, so only the prefix rounds.
enum TSGQi21PrefixCacheType : std::int32_t {
    TSG_QI21_PREFIX_AUTO = 0,
    TSG_QI21_PREFIX_F32 = 1,
    TSG_QI21_PREFIX_F16 = 2,
    TSG_QI21_PREFIX_Q8_0 = 3,
    TSG_QI21_PREFIX_Q8_0_V = 4,
};

// Nonzero TSGgml_QwenImage21Forward results: which graph produced the output.
enum TSGQi21ForwardPath : std::int32_t {
    TSG_QI21_PATH_FULL = 1,      // whole sequence, no cache requested
    TSG_QI21_PATH_EXTRACT = 2,   // whole sequence; prefix K/V stored in the cache
    TSG_QI21_PATH_CACHED = 3,    // target tokens only, attending to the cached prefix
    TSG_QI21_PATH_DECLINED = 4,  // whole sequence; the cache did not fit the device
};

struct TSGQi21Desc {
    const float* images;
    const float* text;
    const float* time_embedding;
    const float* cos;
    const float* sin;
    float* output;
    TSGQi21Weight image_in, text_in, text_out, time_in, time_out, modulation, norm_out, proj_out;
    void* text_norm;
    const TSGQi21Block* blocks;
    const TSGQi21Segment* segments;
    std::int32_t struct_bytes, dim, heads, head_dim, channels, text_dim;
    std::int32_t image_seq, text_seq, total_seq, prefix_seq, num_layers, num_segments;
    float eps;
    // Text and reference-image tokens are modulated at t=0, so their per-layer
    // K/V do not depend on the denoising step. A nonzero key names one request's
    // prefix: the first forward with a key stores the prefix K/V, later ones run
    // only the target tokens. The caller must not reuse a key for other text,
    // references or layout; release it with TSGgml_QwenImage21ReleasePrefixCache.
    std::uint64_t prefix_cache_key;
    std::int32_t prefix_cache_type;
    // Tensor parallelism (TSGgml_QwenImage21ForwardTp): the number of ranks the
    // block weights are sharded over, 0 or 1 when unsharded. A rank holds whole
    // attention heads (heads * head_dim * tp_ranks == dim) and a slice of the
    // MLP; to_out and img_mlp.out are row-parallel partial sums that the ranks
    // all-reduce, twice per block. Everything outside the blocks is replicated.
    std::int32_t tp_ranks;
};

// TSGgml_QwenImage21GetPrefixCacheInfo. state: 0 = no cache for the key,
// 1 = stored, 2 = declined (did not fit). Types are ggml_type values.
struct TSGQi21PrefixCacheInfo {
    std::int32_t state, key_type, value_type, tokens;
    std::int64_t bytes;
};

// Pinned by QwenImageNativeAbiTests on the managed side.
static_assert(sizeof(void*) != 8 || sizeof(TSGQi21Desc) == 464, "QwenImage21ForwardArgs layout");
static_assert(sizeof(void*) != 8 || offsetof(TSGQi21Desc, prefix_cache_key) == 448, "QwenImage21ForwardArgs layout");
static_assert(sizeof(TSGQi21PrefixCacheInfo) == 24, "QwenImage21PrefixCacheInfo layout");
