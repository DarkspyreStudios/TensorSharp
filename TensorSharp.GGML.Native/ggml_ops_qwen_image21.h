// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
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
};
