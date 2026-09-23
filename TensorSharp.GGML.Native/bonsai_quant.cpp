// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "bonsai_quant.h"
#include "ggml.h"

#include <cstring>
#include <limits>

namespace {
constexpr int block_elements = 128;
constexpr size_t pq2_bytes = 34, ptq1_bytes = 28, q2_bytes = 18;

// Return unsigned ternary codes (0=-1, 1=0, 2=+1). PQ2 also represents +2
// with code 3; preserving it makes this transcode lossless for all legal codes.
void unpack(int type, const uint8_t * src, uint8_t (&codes)[block_elements], ggml_fp16_t & scale) {
    if (type == TSG_BONSAI_PQ2_0) {
        std::memcpy(&scale, src, sizeof(scale));
        for (int j = 0; j < block_elements; ++j)
            codes[j] = (src[2 + j / 4] >> (2 * (j % 4))) & 3;
        return;
    }
    std::memcpy(&scale, src + 26, sizeof(scale));
    // PTQ1 packs 16 lanes x 5 trits, then 8 x 5, then 2 x 4.
    // Each encoded byte is ceil(base3_number * 256 / 243). Multiplication
    // modulo 256 exposes the successive most significant ternary digits.
    constexpr unsigned powers[] = {1, 3, 9, 27, 81};
    int out = 0;
    for (int stage = 0; stage < 2; ++stage) {
        const int lanes = stage == 0 ? 16 : 8;
        const int offset = stage == 0 ? 0 : 16;
        for (unsigned power : powers)
            for (int lane = 0; lane < lanes; ++lane)
                codes[out++] = (uint8_t(src[offset + lane] * power) * 3u) >> 8;
    }
    for (int trit = 0; trit < 4; ++trit)
        for (int lane = 0; lane < 2; ++lane)
            codes[out++] = (uint8_t(src[24 + lane] * powers[trit]) * 3u) >> 8;
}

bool valid(int type, const void * src, int64_t elements, const void * dst) {
    return tsg_bonsai_quant_type(type) && src && dst && elements >= 0 &&
        elements % block_elements == 0 &&
        uint64_t(elements) <= std::numeric_limits<size_t>::max() / sizeof(float);
}
} // namespace

bool tsg_bonsai_quant_type(int type) {
    return type == TSG_BONSAI_PQ2_0 || type == TSG_BONSAI_PTQ1_0;
}

size_t tsg_bonsai_row_size(int type, int64_t elements) {
    if (!tsg_bonsai_quant_type(type) || elements <= 0 || elements % block_elements)
        return 0;
    const size_t bytes = type == TSG_BONSAI_PQ2_0 ? pq2_bytes : ptq1_bytes;
    if (uint64_t(elements / block_elements) > std::numeric_limits<size_t>::max() / bytes)
        return 0;
    return size_t(elements / block_elements) * bytes;
}

int tsg_bonsai_dequantize(int type, const void * src, int64_t elements, float * dst) {
    if (!valid(type, src, elements, dst)) return -1;
    const auto * input = static_cast<const uint8_t *>(src);
    const size_t stride = type == TSG_BONSAI_PQ2_0 ? pq2_bytes : ptq1_bytes;
    for (int64_t block = 0; block < elements / block_elements; ++block) {
        uint8_t codes[block_elements];
        ggml_fp16_t scale;
        unpack(type, input + size_t(block) * stride, codes, scale);
        const float d = ggml_fp16_to_fp32(scale);
        for (int j = 0; j < block_elements; ++j)
            dst[block * block_elements + j] = float(int(codes[j]) - 1) * d;
    }
    return 0;
}

int tsg_bonsai_transcode_q2_0(int type, const void * src, int64_t elements, void * dst) {
    if (!valid(type, src, elements, dst)) return -1;
    // A future upstream layout change must fail explicitly, never silently
    // reinterpret the bytes. No floating-point requantization occurs here.
    if (ggml_blck_size(GGML_TYPE_Q2_0) != 64 || ggml_type_size(GGML_TYPE_Q2_0) != q2_bytes)
        return -2;
    const size_t source_bytes = tsg_bonsai_row_size(type, elements);
    const size_t destination_bytes = size_t(elements / 64) * q2_bytes;
    const uintptr_t source_address = reinterpret_cast<uintptr_t>(src);
    const uintptr_t destination_address = reinterpret_cast<uintptr_t>(dst);
    if ((destination_address >= source_address && destination_address - source_address < source_bytes) ||
        (source_address > destination_address && source_address - destination_address < destination_bytes))
        return -1; // expanding in place would overwrite a later source block
    const auto * input = static_cast<const uint8_t *>(src);
    auto * output = static_cast<uint8_t *>(dst);
    const size_t stride = type == TSG_BONSAI_PQ2_0 ? pq2_bytes : ptq1_bytes;
    for (int64_t block = 0; block < elements / block_elements; ++block) {
        const uint8_t * source = input + size_t(block) * stride;
        uint8_t * target = output + size_t(block) * 2 * q2_bytes;
        if (type == TSG_BONSAI_PQ2_0) {
            // PQ2 is already in Q2's sequential 2-bit layout: just duplicate
            // the binary16 scale for the two 64-element blocks.
            std::memcpy(target, source, 2);
            std::memcpy(target + 2, source + 2, 16);
            std::memcpy(target + q2_bytes, source, 2);
            std::memcpy(target + q2_bytes + 2, source + 18, 16);
        } else {
            uint8_t codes[block_elements];
            ggml_fp16_t scale;
            unpack(type, source, codes, scale);
            for (int half = 0; half < 2; ++half) {
                uint8_t * packed = target + size_t(half) * q2_bytes;
                std::memcpy(packed, &scale, 2);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t * c = codes + half * 64 + j * 4;
                    packed[2 + j] = c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6);
                }
            }
        }
    }
    return 0;
}
