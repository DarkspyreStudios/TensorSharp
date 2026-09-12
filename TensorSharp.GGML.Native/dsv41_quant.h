// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define TSG_DSV41_HD __host__ __device__ __forceinline__
#else
#define TSG_DSV41_HD inline
#endif

// The reference checkpoint quantizes BF16 activations, then dequantizes back
// to BF16. The graph stores F32, so retain those two rounding boundaries here.
TSG_DSV41_HD float tsg_dsv41_bf16(float value)
{
    uint32_t bits;
#if defined(__CUDA_ARCH__)
    bits = __float_as_uint(value);
#else
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

TSG_DSV41_HD float tsg_dsv41_round_even_positive(float value)
{
    const float base = floorf(value);
    const float fraction = value - base;
    return base + (fraction > 0.5f || (fraction == 0.5f && (int(base) & 1)) ? 1.0f : 0.0f);
}

TSG_DSV41_HD float tsg_dsv41_divide(float numerator, float denominator)
{
#if defined(__CUDA_ARCH__)
    // This translation unit uses --use_fast_math for the older V4 kernels.
    // Approximate division can move exact FP4 midpoint ties across a bin.
    return __fdiv_rn(numerator, denominator);
#else
    return numerator / denominator;
#endif
}

// Finite E4M3, round-to-nearest-even with saturation. This arithmetic version
// also runs on CUDA devices without native FP8 conversion instructions.
TSG_DSV41_HD float tsg_dsv41_e4m3(float value)
{
    const float magnitude = fminf(fabsf(value), 448.0f);
    int exponent;
    frexpf(fmaxf(magnitude, 0x1p-6f), &exponent);
    const float step = ldexpf(1.0f, exponent - 4);
    return copysignf(tsg_dsv41_round_even_positive(tsg_dsv41_divide(magnitude, step)) * step, value);
}

TSG_DSV41_HD float tsg_dsv41_e2m1(float value)
{
    const float magnitude = fabsf(value);
    // Adjacent representable magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
    // Inclusive boundaries alternate so midpoint ties select an even code.
    const float rounded = magnitude <= .25f ? 0.0f : magnitude < .75f ? .5f :
        magnitude <= 1.25f ? 1.0f : magnitude < 1.75f ? 1.5f :
        magnitude <= 2.5f ? 2.0f : magnitude < 3.5f ? 3.0f :
        magnitude <= 5.0f ? 4.0f : 6.0f;
    return copysignf(rounded, value);
}

TSG_DSV41_HD float tsg_dsv41_ceil_pow2(float value)
{
    // value is positive and normal after the reference's amax clamps.
    uint32_t bits;
#if defined(__CUDA_ARCH__)
    bits = __float_as_uint(value);
#else
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    const int exponent = int((bits >> 23) & 255u) - 127 + ((bits & 0x7fffffu) != 0);
    return ldexpf(1.0f, exponent);
}

TSG_DSV41_HD float tsg_dsv41_quant_scale(float amax, int mode)
{
    if (mode == 0)
        return tsg_dsv41_ceil_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    if (mode == 1)
        return tsg_dsv41_ceil_pow2(fmaxf(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    return tsg_dsv41_e4m3(tsg_dsv41_divide(fmaxf(amax, 6.0f * 0x1p-9f), 6.0f));
}

TSG_DSV41_HD float tsg_dsv41_quant_value(float value, float scale, int mode)
{
    const float normalized = tsg_dsv41_divide(value, scale);
    const float quantized = mode == 0 ? tsg_dsv41_e4m3(normalized) : tsg_dsv41_e2m1(normalized);
    return tsg_dsv41_bf16(quantized * scale);
}

#undef TSG_DSV41_HD
