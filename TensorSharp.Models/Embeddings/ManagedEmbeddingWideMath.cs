// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace TensorSharp.Models.Embeddings;

// Long attention holds four query rows and sixteen output columns in registers.
// Reusing each query/probability vector across four column vectors reduces loads
// while the surrounding bounded K/V tiles remain in cache.
internal static class ManagedEmbeddingWideMath
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void Scores(float* q0, float* q1, float* q2, float* q3,
        float* keys, int keyStride, int dimensions, int length, float scale, float* output)
    {
        if (!AdvSimd.Arm64.IsSupported || dimensions % 4 != 0)
        {
            ManagedEmbeddingMath.Scores(q0, q1, q2, q3, keys, keyStride, dimensions, length, scale, output);
            return;
        }
        for (int key = 0; key < length; key += 16)
        {
            var c00 = Vector128<float>.Zero; var c01 = Vector128<float>.Zero; var c02 = Vector128<float>.Zero; var c03 = Vector128<float>.Zero;
            var c10 = Vector128<float>.Zero; var c11 = Vector128<float>.Zero; var c12 = Vector128<float>.Zero; var c13 = Vector128<float>.Zero;
            var c20 = Vector128<float>.Zero; var c21 = Vector128<float>.Zero; var c22 = Vector128<float>.Zero; var c23 = Vector128<float>.Zero;
            var c30 = Vector128<float>.Zero; var c31 = Vector128<float>.Zero; var c32 = Vector128<float>.Zero; var c33 = Vector128<float>.Zero;
            for (int dim = 0; dim < dimensions; dim += 4)
            {
                var a0 = Vector128.Load(q0 + dim); var a1 = Vector128.Load(q1 + dim);
                var a2 = Vector128.Load(q2 + dim); var a3 = Vector128.Load(q3 + dim);
                {
                    var k0 = Vector128.Load(keys + (long)(dim + 0) * keyStride + key + 0);
                    var k1 = Vector128.Load(keys + (long)(dim + 0) * keyStride + key + 4);
                    var k2 = Vector128.Load(keys + (long)(dim + 0) * keyStride + key + 8);
                    var k3 = Vector128.Load(keys + (long)(dim + 0) * keyStride + key + 12);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, k0, a0, 0);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, k1, a0, 0);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, k2, a0, 0);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, k3, a0, 0);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, k0, a1, 0);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, k1, a1, 0);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, k2, a1, 0);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, k3, a1, 0);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, k0, a2, 0);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, k1, a2, 0);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, k2, a2, 0);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, k3, a2, 0);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, k0, a3, 0);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, k1, a3, 0);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, k2, a3, 0);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, k3, a3, 0);
                }
                {
                    var k0 = Vector128.Load(keys + (long)(dim + 1) * keyStride + key + 0);
                    var k1 = Vector128.Load(keys + (long)(dim + 1) * keyStride + key + 4);
                    var k2 = Vector128.Load(keys + (long)(dim + 1) * keyStride + key + 8);
                    var k3 = Vector128.Load(keys + (long)(dim + 1) * keyStride + key + 12);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, k0, a0, 1);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, k1, a0, 1);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, k2, a0, 1);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, k3, a0, 1);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, k0, a1, 1);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, k1, a1, 1);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, k2, a1, 1);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, k3, a1, 1);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, k0, a2, 1);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, k1, a2, 1);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, k2, a2, 1);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, k3, a2, 1);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, k0, a3, 1);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, k1, a3, 1);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, k2, a3, 1);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, k3, a3, 1);
                }
                {
                    var k0 = Vector128.Load(keys + (long)(dim + 2) * keyStride + key + 0);
                    var k1 = Vector128.Load(keys + (long)(dim + 2) * keyStride + key + 4);
                    var k2 = Vector128.Load(keys + (long)(dim + 2) * keyStride + key + 8);
                    var k3 = Vector128.Load(keys + (long)(dim + 2) * keyStride + key + 12);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, k0, a0, 2);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, k1, a0, 2);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, k2, a0, 2);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, k3, a0, 2);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, k0, a1, 2);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, k1, a1, 2);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, k2, a1, 2);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, k3, a1, 2);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, k0, a2, 2);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, k1, a2, 2);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, k2, a2, 2);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, k3, a2, 2);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, k0, a3, 2);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, k1, a3, 2);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, k2, a3, 2);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, k3, a3, 2);
                }
                {
                    var k0 = Vector128.Load(keys + (long)(dim + 3) * keyStride + key + 0);
                    var k1 = Vector128.Load(keys + (long)(dim + 3) * keyStride + key + 4);
                    var k2 = Vector128.Load(keys + (long)(dim + 3) * keyStride + key + 8);
                    var k3 = Vector128.Load(keys + (long)(dim + 3) * keyStride + key + 12);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, k0, a0, 3);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, k1, a0, 3);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, k2, a0, 3);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, k3, a0, 3);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, k0, a1, 3);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, k1, a1, 3);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, k2, a1, 3);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, k3, a1, 3);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, k0, a2, 3);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, k1, a2, 3);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, k2, a2, 3);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, k3, a2, 3);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, k0, a3, 3);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, k1, a3, 3);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, k2, a3, 3);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, k3, a3, 3);
                }
            }
            int remaining = Math.Min(16, length - key);
            Store(c00 * scale, c01 * scale, c02 * scale, c03 * scale, output + 0 * length + key, remaining);
            Store(c10 * scale, c11 * scale, c12 * scale, c13 * scale, output + 1 * length + key, remaining);
            Store(c20 * scale, c21 * scale, c22 * scale, c23 * scale, output + 2 * length + key, remaining);
            Store(c30 * scale, c31 * scale, c32 * scale, c33 * scale, output + 3 * length + key, remaining);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void Values(float* probabilities, int length, float* values, int valueStride,
        int dimensions, float* output, int outputStride, int queries)
    {
        if (!AdvSimd.Arm64.IsSupported || dimensions % 16 != 0 || ManagedEmbeddingMath.ValueWidth != 4)
        {
            ManagedEmbeddingMath.Values(probabilities, length, values, valueStride, dimensions, output, outputStride, queries);
            return;
        }
        for (int dim = 0; dim < dimensions; dim += 16)
        {
            var c00 = Vector128<float>.Zero; var c01 = Vector128<float>.Zero; var c02 = Vector128<float>.Zero; var c03 = Vector128<float>.Zero;
            var c10 = Vector128<float>.Zero; var c11 = Vector128<float>.Zero; var c12 = Vector128<float>.Zero; var c13 = Vector128<float>.Zero;
            var c20 = Vector128<float>.Zero; var c21 = Vector128<float>.Zero; var c22 = Vector128<float>.Zero; var c23 = Vector128<float>.Zero;
            var c30 = Vector128<float>.Zero; var c31 = Vector128<float>.Zero; var c32 = Vector128<float>.Zero; var c33 = Vector128<float>.Zero;
            int key = 0;
            for (; key + 4 <= length; key += 4)
            {
                var p0 = Vector128.Load(probabilities + key); var p1 = Vector128.Load(probabilities + length + key);
                var p2 = Vector128.Load(probabilities + 2 * length + key); var p3 = Vector128.Load(probabilities + 3 * length + key);
                {
                    var v0 = Vector128.Load(values + (long)(dim + 0) * valueStride + (key + 0) * 4);
                    var v1 = Vector128.Load(values + (long)(dim + 4) * valueStride + (key + 0) * 4);
                    var v2 = Vector128.Load(values + (long)(dim + 8) * valueStride + (key + 0) * 4);
                    var v3 = Vector128.Load(values + (long)(dim + 12) * valueStride + (key + 0) * 4);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, v0, p0, 0);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, v1, p0, 0);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, v2, p0, 0);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, v3, p0, 0);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, v0, p1, 0);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, v1, p1, 0);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, v2, p1, 0);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, v3, p1, 0);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, v0, p2, 0);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, v1, p2, 0);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, v2, p2, 0);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, v3, p2, 0);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, v0, p3, 0);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, v1, p3, 0);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, v2, p3, 0);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, v3, p3, 0);
                }
                {
                    var v0 = Vector128.Load(values + (long)(dim + 0) * valueStride + (key + 1) * 4);
                    var v1 = Vector128.Load(values + (long)(dim + 4) * valueStride + (key + 1) * 4);
                    var v2 = Vector128.Load(values + (long)(dim + 8) * valueStride + (key + 1) * 4);
                    var v3 = Vector128.Load(values + (long)(dim + 12) * valueStride + (key + 1) * 4);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, v0, p0, 1);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, v1, p0, 1);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, v2, p0, 1);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, v3, p0, 1);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, v0, p1, 1);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, v1, p1, 1);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, v2, p1, 1);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, v3, p1, 1);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, v0, p2, 1);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, v1, p2, 1);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, v2, p2, 1);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, v3, p2, 1);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, v0, p3, 1);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, v1, p3, 1);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, v2, p3, 1);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, v3, p3, 1);
                }
                {
                    var v0 = Vector128.Load(values + (long)(dim + 0) * valueStride + (key + 2) * 4);
                    var v1 = Vector128.Load(values + (long)(dim + 4) * valueStride + (key + 2) * 4);
                    var v2 = Vector128.Load(values + (long)(dim + 8) * valueStride + (key + 2) * 4);
                    var v3 = Vector128.Load(values + (long)(dim + 12) * valueStride + (key + 2) * 4);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, v0, p0, 2);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, v1, p0, 2);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, v2, p0, 2);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, v3, p0, 2);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, v0, p1, 2);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, v1, p1, 2);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, v2, p1, 2);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, v3, p1, 2);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, v0, p2, 2);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, v1, p2, 2);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, v2, p2, 2);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, v3, p2, 2);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, v0, p3, 2);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, v1, p3, 2);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, v2, p3, 2);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, v3, p3, 2);
                }
                {
                    var v0 = Vector128.Load(values + (long)(dim + 0) * valueStride + (key + 3) * 4);
                    var v1 = Vector128.Load(values + (long)(dim + 4) * valueStride + (key + 3) * 4);
                    var v2 = Vector128.Load(values + (long)(dim + 8) * valueStride + (key + 3) * 4);
                    var v3 = Vector128.Load(values + (long)(dim + 12) * valueStride + (key + 3) * 4);
                    c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, v0, p0, 3);
                    c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, v1, p0, 3);
                    c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, v2, p0, 3);
                    c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, v3, p0, 3);
                    c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, v0, p1, 3);
                    c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, v1, p1, 3);
                    c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, v2, p1, 3);
                    c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, v3, p1, 3);
                    c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, v0, p2, 3);
                    c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, v1, p2, 3);
                    c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, v2, p2, 3);
                    c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, v3, p2, 3);
                    c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, v0, p3, 3);
                    c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, v1, p3, 3);
                    c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, v2, p3, 3);
                    c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, v3, p3, 3);
                }
            }
            for (; key < length; ++key)
            {
                var p0 = Vector128.Create(probabilities[0 * length + key]);
                var p1 = Vector128.Create(probabilities[1 * length + key]);
                var p2 = Vector128.Create(probabilities[2 * length + key]);
                var p3 = Vector128.Create(probabilities[3 * length + key]);
                var v0 = Vector128.Load(values + (long)(dim + 0) * valueStride + key * 4);
                var v1 = Vector128.Load(values + (long)(dim + 4) * valueStride + key * 4);
                var v2 = Vector128.Load(values + (long)(dim + 8) * valueStride + key * 4);
                var v3 = Vector128.Load(values + (long)(dim + 12) * valueStride + key * 4);
                c00 = AdvSimd.FusedMultiplyAdd(c00, v0, p0);
                c01 = AdvSimd.FusedMultiplyAdd(c01, v1, p0);
                c02 = AdvSimd.FusedMultiplyAdd(c02, v2, p0);
                c03 = AdvSimd.FusedMultiplyAdd(c03, v3, p0);
                c10 = AdvSimd.FusedMultiplyAdd(c10, v0, p1);
                c11 = AdvSimd.FusedMultiplyAdd(c11, v1, p1);
                c12 = AdvSimd.FusedMultiplyAdd(c12, v2, p1);
                c13 = AdvSimd.FusedMultiplyAdd(c13, v3, p1);
                c20 = AdvSimd.FusedMultiplyAdd(c20, v0, p2);
                c21 = AdvSimd.FusedMultiplyAdd(c21, v1, p2);
                c22 = AdvSimd.FusedMultiplyAdd(c22, v2, p2);
                c23 = AdvSimd.FusedMultiplyAdd(c23, v3, p2);
                c30 = AdvSimd.FusedMultiplyAdd(c30, v0, p3);
                c31 = AdvSimd.FusedMultiplyAdd(c31, v1, p3);
                c32 = AdvSimd.FusedMultiplyAdd(c32, v2, p3);
                c33 = AdvSimd.FusedMultiplyAdd(c33, v3, p3);
            }
            if (queries > 0) Store(c00, c01, c02, c03, output + 0 * outputStride + dim, 16);
            if (queries > 1) Store(c10, c11, c12, c13, output + 1 * outputStride + dim, 16);
            if (queries > 2) Store(c20, c21, c22, c23, output + 2 * outputStride + dim, 16);
            if (queries > 3) Store(c30, c31, c32, c33, output + 3 * outputStride + dim, 16);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector128<float> a, Vector128<float> b, Vector128<float> c, Vector128<float> d,
        float* output, int count)
    {
        if (count >= 4) a.Store(output);
        else for (int i = 0; i < count; ++i) output[i] = a.GetElement(i);
        if (count >= 8) b.Store(output + 4);
        else for (int i = 4; i < count; ++i) output[i] = b.GetElement(i - 4);
        if (count >= 12) c.Store(output + 8);
        else for (int i = 8; i < count; ++i) output[i] = c.GetElement(i - 8);
        if (count >= 16) d.Store(output + 12);
        else for (int i = 12; i < count; ++i) output[i] = d.GetElement(i - 12);
    }

    // Fuse subtraction, the runtime's FP32 vector exponential, storage and sum.
    // This uses the same floating-point exp operation, with fewer span passes.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe float ExpSum(float* values, int length, float maximum)
    {
        int index = 0;
        float sum = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            var offset = new Vector<float>(maximum);
            var sum0 = Vector<float>.Zero; var sum1 = Vector<float>.Zero;
            for (; index + 2 * width <= length; index += 2 * width)
            {
                var a = Vector.Exp(TensorComputePrimitives.LoadVector(values + index) - offset);
                var b = Vector.Exp(TensorComputePrimitives.LoadVector(values + index + width) - offset);
                TensorComputePrimitives.StoreVector(values + index, a);
                TensorComputePrimitives.StoreVector(values + index + width, b);
                sum0 += a; sum1 += b;
            }
            for (; index + width <= length; index += width)
            {
                var a = Vector.Exp(TensorComputePrimitives.LoadVector(values + index) - offset);
                TensorComputePrimitives.StoreVector(values + index, a);
                sum0 += a;
            }
            sum = Vector.Sum(sum0 + sum1);
        }
        for (; index < length; ++index)
        {
            float value = MathF.Exp(values[index] - maximum);
            values[index] = value;
            sum += value;
        }
        return sum;
    }
}
