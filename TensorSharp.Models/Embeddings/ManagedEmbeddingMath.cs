// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models.Embeddings;

internal static class ManagedEmbeddingMath
{
    public static int ScoreWidth => AdvSimd.Arm64.IsSupported ? 4 : Vector<float>.Count;
    public static int ValueWidth => Vector<float>.Count;
    public static int PaddedKeyStride(int tokenCount)
    {
        int width = ScoreWidth;
        // Each sequence can start at any token offset, so rounding the total
        // alone does not cover the last unaligned vector load of that sequence.
        return checked((tokenCount + 2 * width - 2) / width * width);
    }

    // Values are packed [channel tile][token][SIMD lane]. Four probability
    // vectors can be reused for every channel in a tile, without horizontal sums.
    public static unsafe void Values(float* probabilities, int length, float* values, int valueStride,
        int dimensions, float* output, int outputStride, int queries)
    {
        if (AdvSimd.Arm64.IsSupported && ValueWidth == 4)
        {
            ValuesArm(probabilities, length, values, valueStride, dimensions, output, outputStride, queries);
            return;
        }
        int width = ValueWidth;
        for (int dim = 0; dim < dimensions; dim += width)
        {
            var s0 = Vector<float>.Zero; var s1 = Vector<float>.Zero;
            var s2 = Vector<float>.Zero; var s3 = Vector<float>.Zero;
            for (int key = 0; key < length; ++key)
            {
                var v = TensorComputePrimitives.LoadVector(values + (long)dim * valueStride + key * width);
                if (Fma.IsSupported || AdvSimd.IsSupported)
                {
                    s0 = Vector.FusedMultiplyAdd(v, new Vector<float>(probabilities[key]), s0);
                    s1 = Vector.FusedMultiplyAdd(v, new Vector<float>(probabilities[length + key]), s1);
                    s2 = Vector.FusedMultiplyAdd(v, new Vector<float>(probabilities[2 * length + key]), s2);
                    s3 = Vector.FusedMultiplyAdd(v, new Vector<float>(probabilities[3 * length + key]), s3);
                }
                else
                {
                    s0 += v * probabilities[key]; s1 += v * probabilities[length + key];
                    s2 += v * probabilities[2 * length + key]; s3 += v * probabilities[3 * length + key];
                }
            }
            TensorComputePrimitives.StoreVector(output + dim, s0);
            if (queries > 1) TensorComputePrimitives.StoreVector(output + outputStride + dim, s1);
            if (queries > 2) TensorComputePrimitives.StoreVector(output + outputStride * 2 + dim, s2);
            if (queries > 3) TensorComputePrimitives.StoreVector(output + outputStride * 3 + dim, s3);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ValuesArm(float* probabilities, int length, float* values, int valueStride,
        int dimensions, float* output, int outputStride, int queries)
    {
        for (int dim = 0; dim < dimensions; dim += 4)
        {
            var s0 = Vector128<float>.Zero; var s1 = Vector128<float>.Zero;
            var s2 = Vector128<float>.Zero; var s3 = Vector128<float>.Zero;
            var t0 = Vector128<float>.Zero; var t1 = Vector128<float>.Zero;
            var t2 = Vector128<float>.Zero; var t3 = Vector128<float>.Zero;
            int key = 0;
            for (; key + 4 <= length; key += 4)
            {
                var p0 = Vector128.Load(probabilities + key); var p1 = Vector128.Load(probabilities + length + key);
                var p2 = Vector128.Load(probabilities + length * 2 + key); var p3 = Vector128.Load(probabilities + length * 3 + key);
                float* v = values + (long)dim * valueStride + key * 4;
                var v0 = Vector128.Load(v); var v1 = Vector128.Load(v + 4);
                var v2 = Vector128.Load(v + 8); var v3 = Vector128.Load(v + 12);
                s0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s0, v0, p0, 0);
                s1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s1, v0, p1, 0);
                s2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s2, v0, p2, 0);
                s3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s3, v0, p3, 0);
                t0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t0, v1, p0, 1);
                t1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t1, v1, p1, 1);
                t2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t2, v1, p2, 1);
                t3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t3, v1, p3, 1);
                s0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s0, v2, p0, 2);
                s1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s1, v2, p1, 2);
                s2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s2, v2, p2, 2);
                s3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s3, v2, p3, 2);
                t0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t0, v3, p0, 3);
                t1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t1, v3, p1, 3);
                t2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t2, v3, p2, 3);
                t3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t3, v3, p3, 3);
            }
            s0 += t0; s1 += t1; s2 += t2; s3 += t3;
            for (; key < length; ++key)
            {
                var v = Vector128.Load(values + (long)dim * valueStride + key * 4);
                s0 = AdvSimd.FusedMultiplyAdd(s0, v, Vector128.Create(probabilities[key]));
                s1 = AdvSimd.FusedMultiplyAdd(s1, v, Vector128.Create(probabilities[length + key]));
                s2 = AdvSimd.FusedMultiplyAdd(s2, v, Vector128.Create(probabilities[length * 2 + key]));
                s3 = AdvSimd.FusedMultiplyAdd(s3, v, Vector128.Create(probabilities[length * 3 + key]));
            }
            s0.Store(output + dim);
            if (queries > 1) s1.Store(output + outputStride + dim);
            if (queries > 2) s2.Store(output + outputStride * 2 + dim);
            if (queries > 3) s3.Store(output + outputStride * 3 + dim);
        }
    }

    // SIMD lanes contain adjacent keys, so scores need no horizontal reduction.
    // ARM loads four coefficients per query and uses FMA-by-lane directly; the
    // portable path lets Vector select the available x86/SIMD register width.
    public static unsafe void Scores(float* q0, float* q1, float* q2, float* q3,
        float* keys, int keyStride, int dimensions, int length, float scale, float* output)
    {
        if (AdvSimd.Arm64.IsSupported && dimensions % 4 == 0)
        {
            ScoresArm(q0, q1, q2, q3, keys, keyStride, dimensions, length, scale, output);
            return;
        }
        int width = Vector<float>.Count;
        for (int key = 0; key < length; key += width)
        {
            var s0 = Vector<float>.Zero; var s1 = Vector<float>.Zero;
            var s2 = Vector<float>.Zero; var s3 = Vector<float>.Zero;
            for (int dim = 0; dim < dimensions; ++dim)
            {
                var k = TensorComputePrimitives.LoadVector(keys + (long)dim * keyStride + key);
                if (Fma.IsSupported || AdvSimd.IsSupported)
                {
                    s0 = Vector.FusedMultiplyAdd(k, new Vector<float>(q0[dim]), s0);
                    s1 = Vector.FusedMultiplyAdd(k, new Vector<float>(q1[dim]), s1);
                    s2 = Vector.FusedMultiplyAdd(k, new Vector<float>(q2[dim]), s2);
                    s3 = Vector.FusedMultiplyAdd(k, new Vector<float>(q3[dim]), s3);
                }
                else
                {
                    s0 += k * q0[dim]; s1 += k * q1[dim];
                    s2 += k * q2[dim]; s3 += k * q3[dim];
                }
            }
            int remaining = Math.Min(width, length - key);
            Store(s0 * scale, output + key, remaining);
            Store(s1 * scale, output + length + key, remaining);
            Store(s2 * scale, output + length * 2 + key, remaining);
            Store(s3 * scale, output + length * 3 + key, remaining);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ScoresArm(float* q0, float* q1, float* q2, float* q3,
        float* keys, int keyStride, int dimensions, int length, float scale, float* output)
    {
        for (int key = 0; key < length; key += 4)
        {
            var s0 = Vector128<float>.Zero; var s1 = Vector128<float>.Zero;
            var s2 = Vector128<float>.Zero; var s3 = Vector128<float>.Zero;
            var t0 = Vector128<float>.Zero; var t1 = Vector128<float>.Zero;
            var t2 = Vector128<float>.Zero; var t3 = Vector128<float>.Zero;
            for (int dim = 0; dim < dimensions; dim += 4)
            {
                var a0 = Vector128.Load(q0 + dim); var a1 = Vector128.Load(q1 + dim);
                var a2 = Vector128.Load(q2 + dim); var a3 = Vector128.Load(q3 + dim);
                float* k = keys + (long)dim * keyStride + key;
                var k0 = Vector128.Load(k); var k1 = Vector128.Load(k + keyStride);
                var k2 = Vector128.Load(k + 2 * keyStride); var k3 = Vector128.Load(k + 3 * keyStride);
                s0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s0, k0, a0, 0);
                s1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s1, k0, a1, 0);
                s2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s2, k0, a2, 0);
                s3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s3, k0, a3, 0);
                t0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t0, k1, a0, 1);
                t1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t1, k1, a1, 1);
                t2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t2, k1, a2, 1);
                t3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t3, k1, a3, 1);
                s0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s0, k2, a0, 2);
                s1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s1, k2, a1, 2);
                s2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s2, k2, a2, 2);
                s3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(s3, k2, a3, 2);
                t0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t0, k3, a0, 3);
                t1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t1, k3, a1, 3);
                t2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t2, k3, a2, 3);
                t3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(t3, k3, a3, 3);
            }
            int remaining = Math.Min(4, length - key);
            Store((s0 + t0) * scale, output + key, remaining);
            Store((s1 + t1) * scale, output + length + key, remaining);
            Store((s2 + t2) * scale, output + length * 2 + key, remaining);
            Store((s3 + t3) * scale, output + length * 3 + key, remaining);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector128<float> value, float* output, int count)
    {
        if (count == 4) value.Store(output);
        else for (int i = 0; i < count; ++i) output[i] = value.GetElement(i);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector<float> value, float* output, int count)
    {
        if (count == Vector<float>.Count) TensorComputePrimitives.StoreVector(output, value);
        else for (int i = 0; i < count; ++i) output[i] = value[i];
    }

    // Attention computes four queries against the same key/value vector. Two
    // accumulators per query hide FMA latency and amortize loads on long rows.
    // Keep the original portable SIMD path on hardware without fused arithmetic.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Dot4(float* a0, float* a1, float* a2, float* a3, float* b, int length,
        out float r0, out float r1, out float r2, out float r3)
    {
        if (!AdvSimd.IsSupported && !Fma.IsSupported)
        {
            TensorComputePrimitives.Dot4(a0, a1, a2, a3, b, length, out r0, out r1, out r2, out r3);
            return;
        }
        int width = Vector<float>.Count;
        var s0 = Vector<float>.Zero; var s1 = Vector<float>.Zero;
        var s2 = Vector<float>.Zero; var s3 = Vector<float>.Zero;
        var t0 = Vector<float>.Zero; var t1 = Vector<float>.Zero;
        var t2 = Vector<float>.Zero; var t3 = Vector<float>.Zero;
        int i = 0;
        for (; i <= length - 2 * width; i += 2 * width)
        {
            var v0 = TensorComputePrimitives.LoadVector(b + i);
            var v1 = TensorComputePrimitives.LoadVector(b + i + width);
            s0 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a0 + i), v0, s0);
            s1 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a1 + i), v0, s1);
            s2 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a2 + i), v0, s2);
            s3 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a3 + i), v0, s3);
            t0 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a0 + i + width), v1, t0);
            t1 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a1 + i + width), v1, t1);
            t2 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a2 + i + width), v1, t2);
            t3 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a3 + i + width), v1, t3);
        }
        s0 += t0; s1 += t1; s2 += t2; s3 += t3;
        for (; i <= length - width; i += width)
        {
            var value = TensorComputePrimitives.LoadVector(b + i);
            s0 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a0 + i), value, s0);
            s1 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a1 + i), value, s1);
            s2 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a2 + i), value, s2);
            s3 = Vector.FusedMultiplyAdd(TensorComputePrimitives.LoadVector(a3 + i), value, s3);
        }
        r0 = Vector.Sum(s0); r1 = Vector.Sum(s1); r2 = Vector.Sum(s2); r3 = Vector.Sum(s3);
        for (; i < length; ++i)
        {
            float value = b[i];
            r0 += a0[i] * value; r1 += a1[i] * value;
            r2 += a2[i] * value; r3 += a3[i] * value;
        }
    }
}
