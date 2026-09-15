// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Embeddings;

namespace InferenceWeb.Tests;

public sealed class ManagedEmbeddingMathTests
{
    [Theory]
    [InlineData(1, 1, 2)]
    [InlineData(63, 4, 8)]
    [InlineData(64, 63, 16)]
    [InlineData(65, 3, 2)]
    [InlineData(65, 64, 8)]
    [InlineData(127, 3, 16)]
    [InlineData(128, 4, 8)]
    [InlineData(129, 7, 8)]
    [InlineData(1025, 64, 16)]
    public unsafe void OnlineAttentionMatchesDoubleSoftmaxAcrossTilesAndGuardsSequenceBoundaries(int length, int queryCount, int headVectors)
    {
        int width = ManagedEmbeddingMath.ValueWidth, head = width * headVectors;
        int stride = length + 11, firstToken = 3, queryStride = head + 5, outputStride = head + 3;
        var queries = new float[queryCount * queryStride + 2];
        var keys = new float[head * stride + 1];
        var values = new float[head * stride + 2];
        var actual = new float[64 * outputStride + 2];
        int scratchSize = ManagedEmbeddingLongAttention.ScratchSize(head);
        var scratch = new float[scratchSize + 7];
        foreach (float[] array in new[] { queries, keys, values, actual, scratch }) Array.Fill(array, float.NaN);
        for (int query = 0; query < queryCount; ++query)
        for (int dim = 0; dim < head; ++dim)
            queries[1 + query * queryStride + dim] = dim == 0 ? (query % 2 == 0 ? 3 : -3) : (float)(0.25 * Math.Sin(query * 7 + dim));
        for (int token = 0; token < length; ++token)
        for (int dim = 0; dim < head; ++dim)
        {
            // Successive key tiles force running maxima to change substantially;
            // alternating query signs also exercise maxima that stay unchanged.
            keys[1 + dim * stride + firstToken + token] = dim == 0 ? token / 64 * 8 : (float)Math.Cos(token * 3 + dim);
            values[2 + dim / width * stride * width + (firstToken + token) * width + dim % width] = (float)Math.Sin(token * 5 + dim);
        }
        fixed (float* q = queries)
        fixed (float* k = keys)
        fixed (float* v = values)
        fixed (float* o = actual)
            ManagedEmbeddingLongAttention.Compute(q + 1, queryStride, k + 1 + firstToken, stride,
                v + 2 + firstToken * width, stride, head, length, queryCount, o + 1, outputStride, scratch, default);
        float scale = 1.0f / MathF.Sqrt(head);
        for (int query = 0; query < queryCount; ++query)
        {
            var scores = new double[length];
            for (int token = 0; token < length; ++token)
            for (int dim = 0; dim < head; ++dim)
                scores[token] += (double)queries[1 + query * queryStride + dim] * keys[1 + dim * stride + firstToken + token] * scale;
            double max = scores.Max();
            double[] probabilities = scores.Select(score => Math.Exp(score - max)).ToArray();
            double sum = probabilities.Sum();
            for (int dim = 0; dim < head; ++dim)
            {
                double expected = 0;
                for (int token = 0; token < length; ++token)
                    expected += probabilities[token] / sum * values[2 + dim / width * stride * width + (firstToken + token) * width + dim % width];
                Assert.InRange(Math.Abs(expected - actual[1 + query * outputStride + dim]), 0, 2e-5);
            }
        }
        for (int query = 0; query < 64; ++query)
        for (int dim = query < queryCount ? head : 0; dim < outputStride; ++dim)
            Assert.True(float.IsNaN(actual[1 + query * outputStride + dim]));
        Assert.True(float.IsNaN(actual[0])); Assert.True(float.IsNaN(actual[^1]));
        Assert.All(scratch.Skip(scratchSize), value => Assert.True(float.IsNaN(value)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    public unsafe void FusedExponentialSumMatchesIndependentScalarWithUnalignedTails(int length)
    {
        var input = Enumerable.Range(0, length).Select(i => (float)(Math.Sin(i * 1.7) * 25 - i % 4 * 40)).ToArray();
        float maximum = input.Max();
        double[] expected = input.Select(value => Math.Exp((float)(value - maximum))).ToArray();
        var buffer = new float[length + 2];
        Array.Fill(buffer, float.NaN); input.CopyTo(buffer, 1);
        float sum;
        fixed (float* values = buffer) sum = ManagedEmbeddingWideMath.ExpSum(values + 1, length, maximum);
        for (int i = 0; i < length; ++i)
            Assert.InRange(Math.Abs(buffer[i + 1] - expected[i]), 0, 2e-6 * Math.Max(expected[i], 1e-38));
        Assert.InRange(Math.Abs(sum - expected.Sum()), 0, 2e-6 * expected.Sum());
        Assert.True(float.IsNaN(buffer[0])); Assert.True(float.IsNaN(buffer[^1]));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 2)]
    [InlineData(5, 3)]
    [InlineData(17, 4)]
    [InlineData(129, 4)]
    public unsafe void PackedValuesMatchDoubleReferenceAndGuardQueryAndTokenTails(int length, int queryCount)
    {
        int width = ManagedEmbeddingMath.ValueWidth, dimensions = width * 3;
        int stride = length + 5, firstToken = 2, outputStride = dimensions + 3;
        var packed = new float[dimensions * stride + 1];
        Array.Fill(packed, float.NaN);
        var probabilities = new float[length * 4];
        var dense = new float[dimensions * length];
        var random = new Random(length * 17 + queryCount);
        for (int i = 0; i < probabilities.Length; ++i) probabilities[i] = (float)random.NextDouble() / length;
        for (int dim = 0; dim < dimensions; ++dim)
        for (int token = 0; token < length; ++token)
        {
            float value = (float)(random.NextDouble() * 4 - 2);
            dense[dim * length + token] = value;
            packed[dim / width * stride * width + (firstToken + token) * width + dim % width + 1] = value;
        }
        var actual = new float[4 * outputStride + 2];
        Array.Fill(actual, float.NaN);
        fixed (float* p = probabilities)
        fixed (float* v = packed)
        fixed (float* o = actual)
            ManagedEmbeddingMath.Values(p, length, v + firstToken * width + 1, stride, dimensions, o + 1, outputStride, queryCount);
        for (int row = 0; row < 4; ++row)
        for (int dim = 0; dim < outputStride; ++dim)
        {
            float value = actual[row * outputStride + dim + 1];
            if (row >= queryCount || dim >= dimensions) Assert.True(float.IsNaN(value));
            else
            {
                double expected = Enumerable.Range(0, length).Sum(token => (double)probabilities[row * length + token] * dense[dim * length + token]);
                Assert.InRange(Math.Abs(expected - value), 0, 2e-6);
            }
        }
        Assert.True(float.IsNaN(actual[0]));
        Assert.True(float.IsNaN(actual[^1]));
    }

    [Fact]
    public void KeyStrideCoversUnalignedLastSequenceWhenTotalTokensAreAligned()
    {
        int width = ManagedEmbeddingMath.ScoreWidth;
        int tokenCount = 2 * width;
        int stride = ManagedEmbeddingMath.PaddedKeyStride(tokenCount);
        Assert.True(stride >= tokenCount + width - 1);
        Assert.Equal(0, stride % width);
    }

    [Theory]
    [InlineData(7, 1)]
    [InlineData(8, 3)]
    [InlineData(32, 5)]
    [InlineData(32, 17)]
    [InlineData(64, 65)]
    public unsafe void ScoreTilesMatchDoubleReferenceAndIgnorePaddingLanes(int dimensions, int length)
    {
        int width = ManagedEmbeddingMath.ScoreWidth;
        int stride = (length + width - 1) / width * width + 3;
        var keys = new float[dimensions * stride + 1];
        Array.Fill(keys, float.NaN);
        var queries = new float[4 * dimensions];
        var random = new Random(dimensions * 13 + length);
        for (int i = 0; i < queries.Length; ++i) queries[i] = (float)(random.NextDouble() * 2 - 1);
        for (int dim = 0; dim < dimensions; ++dim)
        for (int key = 0; key < length; ++key) keys[dim * stride + key + 1] = (float)(random.NextDouble() * 2 - 1);
        var actual = new float[length * 4 + 3];
        Array.Fill(actual, float.NaN);
        float scale = 1.0f / MathF.Sqrt(dimensions);
        fixed (float* q = queries)
        fixed (float* k = keys)
        fixed (float* output = actual)
            ManagedEmbeddingMath.Scores(q, q + dimensions, q + dimensions * 2, q + dimensions * 3,
                k + 1, stride, dimensions, length, scale, output);
        for (int query = 0; query < 4; ++query)
        for (int key = 0; key < length; ++key)
        {
            double expected = Enumerable.Range(0, dimensions).Sum(dim => (double)queries[query * dimensions + dim] * keys[dim * stride + key + 1]) * scale;
            Assert.InRange(Math.Abs(expected - actual[query * length + key]), 0, 2e-6);
        }
        Assert.All(actual.Skip(length * 4), value => Assert.True(float.IsNaN(value)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(17)]
    [InlineData(64)]
    [InlineData(129)]
    [InlineData(512)]
    public unsafe void FourQueryDotMatchesDoubleReferenceWithUnalignedTails(int length)
    {
        var random = new Random(length + 21);
        var a = new float[(length + 3) * 4];
        var b = new float[length + 5];
        for (int i = 0; i < a.Length; ++i) a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; ++i) b[i] = (float)(random.NextDouble() * 2 - 1);
        float[] actual;
        fixed (float* ap = a)
        fixed (float* bp = b)
        {
            ManagedEmbeddingMath.Dot4(ap + 1, ap + length + 4, ap + 2 * length + 7, ap + 3 * length + 10,
                bp + 3, length, out float r0, out float r1, out float r2, out float r3);
            actual = [r0, r1, r2, r3];
        }
        for (int row = 0; row < 4; ++row)
        {
            double expected = Enumerable.Range(0, length).Sum(i => (double)a[row * (length + 3) + 1 + i] * b[i + 3]);
            Assert.InRange(Math.Abs(expected - actual[row]), 0, 5e-6 * Math.Max(1, Math.Sqrt(length)));
        }
    }
}
