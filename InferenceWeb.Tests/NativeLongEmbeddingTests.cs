// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Embeddings;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class NativeLongEmbeddingTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task ContiguousCpuAttentionPreservesPoolingAndMixedSequenceBoundaries(int pooling)
    {
        using var fixture = new EmbeddingModelTests.TinyEncoderFixture(
            pooling: pooling, projectionWeights: true, heads: 4, context: 1031);
        using var native = EmbeddingModel.Load(fixture.Path, new() { Backend = "GGML_CPU", Threads = 3 });
        using var managed = EmbeddingModel.Load(fixture.Path, new() { Backend = "CPU", Threads = 3 });
        // 1008 stays below the native layout threshold after CPU row padding;
        // 1024 selects contiguous K/V, and 1031 also exercises a partial tile.
        int[][] inputs = new[] { 3, 1008, 1024, 1031 }.Select(length =>
            Enumerable.Range(0, length).Select(position => (position * 7 + length) % 3).ToArray()).ToArray();
        var batch = await native.EmbedTokensAsync(inputs);
        var reference = await managed.EmbedTokensAsync(inputs);
        var reversed = await native.EmbedTokensAsync(inputs.Reverse().ToArray());
        var singles = new float[inputs.Length][];
        var expected = new double[inputs.Length][];
        var nativeExpected = new double[inputs.Length][];
        for (int sequence = 0; sequence < inputs.Length; ++sequence)
        {
            singles[sequence] = (await native.EmbedTokensAsync([inputs[sequence]])).Embeddings[0];
            expected[sequence] = EmbeddingModelTests.ScalarFixtureEncoder(inputs[sequence], pooling);
            nativeExpected[sequence] = EmbeddingModelTests.ScalarFixtureEncoder(inputs[sequence], pooling, nativeHalfGelu: true);
            output.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
            {
                pooling, length = inputs[sequence].Length,
                native_scalar_max_error = batch.Embeddings[sequence].Zip(expected[sequence], (a, b) => Math.Abs(a - b)).Max(),
                native_half_gelu_scalar_max_error = batch.Embeddings[sequence].Zip(nativeExpected[sequence], (a, b) => Math.Abs(a - b)).Max(),
                managed_scalar_max_error = reference.Embeddings[sequence].Zip(expected[sequence], (a, b) => Math.Abs(a - b)).Max(),
                native_managed_max_error = batch.Embeddings[sequence].Zip(reference.Embeddings[sequence], (a, b) => Math.Abs(a - b)).Max(),
                native_single_max_error = batch.Embeddings[sequence].Zip(singles[sequence], (a, b) => Math.Abs(a - b)).Max(),
                native_reverse_max_error = batch.Embeddings[sequence].Zip(reversed.Embeddings[inputs.Length - 1 - sequence], (a, b) => Math.Abs(a - b)).Max(),
                native_vector_sha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(batch.Embeddings[sequence].AsSpan()))),
            }));
        }
        Assert.Equal(inputs.Sum(input => input.Length), batch.PromptTokens);
        for (int sequence = 0; sequence < inputs.Length; ++sequence)
        {
            float[] single = singles[sequence];
            Assert.All(batch.Embeddings[sequence], value => Assert.True(float.IsFinite(value)));
            Assert.InRange(batch.Embeddings[sequence].Sum(value => (double)value * value), 0.99999, 1.00001);
            for (int dimension = 0; dimension < native.Dimensions; ++dimension)
            {
                float actual = batch.Embeddings[sequence][dimension];
                Assert.InRange(Math.Abs(actual - nativeExpected[sequence][dimension]), 0, 2e-6);
                Assert.InRange(Math.Abs(reference.Embeddings[sequence][dimension] - expected[sequence][dimension]), 0, 2e-6);
                // ggml's existing half-precision GELU table differs slightly
                // from the managed float GELU; each matches its scalar oracle.
                Assert.InRange(Math.Abs(actual - reference.Embeddings[sequence][dimension]), 0, 6e-6);
                Assert.InRange(Math.Abs(actual - single[dimension]), 0, 2e-6);
                Assert.InRange(Math.Abs(actual - reversed.Embeddings[inputs.Length - 1 - sequence][dimension]), 0, 2e-6);
            }
        }
    }
}
