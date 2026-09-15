// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class ManagedEmbeddingQ8MatrixTests
{
    [Theory]
    [InlineData(32, 1, 1)]
    [InlineData(64, 7, 3)]
    [InlineData(96, 13, 5)]
    [InlineData(1024, 32, 19)]
    public void PackedArmMatrixMatchesDirectQ8WithRowAndColumnTails(int inputSize, int outputSize, int rows)
    {
        if (!ManagedEmbeddingQ8Matrix.IsSupported) return;
        var random = new Random(1729 + inputSize + rows);
        var weights = new float[inputSize * outputSize];
        var input = new float[inputSize * rows];
        for (int i = 0; i < weights.Length; ++i) weights[i] = (float)(random.NextDouble() * 4 - 2);
        for (int i = 0; i < input.Length; ++i) input[i] = (float)(random.NextDouble() * 8 - 4);
        // Exercise zero blocks, mixed signs and round-to-even activation ties.
        Array.Clear(input, 0, Math.Min(32, input.Length));
        if (input.Length >= 64)
        {
            input[32] = 127;
            for (int i = 33; i < 64; ++i) input[i] = (i - 48) + 0.5f;
        }
        var quantized = new byte[inputSize / 32 * 34 * outputSize];
        ManagedQuantizedOps.QuantizeRowFromFloat32((int)GgmlTensorType.Q8_0,
            weights, 0, quantized, 0, weights.Length);
        var expected = new float[rows * outputSize];
        Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32((int)GgmlTensorType.Q8_0,
            quantized, 0, inputSize, outputSize, input, 0, inputSize, rows, expected, 0, outputSize));
        var packed = new ManagedEmbeddingQ8Matrix(quantized, inputSize, outputSize);
        foreach (int threads in new[] { 1, 3 })
        {
            var actual = new float[expected.Length + 4];
            Array.Fill(actual, float.NaN);
            using var pool = new CpuWorkerPool(threads);
            packed.Multiply(input, rows, actual, pool);
            for (int i = 0; i < expected.Length; ++i)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 2e-5f * Math.Max(1, Math.Abs(expected[i])));
            Assert.All(actual.Skip(expected.Length), value => Assert.True(float.IsNaN(value)));
        }
    }
}
