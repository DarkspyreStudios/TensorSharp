// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class JevPromptAttentionTests(ITestOutputHelper output)
{
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void FusedPromptAttention_MatchesIndependentMaterializedReference_LocalAndGlobal_ShortAndLong()
    {
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        // 2048 also exercises attention beyond the sliding window. The compatibility
        // path deliberately preserves default matmul precision rather than using flash.
        // 37 verifies exact non-power-of-two sequence extents and causal masking.
        foreach (int length in new[] { 37, 2048 })
        foreach ((int headDim, int kvHeads, int window) in new[] { (256, 4, 1024), (512, 2, 0) })
        {
            const int heads = 16;
            var random = new Random(317 + headDim);
            using var q = RandomTensor(heads, length, headDim);
            using var k = RandomTensor(kvHeads, length, headDim);
            using var v = RandomTensor(kvHeads, length, headDim);
            float[] expected;
            using (var kExpanded = Ops.RepeatInterleave(null, k, heads / kvHeads, 0))
            using (var vExpanded = Ops.RepeatInterleave(null, v, heads / kvHeads, 0))
            using (var keyTranspose = kExpanded.Transpose(1, 2))
            using (var scores = new Tensor(allocator, DType.Float32, heads, length, length))
            using (var mask = new Tensor(allocator, DType.Float32, 1, length, length))
            using (var values = new Tensor(allocator, DType.Float32, heads, length, headDim))
            {
                var maskValues = new float[length * length];
                for (int row = 0; row < length; row++)
                for (int col = 0; col < length; col++)
                    if (col > row || (window > 0 && col < row - window + 1))
                        maskValues[row * length + col] = -1e30f;
                mask.SetElementsAsFloat(maskValues);
                Ops.AddmmBatch(scores, 0, scores, 1, q, keyTranspose);
                Ops.Add(scores, scores, mask);
                Ops.Softmax(scores, scores);
                Ops.AddmmBatch(values, 0, values, 1, scores, vExpanded);
                using var transposed = values.Transpose(0, 1);
                using var flat = Ops.NewContiguous(transposed);
                expected = flat.GetElementsAsFloat(checked(length * heads * headDim));
            }
            using var actualTensor = new Tensor(allocator, DType.Float32, length, heads * headDim);
            GgmlBasicOps.FusedPrefillAttention(q, k, v, actualTensor, heads, kvHeads, headDim,
                length, length, maskStartPos: 0, slidingWindow: window, scale: 1f, inputFormat: 0,
                matchPerOpPrecision: true);
            float[] actual = actualTensor.GetElementsAsFloat(expected.Length);
            double maxAbs = 0, squaredError = 0, referenceNorm = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]));
                double error = actual[i] - expected[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(error));
                squaredError += error * error;
                referenceNorm += (double)expected[i] * expected[i];
            }
            double relativeRms = Math.Sqrt(squaredError / referenceNorm);
            output.WriteLine($"P={length} hd={headDim} kvHeads={kvHeads} window={window}: maxAbs={maxAbs:G9} relativeRms={relativeRms:G9}");
            // The rejected padded/F32-policy kernel differed by 2e-4 to 1.6e-3.
            // Exact extent/default precision must preserve the established path much more closely.
            Assert.InRange(maxAbs, 0, 2e-5);
            Assert.InRange(relativeRms, 0, 2e-5);

            Tensor RandomTensor(params long[] shape)
            {
                var tensor = new Tensor(allocator, DType.Float32, shape);
                var data = new float[checked((int)tensor.ElementCount())];
                for (int i = 0; i < data.Length; i++) data[i] = random.NextSingle() - 0.5f;
                tensor.SetElementsAsFloat(data);
                return tensor;
            }
        }
    }
}
