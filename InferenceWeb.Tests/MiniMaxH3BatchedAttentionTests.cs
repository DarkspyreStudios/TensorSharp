using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3BatchedAttentionTests
{
    private static Tensor FromFloats(CudaAllocator allocator, float[] values, int rows, int columns)
    {
        using var flat = Tensor.FromArray(allocator, values);
        return flat.View(rows, columns);
    }

    [CudaFact]
    public void PedanticBatchedAttentionMatchesDoubleReferenceAcrossChunksAndHeadStrides()
    {
        using var allocator = new CudaAllocator();
        foreach (var shape in new[] { (Queries: 9, Keys: 13, Heads: 3, Head: 7), (Queries: 17, Keys: 17, Heads: 4, Head: 128) })
        {
            int width = shape.Heads * shape.Head;
            float[] q = Enumerable.Range(0, shape.Queries * width).Select(i => MathF.Sin(i * .131f) + (i % 5) * .000071f).ToArray();
            float[] k = Enumerable.Range(0, shape.Keys * width).Select(i => MathF.Cos(i * .173f) - (i % 7) * .000093f).ToArray();
            float[] v = Enumerable.Range(0, k.Length).Select(i => MathF.Sin(i * .317f) * 80000f).ToArray();
            using var qt = FromFloats(allocator, q, shape.Queries, width);
            using var kt = FromFloats(allocator, k, shape.Keys, width);
            using var vt = FromFloats(allocator, v, shape.Keys, width);
            using var output = new Tensor(allocator, DType.Float32, shape.Queries, width);
            float scale = 1f / MathF.Sqrt(shape.Head);
            foreach (var setting in new[]
            {
                (Chunk: 1, Budget: 64 << 20), (Chunk: 3, Budget: 64 << 20),
                (Chunk: 0, Budget: 64 << 20), (Chunk: 0, Budget: sizeof(float) * shape.Heads * shape.Keys * 2),
            })
            {
                Ops.Fill(output, float.NaN);
                Assert.True(CudaMiniMaxH3Ops.TryBatchedAttention(output, qt, kt, vt, shape.Heads, shape.Head, scale,
                    maximumScoreBytes: setting.Budget, maximumQueryRows: setting.Chunk));
                float[] actual = output.GetElementsAsFloat(q.Length);
                double squaredError = 0, squaredReference = 0;
                for (int row = 0; row < shape.Queries; ++row)
                for (int head = 0; head < shape.Heads; ++head)
                {
                    var probabilities = new double[shape.Keys];
                    for (int key = 0; key < shape.Keys; ++key)
                    for (int channel = 0; channel < shape.Head; ++channel)
                        probabilities[key] += (double)q[row * width + head * shape.Head + channel] *
                            k[key * width + head * shape.Head + channel] * scale;
                    double maximum = probabilities.Max(), sum = 0;
                    for (int key = 0; key < shape.Keys; ++key) sum += probabilities[key] = Math.Exp(probabilities[key] - maximum);
                    for (int channel = 0; channel < shape.Head; ++channel)
                    {
                        double expected = 0;
                        for (int key = 0; key < shape.Keys; ++key)
                            expected += probabilities[key] / sum * v[key * width + head * shape.Head + channel];
                        float value = actual[row * width + head * shape.Head + channel];
                        Assert.True(float.IsFinite(value));
                        squaredError += (value - expected) * (value - expected);
                        squaredReference += expected * expected;
                    }
                }
                Assert.InRange(Math.Sqrt(squaredError / squaredReference), 0, 2e-6);
            }
        }
    }

    [CudaFact]
    public void BatchedAttentionPreservesHostDirtySentinelsAroundOffsetOutput()
    {
        using var allocator = new CudaAllocator();
        const int rows = 7, heads = 3, head = 8, width = heads * head;
        float[] values = Enumerable.Range(0, rows * width).Select(i => MathF.Sin(i * .17f)).ToArray();
        using var q = FromFloats(allocator, values, rows, width);
        using var k = FromFloats(allocator, values, rows, width);
        using var v = FromFloats(allocator, values, rows, width);
        using var storage = FromFloats(allocator, Enumerable.Repeat(123.75f, (rows + 2) * width).ToArray(), rows + 2, width);
        using var output = storage.Narrow(0, 1, rows);
        using var expected = new Tensor(allocator, DType.Float32, rows, width);
        Assert.True(CudaMiniMaxH3Ops.TryBatchedAttention(expected, q, k, v, heads, head, .25f, maximumQueryRows: 3));
        Assert.True(CudaMiniMaxH3Ops.TryBatchedAttention(output, q, k, v, heads, head, .25f, maximumQueryRows: 3));
        float[] actual = storage.GetElementsAsFloat((rows + 2) * width);
        Assert.All(actual[..width], value => Assert.Equal(123.75f, value));
        Assert.All(actual[((rows + 1) * width)..], value => Assert.Equal(123.75f, value));
        Assert.Equal(expected.GetElementsAsFloat(rows * width), actual[width..((rows + 1) * width)]);
    }

    [CudaFact]
    public void BatchedAttentionRejectsAliasingInvalidGeometryAndInsufficientScoreBudget()
    {
        using var allocator = new CudaAllocator();
        using var q = new Tensor(allocator, DType.Float32, 7, 32);
        using var k = new Tensor(allocator, DType.Float32, 9, 32);
        using var v = new Tensor(allocator, DType.Float32, 9, 32);
        using var output = new Tensor(allocator, DType.Float32, 7, 32);
        Ops.Fill(output, 17);
        Assert.False(CudaMiniMaxH3Ops.TryBatchedAttention(q, q, k, v, 2, 16, .25f));
        Assert.False(CudaMiniMaxH3Ops.TryBatchedAttention(output, q, k, v, 3, 16, .25f));
        Assert.False(CudaMiniMaxH3Ops.TryBatchedAttention(output, q, k, v, 2, 16, float.NaN));
        Assert.False(CudaMiniMaxH3Ops.TryBatchedAttention(output, q, k, v, 2, 16, .25f, maximumScoreBytes: 8));
        Assert.False(CudaMiniMaxH3Ops.TryBatchedAttention(output, q, k, q, 2, 16, .25f));
        Assert.All(output.GetElementsAsFloat(7 * 32), value => Assert.Equal(17, value));
    }
}
