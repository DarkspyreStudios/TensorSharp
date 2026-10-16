using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaHeadGateTests
{
    [CudaTheory]
    [InlineData(1, 1, 1)]
    [InlineData(7, 3, 5)]
    [InlineData(17, 32, 64)]
    [InlineData(37, 32, 128)]
    [InlineData(3, 5, 1024)]
    public void NativeHeadGatesExactlyMatchSeparateOperationsIncludingViewsAndFloatRange(int tokens, int heads, int dim)
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator);
        int width = heads * dim;
        float[] input = Enumerable.Repeat(37f, (tokens + 2) * width).ToArray();
        float[] logitValues = Enumerable.Repeat(23f, (tokens + 2) * heads).ToArray();
        float[] samples = [0f, -0f, float.Epsilon, -float.Epsilon, 1e-30f, -1e-30f, 1f, -1f,
            80000f, -80000f, float.MaxValue, -float.MaxValue, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345))];
        float[] logits = [0f, -0f, float.Epsilon, -float.Epsilon, .25f, -.25f, 1f, -1f, 40f, -40f,
            87f, -87f, 88f, -88f, 89f, -89f, 90f, -90f, 100f, -100f, float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(unchecked((int)0x7fc23456))];
        for (int row = 0; row < tokens; ++row)
        {
            for (int channel = 0; channel < width; ++channel)
                input[(row + 1) * width + channel] = samples[(row * width + channel) % samples.Length];
            for (int head = 0; head < heads; ++head)
                logitValues[(row + 1) * heads + head] = logits[(row * heads + head) % logits.Length];
        }
        using var original = new Tensor(allocator, DType.Float32, tokens + 2, width);
        using var optimized = new Tensor(allocator, DType.Float32, tokens + 2, width);
        using var logitsParent = new Tensor(allocator, DType.Float32, tokens + 2, heads);
        original.SetElementsAsFloat(input); optimized.SetElementsAsFloat(input); logitsParent.SetElementsAsFloat(logitValues);
        using var baseline = original.Narrow(0, 1, tokens);
        using var output = optimized.Narrow(0, 1, tokens);
        using var logitView = logitsParent.Narrow(0, 1, tokens);
        for (int iteration = 1; iteration <= 2; ++iteration)
        {
            using var gates = Ops.Sigmoid(null, logitView);
            Ops.Mul(gates, gates, 2f);
            using var headGates = gates.View(tokens, heads, 1);
            using var headOutput = baseline.View(tokens, heads, dim);
            using var expanded = headGates.Expand(headOutput.Sizes);
            using var contiguous = Ops.NewContiguous(expanded);
            Ops.Mul(headOutput, headOutput, contiguous);
            Assert.True(session.TryApplyHeadGates(output, logitView, heads, dim),
                "The current LTX PTX must provide the head-gate export; fallback is not native validation.");
            Assert.Equal(iteration, session.ExecutedHeadGates);
            EqualBits(original.GetElementsAsFloat(input.Length), optimized.GetElementsAsFloat(input.Length));
            EqualBits(logitValues, logitsParent.GetElementsAsFloat(logitValues.Length));
        }
        // Releasing attention workspace must not lose the separately owned
        // optional kernel, and disposal must refuse further launches.
        session.ReleaseWorkspace();
        Assert.True(session.TryApplyHeadGates(output, logitView, heads, dim));
        Assert.Equal(3, session.ExecutedHeadGates);
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.TryApplyHeadGates(output, logitView, heads, dim));
    }

    [CudaFact]
    public void NativeHeadGatesExactlyMatchSeparateOperationsForArbitraryFiniteLogits()
    {
        const int tokens = 4096, heads = 32, dim = 3;
        float[] logits = new float[tokens * heads];
        float[] input = new float[logits.Length * dim];
        uint state = 0x91e10da5u;
        uint Next()
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }
        for (int i = 0; i < logits.Length; ++i)
        {
            uint bits = Next();
            // Half of the logits sample arbitrary finite Float32 encodings,
            // including both signs, subnormals, and very large magnitudes.
            // The remainder concentrate on sigmoid's changing/underflow range.
            if ((i & 1) == 0)
            {
                if ((bits & 0x7f800000u) == 0x7f800000u) bits ^= 0x00800000u;
                logits[i] = BitConverter.Int32BitsToSingle(unchecked((int)bits));
            }
            else
            {
                float unit = (bits >> 8) * (1f / 16777216f);
                logits[i] = (2f * unit - 1f) * ((i & 2) == 0 ? 100f : 16f);
            }
            // A unit channel exposes the gate bits directly; other channels
            // verify that the final multiplication and head broadcast agree.
            input[i * dim] = 1f;
            input[i * dim + 1] = -1f;
            input[i * dim + 2] = (Next() >> 8) * (8f / 16777216f) - 4f;
        }

        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator);
        using var original = new Tensor(allocator, DType.Float32, tokens, heads * dim);
        using var optimized = new Tensor(allocator, DType.Float32, tokens, heads * dim);
        using var deviceLogits = new Tensor(allocator, DType.Float32, tokens, heads);
        original.SetElementsAsFloat(input); optimized.SetElementsAsFloat(input); deviceLogits.SetElementsAsFloat(logits);
        using var gates = Ops.Sigmoid(null, deviceLogits);
        Ops.Mul(gates, gates, 2f);
        using var headGates = gates.View(tokens, heads, 1);
        using var headOutput = original.View(tokens, heads, dim);
        using var expanded = headGates.Expand(headOutput.Sizes);
        using var contiguous = Ops.NewContiguous(expanded);
        Ops.Mul(headOutput, headOutput, contiguous);

        Assert.True(session.TryApplyHeadGates(optimized, deviceLogits, heads, dim),
            "The current LTX PTX must provide the head-gate export; fallback is not native validation.");
        Assert.Equal(1, session.ExecutedHeadGates);
        EqualBits(original.GetElementsAsFloat(input.Length), optimized.GetElementsAsFloat(input.Length));
        EqualBits(logits, deviceLogits.GetElementsAsFloat(logits.Length));
    }

    [CudaFact]
    public void InvalidShapesAliasesAndStridesRefuseBeforeLoadingOrMutation()
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator);
        using var output = new Tensor(allocator, DType.Float32, 3, 8);
        using var logits = new Tensor(allocator, DType.Float32, 3, 2);
        Ops.Fill(output, 37f); Ops.Fill(logits, 1f);
        using var shortLogits = logits.Narrow(0, 0, 2);
        using var transpose = output.Transpose();
        using var transposeLogits = new Tensor(allocator, DType.Float32, 8, 1);
        Assert.False(session.TryApplyHeadGates(output, logits, 0, 4));
        Assert.False(session.TryApplyHeadGates(output, logits, 2, 3));
        Assert.False(session.TryApplyHeadGates(output, shortLogits, 2, 4));
        Assert.False(session.TryApplyHeadGates(output, output, 8, 1));
        Assert.False(session.TryApplyHeadGates(transpose, transposeLogits, 1, 3));
        Assert.Equal(0, session.ExecutedHeadGates);
        Assert.Null(session.HeadGateModulePath);
        Assert.All(output.GetElementsAsFloat(24), value => Assert.Equal(37f, value));
        Assert.All(logits.GetElementsAsFloat(6), value => Assert.Equal(1f, value));
    }

    private static void EqualBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"Float32 bits differ at {i}: expected {BitConverter.SingleToInt32Bits(expected[i]):X8}, actual {BitConverter.SingleToInt32Bits(actual[i]):X8}");
    }
}
