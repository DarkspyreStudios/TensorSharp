using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25PromptMathTests
{
    [Theory]
    [InlineData(1, 1, 2, 0, 10000f, 1f)]
    [InlineData(7, 3, 10, 1017, 10000f, 1f)]
    [InlineData(11, 4, 32, 1013, 1000000f, .25f)]
    [InlineData(3, 2, 14, 47, 500000f, .5f)]
    [InlineData(5, 3, 16, 1019, 10000f, 0f)]
    public void CpuRopeMatchesOriginalMathFOrder(int tokens, int heads, int dim, int position, float theta, float proportion)
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        CheckRope(allocator, tokens, heads, dim, position, theta, proportion, native: false);
    }

    [CudaTheory]
    [InlineData(1, 1, 2, 0, 10000f, 1f)]
    [InlineData(7, 3, 10, 1017, 10000f, 1f)]
    [InlineData(33, 24, 128, 991, 10000f, 1f)]
    [InlineData(11, 4, 256, 1013, 1000000f, .25f)]
    [InlineData(3, 2, 14, 47, 500000f, .5f)]
    public void CudaRopeMatchesOriginalMathFOrderExactly(int tokens, int heads, int dim, int position, float theta, float proportion)
    {
        using var allocator = new CudaAllocator();
        CheckRope(allocator, tokens, heads, dim, position, theta, proportion, native: true);
    }

    [MlxTheory]
    [InlineData(1, 1, 2, 0, 10000f, 1f)]
    [InlineData(7, 3, 10, 1017, 10000f, 1f)]
    [InlineData(33, 24, 128, 991, 10000f, 1f)]
    [InlineData(11, 4, 256, 1013, 1000000f, .25f)]
    [InlineData(3, 2, 14, 47, 500000f, .5f)]
    public void MlxRopeMatchesOriginalMathFOrderExactly(int tokens, int heads, int dim, int position, float theta, float proportion)
    {
        using var allocator = new MlxAllocator();
        CheckRope(allocator, tokens, heads, dim, position, theta, proportion, native: true);
    }

    private static void CheckRope(IAllocator allocator, int tokens, int heads, int dim, int position, float theta, float proportion, bool native)
    {
        int width = heads * dim, rotated = (int)(proportion * (dim / 2));
        var values = new float[(tokens + 2) * width];
        for (int i = 0; i < values.Length; ++i)
            values[i] = MathF.Sin(i * .173f) * (i % 11 == 0 ? 65536f : 4.13f);
        values[width] = -0f;
        var expected = (float[])values.Clone();
        // Independent copy of the original host formula: trig is deliberately
        // evaluated inside each head, before its individual BF16 products.
        for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < heads; ++head)
        for (int pair = 0; pair < rotated; ++pair)
        {
            float angle = (position + token) * (1f / MathF.Pow(theta, 2f * pair / dim));
            float c = Round(MathF.Cos(angle)), s = Round(MathF.Sin(angle));
            int a = (token + 1) * width + head * dim + pair, b = a + dim / 2;
            float x = values[a], y = values[b];
            expected[a] = Round(Round(x * c) - Round(y * s));
            expected[b] = Round(Round(y * c) + Round(x * s));
        }
        using var storage = Tensor.FromArray(allocator, values);
        using var matrix = storage.View(tokens + 2, width);
        using var input = matrix.Narrow(0, 1, tokens);
        if (native)
        {
            // Tables are also offset views. Returning false is a failure, so an
            // old module cannot silently turn this into host fallback coverage.
            var cosine = new float[tokens * rotated + 4];
            var sine = new float[cosine.Length];
            for (int token = 0; token < tokens; ++token)
            for (int pair = 0; pair < rotated; ++pair)
            {
                float angle = (position + token) * (1f / MathF.Pow(theta, 2f * pair / dim));
                cosine[2 + token * rotated + pair] = Round(MathF.Cos(angle));
                sine[2 + token * rotated + pair] = Round(MathF.Sin(angle));
            }
            using var cb = Tensor.FromArray(allocator, cosine); using var sb = Tensor.FromArray(allocator, sine);
            using var c = cb.Narrow(0, 2, tokens * rotated); using var s = sb.Narrow(0, 2, tokens * rotated);
            Assert.True(allocator is CudaAllocator
                ? CudaLtx25PromptOps.TryGemmaRope(input, c, s, heads, dim, rotated)
                : MlxLtx25PromptOps.TryGemmaRope(input, c, s, heads, dim, rotated));
            EqualBits(expected, storage.GetElementsAsFloat(values.Length));
            EqualBits(cosine, cb.GetElementsAsFloat(cosine.Length));
            EqualBits(sine, sb.GetElementsAsFloat(sine.Length));
            storage.SetElementsAsFloat(values);
        }
        Ltx25PromptMath.GemmaRope(input, heads, dim, position, theta, proportion);
        EqualBits(expected, storage.GetElementsAsFloat(values.Length));
    }

    [Fact]
    public void CpuBf16RoundingPreservesOriginalBitContract()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        CheckRound(allocator, native: false);
    }

    [CudaFact]
    public void CudaBf16RoundingPreservesTiesZerosSubnormalsAndSpecialBitPatterns()
    {
        using var allocator = new CudaAllocator();
        CheckRound(allocator, native: true);
    }

    private static void CheckRound(IAllocator allocator, bool native)
    {
        uint[] special = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
            0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x7f7fffff, 0xff7fffff,
            0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f800001, 0xff800001];
        var values = new float[4096 + special.Length + 2];
        values[0] = 37.123f; values[^1] = -37.123f;
        for (int i = 0; i < special.Length; ++i) values[i + 1] = BitConverter.UInt32BitsToSingle(special[i]);
        uint bits = 381793;
        for (int i = special.Length + 1; i < values.Length - 1; ++i)
        { bits = unchecked(bits * 1664525 + 1013904223); values[i] = BitConverter.UInt32BitsToSingle(bits); }
        var expected = (float[])values.Clone();
        for (int i = 1; i < expected.Length - 1; ++i) expected[i] = Round(expected[i]);
        using var storage = Tensor.FromArray(allocator, values);
        using var slice = storage.Narrow(0, 1, values.Length - 2);
        if (native)
        {
            Assert.True(CudaLtx25PromptOps.TryRoundBf16(slice));
            EqualBits(expected, storage.GetElementsAsFloat(values.Length));
            storage.SetElementsAsFloat(values);
        }
        Ltx25PromptMath.RoundBf16(slice);
        EqualBits(expected, storage.GetElementsAsFloat(values.Length));
    }

    [CudaFact]
    public void CudaPromptKernelsRejectUnsupportedLayoutsAndAliases()
    {
        using var allocator = new CudaAllocator();
        CheckRejections(allocator);
    }

    [MlxFact]
    public void MlxPromptRopeRejectsUnsupportedLayoutsAndAliases()
    {
        using var allocator = new MlxAllocator();
        CheckRejections(allocator);
    }

    private static void CheckRejections(IAllocator allocator)
    {
        using var x = Tensor.FromArray(allocator, new float[3, 8]);
        using var c = Tensor.FromArray(allocator, new float[6]);
        using var s = Tensor.FromArray(allocator, new float[6]);
        using var bad = x.Transpose();
        bool Rope(Tensor input, Tensor cosine, Tensor sine, int heads, int dim, int rotated) => allocator is CudaAllocator
            ? CudaLtx25PromptOps.TryGemmaRope(input, cosine, sine, heads, dim, rotated)
            : MlxLtx25PromptOps.TryGemmaRope(input, cosine, sine, heads, dim, rotated);
        Assert.False(Rope(bad, c, s, 1, 3, 1));
        Assert.False(Rope(x, c, s, 2, 4, 3));
        Assert.False(Rope(x, c, s, 2, 4, 0));
        Assert.False(Rope(x, x, s, 2, 4, 2));
        Assert.False(Rope(x, c, s, 4, 4, 2));
        using var shortTable = c.Narrow(0, 0, 5);
        Assert.False(Rope(x, shortTable, s, 2, 4, 2));
        if (allocator is CudaAllocator) Assert.False(CudaLtx25PromptOps.TryRoundBf16(bad));
    }

    private static float Round(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint rounded = (bits & 0x7f800000u) == 0x7f800000u ? bits
            : unchecked(bits + 32767u + ((bits >> 16) & 1u));
        return BitConverter.UInt32BitsToSingle(rounded & 0xffff0000u);
    }

    private static void EqualBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(actual[i]),
                $"[{i}] expected 0x{BitConverter.SingleToUInt32Bits(expected[i]):x8}, actual 0x{BitConverter.SingleToUInt32Bits(actual[i]):x8}");
    }
}
