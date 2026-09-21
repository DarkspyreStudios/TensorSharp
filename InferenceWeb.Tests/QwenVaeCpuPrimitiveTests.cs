// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class QwenVaeCpuPrimitiveTests
{
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 1, 255)]
    [InlineData(96, 1, 256)]
    [InlineData(144, 1, 257)]
    [InlineData(768, 3, 257)]
    [InlineData(1152, 1, 513)]
    public void TiledChannelNormMatchesScalarReferenceExactly(int channels, int height, int width)
    {
        var rng = new Random(210);
        int pixels = height * width;
        var values = new float[channels * pixels];
        var gamma = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            gamma[c] = (float)(rng.NextDouble() * 4 - 2);
            for (int p = 0; p < pixels; p++)
            {
                float value = (float)(rng.NextDouble() * 20 - 10);
                // Exercise epsilon, signed zero and the need for double sums.
                values[c * pixels + p] = (p % 7) switch
                {
                    0 => c % 2 == 0 ? 0f : -0f,
                    1 => value * 1e-20f,
                    2 => value * 1e20f,
                    _ => value,
                };
            }
        }
        var original = (float[])values.Clone();
        var expected = ScalarNorm(values, gamma, channels, pixels);
        var actual = VaeReferenceMath.RmsNormChannel(new Feature(channels, height, width, values), gamma);
        Assert.Equal((channels, height, width), (actual.C, actual.H, actual.W));
        Assert.Equal(original, values);
        AssertSameBits(expected, actual.D);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(31)]
    [InlineData(65536)]
    [InlineData(131071)]
    [InlineData(131072)]
    [InlineData(196621)]
    public void ChunkedSiluMatchesScalarReferenceExactly(int length)
    {
        var values = new float[length];
        float[] boundaries = { -100f, -20f, -1f, -0f, 0f, 1e-20f, 1f, 20f, 100f };
        var rng = new Random(21);
        for (int i = 0; i < values.Length; i++)
            values[i] = i % 2 == 0 ? boundaries[i % boundaries.Length] : (float)(rng.NextDouble() * 40 - 20);
        var expected = (float[])values.Clone();
        for (int i = 0; i < expected.Length; i++)
        {
            float value = expected[i];
            expected[i] = value / (1f + MathF.Exp(-value));
        }
        VaeReferenceMath.SiluInPlace(values);
        AssertSameBits(expected, values);
    }

    private static float[] ScalarNorm(float[] values, float[] gamma, int channels, int pixels)
    {
        var result = new float[values.Length];
        float scale = MathF.Sqrt(channels);
        for (int p = 0; p < pixels; p++)
        {
            double squares = 0;
            for (int c = 0; c < channels; c++)
            {
                float value = values[c * pixels + p];
                squares += (double)value * value;
            }
            float inverse = (float)(1.0 / Math.Sqrt(squares + 1e-12));
            for (int c = 0; c < channels; c++)
                result[c * pixels + p] = values[c * pixels + p] * inverse * scale * gamma[c];
        }
        return result;
    }

    private static void AssertSameBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }
}
