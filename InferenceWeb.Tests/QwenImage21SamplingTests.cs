using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public class QwenImage21SamplingTests
{
    [Fact]
    public void PhiloxNoiseMatchesStableDiffusionCppCudaRng()
    {
        // Golden vector from unchanged sd.cpp c678dfe core/rng_philox.hpp, seed=42.
        float[] expected = { .194018871f, 2.16137385f, -.172050610f, .849060059f,
            -1.92439914f, .652985454f, -.649441063f, -.817524731f, .527964652f,
            -1.27534986f, -1.66212630f, -.303313762f, -.0925699323f, .199237078f,
            -1.12043273f, 1.85765874f };
        var actual = QwenImage21Sampling.Noise(expected.Length, 42);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6f);
        Assert.NotEqual(actual, QwenImage21Sampling.Noise(expected.Length, 42L + (1L << 32)));
    }

    [Theory]
    [InlineData(256, .6224593312)]
    [InlineData(4096, .7595109169)]
    public void ResolutionScheduleUsesTwoPointOneAnchors(int tokens, double middleSigma)
    {
        var actual = QwenImage21Sampling.Sigmas(2, tokens);
        Assert.Equal(1f, actual[0]);
        Assert.InRange(Math.Abs(actual[1] - middleSigma), 0, 1e-7);
        Assert.Equal(0f, actual[^1]);
    }

    [Fact]
    public void LatentTransposePreservesChannelsWithoutOldPatchPacking()
    {
        float[] chw = Enumerable.Range(0, 64 * 2 * 4).Select(i => (float)i).ToArray();
        var tokens = QwenImage21Pipeline.ToTokens(chw, 2, 4);
        Assert.Equal(new float[] { 0, 8, 16, 24 }, tokens.Take(4));
        Assert.Equal(1, tokens[64]);
        Assert.Equal(chw, QwenImage21Pipeline.ToChannels(tokens, 2, 4));
    }

    [Theory]
    [InlineData(512, 0)]
    [InlineData(0, 512)]
    [InlineData(513, 512)]
    [InlineData(-32, 512)]
    public void InvalidExplicitGeometryFailsBeforeAllocating(int width, int height)
    {
        Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = width, Height = height }, null));
    }
}
