using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public class QwenImage21SamplingTests
{
    [Theory]
    [InlineData(1f, 0.875f)]
    [InlineData(0.75f, 0.5f)]
    [InlineData(0.5f, 0f)]
    public void PreviewRecoversCleanFlowLatentWithoutChangingSamplingState(float sigma, float nextSigma)
    {
        // For an exact linear flow, x_t = (1-t)*clean + t*noise and
        // velocity = noise-clean. Preview must recover clean after any Euler
        // interval, including an early step whose actual state is mostly noise.
        float[] clean = { -3f, 0f, 1f, 4f };
        float[] noise = { 1f, -2f, 5f, 0f };
        float[] velocity = clean.Zip(noise, (c, n) => n - c).ToArray();
        float[] updated = clean.Zip(noise, (c, n) => (1f - sigma) * c + sigma * n).ToArray();
        for (int i = 0; i < updated.Length; i++) updated[i] += (nextSigma - sigma) * velocity[i];
        float[] expectedState = (float[])updated.Clone();
        float[] expectedVelocity = (float[])velocity.Clone();

        var preview = QwenImage21Sampling.PreviewLatents(updated, velocity, nextSigma);

        Assert.Equal(clean, preview);
        Assert.Equal(expectedState, updated);
        Assert.Equal(expectedVelocity, velocity);
        Assert.NotSame(updated, preview);
        if (nextSigma == 0f) Assert.Equal(updated, preview);
        else Assert.NotEqual(updated, preview);
    }

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
    [InlineData(256, .9843579531, .8282203674, .6143688560, .3408319354, .0601277351)]
    [InlineData(4096, .9869637489, .8528681993, .6566662788, .3819332719, .0678805709)] // 1024²
    [InlineData(8192, .9892514348, .8756618500, .6988655925, .4275377989, .0776016116)]
    [InlineData(16384, .9926459789, .9116604924, .7724375129, .5205857754, .1022295356)] // 2048²
    public void FortyStepScheduleMatchesOfficialQwenImage21Scheduler(
        int tokens, double first, double quarter, double middle, double threeQuarter, double penultimate)
    {
        // Golden values evaluated by the unmodified NumPy time_shift and
        // stretch_shift_to_terminal methods in Diffusers 6256aa7666cedd47443adc8f82da9a10e110b09c,
        // using https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json.
        // Covers both official anchors and extrapolation to native 2K (16384 tokens).
        var actual = QwenImage21Sampling.Sigmas(40, tokens);
        Assert.Equal(41, actual.Length);
        Assert.Equal(1f, actual[0]);
        int[] indices = { 1, 10, 20, 30, 38 };
        double[] expected = { first, quarter, middle, threeQuarter, penultimate };
        for (int i = 0; i < indices.Length; i++)
            Assert.InRange(Math.Abs(actual[indices[i]] - expected[i]), 0, 2e-7);
        Assert.Equal(.02f, actual[^2]);
        Assert.Equal(0f, actual[^1]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(40)]
    [InlineData(100)]
    public void ResolutionSchedulesStayFiniteAndStrictlyDecrease(int steps)
    {
        foreach (int tokens in new[] { 4, 256, 4096, 8192, 16384, 65536 })
        {
            var actual = QwenImage21Sampling.Sigmas(steps, tokens);
            Assert.Equal(steps + 1, actual.Length);
            Assert.Equal(1f, actual[0]);
            Assert.Equal(0f, actual[^1]);
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.InRange(actual[i], 0f, 1f);
                if (i > 0) Assert.True(actual[i] < actual[i - 1]);
            }
            if (steps > 1) Assert.Equal(.02f, actual[^2]);
        }
    }

    [Fact]
    public void SingleStepKeepsOneEulerUpdateWithoutDividingByZero()
    {
        Assert.Equal(new[] { 1f, 0f }, QwenImage21Sampling.Sigmas(1, 16384));
    }

    [Theory]
    [InlineData(0, 16384)]
    [InlineData(-1, 16384)]
    [InlineData(40, 0)]
    [InlineData(40, -1)]
    public void InvalidScheduleParametersAreRejected(int steps, int tokens)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImage21Sampling.Sigmas(steps, tokens));
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
