// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public class DiffusionChatSeedTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(42)]
    [InlineData(1234)]
    [InlineData(int.MaxValue)]
    public void ProtocolSeedsReachDiffusionSamplerUnchanged(int seed)
    {
        using var flat = JsonDocument.Parse($"{{\"seed\":{seed}}}");
        using var nested = JsonDocument.Parse($"{{\"options\":{{\"seed\":{seed}}}}}");
        SamplingConfig[] parsed = [SamplingConfigParser.ParseOpenAI(flat.RootElement),
            SamplingConfigParser.ParseWebUi(flat.RootElement), SamplingConfigParser.ParseOllama(nested.RootElement)];
        foreach (var sampling in parsed)
        {
            var first = ChatGenerationPipeline.CreateDiffusionParameters(256, 256, sampling);
            var repeat = ChatGenerationPipeline.CreateDiffusionParameters(256, 256, sampling);
            Assert.Equal(seed, first.Seed);
            Assert.Equal(first.Seed, repeat.Seed);
            Assert.Equal(seed, sampling.Seed);
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-123)]
    public void NegativeSeedKeepsNondeterministicPolicyWithoutMutatingConfiguration(int seed)
    {
        var sampling = new SamplingConfig { Seed = seed };
        var parameters = ChatGenerationPipeline.CreateDiffusionParameters(256, 256, sampling);
        Assert.InRange(parameters.Seed, 0, int.MaxValue - 1);
        Assert.Equal(seed, sampling.Seed);
        Assert.InRange(ChatGenerationPipeline.CreateDiffusionParameters(256, 256, null).Seed, 0, int.MaxValue - 1);
    }

    [Fact]
    public void SeedPropagationPreservesCanvasBudgetAndDoesNotOverflowLargeBudgets()
    {
        var sampling = new SamplingConfig { Seed = 42 };
        Assert.Equal(1, ChatGenerationPipeline.CreateDiffusionParameters(16, 256, sampling).MaxBlocks);
        Assert.Equal(2, ChatGenerationPipeline.CreateDiffusionParameters(300, 256, sampling).MaxBlocks);
        Assert.Equal(8388608, ChatGenerationPipeline.CreateDiffusionParameters(int.MaxValue, 256, sampling).MaxBlocks);
    }
}
