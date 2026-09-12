// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace InferenceWeb.Tests;

public sealed class DeepSeek41AudioInputTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedPipelineRejectsAudioIncludingMixedMediaAndEarlierTurns(bool includeImage)
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Describe this", AudioPaths = new() { "clip.wav" },
                ImagePaths = includeImage ? new() { "photo.png" } : null },
            new() { Role = "assistant", Content = "Previous reply" },
            new() { Role = "user", Content = "What was said?" },
        };
        Assert.Contains("does not support audio input",
            ChatGenerationPipeline.UnsupportedAudioInputError("deepseek41", history));
    }

    [Theory]
    [InlineData("gemma4", true)]
    [InlineData("deepseek41", false)]
    public void AudioGatePreservesOtherArchitecturesAndImageOnlyRequests(string architecture, bool audio)
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = "Describe this",
            ImagePaths = new() { "photo.png" }, AudioPaths = audio ? new() { "clip.wav" } : null } };
        Assert.Null(ChatGenerationPipeline.UnsupportedAudioInputError(architecture, history));
    }
}
