using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public sealed class Ltx25GenerationSettingsTests
{
    [Fact]
    public void DefaultRequestUsesTheInteractiveSeventeenFramePreview()
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Seed = 42 });
        Assert.Equal((512, 512, 17, 24), (settings.Width, settings.Height, settings.Frames, settings.Fps));
        Assert.Equal(768, settings.VideoTokens);
        Assert.Equal(18, settings.AudioTokens);
    }

    [Theory]
    [InlineData(17, 24, 18)]
    [InlineData(33, 24, 34)]
    [InlineData(9, 20, 11)]
    [InlineData(17, 20, 21)]
    [InlineData(17, 10, 42)]
    [InlineData(9, 10, 22)]
    public void AudioGeometryFollowsOriginalDurationRounding(int frames, int fps, int audioTokens)
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = 64, Height = 96, Frames = frames, Fps = fps, Seed = 42 });
        Assert.Equal(audioTokens, settings.AudioTokens);
        Assert.Equal((frames - 1) / 8 + 1, settings.LatentFrames);
        Assert.Equal(2, settings.LatentWidth);
        Assert.Equal(3, settings.LatentHeight);
    }

    [Theory]
    [InlineData(16, 64, 64)]
    [InlineData(17, 63, 64)]
    [InlineData(17, 64, -1)]
    public void InvalidExplicitGeometryFailsBeforeLoadingNetworks(int frames, int width, int height)
    {
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = width, Height = height, Frames = frames }));
    }

    [Fact]
    public void UnsupportedRecipeOrConditioningDoesNotSilentlyGenerateADifferentRequest()
    {
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Steps = 4 }));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { CfgScale = 3 }));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Sampler = "unipc" }));
        Assert.NotNull(Ltx25GenerationSettings.Resolve(new VideoGenerationParams { ImagePath = "image.png" }));
        Assert.Throws<NotSupportedException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { ReferenceAudioPaths = ["reference.wav"] }));
    }
}
