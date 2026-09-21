// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class QwenImageResolutionTests : IDisposable
{
    private readonly string _width = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH");
    private readonly string _height = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT");

    public QwenImageResolutionTests()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", null);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", _width);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", _height);
    }

    [Fact]
    public void AutomaticTextToImageUsesNativeTwoKGeometry()
    {
        Assert.Equal((2048, 2048), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
    }

    [Theory]
    [InlineData(4, 3, 2368, 1760)]
    [InlineData(3, 4, 1760, 2368)]
    public void AutomaticEditRetainsAspectRatioAtNativeArea(int width, int height, int expectedWidth, int expectedHeight)
    {
        var reference = new RgbImage(width, height, new float[width * height * 3]);
        Assert.Equal((expectedWidth, expectedHeight),
            QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), reference));
    }

    [Fact]
    public void ExplicitDraftAreaKeepsOneKAvailable()
    {
        Assert.Equal((1024, 1024), QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { TargetArea = 1024L * 1024 }, null));
    }

    [Fact]
    public void ExplicitGeometryWinsOverAreaAndEnvironment()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "2048");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "2048");
        Assert.Equal((1024, 768), QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = 1024, Height = 768, TargetArea = 512L * 512 }, null));
    }

    [Fact]
    public void EnvironmentGeometryStillOverridesAutomaticSize()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        Assert.Equal((1536, 1024), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
    }

    [Fact]
    public void EarlierModelsRetainTheirOriginalAreaAndExplicitOverrides()
    {
        Assert.Equal(1024L * 1024, new QwenImageParams().ResolveTargetArea(version21: false));
        Assert.Equal(512L * 512,
            new QwenImageParams { TargetArea = 512L * 512 }.ResolveTargetArea(version21: false));
    }
}
