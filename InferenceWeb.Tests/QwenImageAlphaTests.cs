// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class QwenImageAlphaTests
{
    [Fact]
    public void RgbaRoundTrip_PreservesStraightColorAndAlpha()
    {
        byte[] rgba = { 255, 0, 128, 0, 32, 64, 96, 128, 0, 255, 64, 255 };
        byte[] input = MediaCodecs.Image.EncodePng(rgba, 3, 1, 4);
        var decoded = ImageIO.Decode(input, preserveAlpha: true);
        Assert.NotNull(decoded.Alpha);
        Assert.Equal(1f, decoded.Pixels[0]); // transparent RGB is retained, not premultiplied.
        Assert.Equal(0f, decoded.Alpha![0]);
        Assert.Equal(128f / 255f, decoded.Alpha[1]);
        byte[] output = MediaCodecs.Image.DecodeRgba(ImageIO.EncodePng(decoded), out int w, out int h);
        Assert.Equal((3, 1), (w, h));
        Assert.Equal(rgba, output);
    }

    [Fact]
    public void DefaultDecodeAndRgbConstructor_PreserveTheOpaqueContract()
    {
        byte[] input = MediaCodecs.Image.EncodePng(new byte[] { 20, 40, 60, 0 }, 1, 1, 4);
        var legacy = ImageIO.Decode(input);
        Assert.Null(legacy.Alpha);
        byte[] encoded = MediaCodecs.Image.DecodeRgba(ImageIO.EncodePng(legacy), out _, out _);
        Assert.Equal(new byte[] { 20, 40, 60, 255 }, encoded);
        Assert.Null(new RgbImage(1, 1, new float[3]).Alpha);
    }

    [Fact]
    public void RgbImage_RejectsAnAlphaPlaneWithDifferentGeometry() =>
        Assert.Throws<ArgumentException>(() => new RgbImage(2, 1, new float[6], new float[1]));

    [Fact]
    public void ResizeCover_CropsAlphaWithTheIdenticalPixelCoordinates()
    {
        var image = new RgbImage(4, 2, new float[24], new[] { 0f, .2f, .4f, .6f, .1f, .3f, .5f, .7f });
        var cropped = ImageIO.ResizeCover(image, 2, 2);
        Assert.Equal(new[] { .2f, .4f, .3f, .5f }, cropped.Alpha);
    }

    [Fact]
    public void Resize_FiltersAlphaOnTheSameGeometryAsRgb()
    {
        var alpha = new[] { 0f, 1f, 128f / 255f, 64f / 255f };
        var rgb = new float[12];
        for (int i = 0; i < alpha.Length; i++)
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = alpha[i];
        var resized = ImageIO.Resize(new RgbImage(2, 2, rgb, alpha), 6, 4);
        Assert.Equal(24, resized.Alpha!.Length);
        for (int i = 0; i < resized.Alpha.Length; i++)
            Assert.Equal(resized.Pixels[i * 3], resized.Alpha[i]);
    }

    [Fact]
    public void PngFileLoadAndSave_PreserveAlphaWhenRequested()
    {
        string path = Path.Combine(Path.GetTempPath(), "qwen21-alpha-" + Guid.NewGuid().ToString("N") + ".png");
        try
        {
            ImageIO.SavePng(path, new RgbImage(1, 1, new[] { 1f, 0f, 0f }, new[] { 64f / 255f }));
            Assert.Equal(64f / 255f, ImageIO.Load(path, preserveAlpha: true).Alpha![0]);
            Assert.Null(ImageIO.Load(path).Alpha);
        }
        finally { File.Delete(path); }
    }
}
