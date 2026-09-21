// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class QwenImage21CompanionTests
{
    [Fact]
    public void InterleavedRope_Uses242020Axes_WithTemporalTail()
    {
        var axes = Enumerable.Range(0, 64).Select(QwenImageTextEncoder.InterleavedRopeAxis).ToArray();
        Assert.Equal(24, axes.Count(x => x == 0));
        Assert.Equal(20, axes.Count(x => x == 1));
        Assert.Equal(20, axes.Count(x => x == 2));
        Assert.Equal(new[] { 0, 1, 2, 0, 1, 2 }, axes.Take(6));
        Assert.Equal(new[] { 0, 0, 0, 0 }, axes.Skip(60));
    }

    [Fact]
    public void ImagePositions_AdvanceByLargestGridAxis()
    {
        var images = new[]
        {
            new ImageCond { Start = 2, Count = 6, GridH = 4, GridW = 6 },
            new ImageCond { Start = 10, Count = 2, GridH = 4, GridW = 2 },
        };
        var pos = QwenImageTextEncoder.BuildPositions(14, images);
        Assert.Equal(new[] { 0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 9, 10 }, pos.Take(14));
        Assert.Equal(new[] { 0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 8, 9, 10 }, pos.Skip(14).Take(14));
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 7, 9, 10 }, pos.Skip(28));
    }

    [Fact]
    public void EncoderShortcut_TemporalFrontPaddingRemainsInAverage()
    {
        var input = new Feature(2, 2, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var output = VaeReferenceMath.AverageDown21(input, outChannels: 4, timeFactor: 2, spatialFactor: 2);
        Assert.Equal(4, output.C);
        Assert.Equal(1, output.H);
        Assert.Equal(1, output.W);
        Assert.Equal(new float[] { 0, 2.5f, 0, 6.5f }, output.D);
    }

    [Fact]
    public void DecoderShortcut_FirstTemporalChunkSelectsLastTimeSlice()
    {
        var input = new Feature(4, 1, 1, new float[] { 10, 20, 30, 40 });
        var output = VaeReferenceMath.DuplicateUp21(input, outChannels: 2, timeFactor: 2, spatialFactor: 2);
        Assert.Equal(new float[] { 20, 20, 20, 20, 40, 40, 40, 40 }, output.D);
    }

    [Fact]
    public void DecoderShortcut_SpatialOnlyStageUnpacksDifferentChannelRows()
    {
        var input = new Feature(4, 1, 1, new float[] { 10, 20, 30, 40 });
        var output = VaeReferenceMath.DuplicateUp21(input, outChannels: 2, timeFactor: 1, spatialFactor: 2);
        Assert.Equal(new float[] { 10, 10, 20, 20, 30, 30, 40, 40 }, output.D);
    }

    [Fact]
    public void VaeStatistics_Contain64FiniteNondegenerateChannels()
    {
        Assert.Equal(64, VaeReferenceMath.Qwen21Mean.Length);
        Assert.Equal(64, VaeReferenceMath.Qwen21Std.Length);
        Assert.All(VaeReferenceMath.Qwen21Mean, x => Assert.True(float.IsFinite(x)));
        Assert.All(VaeReferenceMath.Qwen21Std, x => Assert.True(float.IsFinite(x) && x > 0));
        // Sentinel values catch accidental substitution of the older Wan/Qwen VAE statistics.
        Assert.Equal(0.5126f, VaeReferenceMath.Qwen21Mean[0]);
        Assert.Equal(3.8161f, VaeReferenceMath.Qwen21Std[63]);
    }
}
