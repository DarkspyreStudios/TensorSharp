// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class QwenImage21DiTLayoutTests
{
    [Fact]
    public void CachedLayoutsRetainBothCfgBranchesAndObserveMutableKeys()
    {
        var cache = new QwenImage21DiT.LayoutCache();
        var shapes = new[] { (2, 2), (2, 3) };
        int[] slots = { 0, 1, 0 };
        var first = cache.Get(3, slots, shapes);
        var negative = cache.Get(2, new[] { 1, 0 }, shapes);
        Assert.Same(first.Cos, cache.Get(3, slots, shapes).Cos);
        Assert.Same(negative.Cos, cache.Get(2, new[] { 1, 0 }, shapes).Cos);

        // Changing geometry without changing the number of tokens changes RoPE.
        shapes[1] = (3, 2);
        var reshaped = cache.Get(3, slots, shapes);
        Assert.NotSame(first.Cos, reshaped.Cos);
        Assert.NotEqual(first.Cos[6 * 64 + 8], reshaped.Cos[6 * 64 + 8]);
        slots[0] = 1;
        slots[1] = 0;
        var moved = cache.Get(3, slots, shapes);
        Assert.NotSame(reshaped.Cos, moved.Cos);
        Assert.Equal(1, moved.Segments[0].IsImage);
        cache.Clear();
        Assert.NotSame(moved.Cos, cache.Get(3, slots, shapes).Cos);
    }

    [Fact]
    public void AutomaticReferenceGeometryMatchesHalfAwayFromZeroRounding()
    {
        // sqrt(512² * 1089/1024) = 528 = 16.5 grid cells.
        var reference = new RgbImage(1089, 1024, new float[1089 * 1024 * 3]);
        var geometry = QwenImage21Pipeline.ResolveDimensions(new QwenImageParams { TargetArea = 512 * 512 }, reference);
        Assert.Equal((544, 512), geometry);
    }

    [Fact]
    public void AutomaticGeometryRejectsIntegerOverflowInsteadOfClampingTo32()
    {
        Assert.Throws<OverflowException>(() => QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { TargetArea = long.MaxValue }, null));
    }

    [Fact]
    public void TextThenTargetUsesCausalTextAndCenteredImageCoordinates()
    {
        var layout = QwenImage21DiT.BuildLayout(2, null, new[] { (2, 3) });
        Assert.Equal(2, layout.Prefix);
        Assert.Equal(new[] { 0, 2 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 2, 8 }, layout.Segments.Select(s => s.End));
        Assert.Equal(new[] { 0, 1 }, layout.Segments.Select(s => s.IsImage));
        AssertPosition(layout.Cos, layout.Sin, 0, 0, 0, 0);
        AssertPosition(layout.Cos, layout.Sin, 1, 1, 1, 1);
        // Width=3 uses [-2,-1,0], matching upstream's ceil-centered coordinates.
        AssertPosition(layout.Cos, layout.Sin, 2, 2, -1, -2);
        AssertPosition(layout.Cos, layout.Sin, 7, 2, 0, 0);
    }

    [Fact]
    public void VisionSlotExpandsToFourLatentsAndResumesTextCoordinates()
    {
        var layout = QwenImage21DiT.BuildLayout(5, new[] { 0, 0, 1, 0, 0 }, new[] { (2, 2), (2, 3) });
        Assert.Equal(8, layout.Prefix);
        Assert.Equal(new[] { 0, 2, 6, 8 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 2, 6, 8, 14 }, layout.Segments.Select(s => s.End));
        Assert.Equal(new[] { 0, 0, 3, 4 }, layout.Segments.Select(s => s.SourceStart));
        Assert.Equal(new[] { 0, 1, 0, 1 }, layout.Segments.Select(s => s.IsImage));
        AssertPosition(layout.Cos, layout.Sin, 2, 2, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 6, 4, 4, 4);
        AssertPosition(layout.Cos, layout.Sin, 8, 6, -1, -2);
    }

    [Fact]
    public void MultipleReferencesPreserveTheirOrderAndDistinctSources()
    {
        var layout = QwenImage21DiT.BuildLayout(5, new[] { 0, 1, 0, 2, 0 }, new[] { (2, 2), (2, 2), (2, 2) });
        Assert.Equal(11, layout.Prefix);
        Assert.Equal(new[] { 0, 1, 5, 6, 10, 11 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 0, 4, 8 }, layout.Segments.Where(s => s.IsImage == 1).Select(s => s.SourceStart));
        AssertPosition(layout.Cos, layout.Sin, 1, 1, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 6, 4, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 11, 7, -1, -1);
    }

    [Theory]
    [InlineData(new[] { 0, 0, 0 })] // missing reference
    [InlineData(new[] { 0, 2, 0 })] // out of order
    [InlineData(new[] { 0, -1, 0 })] // invalid tag
    [InlineData(new[] { 1, 1, 0 })] // two vision slots cannot represent four latents
    [InlineData(new[] { 1, 0, 1 })] // repeated noncontiguous reference
    public void InvalidReferenceSlotsFailBeforeNativeDispatch(int[] slots)
    {
        Assert.Throws<ArgumentException>(() => QwenImage21DiT.BuildLayout(3, slots, new[] { (2, 2), (2, 2) }));
    }

    private static void AssertPosition(float[] cos, float[] sin, int token, int t, int h, int w)
    {
        // The first pair of each RoPE axis has frequency one. These checks use
        // analytic coordinates rather than reproducing the layout implementation.
        foreach (var (channel, coordinate) in new[] { (0, t), (8, h), (36, w) })
        {
            Assert.InRange(Math.Abs(cos[token * 64 + channel] - Math.Cos(coordinate)), 0, 1e-7);
            Assert.InRange(Math.Abs(sin[token * 64 + channel] - Math.Sin(coordinate)), 0, 1e-7);
        }
    }
}
