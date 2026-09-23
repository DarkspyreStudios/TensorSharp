// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class QwenImage21VaeTensorStoreTests
{
    [Theory]
    [InlineData("conv1.weight", "quant_conv.weight")]
    [InlineData("conv2.bias", "post_quant_conv.bias")]
    [InlineData("encoder.conv1.weight", "encoder.conv_in.weight")]
    [InlineData("decoder.head.2.weight", "decoder.conv_out.weight")]
    [InlineData("encoder.head.0.gamma", "encoder.norm_out.gamma")]
    [InlineData("decoder.middle.0.residual.0.gamma", "decoder.mid_block.resnets.0.norm1.gamma")]
    [InlineData("encoder.middle.2.residual.6.weight", "encoder.mid_block.resnets.1.conv2.weight")]
    [InlineData("decoder.middle.1.to_qkv.weight", "decoder.mid_block.attentions.0.to_qkv.weight")]
    [InlineData("encoder.downsamples.4.downsamples.0.residual.2.weight", "encoder.down_blocks.4.resnets.0.conv1.weight")]
    [InlineData("decoder.upsamples.4.upsamples.0.shortcut.weight", "decoder.up_blocks.4.resnets.0.conv_shortcut.weight")]
    [InlineData("decoder.upsamples.4.upsamples.2.residual.3.gamma", "decoder.up_blocks.4.resnets.2.norm2.gamma")]
    [InlineData("encoder.downsamples.1.downsamples.2.resample.1.weight", "encoder.down_blocks.1.downsampler.resample.1.weight")]
    [InlineData("decoder.upsamples.2.upsamples.3.time_conv.weight", "decoder.up_blocks.2.upsampler.time_conv.weight")]
    public void ReleasedDiffusersNames_ResolveThroughOriginalArchitecture(string original, string diffusers)
    {
        var source = new MemoryStore(diffusers, new long[] { 2, 3, 2, 2 }, Enumerable.Range(0, 24).Select(i => (float)i).ToArray());
        var weights = new QwenImage21VaeTensorStore(source);
        Assert.True(weights.HasTensor(original));
        Assert.Equal(source.Values, weights.ReadFloat32(original));
    }

    [Fact]
    public void SpatialKernel_AddsOnlySingletonTimeAxis_WithoutReorderingValues()
    {
        var source = new MemoryStore("encoder.conv_in.weight", new long[] { 2, 3, 2, 2 }, Enumerable.Range(0, 24).Select(i => (float)i).ToArray());
        var weights = new QwenImage21VaeTensorStore(source);
        Assert.Equal(new long[] { 2, 3, 1, 2, 2 }, weights.TensorShape("encoder.conv1.weight"));
        Assert.Same(source.Values, weights.ReadFloat32("encoder.conv1.weight"));
        Assert.Equal(new long[] { 2, 3, 2, 2 }, source.Shape);
    }

    [Fact]
    public void OriginalTemporalKernelAndNorm_PreserveTheirShape()
    {
        var conv = new QwenImage21VaeTensorStore(new MemoryStore("encoder.conv1.weight", new long[] { 2, 3, 3, 2, 2 }, Array.Empty<float>()));
        Assert.Equal(new long[] { 2, 3, 3, 2, 2 }, conv.TensorShape("encoder.conv1.weight"));
        var norm = new QwenImage21VaeTensorStore(new MemoryStore("encoder.norm_out.gamma", new long[] { 768, 1, 1, 1 }, Array.Empty<float>()));
        Assert.Equal(new long[] { 768, 1, 1, 1 }, norm.TensorShape("encoder.head.0.gamma"));
        Assert.False(norm.HasTensor("encoder.head.2.weight"));
        Assert.Empty(norm.TensorShape("encoder.head.2.weight"));
    }

    [Theory]
    [InlineData("decoder.middle.1.proj.weight", "decoder.middle.1.proj.weight")]
    [InlineData("encoder.middle.1.to_qkv.weight", "encoder.mid_block.attentions.0.to_qkv.weight")]
    [InlineData("decoder.upsamples.0.upsamples.3.resample.1.weight", "decoder.up_blocks.0.upsampler.resample.1.weight")]
    public void SpatialOnlyOperators_PreserveConv2dShape(string original, string stored)
    {
        var source = new MemoryStore(stored, new long[] { 2, 3, 3, 3 }, Array.Empty<float>());
        Assert.Equal(source.Shape, new QwenImage21VaeTensorStore(source).TensorShape(original));
    }

    private sealed class MemoryStore(string name, long[] shape, float[] values) : IFloatTensorStore
    {
        internal float[] Values => values;
        internal long[] Shape => shape;
        public bool HasTensor(string key) => key == name;
        public long[] TensorShape(string key) => key == name ? shape : Array.Empty<long>();
        public float[] ReadFloat32(string key) => key == name ? values : throw new KeyNotFoundException(key);
    }
}
