// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>Request-side rules of the Qwen-Image-2.1 prefix KV cache. The numerical
/// equivalence of cached and uncached predictions is checked natively
/// (tests/qwen_image21_test.cpp) and end to end by PNG hash.</summary>
public sealed class QwenImage21PrefixCacheTests
{
    [Theory]
    [InlineData(null, QwenImage21PrefixCacheType.Auto)]
    [InlineData("", QwenImage21PrefixCacheType.Auto)]
    [InlineData("auto", QwenImage21PrefixCacheType.Auto)]
    [InlineData(" F16 ", QwenImage21PrefixCacheType.F16)]
    [InlineData("f32", QwenImage21PrefixCacheType.F32)]
    [InlineData("q8_0", QwenImage21PrefixCacheType.Q8_0)]
    [InlineData("Q8_0_V", QwenImage21PrefixCacheType.Q8_0V)]
    public void StorageTypeParses(string value, QwenImage21PrefixCacheType expected) =>
        Assert.Equal(expected, QwenImage21DiT.ParsePrefixCacheType(value));

    [Theory]
    [InlineData("fp8")]
    [InlineData("q4_0")]
    [InlineData("bf16")]
    public void UnknownStorageTypeIsRefusedEvenWhenDisabled(string value)
    {
        var error = Assert.Throws<ArgumentException>(() => QwenImage21DiT.ParsePrefixCacheType(value));
        Assert.Contains("TS_QWEN21_PREFIX_CACHE_TYPE", error.Message);
        Assert.Throws<ArgumentException>(() => QwenImage21DiT.CreatePrefixCache(new float[1], null, null, "0", value));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("off")]
    [InlineData("FALSE")]
    [InlineData("no")]
    public void DisabledCacheIsNull(string enabled) =>
        Assert.Null(QwenImage21DiT.CreatePrefixCache(new float[1], null, null, enabled, null));

    [Fact]
    public void CachesAreEnabledByDefaultWithDistinctKeys()
    {
        var text = new float[4];
        var a = QwenImage21DiT.CreatePrefixCache(text, null, null, null, null);
        var b = QwenImage21DiT.CreatePrefixCache(text, null, null, "1", "q8_0_v");
        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.NotEqual(0UL, a.Key);
        Assert.NotEqual(a.Key, b.Key);
        Assert.Equal(QwenImage21PrefixCacheType.Auto, a.Type);
        Assert.Equal(QwenImage21PrefixCacheType.Q8_0V, b.Type);
    }

    [Fact]
    public void CacheDescribesOnlyItsOwnConditioningArrays()
    {
        var text = new float[8];
        var slots = new[] { 0, 1, 1, 0 };
        var references = new[] { new float[64], new float[64] };
        var cache = QwenImage21DiT.CreatePrefixCache(text, slots, references, null, null);

        Assert.True(cache.Describes(text, slots, references));
        // Same references in a new outer array: the per-image arrays are what the K/V encode.
        Assert.True(cache.Describes(text, slots, new[] { references[0], references[1] }));
        // Equal content is not enough: a different prompt array may be mutated independently.
        Assert.False(cache.Describes((float[])text.Clone(), slots, references));
        Assert.False(cache.Describes(text, (int[])slots.Clone(), references));
        Assert.False(cache.Describes(text, slots, new[] { references[1], references[0] }));
        Assert.False(cache.Describes(text, slots, new[] { references[0] }));
        Assert.False(cache.Describes(text, null, references));

        var textOnly = QwenImage21DiT.CreatePrefixCache(text, null, null, null, null);
        Assert.True(textOnly.Describes(text, null, null));
        Assert.True(textOnly.Describes(text, null, Array.Empty<float[]>()));
    }
}
