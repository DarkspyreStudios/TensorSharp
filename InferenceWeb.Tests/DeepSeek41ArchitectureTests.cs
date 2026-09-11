// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public class DeepSeek41ArchitectureTests : IDisposable
{
    private readonly EnvScope _env = new();

    public DeepSeek41ArchitectureTests()
    {
        _env.ClearSpeculationVars();
        _env.Set("TS_DSV41_TP", null);
        _env.Set("TS_DSV4_NGPU", null);
    }

    public void Dispose() => _env.Dispose();

    [Fact]
    public void DescriptorIsIndependentAndDeclaresLayerSplit()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("deepseek41", out var v41));
        Assert.True(ModelArchitectureRegistry.TryGet("deepseek4", out var v4));
        Assert.NotSame(v4, v41);
        Assert.Equal(MultiGpuMode.LayerSplit, v41.MultiGpu);
        Assert.Contains("not implemented", v41.MultiGpuLimitation);
        TensorSharp.ITensorParallelGroup group = null;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(v41, BackendType.GgmlCuda, 4, ref group, out int split));
        Assert.Equal(4, split);
        Assert.Contains("LAYER SPLIT", v41.DescribeMultiGpuPlacement(4));
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Mlx)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlVulkan)]
    public void UnsupportedBackendsRefusedBeforeLoadingWeights(BackendType backend)
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null));
        Assert.Contains("ggml_cuda", error.Message);
    }

    [Fact]
    public void MissingEngramSidecarGivesPreparationInstructions()
    {
        string model = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "model.gguf");
        var error = Assert.Throws<FileNotFoundException>(() =>
            DeepSeek41Architecture.ValidateLoad(model, BackendType.GgmlCuda, null));
        Assert.EndsWith("deepseek41.engram.bin", error.FileName);
        Assert.Contains("eng/dsv41-prepare.py", error.Message);
    }

    [Fact]
    public void V4DraftCannotBeAppliedToV41()
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCuda, "v4-draft.gguf"));
        Assert.Contains("DSpark", error.Message);
    }

    [Fact]
    public void V4DraftEnvironmentCannotBeAppliedToV41()
    {
        _env.Set("TS_DSV4_DSPARK", "v4-draft.gguf");
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCuda, null));
        Assert.Contains("DSpark", error.Message);
    }

    [Theory]
    [InlineData(null, 4, 0)]
    [InlineData("0", 4, 0)]
    [InlineData("2", 2, 2)]
    [InlineData("4", 4, 4)]
    [InlineData("8", 8, 8)]
    [InlineData("4", 0, 4)] // Automatic device enumeration is validated natively.
    [InlineData(" +4", 4, 4)] // Native strtol accepts a leading sign/whitespace.
    public void RoutedMoeTensorParallelRanksValidateKnownGpuCount(string value, int gpuCount, int expected)
        => Assert.Equal(expected, DeepSeek41Architecture.ParseRoutedMoeTensorParallelRanks(value, gpuCount));

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("1")]
    [InlineData("-2")]
    [InlineData("9")]
    [InlineData("2.0")]
    [InlineData("2x")]
    [InlineData("2 ")]
    [InlineData("999999999999999999999999")]
    public void MalformedRoutedMoeTensorParallelSettingIsRejected(string value)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            DeepSeek41Architecture.ParseRoutedMoeTensorParallelRanks(value, 0));
        Assert.Contains("TS_DSV41_TP", error.Message);
    }

    [Fact]
    public void MismatchedRoutedMoeRanksAreRejectedBeforeSidecarOrWeights()
    {
        _env.Set("TS_DSV41_TP", "2");
        var error = Assert.Throws<ArgumentException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCuda, null, 4));
        Assert.Contains("selected GPU count (4)", error.Message);
    }

    [Fact]
    public void NativeGpuCountOverrideDeterminesRankValidationAndPlacementMessage()
    {
        _env.Set("TS_DSV4_NGPU", "2");
        _env.Set("TS_DSV41_TP", "2");
        Assert.Equal(2, DeepSeek41Architecture.ResolveRoutedMoeTensorParallelRanks(4));
        string message = DeepSeek41Architecture.Descriptor.DescribeMultiGpuPlacement(4);
        Assert.Contains("across 2 GPUs", message);
        Assert.Contains("gate/up/down", message);
        Assert.Contains("host-staged F32", message);
        Assert.Contains("CPU-offloaded layers", message);
        Assert.Contains("Attention and shared experts retain layer placement", message);
        Assert.DoesNotContain("shards no weights", message);

        _env.Set("TS_DSV41_TP", "0");
        Assert.Contains("2 GPUs by LAYER SPLIT", DeepSeek41Architecture.Descriptor.DescribeMultiGpuPlacement(4));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    public void ExplicitAutomaticGpuOverrideDefersRankCountCheckToNative(string gpuOverride)
    {
        _env.Set("TS_DSV4_NGPU", gpuOverride);
        _env.Set("TS_DSV41_TP", "8");
        Assert.Equal(8, DeepSeek41Architecture.ResolveRoutedMoeTensorParallelRanks(2));
        _env.Set("TS_DSV41_TP", "0");
        Assert.Contains("automatically selected visible GPUs", DeepSeek41Architecture.Descriptor.DescribeMultiGpuPlacement(2));
    }

    [Fact]
    public void NativeRoutedMoeShardingDoesNotCreateManagedCollectives()
    {
        _env.Set("TS_DSV41_TP", "4");
        TensorSharp.ITensorParallelGroup group = null;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, 4, ref group, out int split));
        Assert.Equal(4, split);
        Assert.Null(group);
    }
}
