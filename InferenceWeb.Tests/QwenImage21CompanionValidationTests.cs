// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class QwenImage21CompanionValidationTests
{
    private sealed class Shapes : IFloatTensorStore
    {
        internal readonly Dictionary<string, long[]> Tensors = new()
        {
            ["encoder.conv1.weight"] = new long[] { 96, 4, 1, 3, 3 },
            ["encoder.head.2.weight"] = new long[] { 128, 768, 1, 3, 3 },
            ["conv1.weight"] = new long[] { 128, 128, 1, 1, 1 },
            ["conv2.weight"] = new long[] { 64, 64, 1, 1, 1 },
            ["decoder.conv1.weight"] = new long[] { 1152, 64, 1, 3, 3 },
            ["decoder.head.2.weight"] = new long[] { 4, 144, 1, 3, 3 },
            ["encoder.downsamples.4.downsamples.0.residual.2.weight"] = new long[] { 768, 768, 1, 3, 3 },
            ["decoder.upsamples.4.upsamples.0.residual.2.weight"] = new long[] { 144, 288, 1, 3, 3 },
        };
        public bool HasTensor(string name) => Tensors.ContainsKey(name);
        public long[] TensorShape(string name) => Tensors[name];
        public float[] ReadFloat32(string name) => throw new InvalidOperationException("Validation must not load tensor data.");
    }

    [Fact]
    public void VaeValidation_UsesMetadataWithoutLoadingWeights() => QwenImage21CompanionValidation.ValidateVae(new Shapes());

    [Theory]
    [InlineData(3)] // RGB Wan 2.2 is not the RGBA 2.1 VAE.
    [InlineData(16)]
    public void VaeValidation_RejectsWrongInputRepresentation(int channels)
    {
        var weights = new Shapes();
        weights.Tensors["encoder.conv1.weight"][1] = channels;
        Assert.Contains("RGBA VAE", Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateVae(weights)).Message);
    }

    [Fact]
    public void VaeValidation_RejectsOriginal16ChannelQwenVae()
    {
        var weights = new Shapes();
        weights.Tensors["conv2.weight"] = new long[] { 16, 16, 1, 1, 1 };
        Assert.Contains("conv2.weight", Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateVae(weights)).Message);
    }

    [Fact]
    public void VaeValidation_RejectsMissingFinalDownsample()
    {
        var weights = new Shapes();
        weights.Tensors.Remove("encoder.downsamples.4.downsamples.0.residual.2.weight");
        Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateVae(weights));
    }

    [Fact]
    public void TextValidation_AcceptsTheDownloadedCheckpointLayout()
    {
        using var file = new MetadataFile();
        SetTextMetadata(file.Gguf);
        QwenImage21CompanionValidation.ValidateText(file.Gguf);
    }

    [Theory]
    [InlineData("general.architecture", "qwen2vl")]
    [InlineData("qwen3vl.embedding_length", 3584)]
    [InlineData("qwen3vl.block_count", 28)]
    [InlineData("qwen3vl.n_deepstack_layers", 0)]
    [InlineData("qwen3vl.attention.head_count_kv", 4)]
    public void TextValidation_RejectsEarlierAndMismatchedEncoders(string key, object value)
    {
        using var file = new MetadataFile();
        SetTextMetadata(file.Gguf);
        file.Gguf.Metadata[key] = value;
        Assert.Contains(key, Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateText(file.Gguf)).Message);
    }

    [Fact]
    public void TextValidation_RejectsMissingQkNormAndWrongRope()
    {
        using var file = new MetadataFile();
        SetTextMetadata(file.Gguf);
        file.Gguf.Tensors.Remove("blk.35.attn_k_norm.weight");
        Assert.Contains("blk.35.attn_k_norm", Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateText(file.Gguf)).Message);
        SetTextMetadata(file.Gguf);
        file.Gguf.Metadata["qwen3vl.rope.dimension_sections"] = new[] { 16, 24, 24 };
        Assert.Contains("dimension_sections", Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateText(file.Gguf)).Message);
    }

    [Fact]
    public void VisionValidation_AcceptsTheDownloadedCheckpointLayout()
    {
        using var file = new MetadataFile();
        SetVisionMetadata(file.Gguf);
        QwenImage21CompanionValidation.ValidateVision(file.Gguf);
    }

    [Theory]
    [InlineData("clip.projector_type", "qwen2vl_merger")]
    [InlineData("clip.vision.patch_size", 14)]
    [InlineData("clip.vision.projection_dim", 3584)]
    public void VisionValidation_RejectsEarlierProjectors(string key, object value)
    {
        using var file = new MetadataFile();
        SetVisionMetadata(file.Gguf);
        file.Gguf.Metadata[key] = value;
        Assert.Contains(key, Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateVision(file.Gguf)).Message);
    }

    [Fact]
    public void VisionValidation_RejectsMissingDeepStack()
    {
        using var file = new MetadataFile();
        SetVisionMetadata(file.Gguf);
        file.Gguf.Tensors.Remove("v.deepstack.24.fc2.weight");
        Assert.Contains("deepstack.24", Assert.Throws<NotSupportedException>(() => QwenImage21CompanionValidation.ValidateVision(file.Gguf)).Message);
    }

    [ModelFact("TENSORSHARP_QWEN21_DIT")]
    public void RealCompanions_PassArchitectureValidation()
    {
        string directory = Path.GetDirectoryName(Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_DIT"))!;
        using var vae = SafetensorsModel.Open(Path.Combine(directory, "qwen_image_2.1_vae_bf16.safetensors"));
        using var text = new GgufFile(CompanionPath(directory, "TS_QWEN_IMAGE_TE",
            "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"));
        using var vision = new GgufFile(CompanionPath(directory, "TS_QWEN_IMAGE_MMPROJ",
            "mmproj-BF16.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"));
        QwenImage21CompanionValidation.ValidateVae(new QwenImage21VaeTensorStore(vae));
        QwenImage21CompanionValidation.ValidateText(text);
        QwenImage21CompanionValidation.ValidateVision(vision);
    }

    private static string CompanionPath(string directory, string variable, params string[] candidates)
    {
        string? explicitPath = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrWhiteSpace(explicitPath)) return explicitPath;
        return candidates.Select(name => Path.Combine(directory, name)).FirstOrDefault(File.Exists)
            ?? throw new FileNotFoundException($"No {variable} test companion found in {directory}.");
    }

    private static void SetTextMetadata(GgufFile file)
    {
        file.Metadata["general.architecture"] = "qwen3vl";
        foreach (var (key, value) in new[] { ("embedding_length", 4096), ("block_count", 36), ("attention.head_count", 32),
                     ("attention.head_count_kv", 8), ("attention.key_length", 128), ("attention.value_length", 128), ("n_deepstack_layers", 3) })
            file.Metadata["qwen3vl." + key] = value;
        file.Metadata["qwen3vl.rope.dimension_sections"] = new[] { 24, 20, 20, 0 };
        for (int layer = 0; layer < 36; layer++)
        {
            Tensor(file, $"blk.{layer}.attn_q_norm.weight", 128);
            Tensor(file, $"blk.{layer}.attn_k_norm.weight", 128);
        }
        Tensor(file, "blk.35.attn_q.weight", 4096, 4096);
    }

    private static void SetVisionMetadata(GgufFile file)
    {
        file.Metadata["general.architecture"] = "clip";
        file.Metadata["clip.projector_type"] = "qwen3vl_merger";
        foreach (var (key, value) in new[] { ("patch_size", 16), ("spatial_merge_size", 2), ("embedding_length", 1152),
                     ("projection_dim", 4096), ("block_count", 27), ("attention.head_count", 16) })
            file.Metadata["clip.vision." + key] = value;
        file.Metadata["clip.vision.is_deepstack_layers"] = Enumerable.Range(0, 27).Select(i => i is 8 or 16 or 24).ToArray();
        Tensor(file, "v.patch_embd.weight", 16, 16, 3, 1152);
        Tensor(file, "v.patch_embd.weight.1", 16, 16, 3, 1152);
        Tensor(file, "mm.2.weight", 4608, 4096);
        foreach (int layer in new[] { 8, 16, 24 }) Tensor(file, $"v.deepstack.{layer}.fc2.weight", 4608, 4096);
    }

    private static void Tensor(GgufFile file, string name, params ulong[] shape) =>
        file.Tensors[name] = new GgufTensorInfo { Name = name, Shape = shape, Type = GgmlTensorType.F32 };

    // Metadata-only fixture: descriptors are injected after opening an empty valid GGUF.
    // No enormous placeholder tensor buffers or native backend are needed.
    private sealed class MetadataFile : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), "qwen21-metadata-" + Guid.NewGuid().ToString("N") + ".gguf");
        internal GgufFile Gguf { get; }
        internal MetadataFile()
        {
            using (var writer = new BinaryWriter(File.Create(_path)))
            {
                writer.Write(0x46554747u); writer.Write(3u);
                writer.Write(0UL); writer.Write(0UL); writer.Write(0UL);
            }
            Gguf = new GgufFile(_path);
        }
        public void Dispose() { Gguf.Dispose(); File.Delete(_path); }
    }
}
