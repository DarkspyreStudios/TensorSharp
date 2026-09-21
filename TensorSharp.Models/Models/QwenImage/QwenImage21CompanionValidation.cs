// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// Check companion architecture metadata before reading large tensors or calling native kernels.
/// Filenames are discovery hints only: an explicitly supplied or renamed file must match the
/// model's conditioning and latent representation too.
/// </summary>
internal static class QwenImage21CompanionValidation
{
    internal static void ValidateVae(IFloatTensorStore weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        // These boundaries distinguish the RGBA /16 64-channel VAE from both
        // the original 16-channel Qwen-Image VAE and the RGB Wan 2.2 VAE.
        RequireVaeShape(weights, "encoder.conv1.weight", 96, 4, 1, 3, 3);
        RequireVaeShape(weights, "encoder.head.2.weight", 128, 768, 1, 3, 3);
        RequireVaeShape(weights, "conv1.weight", 128, 128, 1, 1, 1);
        RequireVaeShape(weights, "conv2.weight", 64, 64, 1, 1, 1);
        RequireVaeShape(weights, "decoder.conv1.weight", 1152, 64, 1, 3, 3);
        RequireVaeShape(weights, "decoder.head.2.weight", 4, 144, 1, 3, 3);
        RequireVaeShape(weights, "encoder.downsamples.4.downsamples.0.residual.2.weight", 768, 768, 1, 3, 3);
        RequireVaeShape(weights, "decoder.upsamples.4.upsamples.0.residual.2.weight", 144, 288, 1, 3, 3);
    }

    internal static void ValidateText(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        const string companion = "Qwen3-VL-8B text encoder";
        RequireString(gguf, "general.architecture", "qwen3vl", companion);
        RequireInt(gguf, "qwen3vl.embedding_length", 4096, companion);
        RequireInt(gguf, "qwen3vl.block_count", 36, companion);
        RequireInt(gguf, "qwen3vl.attention.head_count", 32, companion);
        RequireInt(gguf, "qwen3vl.attention.head_count_kv", 8, companion);
        RequireInt(gguf, "qwen3vl.attention.key_length", 128, companion);
        RequireInt(gguf, "qwen3vl.attention.value_length", 128, companion);
        RequireInt(gguf, "qwen3vl.n_deepstack_layers", 3, companion);
        var sections = gguf.GetInt32Array("qwen3vl.rope.dimension_sections");
        if (sections == null || !sections.SequenceEqual(new[] { 24, 20, 20, 0 }))
            throw Incompatible(companion, "qwen3vl.rope.dimension_sections must be [24,20,20,0]");
        for (int layer = 0; layer < 36; layer++)
        {
            RequireGgufShape(gguf, $"blk.{layer}.attn_q_norm.weight", companion, 128);
            RequireGgufShape(gguf, $"blk.{layer}.attn_k_norm.weight", companion, 128);
        }
        RequireGgufShape(gguf, "blk.35.attn_q.weight", companion, 4096, 4096);
    }

    internal static void ValidateVision(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        const string companion = "Qwen3-VL-8B vision projector";
        RequireString(gguf, "general.architecture", "clip", companion);
        RequireString(gguf, "clip.projector_type", "qwen3vl_merger", companion);
        RequireInt(gguf, "clip.vision.patch_size", 16, companion);
        RequireInt(gguf, "clip.vision.spatial_merge_size", 2, companion);
        RequireInt(gguf, "clip.vision.embedding_length", 1152, companion);
        RequireInt(gguf, "clip.vision.projection_dim", 4096, companion);
        RequireInt(gguf, "clip.vision.block_count", 27, companion);
        RequireInt(gguf, "clip.vision.attention.head_count", 16, companion);
        var deepStack = gguf.GetBoolArray("clip.vision.is_deepstack_layers");
        if (deepStack == null || deepStack.Length != 27 ||
            !deepStack.Select((enabled, index) => (enabled, index)).Where(p => p.enabled)
                .Select(p => p.index).SequenceEqual(new[] { 8, 16, 24 }))
            throw Incompatible(companion, "DeepStack mergers must follow vision blocks 8, 16 and 24");
        RequireGgufShape(gguf, "v.patch_embd.weight", companion, 16, 16, 3, 1152);
        RequireGgufShape(gguf, "v.patch_embd.weight.1", companion, 16, 16, 3, 1152);
        RequireGgufShape(gguf, "mm.2.weight", companion, 4608, 4096);
        foreach (int layer in new[] { 8, 16, 24 })
            RequireGgufShape(gguf, $"v.deepstack.{layer}.fc2.weight", companion, 4608, 4096);
    }

    private static void RequireVaeShape(IFloatTensorStore weights, string tensor, params long[] shape)
    {
        if (!weights.HasTensor(tensor) || !weights.TensorShape(tensor).SequenceEqual(shape))
            throw Incompatible("dedicated Qwen-Image-2.1 RGBA VAE", $"{tensor} must have shape [{string.Join(",", shape)}]");
    }

    private static void RequireGgufShape(GgufFile gguf, string tensor, string companion, params ulong[] shape)
    {
        if (!gguf.Tensors.TryGetValue(tensor, out var info) || !info.Shape.SequenceEqual(shape))
            throw Incompatible(companion, $"{tensor} must have GGUF shape [{string.Join(",", shape)}]");
    }

    private static void RequireString(GgufFile gguf, string key, string value, string companion)
    {
        if (!string.Equals(gguf.GetString(key), value, StringComparison.Ordinal))
            throw Incompatible(companion, $"{key} must be '{value}'");
    }

    private static void RequireInt(GgufFile gguf, string key, uint value, string companion)
    {
        try
        {
            if (gguf.GetUint32(key) == value) return;
        }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException) { }
        throw Incompatible(companion, $"{key} must be {value}");
    }

    private static NotSupportedException Incompatible(string companion, string detail) =>
        new($"Qwen-Image-2.1 requires its matching {companion}: {detail}. Earlier Qwen-Image companions are not interchangeable.");
}
