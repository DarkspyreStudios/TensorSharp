// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// Presents the released Diffusers Qwen-Image-2.1 VAE using the original Wan names
/// consumed by our single-frame VAE. Spatial kernels have an implicit unit time
/// axis; adding that axis changes only shape metadata, never tensor values.
/// The underlying model continues to own and dispose its file handles.
/// </summary>
internal sealed class QwenImage21VaeTensorStore : IFloatTensorStore
{
    private readonly IFloatTensorStore _source;
    private static readonly (string Original, string Diffusers, bool Residual)[] Prefixes = BuildPrefixes();

    internal QwenImage21VaeTensorStore(IFloatTensorStore source) =>
        _source = source ?? throw new ArgumentNullException(nameof(source));

    public bool HasTensor(string name) => _source.HasTensor(ResolveName(name));

    public float[] ReadFloat32(string name) => _source.ReadFloat32(ResolveName(name));

    public long[] TensorShape(string name)
    {
        long[] shape = _source.TensorShape(ResolveName(name));
        // Attention projections and spatial resampling are ordinary Conv2D in
        // both layouts. Only causal convolutions carry the implicit time axis.
        if (shape.Length == 4 && name.EndsWith(".weight", StringComparison.Ordinal) &&
            !name.Contains(".middle.1.", StringComparison.Ordinal) &&
            !name.Contains(".resample.", StringComparison.Ordinal))
            return new[] { shape[0], shape[1], 1L, shape[2], shape[3] };
        return shape;
    }

    private string ResolveName(string name) => _source.HasTensor(name) ? name : DiffusersName(name);

    internal static string DiffusersName(string name)
    {
        foreach (var prefix in Prefixes)
        {
            if (!name.StartsWith(prefix.Original, StringComparison.Ordinal)) continue;
            string suffix = name[prefix.Original.Length..];
            if (prefix.Residual)
            {
                string[] parts = suffix.Split('.', 2);
                string operation = parts[0] switch
                {
                    "0" => "norm1",
                    "2" => "conv1",
                    "3" => "norm2",
                    "6" => "conv2",
                    _ => parts[0],
                };
                suffix = parts.Length == 2 ? operation + "." + parts[1] : operation;
            }
            return prefix.Diffusers + suffix;
        }
        return name;
    }

    private static (string, string, bool)[] BuildPrefixes()
    {
        var result = new List<(string, string, bool)>
        {
            ("conv1.", "quant_conv.", false),
            ("conv2.", "post_quant_conv.", false),
        };
        foreach (string side in new[] { "encoder", "decoder" })
        {
            bool encoder = side == "encoder";
            result.Add(($"{side}.conv1.", $"{side}.conv_in.", false));
            result.Add(($"{side}.head.0.", $"{side}.norm_out.", false));
            result.Add(($"{side}.head.2.", $"{side}.conv_out.", false));
            result.Add(($"{side}.middle.1.", $"{side}.mid_block.attentions.0.", false));
            result.Add(($"{side}.middle.0.residual.", $"{side}.mid_block.resnets.0.", true));
            result.Add(($"{side}.middle.2.residual.", $"{side}.mid_block.resnets.1.", true));
            string samples = encoder ? "downsamples" : "upsamples";
            string blocks = encoder ? "down_blocks" : "up_blocks";
            int residualCount = encoder ? 2 : 3;
            for (int stage = 0; stage < 5; stage++)
            {
                string original = $"{side}.{samples}.{stage}.{samples}.";
                string diffusers = $"{side}.{blocks}.{stage}.";
                for (int block = 0; block < residualCount; block++)
                {
                    result.Add(($"{original}{block}.residual.", $"{diffusers}resnets.{block}.", true));
                    result.Add(($"{original}{block}.shortcut.", $"{diffusers}resnets.{block}.conv_shortcut.", false));
                }
                result.Add(($"{original}{residualCount}.",
                    diffusers + (encoder ? "downsampler." : "upsampler."), false));
            }
        }
        return result.ToArray();
    }
}
