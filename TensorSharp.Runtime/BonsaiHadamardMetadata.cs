// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;

namespace TensorSharp.Runtime;

/// <summary>The explicit PRISM rotation carried by Bonsai2 GGUF checkpoints.</summary>
public sealed class BonsaiHadamardMetadata
{
    private const string Prefix = "prism.hadamard.";
    public int BlockSize { get; private init; }
    public bool GdnVGrouped { get; private init; }
    public IReadOnlySet<string> WeightNames { get; private init; } = null!;
    public IReadOnlySet<string> InverseWeightNames { get; private init; } = null!;
    public IReadOnlyDictionary<int, float[]> SignsByWidth { get; private init; } = null!;

    /// <summary>Projection rows may be concatenated only when they use the same input basis.</summary>
    public bool CanFuseProjections(params string[] weightNames)
    {
        if (weightNames == null || weightNames.Length == 0)
            return false;
        bool rotated = WeightNames.Contains(weightNames[0]);
        foreach (string name in weightNames)
            if (InverseWeightNames.Contains(name) || WeightNames.Contains(name) != rotated)
                return false;
        return true;
    }

    /// <summary>Returns null for ordinary GGUFs; rejects incomplete or unknown transforms.</summary>
    public static BonsaiHadamardMetadata? Read(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        bool present = false;
        foreach (string key in file.Metadata.Keys)
            present |= key.StartsWith(Prefix, StringComparison.Ordinal);
        bool customQuant = false;
        foreach (var tensor in file.Tensors.Values)
            customQuant |= tensor.Type is GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0;
        if (!present)
        {
            if (customQuant)
                throw new InvalidDataException("Bonsai2 PQ2_0/PTQ1_0 weights require prism.hadamard metadata.");
            return null;
        }

        if (file.GetUint32(Prefix + "version") != 1 ||
            file.GetString(Prefix + "transform") != "normalized-sylvester-walsh-hadamard" ||
            file.GetString(Prefix + "axis") != "input-last-dimension" ||
            file.GetString(Prefix + "sign_mode") != "explicit")
            throw new NotSupportedException("Unsupported Bonsai2 PRISM Hadamard version, transform, axis, or sign mode.");

        int blockSize = checked((int)file.GetUint32(Prefix + "block_size"));
        if (blockSize is not (64 or 128 or 256 or 512 or 1024))
            throw new InvalidDataException("Bonsai2 Hadamard block size must be 64, 128, 256, 512, or 1024.");
        int[] widths = file.GetInt32Array(Prefix + "sign_widths")
            ?? throw new InvalidDataException("Missing Bonsai2 Hadamard sign_widths.");
        if (!file.Metadata.TryGetValue(Prefix + "sign_values", out object? values) || values is not Array signValues)
            throw new InvalidDataException("Missing Bonsai2 Hadamard sign_values.");
        var signs = new Dictionary<int, float[]>();
        int offset = 0;
        foreach (int width in widths)
        {
            if (width <= 0 || width % blockSize != 0 || signs.ContainsKey(width) || width > signValues.Length - offset)
                throw new InvalidDataException("Bonsai2 Hadamard sign widths must be unique, block aligned, and match sign_values.");
            var row = new float[width];
            for (int i = 0; i < width; i++)
            {
                float sign = Convert.ToSingle(signValues.GetValue(offset++));
                if (sign != -1f && sign != 1f)
                    throw new InvalidDataException("Bonsai2 Hadamard signs must be exactly -1 or +1.");
                row[i] = sign;
            }
            signs.Add(width, row);
        }
        if (offset != signValues.Length || signs.Count == 0)
            throw new InvalidDataException("Bonsai2 Hadamard sign_values length does not match sign_widths.");

        var forward = ReadNames("weight_names");
        var inverse = ReadNames("inverse_weight_names");
        foreach (string name in inverse)
        {
            if (forward.Contains(name))
                throw new InvalidDataException($"Bonsai2 tensor '{name}' cannot have both forward and inverse rotations.");
            if (name != "token_embd.weight")
                throw new NotSupportedException($"Unsupported Bonsai2 inverse rotation target '{name}'.");
        }
        foreach (var tensor in file.Tensors.Values)
            if ((tensor.Type is GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0) &&
                !forward.Contains(tensor.Name) && !inverse.Contains(tensor.Name))
                throw new InvalidDataException($"Bonsai2 tensor '{tensor.Name}' is missing its Hadamard rotation metadata.");

        return new BonsaiHadamardMetadata
        {
            BlockSize = blockSize,
            GdnVGrouped = file.GetBool(Prefix + "gdn_v_grouped"),
            WeightNames = forward,
            InverseWeightNames = inverse,
            SignsByWidth = signs,
        };

        HashSet<string> ReadNames(string suffix)
        {
            string[] names = file.GetStringArray(Prefix + suffix)
                ?? throw new InvalidDataException($"Missing Bonsai2 Hadamard {suffix}.");
            var result = new HashSet<string>(StringComparer.Ordinal);
            foreach (string name in names)
            {
                if (!result.Add(name) || !file.Tensors.TryGetValue(name, out var tensor) ||
                    tensor.Shape.Length != 2 || tensor.Shape[0] > int.MaxValue ||
                    !signs.ContainsKey((int)tensor.Shape[0]))
                    throw new InvalidDataException($"Invalid or duplicate Bonsai2 Hadamard tensor '{name}'.");
            }
            return result;
        }
    }

    /// <summary>Recover an embedding row: normalized H first, then explicit signs.</summary>
    public static void InverseTransform(Span<float> row, ReadOnlySpan<float> signs, int blockSize)
    {
        if (row.Length != signs.Length || blockSize < 2 || (blockSize & (blockSize - 1)) != 0 || row.Length % blockSize != 0)
            throw new ArgumentException("Invalid Bonsai2 inverse transform dimensions.");
        float scale = 1f / MathF.Sqrt(blockSize);
        for (int start = 0; start < row.Length; start += blockSize)
            for (int stride = 1; stride < blockSize; stride *= 2)
                for (int group = start; group < start + blockSize; group += stride * 2)
                    for (int i = 0; i < stride; i++)
                    {
                        float a = row[group + i], b = row[group + stride + i];
                        row[group + i] = a + b;
                        row[group + stride + i] = a - b;
                    }
        for (int i = 0; i < row.Length; i++)
            row[i] *= scale * signs[i];
    }
}
