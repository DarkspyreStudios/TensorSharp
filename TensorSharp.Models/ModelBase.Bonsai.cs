// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public abstract partial class ModelBase
{
    protected BonsaiHadamardMetadata BonsaiHadamard { get; private set; }

    protected bool HasBonsaiCheckpointMetadata
    {
        get
        {
            foreach (string key in _gguf.Metadata.Keys)
                if (key.StartsWith("prism.hadamard.", StringComparison.Ordinal))
                    return true;
            foreach (var tensor in _gguf.Tensors.Values)
                if (tensor.Type is GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0)
                    return true;
            return false;
        }
    }

    // The ordinary Dispose paths retire captured graphs first. If one of those
    // paths throws (for example an older native library lacks a reset export),
    // still release each load-owned object independently. This method runs only
    // while unwinding a failed constructor and must preserve that exception.
    protected void CleanUpFailedBonsaiConstruction(Action releaseDerivedResources, Action releaseBaseResources)
    {
        try { releaseDerivedResources(); } catch { }
        bool baseReleased = false;
        try { releaseBaseResources(); baseReleased = true; } catch { }
        foreach (var weight in _quantWeights.Values)
            try { weight.Dispose(); } catch { }
        _quantWeights.Clear();
        foreach (var weight in _weights.Values)
            try { weight.Dispose(); } catch { }
        _weights.Clear();
        try { _gguf.Dispose(); } catch { }
        if (!baseReleased && _allocator is IDisposable allocator)
            try { allocator.Dispose(); } catch { }
        try { _ggmlContext?.ReleasePooledMemory(); } catch { }
    }

    private void ReadBonsaiMetadata()
    {
        BonsaiHadamard = BonsaiHadamardMetadata.Read(_gguf);
        if (BonsaiHadamard == null)
            return;
        if (!IsGgmlBackend || IsTensorParallel)
            throw new NotSupportedException("Bonsai2 PRISM inference currently requires a single-device GGML backend.");
        if (Config?.Architecture != "qwen35")
            throw new NotSupportedException("Bonsai2 PRISM transforms currently support the qwen35 architecture.");
        foreach (string name in BonsaiHadamard.WeightNames)
        {
            bool supported = name == "output.weight";
            if (name.StartsWith("blk.", StringComparison.Ordinal))
            {
                int endLayer = name.IndexOf('.', 4);
                if (endLayer > 4 && int.TryParse(name.AsSpan(4, endLayer - 4), out int layer) &&
                    layer >= 0 && layer < Config.NumLayers)
                    supported = name[(endLayer + 1)..] is "attn_qkv.weight" or "attn_gate.weight" or
                        "ssm_out.weight" or "attn_q.weight" or "attn_k.weight" or "attn_v.weight" or
                        "attn_output.weight" or "ffn_gate.weight" or "ffn_up.weight" or "ffn_down.weight";
            }
            if (!supported)
                throw new NotSupportedException($"Bonsai2 rotation for '{name}' is not supported by the Qwen3.5 projection paths.");
            ValidateEncoding(name);
        }
        foreach (string name in BonsaiHadamard.InverseWeightNames)
            ValidateEncoding(name);

        void ValidateEncoding(string name)
        {
            if (_gguf.Tensors[name].Type is not (GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0))
                throw new NotSupportedException($"Bonsai2 rotated tensor '{name}' must use PQ2_0 or PTQ1_0 encoding.");
        }
    }

    private QuantizedWeight LoadBonsaiQuantizedWeight(GgufTensorInfo info)
    {
        if (info.Shape.Length != 2 || info.Shape[0] == 0 || info.Shape[0] % 128 != 0)
            throw new InvalidDataException($"Bonsai2 tensor '{info.Name}' must be a matrix with 128-aligned rows.");
        EnsureQuantBackendAvailable();
        if (!_gguf.TryGetTensorDataPointer(info, out IntPtr source))
            throw new IOException($"Unable to map Bonsai2 tensor '{info.Name}'.");
        long elements = checked((long)info.Shape[0] * (long)info.Shape[1]);
        long bytes = checked(elements / 64 * 18);
        IntPtr destination = QuantizedWeight.AllocateBuffer(bytes);
        try
        {
            GgmlBonsai.TranscodeToQ2_0((int)info.Type, source, elements, destination);
            return new QuantizedWeight(destination, bytes, (int)GgmlTensorType.Q2_0,
                (long)info.Shape[0], (long)info.Shape[1]);
        }
        catch
        {
            QuantizedWeight.FreeBuffer(destination);
            throw;
        }
    }

    protected void RegisterBonsaiWeightTransforms(int headVDim, int keyHeads, int valueHeads)
    {
        if (BonsaiHadamard == null)
            return;
        foreach (var (name, weight) in _quantWeights)
        {
            bool inverse = BonsaiHadamard.InverseWeightNames.Contains(name);
            bool forward = BonsaiHadamard.WeightNames.Contains(name);
            if (!forward && name.EndsWith(".ffn_gate_up.weight", StringComparison.Ordinal))
            {
                string prefix = name[..^"ffn_gate_up.weight".Length];
                forward = BonsaiHadamard.WeightNames.Contains(prefix + "ffn_gate.weight") &&
                    BonsaiHadamard.WeightNames.Contains(prefix + "ffn_up.weight");
            }
            if (!forward && name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal))
            {
                string prefix = name[..^"attn_qkv.weight".Length];
                forward = BonsaiHadamard.WeightNames.Contains(prefix + "attn_q.weight") &&
                    BonsaiHadamard.WeightNames.Contains(prefix + "attn_k.weight") &&
                    BonsaiHadamard.WeightNames.Contains(prefix + "attn_v.weight");
            }
            if (!forward && !inverse)
                continue;
            bool grouped = forward && BonsaiHadamard.GdnVGrouped && name.EndsWith(".ssm_out.weight", StringComparison.Ordinal);
            if (grouped && (keyHeads <= 0 || valueHeads % keyHeads != 0 || weight.Ne0 != (long)headVDim * valueHeads))
                throw new InvalidDataException($"Invalid Bonsai2 grouped GDN dimensions for '{name}'.");
            weight.SetBonsaiTransform(BonsaiHadamard.SignsByWidth[checked((int)weight.Ne0)],
                BonsaiHadamard.BlockSize, inverse,
                grouped ? headVDim : 0, grouped ? keyHeads : 0, grouped ? valueHeads / keyHeads : 1);
        }
    }
}
