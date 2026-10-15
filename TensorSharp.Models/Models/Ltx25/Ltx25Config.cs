// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Tensor-derived LTX 2.5 denoiser geometry. Norm/time/RoPE conventions
/// follow the original default AV architecture; shapes cannot establish absent
/// checkpoint configuration such as theta or frequency precision. This does
/// not register a pipeline or establish text encoder/VAE compatibility.</summary>
public sealed class Ltx25Config
{
    public string Prefix { get; private init; }
    public int VideoDim { get; private init; }
    public int AudioDim { get; private init; }
    public int VideoChannels { get; private init; }
    public int AudioChannels { get; private init; }
    public int VideoHeads { get; private init; }
    public int AudioHeads { get; private init; }
    public int VideoHeadDim => VideoDim / VideoHeads;
    public int AudioHeadDim => AudioDim / AudioHeads;
    public int Layers { get; private init; }
    public int VideoContextDim { get; private init; }
    public int AudioContextDim { get; private init; }
    public bool VideoFfnBias { get; private init; }
    public bool AudioFfnBias { get; private init; }
    public bool KeyframeEmbedding { get; private init; }
    public const int TimestepDimension = 256;
    public const float NormEpsilon = 1e-6f;
    public const float TimestepMultiplier = 1000f;
    public const float CrossGateMultiplier = 1f;

    public static Ltx25Config FromGguf(GgufFile file)
        => FromShapes(file.Tensors.ToDictionary(pair => pair.Key, pair => pair.Value.Shape, StringComparer.Ordinal));

    /// <summary>Validate an AV checkpoint with nine-way modulation and gated
    /// attention, the LTX 2.5 variant implemented by this denoiser.</summary>
    public static Ltx25Config FromShapes(IReadOnlyDictionary<string, ulong[]> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        string[] prefixes = ["", "model.diffusion_model."];
        string[] found = prefixes.Where(prefix => tensors.ContainsKey(prefix + "patchify_proj.weight")).ToArray();
        if (found.Length != 1) throw new InvalidDataException("Expected one LTX denoiser tensor namespace.");
        string p = found[0];
        int Axis(string name, int axis)
        {
            if (!tensors.TryGetValue(p + name, out var shape) || axis >= shape.Length || shape[axis] == 0 || shape[axis] > int.MaxValue)
                throw new InvalidDataException("Missing or invalid LTX tensor: " + p + name);
            return (int)shape[axis];
        }
        void Shape(string name, params int[] expected)
        {
            if (!tensors.TryGetValue(p + name, out var shape) || shape.Length != expected.Length ||
                !shape.SequenceEqual(expected.Select(value => (ulong)value)))
                throw new InvalidDataException("Unexpected LTX tensor geometry: " + p + name);
        }
        int v = Axis("patchify_proj.weight", 1), a = Axis("audio_patchify_proj.weight", 1);
        int vc = Axis("patchify_proj.weight", 0), ac = Axis("audio_patchify_proj.weight", 0);
        int vh = Axis("transformer_blocks.0.attn1.to_gate_logits.bias", 0);
        int ah = Axis("transformer_blocks.0.audio_attn1.to_gate_logits.bias", 0);
        if (v > 16384 || a > 16384 || vc > 4096 || ac > 4096 || vh > 128 || ah > 128 ||
            vh != ah || v % vh != 0 || a % ah != 0 || v / vh % 2 != 0 || a / ah % 2 != 0)
            throw new InvalidDataException("Unsupported LTX attention geometry.");
        int contextV = Axis("transformer_blocks.0.attn2.to_k.weight", 0);
        int contextA = Axis("transformer_blocks.0.audio_attn2.to_k.weight", 0);
        // Prompt-side modulation is defined over the corresponding stream dim.
        if (contextV != v || contextA != a) throw new InvalidDataException("LTX 2.5 context/modulation widths differ.");
        int layers = 0;
        string blockPrefix = p + "transformer_blocks.";
        foreach (string name in tensors.Keys)
        {
            if (!name.StartsWith(blockPrefix, StringComparison.Ordinal)) continue;
            int end = name.IndexOf('.', blockPrefix.Length);
            if (end > blockPrefix.Length && int.TryParse(name.AsSpan(blockPrefix.Length, end - blockPrefix.Length), out int index))
                layers = Math.Max(layers, checked(index + 1));
        }
        if (layers is < 1 or > 128) throw new InvalidDataException("Invalid LTX transformer depth.");
        bool vb = tensors.ContainsKey(p + "transformer_blocks.0.ff.net.0.proj.bias");
        bool ab = tensors.ContainsKey(p + "transformer_blocks.0.audio_ff.net.0.proj.bias");
        void Linear(string name, int input, int output, bool bias = true)
        {
            Shape(name + ".weight", input, output);
            if (bias) Shape(name + ".bias", output);
            else if (tensors.ContainsKey(p + name + ".bias")) throw new InvalidDataException("Inconsistent LTX feed-forward bias: " + p + name);
        }
        void Ada(string name, int dim, int coefficients)
        {
            Linear(name + ".emb.timestep_embedder.linear_1", TimestepDimension, dim);
            Linear(name + ".emb.timestep_embedder.linear_2", dim, dim);
            Linear(name + ".linear", dim, checked(coefficients * dim));
        }
        void Attention(string name, int query, int context, int inner, int heads)
        {
            Linear(name + ".to_q", query, inner); Linear(name + ".to_k", context, inner);
            Linear(name + ".to_v", context, inner); Linear(name + ".to_out.0", inner, query);
            Linear(name + ".to_gate_logits", query, heads);
            Shape(name + ".q_norm.weight", inner); Shape(name + ".k_norm.weight", inner);
        }
        Linear("patchify_proj", vc, v); Linear("audio_patchify_proj", ac, a);
        Linear("proj_out", v, vc); Linear("audio_proj_out", a, ac);
        Shape("scale_shift_table", v, 2); Shape("audio_scale_shift_table", a, 2);
        Ada("adaln_single", v, 9); Ada("audio_adaln_single", a, 9);
        Ada("prompt_adaln_single", v, 2); Ada("audio_prompt_adaln_single", a, 2);
        Ada("av_ca_video_scale_shift_adaln_single", v, 4); Ada("av_ca_audio_scale_shift_adaln_single", a, 4);
        Ada("av_ca_a2v_gate_adaln_single", v, 1); Ada("av_ca_v2a_gate_adaln_single", a, 1);
        for (int i = 0; i < layers; ++i)
        {
            string b = $"transformer_blocks.{i}.";
            Shape(b + "scale_shift_table", v, 9); Shape(b + "audio_scale_shift_table", a, 9);
            Shape(b + "prompt_scale_shift_table", v, 2); Shape(b + "audio_prompt_scale_shift_table", a, 2);
            Shape(b + "scale_shift_table_a2v_ca_video", v, 5); Shape(b + "scale_shift_table_a2v_ca_audio", a, 5);
            Attention(b + "attn1", v, v, v, vh); Attention(b + "attn2", v, contextV, v, vh);
            Attention(b + "audio_attn1", a, a, a, ah); Attention(b + "audio_attn2", a, contextA, a, ah);
            Attention(b + "audio_to_video_attn", v, a, a, ah); Attention(b + "video_to_audio_attn", a, v, a, ah);
            Linear(b + "ff.net.0.proj", v, checked(4 * v), vb); Linear(b + "ff.net.2", checked(4 * v), v, vb);
            Linear(b + "audio_ff.net.0.proj", a, checked(4 * a), ab); Linear(b + "audio_ff.net.2", checked(4 * a), a, ab);
        }
        bool keyframe = tensors.ContainsKey(p + "keyframes_abs_pos_embedding");
        if (keyframe) Shape("keyframes_abs_pos_embedding", v, 1);
        return new Ltx25Config
        {
            Prefix = p, VideoDim = v, AudioDim = a, VideoChannels = vc, AudioChannels = ac,
            VideoHeads = vh, AudioHeads = ah, Layers = layers, VideoContextDim = contextV, AudioContextDim = contextA,
            VideoFfnBias = vb, AudioFfnBias = ab, KeyframeEmbedding = keyframe,
        };
    }
}
