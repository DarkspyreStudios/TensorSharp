// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>Qwen-Image-2.1's single-stream DiT; GGUF projections stay quantized.
/// Latent tokens are 64-channel VAE pixels, without the original Qwen-Image's 2x2 packing.</summary>
internal sealed class QwenImage21DiT : ModelBase
{
    internal const int HiddenSize = 4096, HeadDim = 128, Heads = 32, Channels = 64, Layers = 32, TextDim = 4096;
    private readonly Dictionary<string, IntPtr> _pointers = new();
    private readonly List<IntPtr> _owned = new();
    private readonly QwenImage21Block[] _blocks;
    private readonly QwenImage21ForwardArgs _nativeWeights;
    private readonly string _prefix;

    public QwenImage21DiT(string ggufPath, BackendType backend) : base(ggufPath, backend)
    {
        try
        {
            if (!IsGgmlBackend) throw new NotSupportedException("Qwen-Image-2.1 requires a GGML backend (ggml-metal, ggml-cuda, ggml-vulkan or ggml-cpu).");
            EnsureQuantBackendAvailable();
            Config = new ModelConfig { Architecture = "qwen_image_2_1", HiddenSize = HiddenSize, NumLayers = Layers };
            _prefix = _gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
            _nativeWeights = new QwenImage21ForwardArgs
            {
                ImageIn = Weight("img_in.weight", Channels, HiddenSize),
                TextIn = Weight("txt_in.in_layer.weight", TextDim, HiddenSize),
                TextOut = Weight("txt_in.out_layer.weight", HiddenSize, HiddenSize),
                TimeIn = Weight("time_text_embed.timestep_embedder.linear_1.weight", 256, HiddenSize),
                TimeOut = Weight("time_text_embed.timestep_embedder.linear_2.weight", HiddenSize, HiddenSize),
                Modulation = Weight("modulation.1.weight", HiddenSize, 4 * HiddenSize),
                NormOut = Weight("norm_out.linear.weight", HiddenSize, HiddenSize),
                ProjOut = Weight("proj_out.weight", HiddenSize, Channels),
                TextNorm = F32("txt_in.text_norm.weight", TextDim),
                StructBytes = Marshal.SizeOf<QwenImage21ForwardArgs>(),
                Dim = HiddenSize, Heads = Heads, HeadDim = HeadDim, Channels = Channels, TextDim = TextDim,
                NumLayers = Layers, Eps = 1e-6f,
            };
            _blocks = new QwenImage21Block[Layers];
            for (int i = 0; i < Layers; ++i)
            {
                string p = $"transformer_blocks.{i}.";
                bool fused = _gguf.Tensors.ContainsKey(_prefix + p + "img_mlp.gate_up.weight");
                _blocks[i] = new QwenImage21Block
                {
                    Q = Weight(p + "attn.to_q.weight", HiddenSize, HiddenSize),
                    K = Weight(p + "attn.to_k.weight", HiddenSize, HiddenSize),
                    V = Weight(p + "attn.to_v.weight", HiddenSize, HiddenSize),
                    Out = Weight(p + "attn.to_out.0.weight", HiddenSize, HiddenSize),
                    Gate = Weight(p + (fused ? "img_mlp.gate_up.weight" : "img_mlp.gate_layer.weight"), HiddenSize, fused ? 24576 : 12288),
                    Up = fused ? default : Weight(p + "img_mlp.proj.weight", HiddenSize, 12288),
                    Down = Weight(p + "img_mlp.out.weight", 12288, HiddenSize),
                    NormQ = F32(p + "attn.norm_q.weight", HeadDim),
                    NormK = F32(p + "attn.norm_k.weight", HeadDim),
                };
            }
            if (_gguf.Tensors.ContainsKey(_prefix + $"transformer_blocks.{Layers}.attn.to_q.weight"))
                throw new NotSupportedException("Expected a 32-layer Qwen-Image-2.1 transformer.");
            Console.WriteLine($"Qwen-Image-2.1 DiT: {Layers} layers, {HiddenSize} hidden, {Heads} heads, quantized resident GGML graph.");
        }
        catch { Dispose(); throw; }
    }

    private QwenImage21Weight Weight(string name, int input, int output)
    {
        name = _prefix + name;
        if (!_gguf.Tensors.TryGetValue(name, out var info) || info.Shape.Length != 2 ||
            (long)info.Shape[0] != input || (long)info.Shape[1] != output)
            throw new NotSupportedException($"Qwen-Image-2.1 requires {name} with GGUF shape [{input},{output}].");
        if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
        {
            ptr = QuantizedWeight.AllocateBuffer(_gguf.GetTensorByteCount(info));
            _owned.Add(ptr);
            _gguf.ReadTensorDataToNative(info, ptr, _gguf.GetTensorByteCount(info));
        }
        return new QwenImage21Weight { Data = ptr, Type = (int)info.Type, Ne0 = input, Ne1 = output, Bytes = _gguf.GetTensorByteCount(info) };
    }

    private IntPtr F32(string name, int count)
    {
        name = _prefix + name;
        if (_pointers.TryGetValue(name, out var ptr)) return ptr;
        if (!_gguf.Tensors.TryGetValue(name, out var info) || info.NumElements != count)
            throw new NotSupportedException($"Qwen-Image-2.1 requires {name} with {count} elements.");
        if (info.Type == GgmlTensorType.F32 && _gguf.TryGetTensorDataPointer(info, out ptr))
            return _pointers[name] = ptr;
        ptr = QuantizedWeight.AllocateBuffer(count * sizeof(float));
        _owned.Add(ptr);
        long bytes = _gguf.GetTensorByteCount(info);
        IntPtr source = QuantizedWeight.AllocateBuffer(bytes);
        try
        {
            _gguf.ReadTensorDataToNative(info, source, bytes);
            NativeDequant.DequantizeToFloat32Native((int)info.Type, source, ptr, count);
        }
        finally { QuantizedWeight.FreeBuffer(source); }
        return _pointers[name] = ptr;
    }

    /// <param name="imageSlots">One tag per text token: 0=text, 1..N=reference image.
    /// Each contiguous vision-slot run is replaced by four times as many latent tokens.</param>
    internal float[] Predict(float[] targetTokens, int latentH, int latentW, float[] textCond, int textSeq,
        float timestep01, int[] imageSlots = null, float[][] referenceTokens = null,
        int[] referenceHeights = null, int[] referenceWidths = null)
    {
        if (latentH <= 0 || latentW <= 0 || targetTokens == null || targetTokens.Length != checked(latentH * latentW * Channels))
            throw new ArgumentException("Target must contain latentH*latentW*64 token-major floats.");
        if (textSeq <= 0 || textCond == null || textCond.Length != checked(textSeq * TextDim))
            throw new ArgumentException("Conditioning must contain textSeq*4096 token-major floats.");
        if (!float.IsFinite(timestep01) || timestep01 < 0 || timestep01 > 1)
            throw new ArgumentOutOfRangeException(nameof(timestep01));
        referenceTokens ??= Array.Empty<float[]>();
        referenceHeights ??= Array.Empty<int>();
        referenceWidths ??= Array.Empty<int>();
        if (referenceHeights.Length != referenceTokens.Length || referenceWidths.Length != referenceTokens.Length)
            throw new ArgumentException("Reference latent shapes are required for each reference image.");
        for (int i = 0; i < referenceTokens.Length; ++i)
            if (referenceHeights[i] <= 0 || referenceWidths[i] <= 0 || referenceTokens[i] == null ||
                referenceTokens[i].Length != checked(referenceHeights[i] * referenceWidths[i] * Channels))
                throw new ArgumentException($"Invalid reference latent {i}.");
        var shapes = referenceHeights.Select((h, i) => (Height: h, Width: referenceWidths[i])).Append((latentH, latentW)).ToArray();
        var layout = BuildLayout(textSeq, imageSlots, shapes);
        float[] images = new float[checked(referenceTokens.Sum(x => x.Length) + targetTokens.Length)];
        int offset = 0;
        foreach (var reference in referenceTokens) { reference.CopyTo(images, offset); offset += reference.Length; }
        targetTokens.CopyTo(images, offset);
        var time = new float[512];
        for (int i = 0; i < 128; ++i)
        {
            float angle = timestep01 * 1000f * MathF.Exp(-MathF.Log(10000f) * i / 128);
            time[i] = MathF.Cos(angle); time[i + 128] = MathF.Sin(angle);
            time[256 + i] = 1f;
        }
        var output = new float[targetTokens.Length];
        var pins = new List<GCHandle>();
        IntPtr Pin<T>(T[] data) where T : struct
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            pins.Add(handle);
            return handle.AddrOfPinnedObject();
        }
        try
        {
            var args = _nativeWeights;
            args.Images = Pin(images); args.Text = Pin(textCond); args.TimeEmbedding = Pin(time);
            args.Cos = Pin(layout.Cos); args.Sin = Pin(layout.Sin); args.Output = Pin(output);
            args.Blocks = Pin(_blocks); args.Segments = Pin(layout.Segments);
            args.ImageSeq = images.Length / Channels; args.TextSeq = textSeq;
            args.TotalSeq = layout.Cos.Length / (HeadDim / 2); args.PrefixSeq = layout.Prefix;
            args.NumSegments = layout.Segments.Length;
            GgmlBasicOps.QwenImage21Forward(in args);
        }
        finally { foreach (var pin in pins) pin.Free(); }
        if (output.Any(v => !float.IsFinite(v)))
            throw new InvalidOperationException("Qwen-Image-2.1 DiT produced non-finite latent velocities.");
        return output;
    }

    internal static (QwenImage21Segment[] Segments, float[] Cos, float[] Sin, int Prefix) BuildLayout(
        int textLength, int[] imageSlots, (int Height, int Width)[] shapes)
    {
        if (textLength <= 0 || shapes == null || shapes.Length == 0 ||
            shapes.Any(s => s.Height <= 0 || s.Width <= 0) || (imageSlots != null && imageSlots.Length != textLength))
            throw new ArgumentException("Invalid Qwen-Image-2.1 token layout.");
        var segments = new List<QwenImage21Segment>();
        var positions = new List<(int T, int H, int W)>();
        int position = 0, nextImage = 0, imageOffset = 0;
        void AppendImage(int index)
        {
            var (height, width) = shapes[index];
            int count = checked(height * width), start = positions.Count;
            segments.Add(new QwenImage21Segment { Start = start, End = checked(start + count), SourceStart = imageOffset, IsImage = 1 });
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w)
                    positions.Add((position, h - (height - height / 2), w - (width - width / 2)));
            position += Math.Max(height, width); imageOffset += count;
        }
        for (int i = 0; i < textLength;)
        {
            int begin = i, tag = imageSlots?[i] ?? 0;
            while (i < textLength && (imageSlots?[i] ?? 0) == tag) ++i;
            if (tag != 0)
            {
                if (tag != nextImage + 1 || nextImage + 1 >= shapes.Length ||
                    (long)(i - begin) * 4 != (long)shapes[nextImage].Height * shapes[nextImage].Width)
                    throw new ArgumentException("Vision slots and reference latents must have matching sizes and ordered image tags.");
                AppendImage(nextImage++);
            }
            else
            {
                int start = positions.Count;
                segments.Add(new QwenImage21Segment { Start = start, End = start + i - begin, SourceStart = begin });
                for (int j = begin; j < i; ++j, ++position) positions.Add((position, position, position));
            }
        }
        if (nextImage + 1 != shapes.Length) throw new ArgumentException("Missing reference image slots.");
        int prefix = positions.Count;
        AppendImage(nextImage);
        var cos = new float[checked(positions.Count * HeadDim / 2)];
        var sin = new float[cos.Length];
        int[] dims = { 16, 56, 56 };
        for (int token = 0; token < positions.Count; ++token)
        {
            var p = positions[token];
            int[] coords = { p.T, p.H, p.W };
            int channel = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int j = 0; j < dims[axis] / 2; ++j, ++channel)
                {
                    double angle = coords[axis] / Math.Pow(10000.0, 2.0 * j / dims[axis]);
                    cos[token * HeadDim / 2 + channel] = (float)Math.Cos(angle);
                    sin[token * HeadDim / 2 + channel] = (float)Math.Sin(angle);
                }
        }
        return (segments.ToArray(), cos, sin, prefix);
    }

    protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException("Use Predict for image inference.");
    protected override void ResetKVCacheCore() { }
    public override void Dispose()
    {
        base.Dispose();
        foreach (var ptr in _owned) QuantizedWeight.FreeBuffer(ptr);
        _owned.Clear(); _pointers.Clear();
    }
}
