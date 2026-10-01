// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    internal static unsafe int TSGgml_QwenVaeAttention(float* qkv, float* output, int channels, int sequence)
    {
        using var call = GgmlNativeLoader.EnterNativeCall();
        return Native_TSGgml_QwenVaeAttention(qkv, output, channels, sequence);
    }

    [LibraryImport(DllName, EntryPoint = "TSGgml_QwenVaeAttention")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int Native_TSGgml_QwenVaeAttention(float* qkv, float* output, int channels, int sequence);
}

public partial class GgmlBasicOps
{
    /// <summary>Spatial attention over all keys in channel-planar Q/K/V, with bounded score scratch.</summary>
    public static unsafe bool TryQwenVaeAttention(float[] qkv, float[] output, int channels, int sequence)
    {
        if (channels <= 0 || sequence <= 0 || qkv == null || output == null ||
            qkv.LongLength != 3L * channels * sequence || output.LongLength != (long)channels * sequence)
            throw new ArgumentException("Expected Q/K/V [3,C,sequence] and output [C,sequence].");
        fixed (float* input = qkv, result = output)
            return GgmlNative.TSGgml_QwenVaeAttention(input, result, channels, sequence) != 0;
    }
}
