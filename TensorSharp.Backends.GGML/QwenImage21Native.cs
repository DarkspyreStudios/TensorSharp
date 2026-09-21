// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Weight
{
    public IntPtr Data;
    public int Type, Reserved;
    public long Ne0, Ne1, Bytes;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Block
{
    public QwenImage21Weight Q, K, V, Out, Gate, Up, Down;
    public IntPtr NormQ, NormK;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Segment
{
    public int Start, End, SourceStart, IsImage;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21ForwardArgs
{
    public IntPtr Images, Text, TimeEmbedding, Cos, Sin, Output;
    public QwenImage21Weight ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
    public IntPtr TextNorm, Blocks, Segments;
    public int StructBytes, Dim, Heads, HeadDim, Channels, TextDim;
    public int ImageSeq, TextSeq, TotalSeq, PrefixSeq, NumLayers, NumSegments;
    public float Eps;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_QwenImage21Forward(in QwenImage21ForwardArgs desc);

    public static void QwenImage21Forward(in QwenImage21ForwardArgs desc)
    {
        if (TSGgml_QwenImage21Forward(in desc) == 0)
            throw new InvalidOperationException(GetLastErrorMessage("Qwen-Image-2.1 native inference failed."));
    }
}

public partial class GgmlBasicOps
{
    /// <summary>Complete Qwen-Image-2.1 velocity prediction in one resident-weight GGML graph.</summary>
    public static void QwenImage21Forward(in QwenImage21ForwardArgs args) => GgmlNative.QwenImage21Forward(in args);
}
