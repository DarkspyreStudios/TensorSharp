// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

/// <summary>TensorSharp-owned integration for Bonsai2's PRISM checkpoint format.</summary>
public static class GgmlBonsai
{
    /// <summary>Losslessly repack PQ2_0/PTQ1_0 into upstream ggml Q2_0 blocks.</summary>
    public static void TranscodeToQ2_0(int sourceType, IntPtr source, long elements, IntPtr destination)
    {
        if (sourceType is not (142 or 143) || source == IntPtr.Zero || destination == IntPtr.Zero || elements <= 0 || elements % 128 != 0)
            throw new ArgumentException("Invalid Bonsai2 transcode type, pointer, or block alignment.");
        int status = GgmlNative.TSGgml_TranscodeBonsaiToQ2_0(sourceType, source, elements, destination);
        if (status != 0)
            throw new InvalidOperationException($"Bonsai2 transcode failed with status {status}.");
    }

    public static unsafe void RegisterWeight(IntPtr key, float[] signs, int blockSize, bool inverse,
        int permutationHeadDim = 0, int permutationKeyHeads = 0, int permutationRepeat = 1)
    {
        ArgumentNullException.ThrowIfNull(signs);
        fixed (float* values = signs)
        {
            int status = GgmlNative.TSGgml_BonsaiRegisterWeight(key, signs.Length, (IntPtr)values,
                blockSize, inverse ? 1 : 0, permutationHeadDim, permutationKeyHeads, permutationRepeat);
            if (status == 0)
                throw new InvalidOperationException("Native Bonsai2 Hadamard weight registration failed.");
        }
    }

    public static void UnregisterWeight(IntPtr key) => GgmlNative.TSGgml_BonsaiUnregisterWeight(key);
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    internal static partial int TSGgml_TranscodeBonsaiToQ2_0(int sourceType, IntPtr source, long elements, IntPtr destination);

    [LibraryImport(DllName)]
    internal static partial int TSGgml_BonsaiRegisterWeight(IntPtr key, int width, IntPtr signs,
        int blockSize, int inverse, int permutationHeadDim, int permutationKeyHeads, int permutationRepeat);

    [LibraryImport(DllName)]
    internal static partial void TSGgml_BonsaiUnregisterWeight(IntPtr key);
}
