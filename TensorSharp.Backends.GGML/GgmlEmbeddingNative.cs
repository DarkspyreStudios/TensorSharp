// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public static partial class GgmlEmbeddingNative
{
    private const string DllName = "GgmlOps";
    static GgmlEmbeddingNative() => GgmlNative.EnsureImportResolverRegistered();

    public static string LastError(string fallback) => GgmlNative.LastNativeError(fallback);

    public static IntPtr TSGgml_EmbeddingLoad(string path, string backend, int device, int threads)
    {
        backend = GgmlNativeLoader.PrepareModelBackend(backend);
        using var reservation = GgmlNativeLoader.ReserveNativeHandle("embedding-model");
        return GgmlNativeLoader.TrackNativeHandle(reservation, Native_TSGgml_EmbeddingLoad(path, backend, device, threads));
    }

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8, EntryPoint = "TSGgml_EmbeddingLoad")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial IntPtr Native_TSGgml_EmbeddingLoad(string path, string backend, int device, int threads);

    public static void TSGgml_EmbeddingFree(IntPtr handle)
    {
        GgmlNativeLoader.ReleaseNativeHandle("embedding-model", handle, () => Native_TSGgml_EmbeddingFree(handle));
    }

    [LibraryImport(DllName, EntryPoint = "TSGgml_EmbeddingFree")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial void Native_TSGgml_EmbeddingFree(IntPtr handle);

    private static unsafe int TSGgml_EmbeddingEncode(IntPtr handle, int* tokens, int* lengths, int batch, float* output, int capacity)
    {
        using var call = GgmlNativeLoader.EnterNativeCall("embedding-model", handle);
        return Native_TSGgml_EmbeddingEncode(handle, tokens, lengths, batch, output, capacity);
    }

    [LibraryImport(DllName, EntryPoint = "TSGgml_EmbeddingEncode")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int Native_TSGgml_EmbeddingEncode(IntPtr handle, int* tokens, int* lengths, int batch, float* output, int capacity);

    public static unsafe void Encode(IntPtr handle, int[] tokens, int[] lengths, float[] output)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentNullException.ThrowIfNull(output);
        long count = 0;
        foreach (int length in lengths)
        {
            if (length <= 0) throw new ArgumentException("Every sequence must have at least one token.", nameof(lengths));
            count += length;
        }
        if (count != tokens.Length) throw new ArgumentException("Lengths do not match the token buffer.", nameof(lengths));
        fixed (int* t = tokens)
        fixed (int* l = lengths)
        fixed (float* o = output)
            if (TSGgml_EmbeddingEncode(handle, t, l, lengths.Length, o, output.Length) != 0)
                throw new InvalidOperationException(GgmlNative.LastNativeError("Embedding inference failed."));
    }
}
