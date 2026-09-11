// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static partial class GgmlDeepSeek41VisionNative
    {
        private const string DllName = "GgmlOps";
        static GgmlDeepSeek41VisionNative() => GgmlNative.EnsureImportResolverRegistered();

        [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr TSGgml_Dsv41VisionLoad(string path, string backendName, int device, int nThreads);
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void TSGgml_Dsv41VisionFree(IntPtr handle);
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv41VisionInfo(IntPtr handle, int* info, int count);
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv41VisionEncode(IntPtr handle, float* patches,
            int nH, int nW, float* output, int capacity);
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int TSGgml_Dsv41AttachVision(IntPtr text, IntPtr vision);
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv41ForwardVision(IntPtr text, int* tokens,
            byte* imageMask, float* imageEmbeddings, int nTokens, int nImageTokens, float* logits);

        public static unsafe int[] Info(IntPtr handle)
        {
            var info = new int[8];
            fixed (int* p = info)
                if (TSGgml_Dsv41VisionInfo(handle, p, info.Length) != 0)
                    throw new InvalidOperationException("DeepSeek V4.1 vision metadata query failed (see stderr).");
            return info;
        }

        public static unsafe int Encode(IntPtr handle, float[] patches, int rows, int columns, float[] output)
        {
            ArgumentNullException.ThrowIfNull(patches);
            ArgumentNullException.ThrowIfNull(output);
            int patchSize = Info(handle)[0];
            if (rows <= 0 || columns <= 0 || (long)rows * columns * patchSize * patchSize * 3 != patches.Length)
                throw new ArgumentException("Vision patch buffer does not match the declared patch grid.");
            fixed (float* p = patches)
            fixed (float* o = output)
                return TSGgml_Dsv41VisionEncode(handle, p, rows, columns, o, output.Length);
        }

        public static unsafe int Forward(IntPtr handle, int[] tokens, byte[] imageMask,
            float[] embeddings, int imageRows, float[] logits)
        {
            if (imageMask.Length != tokens.Length)
                throw new ArgumentException("Image mask must contain one entry per input token.");
            fixed (int* t = tokens)
            fixed (byte* m = imageMask)
            fixed (float* e = embeddings)
            fixed (float* l = logits)
                return TSGgml_Dsv41ForwardVision(handle, t, m, e, tokens.Length, imageRows, l);
        }
    }
}
