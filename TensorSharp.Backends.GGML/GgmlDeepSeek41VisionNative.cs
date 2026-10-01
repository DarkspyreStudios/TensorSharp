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

        public static IntPtr TSGgml_Dsv41VisionLoad(string path, string backendName, int device, int nThreads)
        {
            backendName = GgmlNativeLoader.PrepareModelBackend(backendName);
            using var call = GgmlNativeLoader.EnterNativeCall();
            return GgmlNativeLoader.TrackNativeHandle("deepseek-vision", Native_TSGgml_Dsv41VisionLoad(path, backendName, device, nThreads));
        }

        [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8, EntryPoint = "TSGgml_Dsv41VisionLoad")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr Native_TSGgml_Dsv41VisionLoad(string path, string backendName, int device, int nThreads);
        public static void TSGgml_Dsv41VisionFree(IntPtr handle)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            using var resource = GgmlNativeLoader.BeginNativeHandleRelease("deepseek-vision", handle);
            Native_TSGgml_Dsv41VisionFree(handle);
            resource.Complete();
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Dsv41VisionFree")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void Native_TSGgml_Dsv41VisionFree(IntPtr handle);
        private static unsafe int TSGgml_Dsv41VisionInfo(IntPtr handle, int* info, int count)
        {
            using var call = GgmlNativeLoader.EnterNativeCall("deepseek-vision", handle);
            return Native_TSGgml_Dsv41VisionInfo(handle, info, count);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Dsv41VisionInfo")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int Native_TSGgml_Dsv41VisionInfo(IntPtr handle, int* info, int count);
        private static unsafe int TSGgml_Dsv41VisionEncode(IntPtr handle, float* patches, int nH, int nW, float* output, int capacity)
        {
            using var call = GgmlNativeLoader.EnterNativeCall("deepseek-vision", handle);
            return Native_TSGgml_Dsv41VisionEncode(handle, patches, nH, nW, output, capacity);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Dsv41VisionEncode")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int Native_TSGgml_Dsv41VisionEncode(IntPtr handle, float* patches, int nH, int nW, float* output, int capacity);
        public static int TSGgml_Dsv41AttachVision(IntPtr text, IntPtr vision)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            return Native_TSGgml_Dsv41AttachVision(text, vision);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Dsv41AttachVision")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int Native_TSGgml_Dsv41AttachVision(IntPtr text, IntPtr vision);
        private static unsafe int TSGgml_Dsv41ForwardVision(IntPtr text, int* tokens, byte* imageMask, float* imageEmbeddings, int nTokens, int nImageTokens, float* logits)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            return Native_TSGgml_Dsv41ForwardVision(text, tokens, imageMask, imageEmbeddings, nTokens, nImageTokens, logits);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Dsv41ForwardVision")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int Native_TSGgml_Dsv41ForwardVision(IntPtr text, int* tokens, byte* imageMask, float* imageEmbeddings, int nTokens, int nImageTokens, float* logits);

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
