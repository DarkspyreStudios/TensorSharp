// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        private static int TSGgml_Qwen3ModelDecodeLogits([In] Qwen3LayerDecodeArgs[] layers, int numLayers, int tokenId, IntPtr tokenEmbedding, int tokenEmbeddingType, long tokenEmbeddingNe0, long tokenEmbeddingNe1, long tokenEmbeddingBytes, int hiddenSize, int position, int numHeads, int numKvHeads, int headDim, int cacheSize, int intermediateSize, int kvCacheType, float eps, float ropeBase, float ropeFreqScale, int ropeMode, int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor, float ropeBetaFast, float ropeBetaSlow, IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            return Native_TSGgml_Qwen3ModelDecodeLogits(layers, numLayers, tokenId, tokenEmbedding, tokenEmbeddingType, tokenEmbeddingNe0, tokenEmbeddingNe1, tokenEmbeddingBytes, hiddenSize, position, numHeads, numKvHeads, headDim, cacheSize, intermediateSize, kvCacheType, eps, ropeBase, ropeFreqScale, ropeMode, ropeOriginalContext, ropeExtFactor, ropeAttnFactor, ropeBetaFast, ropeBetaSlow, logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen3ModelDecodeLogits")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int Native_TSGgml_Qwen3ModelDecodeLogits([In] Qwen3LayerDecodeArgs[] layers, int numLayers, int tokenId, IntPtr tokenEmbedding, int tokenEmbeddingType, long tokenEmbeddingNe0, long tokenEmbeddingNe1, long tokenEmbeddingBytes, int hiddenSize, int position, int numHeads, int numKvHeads, int headDim, int cacheSize, int intermediateSize, int kvCacheType, float eps, float ropeBase, float ropeFreqScale, int ropeMode, int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor, float ropeBetaFast, float ropeBetaSlow, IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm);

        private static void TSGgml_Qwen3ResetDecodeCache()
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            Native_TSGgml_Qwen3ResetDecodeCache();
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen3ResetDecodeCache")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void Native_TSGgml_Qwen3ResetDecodeCache();

        private static void TSGgml_Qwen3DropDecodeCache(IntPtr firstKCache)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            Native_TSGgml_Qwen3DropDecodeCache(firstKCache);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen3DropDecodeCache")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void Native_TSGgml_Qwen3DropDecodeCache(IntPtr firstKCache);

        internal static void Qwen3ModelDecodeLogits(
            Qwen3LayerDecodeArgs[] layers, int numLayers,
            int tokenId,
            IntPtr tokenEmbedding, int tokenEmbeddingType,
            long tokenEmbeddingNe0, long tokenEmbeddingNe1,
            long tokenEmbeddingBytes,
            int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int intermediateSize, int kvCacheType,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeMode, int ropeOriginalContext,
            float ropeExtFactor, float ropeAttnFactor,
            float ropeBetaFast, float ropeBetaSlow,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm)
        {
            CheckResult(TSGgml_Qwen3ModelDecodeLogits(
                layers, numLayers,
                tokenId,
                tokenEmbedding, tokenEmbeddingType,
                tokenEmbeddingNe0, tokenEmbeddingNe1,
                tokenEmbeddingBytes,
                hiddenSize, position,
                numHeads, numKvHeads, headDim, cacheSize,
                intermediateSize, kvCacheType,
                eps, ropeBase, ropeFreqScale,
                ropeMode, ropeOriginalContext,
                ropeExtFactor, ropeAttnFactor,
                ropeBetaFast, ropeBetaSlow,
                logits, vocabSize,
                lmHead, lmHeadType,
                lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm), "qwen3_model_decode_logits");
        }

        internal static void Qwen3ResetDecodeCache() => TSGgml_Qwen3ResetDecodeCache();
        internal static void Qwen3DropDecodeCache(IntPtr firstKCache)
            => TSGgml_Qwen3DropDecodeCache(firstKCache);
    }
}
