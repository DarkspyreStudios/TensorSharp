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
        private static int TSGgml_Qwen3ModelPrefill([In] Qwen3PrefillLayerArgs[] layers, int numLayers, [In] int[] tokenIds, int numTokens, int startPos, IntPtr logits, int vocabSize, int computeLogits, IntPtr tokenEmbd, int tokenEmbdType, long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes, IntPtr outputNorm, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, int hiddenSize, int numHeads, int numKvHeads, int headDim, int intermediateSize, int maxSeqLen, int kvCacheType, float eps, float ropeBase, float ropeFreqScale, int ropeMode, int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor, float ropeBetaFast, float ropeBetaSlow)
        {
            using var call = GgmlNativeLoader.EnterNativeCall();
            return Native_TSGgml_Qwen3ModelPrefill(layers, numLayers, tokenIds, numTokens, startPos, logits, vocabSize, computeLogits, tokenEmbd, tokenEmbdType, tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes, outputNorm, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, hiddenSize, numHeads, numKvHeads, headDim, intermediateSize, maxSeqLen, kvCacheType, eps, ropeBase, ropeFreqScale, ropeMode, ropeOriginalContext, ropeExtFactor, ropeAttnFactor, ropeBetaFast, ropeBetaSlow);
        }

        [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen3ModelPrefill")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int Native_TSGgml_Qwen3ModelPrefill([In] Qwen3PrefillLayerArgs[] layers, int numLayers, [In] int[] tokenIds, int numTokens, int startPos, IntPtr logits, int vocabSize, int computeLogits, IntPtr tokenEmbd, int tokenEmbdType, long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes, IntPtr outputNorm, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, int hiddenSize, int numHeads, int numKvHeads, int headDim, int intermediateSize, int maxSeqLen, int kvCacheType, float eps, float ropeBase, float ropeFreqScale, int ropeMode, int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor, float ropeBetaFast, float ropeBetaSlow);

        internal static bool TryQwen3ModelPrefill(
            Qwen3PrefillLayerArgs[] layers, int numLayers,
            int[] tokenIds, int numTokens, int startPos,
            IntPtr logits, int vocabSize, bool computeLogits,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr outputNorm,
            IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            int hiddenSize, int numHeads, int numKvHeads, int headDim,
            int intermediateSize, int maxSeqLen, int kvCacheType,
            float eps, float ropeBase, float ropeFreqScale, int ropeMode,
            int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor,
            float ropeBetaFast, float ropeBetaSlow)
            => TSGgml_Qwen3ModelPrefill(
                layers, numLayers, tokenIds, numTokens, startPos,
                logits, vocabSize, computeLogits ? 1 : 0,
                tokenEmbd, tokenEmbdType,
                tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                outputNorm,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                hiddenSize, numHeads, numKvHeads, headDim,
                intermediateSize, maxSeqLen, kvCacheType,
                eps, ropeBase, ropeFreqScale, ropeMode,
                ropeOriginalContext, ropeExtFactor, ropeAttnFactor,
                ropeBetaFast, ropeBetaSlow) != 0;
    }
}
