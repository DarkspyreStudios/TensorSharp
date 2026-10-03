// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    /// <summary>
    /// One dense Qwen 3 layer consumed by the native whole-model prefill graph.
    /// Field order and native size must match TSGgmlQwen3PrefillLayerDesc.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen3PrefillLayerArgs
    {
        public IntPtr AttnNormW;
        public IntPtr QkvW;
        public IntPtr QNormW;
        public IntPtr KNormW;
        public IntPtr OW;
        public IntPtr FfnNormW;
        public IntPtr GuW;
        public IntPtr DownW;
        public IntPtr KCache;
        public IntPtr VCache;
        public IntPtr QkvBias;

        public long QkvNe0, QkvNe1, QkvBytes;
        public long ONe0, ONe1, OBytes;
        public long GuNe0, GuNe1, GuBytes;
        public long DownNe0, DownNe1, DownBytes;

        public int StructBytes;
        public int QkvType;
        public int OType;
        public int GuType;
        public int DownType;

        public static int NativeSize => Marshal.SizeOf<Qwen3PrefillLayerArgs>();
    }

    public partial class GgmlBasicOps
    {
        /// <summary>
        /// Runs embedding, all dense Qwen 3 layers, and optionally the final
        /// norm/LM head as one native graph. The native kernel writes this
        /// chunk's K/V rows in place and returns false when the shape is not
        /// supported, allowing the existing managed path to remain the fallback.
        /// </summary>
        public static bool TryQwen3ModelPrefill(
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
            => GgmlNative.TryQwen3ModelPrefill(
                layers, numLayers, tokenIds, numTokens, startPos,
                logits, vocabSize, computeLogits,
                tokenEmbd, tokenEmbdType,
                tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                outputNorm,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                hiddenSize, numHeads, numKvHeads, headDim,
                intermediateSize, maxSeqLen, kvCacheType,
                eps, ropeBase, ropeFreqScale, ropeMode,
                ropeOriginalContext, ropeExtFactor, ropeAttnFactor,
                ropeBetaFast, ropeBetaSlow);

        /// <summary>Runs the prefill graph and writes the last token's final-normed hidden state instead of logits.</summary>
        public static bool TryQwen3ModelPrefillHiddenState(
            Qwen3PrefillLayerArgs[] layers, int numLayers,
            int[] tokenIds, int numTokens, int startPos,
            IntPtr hidden, int vocabSize,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr outputNorm,
            int hiddenSize, int numHeads, int numKvHeads, int headDim,
            int intermediateSize, int maxSeqLen, int kvCacheType,
            float eps, float ropeBase, float ropeFreqScale, int ropeMode,
            int ropeOriginalContext, float ropeExtFactor, float ropeAttnFactor,
            float ropeBetaFast, float ropeBetaSlow)
            => GgmlNative.TryQwen3ModelPrefillHiddenState(
                layers, numLayers, tokenIds, numTokens, startPos,
                hidden, vocabSize,
                tokenEmbd, tokenEmbdType,
                tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                outputNorm,
                hiddenSize, numHeads, numKvHeads, headDim,
                intermediateSize, maxSeqLen, kvCacheType,
                eps, ropeBase, ropeFreqScale, ropeMode,
                ropeOriginalContext, ropeExtFactor, ropeAttnFactor,
                ropeBetaFast, ropeBetaSlow);
    }
}
