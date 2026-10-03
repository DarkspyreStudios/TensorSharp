// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen3Model
    {
        /// <summary>
        /// Runs <paramref name="tokens"/> as a fresh sequence and returns the final-normed hidden state of the
        /// last token, the vector a Qwen3-Embedding checkpoint pools. The KV cache is reset first and the LM head
        /// never runs.
        /// </summary>
        /// <remarks>
        /// On a GGML backend the whole sequence runs through the native prefill graph, which returns the hidden
        /// state in place of logits. Otherwise every token but the last is prefilled one at a time and the last
        /// token runs the per-operation layer loop.
        /// </remarks>
        internal float[] ForwardLastHiddenState(int[] tokens)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (tokens.Length == 0)
                throw new ArgumentException("At least one token is required.", nameof(tokens));
            if (IsTensorParallel)
                throw new NotSupportedException("Qwen3 embeddings do not support tensor parallelism.");

            ResetKVCache();
            if (tokens.Length >= 2 && CanUseNativeQwen3Prefill)
            {
                // Leading chunks commit K/V only; the last chunk keeps at least two tokens for the native graph.
                int chunkSize = ResolvePrefillChunkSize();
                int lastStart = (tokens.Length - 2) / chunkSize * chunkSize;
                bool prefilled = true;
                for (int pos = 0; pos < lastStart && prefilled; pos += chunkSize)
                {
                    int[] chunk = tokens.AsSpan(pos, chunkSize).ToArray();
                    prefilled = TryNativeQwen3Prefill(chunk, _cacheSeqLen, false, out _);
                    if (prefilled)
                    {
                        _kvCacheHostDirty = true;
                        _cacheSeqLen += chunk.Length;
                    }
                }
                if (prefilled)
                {
                    int[] last = tokens.AsSpan(lastStart).ToArray();
                    if (TryNativeQwen3HiddenState(last, _cacheSeqLen, out float[] native))
                    {
                        _kvCacheHostDirty = true;
                        _cacheSeqLen += last.Length;
                        return native;
                    }
                }
                ResetKVCache();
            }

            for (int pos = 0; pos < tokens.Length - 1; pos++)
                PrefillWithoutLogits(new[] { tokens[pos] });

            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + 1);
            Tensor hidden = Embedding(new[] { tokens[^1] });
            EnsureKvCacheHostSynchronized();
            for (int layer = 0; layer < Config.NumLayers; layer++)
                hidden = TransformerBlock(hidden, layer, 1, startPos);

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();
            float[] state = TensorToFloatArray(normed);
            normed.Dispose();
            _cacheSeqLen += 1;
            return state;
        }

        private unsafe bool TryNativeQwen3HiddenState(int[] tokens, int startPos, out float[] hidden)
        {
            hidden = null;
            if (tokens.Length <= 1 || !CanUseNativeQwen3Prefill || !EnsureQwen3PrefillDescriptors())
                return false;

            DropNativeQwen3DecodeForActiveCache();
            int kvType = _kvCacheDtype.GgmlType();
            if (kvType != 0 && kvType != 1 && kvType != 2 && kvType != 8)
                return false;

            EnsureCacheCapacity(startPos + tokens.Length);
            int maxSeqLen = (int)_kvCacheK[0].Sizes[1];
            var state = new float[Config.HiddenSize];
            bool ok;
            fixed (float* statePtr = state)
            {
                ok = GgmlBasicOps.TryQwen3ModelPrefillHiddenState(
                    _qwen3PrefillLayers, Config.NumLayers,
                    tokens, tokens.Length, startPos,
                    (IntPtr)statePtr, Config.VocabSize,
                    _qwen3PrefillEmbedding.Data,
                    _qwen3PrefillEmbedding.Type,
                    _qwen3PrefillEmbedding.Ne0,
                    _qwen3PrefillEmbedding.Ne1,
                    _qwen3PrefillEmbedding.Bytes,
                    _qwen3PrefillOutputNorm,
                    Config.HiddenSize, Config.NumHeads,
                    Config.NumKVHeads, Config.HeadDim,
                    Config.IntermediateSize, maxSeqLen, kvType,
                    Config.Eps, Config.RopeBase,
                    1.0f / Config.RopeScale, 2,
                    _ropeOriginalContext, _ropeExtFactor,
                    _ropeAttnFactor, _ropeBetaFast, _ropeBetaSlow);
            }
            if (ok)
                hidden = state;
            return ok;
        }
    }
}
