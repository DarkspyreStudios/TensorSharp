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

namespace TensorSharp.Models
{
    public partial class Qwen3Model
    {
        /// <summary>
        /// Runs <paramref name="tokens"/> as a fresh sequence and returns the final-normed hidden state of the
        /// last token, the vector a Qwen3-Embedding checkpoint pools. The KV cache is reset first and the LM head
        /// never runs. Leading chunks take the same no-logits prefill as <see cref="ForwardRefillCore"/>; the last
        /// chunk runs the per-operation layer loop so its hidden state stays in managed memory.
        /// </summary>
        internal float[] ForwardLastHiddenState(int[] tokens)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (tokens.Length == 0)
                throw new ArgumentException("At least one token is required.", nameof(tokens));
            if (IsTensorParallel)
                throw new NotSupportedException("Qwen3 embeddings do not support tensor parallelism.");

            ResetKVCache();
            int chunkSize = ResolvePrefillChunkSize();
            int lastStart = (tokens.Length - 1) / chunkSize * chunkSize;
            for (int pos = 0; pos < lastStart; pos += chunkSize)
                PrefillWithoutLogits(tokens.AsSpan(pos, chunkSize).ToArray());

            int[] last = tokens.AsSpan(lastStart).ToArray();
            int seqLen = last.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor hidden = Embedding(last);
            if (seqLen > 1)
                DropNativeQwen3DecodeForActiveCache();
            EnsureKvCacheHostSynchronized();
            for (int layer = 0; layer < Config.NumLayers; layer++)
                hidden = TransformerBlock(hidden, layer, seqLen, startPos);

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();
            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            float[] state = TensorToFloatArray(lastHidden);
            lastHidden.Dispose();
            _cacheSeqLen += seqLen;
            return state;
        }
    }
}
