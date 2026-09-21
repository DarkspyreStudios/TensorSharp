// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime;
using System.IO;

namespace TensorSharp.Models
{
    /// <summary>
    /// Engram lookup metadata embedded in the checkpoint GGUF and the n-gram
    /// hashing it drives. This is the managed counterpart of dsv41_engram.h;
    /// multipliers, bucket geometry and token IDs must match exactly.
    /// </summary>
    internal sealed class Dsv41EngramData
    {
        /// <summary>One Engram table: which layer owns it, how many rows it has,
        /// and the hash parameters that address them.</summary>
        internal sealed class LayerLayout
        {
            public int Id;
            public long Rows;
            public ulong[] Multipliers;   // [MaxNgramSize]
            public uint[] Primes;         // [HashColumns] - bucket sizes
            public long[] Offsets;        // [HashColumns] - running sum of Primes
        }

        public uint VocabSize;
        public uint CompressedVocabSize;
        public uint PadId;               // already compressed; do not apply TokenMap again
        public uint MaxNgramSize;
        public uint HeadCount;
        public uint HeadDim;
        public ulong TokenizerHash;
        public int CandidateSourceLayerId = -1;
        public uint CandidateTopkBlocks;
        public uint CandidateBlockSize;
        public int[] KvSourceLayerIds;
        public int[] IndexSourceLayerIds;
        public int[] TokenMap;            // [VocabSize] -> compressed id
        public LayerLayout[] Layers;

        /// <summary>Hash columns per token per table: one per (lookback, head).</summary>
        public uint HashColumns => (MaxNgramSize - 1) * HeadCount;

        /// <summary>
        /// FNV-1a over each token's byte length then its bytes, folded across the
        /// whole vocabulary. Vision artifacts use this fingerprint to identify their
        /// parent checkpoint.
        /// </summary>
        public static ulong FingerprintToken(ulong hash, string token)
        {
            ulong size = (ulong)System.Text.Encoding.UTF8.GetByteCount(token);
            for (int i = 0; i < 8; i++)
                hash = (hash ^ (byte)(size >> (8 * i))) * 1099511628211UL;
            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(token))
                hash = (hash ^ b) * 1099511628211UL;
            return hash;
        }

        public static Dsv41EngramData Load(GgufFile file,
            Func<string, GgufTensorInfo> findTensor = null)
        {
            findTensor ??= name => file.Tensors.TryGetValue(name, out var tensor) ? tensor : null;
            InvalidDataException Invalid(string key) =>
                new InvalidDataException("Missing or invalid DeepSeek V4.1 GGUF metadata: " + key);
            long Integer(string key, long? fallback = null)
            {
                if (!file.Metadata.TryGetValue(key, out object raw))
                    return fallback ?? throw Invalid(key);
                try
                {
                    return raw switch
                    {
                        uint u32 => u32, int i32 => i32,
                        ulong u64 => checked((long)u64), long i64 => i64,
                        _ => throw Invalid(key),
                    };
                }
                catch (OverflowException) { throw Invalid(key); }
            }
            uint U32(string key, long? fallback = null)
            {
                long value = Integer(key, fallback);
                if (value < 0 || value > uint.MaxValue) throw Invalid(key);
                return (uint)value;
            }
            ulong[] ArrayValues(string key, int maxCount)
            {
                if (!file.Metadata.TryGetValue(key, out object raw) || raw is not Array array ||
                    array.Length == 0 || array.Length > maxCount)
                    throw Invalid(key);
                var result = new ulong[array.Length];
                try
                {
                    for (int i = 0; i < result.Length; i++)
                        result[i] = array.GetValue(i) switch
                        {
                            uint u32 => u32, int i32 => checked((ulong)i32),
                            ulong u64 => u64, long i64 => checked((ulong)i64),
                            _ => throw Invalid(key),
                        };
                }
                catch (OverflowException) { throw Invalid(key); }
                return result;
            }
            string[] tokens = file.GetStringArray("tokenizer.ggml.tokens");
            if (tokens == null || tokens.Length == 0 || tokens.Length > 1048576)
                throw Invalid("tokenizer.ggml.tokens");
            var embedding = findTensor("token_embd.weight");
            if (embedding == null || embedding.Shape.Length < 2 || embedding.Shape[1] != (ulong)tokens.Length)
                throw Invalid("token embedding vocabulary does not match tokenizer");
            var data = new Dsv41EngramData
            {
                VocabSize = (uint)tokens.Length,
                PadId = U32("deepseek41.engram.pad_id"),
                MaxNgramSize = U32("deepseek41.engram.max_ngram_size"),
                HeadCount = U32("deepseek41.engram.head_count"),
                HeadDim = U32("deepseek41.engram.key_length"),
                TokenizerHash = 14695981039346656037UL,
            };
            foreach (string token in tokens)
                data.TokenizerHash = FingerprintToken(data.TokenizerHash, token);
            if (data.MaxNgramSize < 2 || data.MaxNgramSize > 16 || data.HeadCount == 0 ||
                data.HeadCount > 128 || data.HeadDim == 0 || data.HeadDim > 65536)
                throw Invalid("deepseek41.engram dimensions");
            ulong[] ids = ArrayValues("deepseek41.engram.layer_ids", 128);
            int columns = (int)data.HashColumns;
            ulong[] multipliers = ArrayValues("deepseek41.engram.multipliers", ids.Length * (int)data.MaxNgramSize);
            ulong[] primes = ArrayValues("deepseek41.engram.primes", ids.Length * columns);
            ulong[] offsets = ArrayValues("deepseek41.engram.offsets", ids.Length * columns);
            if (multipliers.Length != ids.Length * data.MaxNgramSize ||
                primes.Length != ids.Length * columns || offsets.Length != ids.Length * columns)
                throw Invalid("deepseek41.engram flattened array lengths");
            ulong[] map = ArrayValues("deepseek41.engram.token_map", tokens.Length);
            data.TokenMap = new int[map.Length];
            for (int i = 0; i < map.Length; i++)
            {
                if (map[i] >= data.VocabSize) throw Invalid("deepseek41.engram.token_map");
                data.TokenMap[i] = (int)map[i];
                data.CompressedVocabSize = Math.Max(data.CompressedVocabSize, (uint)map[i] + 1);
            }
            uint nLayer = U32("deepseek41.block_count");
            if (nLayer == 0 || nLayer > 128) throw Invalid("deepseek41.block_count");
            data.Layers = new LayerLayout[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] >= nLayer) throw Invalid("deepseek41.engram.layer_ids");
                var tensor = findTensor($"blk.{ids[i]}.engram_embd.weight");
                if (tensor == null || tensor.Shape.Length < 2 || tensor.Shape[0] != data.HeadDim ||
                    tensor.Shape[1] == 0 || tensor.Shape[1] > int.MaxValue ||
                    tensor.Shape.Skip(2).Any(d => d != 1))
                    throw Invalid($"Engram embedding tensor dimensions at layer {ids[i]}");
                var layer = new LayerLayout
                {
                    Id = (int)ids[i], Rows = (long)tensor.Shape[1],
                    Multipliers = multipliers.AsSpan(i * (int)data.MaxNgramSize, (int)data.MaxNgramSize).ToArray(),
                    Primes = new uint[columns], Offsets = new long[columns],
                };
                for (int j = 0; j < columns; j++)
                {
                    if (primes[i * columns + j] > uint.MaxValue || offsets[i * columns + j] > long.MaxValue)
                        throw Invalid("deepseek41.engram bucket dimensions");
                    layer.Primes[j] = (uint)primes[i * columns + j];
                    layer.Offsets[j] = (long)offsets[i * columns + j];
                }
                data.Layers[i] = layer;
            }
            ulong[] ratios = ArrayValues("deepseek41.attention.compress_ratios", 256);
            if (ratios.Length < nLayer) throw Invalid("deepseek41.attention.compress_ratios");
            var kv = new List<int>();
            var index = new List<int>();
            int candidate = -1;
            for (int i = 0; i < nLayer; i++)
            {
                if (findTensor($"blk.{i}.engram_embd.weight") != null && !ids.Contains((ulong)i))
                    throw Invalid("Engram embedding tensor is absent from deepseek41.engram.layer_ids");
                if (findTensor($"blk.{i}.attn_compressor_kv.weight") != null) kv.Add(i);
                if (findTensor($"blk.{i}.indexer.attn_q_b.weight") != null)
                {
                    index.Add(i);
                    if (candidate < 0 && ratios[i] == 1) candidate = i;
                }
            }
            data.KvSourceLayerIds = kv.ToArray();
            data.IndexSourceLayerIds = index.ToArray();
            // Published Flash GGUF omits pruning geometry. The first ratio-1
            // indexer owns the mask; these are its original checkpoint defaults.
            long source = Integer("deepseek41.attention.indexer.candidate_source_layer", candidate);
            if (source < -1 || source >= nLayer) throw Invalid("deepseek41.attention.indexer.candidate_source_layer");
            data.CandidateSourceLayerId = (int)source;
            if (source >= 0)
            {
                data.CandidateTopkBlocks = U32("deepseek41.attention.indexer.candidate_top_k", 2048);
                data.CandidateBlockSize = U32("deepseek41.attention.indexer.candidate_block_size", 8);
                if (!index.Contains((int)source)) throw Invalid("candidate source must own an indexer");
            }
            data.Validate((uint)tokens.Length);
            return data;
        }

        private void Validate(uint expectedVocab)
        {
            if (VocabSize != expectedVocab || VocabSize == 0 || VocabSize > 1048576 ||
                CompressedVocabSize == 0 || CompressedVocabSize > VocabSize || PadId >= CompressedVocabSize ||
                Layers.Length == 0 || Layers.Length > 128 || MaxNgramSize < 2 || MaxNgramSize > 16 ||
                HeadCount == 0 || HeadCount > 128 || HeadDim == 0 || HeadDim > 65536 ||
                KvSourceLayerIds.Length == 0 || KvSourceLayerIds.Length > 128 ||
                IndexSourceLayerIds.Length == 0 || IndexSourceLayerIds.Length > 128 ||
                CandidateSourceLayerId < -1 ||
                (CandidateSourceLayerId >= 0 && (CandidateTopkBlocks == 0 || CandidateBlockSize == 0 ||
                    CandidateTopkBlocks > int.MaxValue || CandidateBlockSize > int.MaxValue)))
                throw new InvalidDataException("DeepSeek V4.1 GGUF Engram metadata has invalid dimensions");
            if (TokenMap.Length != VocabSize)
                throw new InvalidDataException("DeepSeek V4.1 GGUF Engram token map does not match the tokenizer vocabulary");
            var seen = new bool[CompressedVocabSize];
            foreach (int value in TokenMap)
            {
                if (value < 0 || value >= CompressedVocabSize)
                    throw new InvalidDataException("DeepSeek V4.1 compressed token id is out of bounds");
                seen[value] = true;
            }
            if (seen.Any(present => !present))
                throw new InvalidDataException("DeepSeek V4.1 compressed token map has missing ids");
            for (int i = 0; i < Layers.Length; i++)
            {
                LayerLayout layer = Layers[i];
                if (layer.Id < 0 || (i > 0 && layer.Id <= Layers[i - 1].Id) ||
                    layer.Rows <= 0 || layer.Rows > int.MaxValue || layer.Multipliers.Length != MaxNgramSize ||
                    layer.Primes.Length != HashColumns || layer.Offsets.Length != HashColumns)
                    throw new InvalidDataException("Invalid DeepSeek V4.1 Engram table dimensions");
                foreach (ulong multiplier in layer.Multipliers)
                    if ((multiplier & 1) == 0 || multiplier > (ulong)long.MaxValue / CompressedVocabSize)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram hash multiplier");
                long total = 0;
                for (int j = 0; j < layer.Primes.Length; j++)
                {
                    if (layer.Primes[j] < 2 || layer.Offsets[j] != total)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram bucket layout");
                    total += layer.Primes[j];
                }
                if (total != layer.Rows)
                    throw new InvalidDataException("DeepSeek V4.1 Engram bucket sizes do not match table rows");
            }
        }

        /// <summary>
        /// Row ids for one ubatch, laid out [layer][token][hash column].
        ///
        /// <para><paramref name="history"/> is the sequence's compressed-token
        /// history and is extended in place, because a lookback reaches tokens
        /// from earlier calls. A negative input token marks an image position: it
        /// stores -1, which blocks that position and every later lookback that
        /// reaches it, substituting the pad token instead.</para>
        /// </summary>
        public int[] HashTokens(ReadOnlySpan<int> tokens, int startPos, ref int[] history, ref int historyLength)
        {
            if (startPos > historyLength)
                throw new InvalidOperationException("DeepSeek V4.1 Engram history is not contiguous");
            foreach (int token in tokens)
            {
                if (token >= 0 && (uint)token >= VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokens), "DeepSeek V4.1 token is out of bounds");
            }

            int needed = startPos + tokens.Length;
            if (history == null || history.Length < needed)
                Array.Resize(ref history, Math.Max(needed, (history?.Length ?? 0) * 2));
            for (int i = 0; i < tokens.Length; i++)
                history[startPos + i] = tokens[i] < 0 ? -1 : TokenMap[tokens[i]];
            historyLength = needed;

            uint columns = HashColumns;
            var hashes = new int[Layers.Length * tokens.Length * columns];
            for (int li = 0; li < Layers.Length; li++)
            {
                LayerLayout layer = Layers[li];
                for (int i = 0; i < tokens.Length; i++)
                {
                    int pos = startPos + i;
                    ulong rolling = 0;
                    bool blocked = false;
                    for (uint shift = 0; shift < MaxNgramSize; shift++)
                    {
                        // Once a lookback runs off the start of the sequence or
                        // hits an image position, every longer one is blocked too.
                        blocked = blocked || pos < shift || history[pos - shift] < 0;
                        int token = blocked ? (int)PadId : history[pos - shift];
                        rolling ^= (ulong)token * layer.Multipliers[shift];
                        if (shift == 0)
                            continue;
                        for (uint head = 0; head < HeadCount; head++)
                        {
                            uint column = (shift - 1) * HeadCount + head;
                            // Every head at this lookback sees the same rolling
                            // value; the prime and offset are what separate them.
                            hashes[(li * tokens.Length + i) * columns + column] =
                                (int)(rolling % layer.Primes[column] + (ulong)layer.Offsets[column]);
                        }
                    }
                }
            }
            return hashes;
        }
    }
}
