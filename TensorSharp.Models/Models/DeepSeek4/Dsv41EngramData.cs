// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.IO;

namespace TensorSharp.Models
{
    /// <summary>
    /// The tokenizer-derived Engram sidecar (<c>deepseek41.engram.bin</c>) and the
    /// n-gram hashing it drives.
    ///
    /// <para>The GGUF conversion keeps neither the compressed token map nor the
    /// per-layer bucket layout, so V4.1 cannot select an Engram row without this
    /// file. It is produced by <c>eng/dsv41-prepare.py</c> from the checkpoint's
    /// tokenizer and config.</para>
    ///
    /// <para>This is the managed counterpart of
    /// <c>TensorSharp.GGML.Native/dsv41_engram.h</c>. The two must agree exactly:
    /// a single differing multiplier, prime or offset selects a different row and
    /// silently changes every embedding.</para>
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
        public uint PadTokenId;
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
        /// whole vocabulary. The sidecar records the value its tokenizer produced;
        /// a mismatch means the sidecar belongs to a different checkpoint, and
        /// every row it selects would be wrong.
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

        public static Dsv41EngramData Load(string path, uint expectedVocab, ulong expectedTokenizerHash)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    "DeepSeek V4.1 requires its tokenizer-derived Engram lookup sidecar. " +
                    "Run python eng/dsv41-prepare.py --help for preparation instructions.", path);

            byte[] raw = File.ReadAllBytes(path);
            int at = 0;

            ulong Read(int size)
            {
                if (at + size > raw.Length)
                    throw new InvalidDataException("Truncated DeepSeek V4.1 Engram metadata");
                ulong value = 0;
                for (int i = 0; i < size; i++)
                    value |= (ulong)raw[at + i] << (8 * i);
                at += size;
                return value;
            }

            const string magic = "TSD41E01";
            foreach (char c in magic)
            {
                if (at >= raw.Length || raw[at++] != (byte)c)
                    throw new InvalidDataException("Invalid DeepSeek V4.1 Engram metadata version");
            }

            var result = new Dsv41EngramData
            {
                VocabSize = (uint)Read(4),
                CompressedVocabSize = (uint)Read(4),
                PadTokenId = (uint)Read(4),
            };
            uint layerCount = (uint)Read(4);
            result.MaxNgramSize = (uint)Read(4);
            result.HeadCount = (uint)Read(4);
            result.HeadDim = (uint)Read(4);
            result.TokenizerHash = Read(8);
            result.CandidateSourceLayerId = (int)(uint)Read(4);
            result.CandidateTopkBlocks = (uint)Read(4);
            result.CandidateBlockSize = (uint)Read(4);
            uint kvCount = (uint)Read(4), indexCount = (uint)Read(4);

            if (result.VocabSize != expectedVocab || result.VocabSize > 1048576 ||
                result.TokenizerHash != expectedTokenizerHash || result.CompressedVocabSize == 0 ||
                result.CompressedVocabSize > result.VocabSize || result.PadTokenId >= result.VocabSize ||
                layerCount == 0 || layerCount > 128 || result.MaxNgramSize < 2 || result.MaxNgramSize > 16 ||
                result.HeadCount == 0 || result.HeadCount > 128 || result.HeadDim == 0 || result.HeadDim > 65536 ||
                kvCount == 0 || kvCount > 128 || indexCount == 0 || indexCount > 128 ||
                (result.CandidateSourceLayerId >= 0 && (result.CandidateTopkBlocks == 0 || result.CandidateBlockSize == 0)))
            {
                throw new InvalidDataException(
                    "DeepSeek V4.1 Engram metadata does not match the model tokenizer or has invalid dimensions");
            }

            int[] SourceIds(uint count)
            {
                var values = new int[count];
                for (uint i = 0; i < count; i++)
                {
                    int value = (int)(uint)Read(4);
                    if (value < 0 || (i > 0 && value <= values[i - 1]))
                        throw new InvalidDataException("Invalid DeepSeek V4.1 shared-cache source layers");
                    values[i] = value;
                }
                return values;
            }

            result.KvSourceLayerIds = SourceIds(kvCount);
            result.IndexSourceLayerIds = SourceIds(indexCount);

            result.TokenMap = new int[result.VocabSize];
            var seen = new bool[result.CompressedVocabSize];
            for (int i = 0; i < result.TokenMap.Length; i++)
            {
                int value = (int)(uint)Read(4);
                if (value < 0 || (uint)value >= result.CompressedVocabSize)
                    throw new InvalidDataException("DeepSeek V4.1 compressed token id is out of bounds");
                result.TokenMap[i] = value;
                seen[value] = true;
            }
            foreach (bool present in seen)
            {
                if (!present)
                    throw new InvalidDataException("DeepSeek V4.1 compressed token map has missing ids");
            }

            uint columns = result.HashColumns;
            result.Layers = new LayerLayout[layerCount];
            for (uint i = 0; i < layerCount; i++)
            {
                var layer = new LayerLayout
                {
                    Id = (int)(uint)Read(4),
                    Rows = (long)Read(8),
                };
                if (layer.Id < 0 || (i > 0 && layer.Id <= result.Layers[i - 1].Id) ||
                    layer.Rows <= 0 || layer.Rows > int.MaxValue)
                    throw new InvalidDataException("Invalid DeepSeek V4.1 Engram table dimensions");

                layer.Multipliers = new ulong[result.MaxNgramSize];
                for (uint j = 0; j < result.MaxNgramSize; j++)
                {
                    ulong multiplier = Read(8);
                    // Odd keeps the multiply invertible; the bound keeps
                    // token * multiplier inside a signed 64-bit value.
                    if ((multiplier & 1) == 0 || multiplier > (ulong)long.MaxValue / result.CompressedVocabSize)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram hash multiplier");
                    layer.Multipliers[j] = multiplier;
                }

                layer.Primes = new uint[columns];
                layer.Offsets = new long[columns];
                for (uint j = 0; j < columns; j++)
                    layer.Primes[j] = (uint)Read(4);
                long total = 0;
                for (uint j = 0; j < columns; j++)
                {
                    long offset = (long)Read(8);
                    // Buckets must tile the table exactly, in order and without
                    // gaps: that is what makes a column's rows disjoint.
                    if (layer.Primes[j] < 2 || offset != total)
                        throw new InvalidDataException("Invalid DeepSeek V4.1 Engram bucket layout");
                    layer.Offsets[j] = offset;
                    total += layer.Primes[j];
                }
                if (total != layer.Rows)
                    throw new InvalidDataException("DeepSeek V4.1 Engram bucket sizes do not match table rows");
                result.Layers[i] = layer;
            }

            if (at != raw.Length)
                throw new InvalidDataException("Unexpected trailing DeepSeek V4.1 Engram metadata");
            return result;
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
                        int token = blocked ? TokenMap[PadTokenId] : history[pos - shift];
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
