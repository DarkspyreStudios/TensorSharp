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
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>
    /// A byte trie over every token's exact bytes, built once per tokenizer and
    /// shared by every request.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is what makes constrained decoding cheap. The obvious way to build a
    /// token mask — llama.cpp's way — is to test all V tokens against the grammar
    /// independently, which at a 256k vocabulary is 256k grammar walks <i>per
    /// generated token</i>. Sharing prefixes instead means a token's first byte
    /// is tested once for all tokens starting with it, and when a prefix is
    /// rejected the entire subtree below it is skipped. This is the same
    /// structure xgrammar uses, and the reason it and outlines are usable at
    /// production vocab sizes.
    /// </para>
    /// <para>
    /// Nodes are stored in flat arrays rather than as objects: children live in
    /// one contiguous span per node, sorted by byte, so a walk is a couple of
    /// array reads with no pointer chasing or dictionary hashing on the hot path.
    /// </para>
    /// </remarks>
    public sealed class GrammarTokenVocabulary
    {
        // Children of node i occupy [_childStart[i], _childStart[i+1]) of the
        // _childByte/_childNode pair of arrays, sorted ascending by byte.
        private readonly int[] _childStart;
        private readonly byte[] _childByte;
        private readonly int[] _childNode;

        /// <summary>Token that ends at each node, or -1.</summary>
        private readonly int[] _nodeToken;
        private readonly Dictionary<int, int[]> _tokenAliases;

        private sealed class PerTokenizer
        {
            public readonly Dictionary<string, GrammarTokenVocabulary> ByAllowedControls = new(StringComparer.Ordinal);
        }
        private static readonly ConditionalWeakTable<ITokenizer, PerTokenizer> Cache = new();

        public int VocabSize { get; }
        public int NodeCount => _nodeToken.Length;

        /// <summary>Ids excluded from grammar matching (control tokens, EOS).</summary>
        public IReadOnlyCollection<int> SpecialTokenIds { get; }

        /// <summary>Number of ulong words a full-vocabulary bitmask needs.</summary>
        public int MaskWords => (VocabSize + 63) >> 6;

        private GrammarTokenVocabulary(
            int[] childStart, byte[] childByte, int[] childNode, int[] nodeToken,
            int vocabSize, IReadOnlyCollection<int> special, Dictionary<int, int[]> tokenAliases)
        {
            _childStart = childStart;
            _childByte = childByte;
            _childNode = childNode;
            _nodeToken = nodeToken;
            VocabSize = vocabSize;
            SpecialTokenIds = special;
            _tokenAliases = tokenAliases;
        }

        /// <summary>
        /// Get (building on first use) the trie for <paramref name="tokenizer"/>.
        /// Cached weakly against the tokenizer, so it is built once per model and
        /// released with it.
        /// </summary>
        public static GrammarTokenVocabulary ForTokenizer(ITokenizer tokenizer,
            IReadOnlyCollection<int>? allowedControlTokens = null)
        {
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));
            int[] allowed = allowedControlTokens?.Distinct().OrderBy(i => i).ToArray() ?? Array.Empty<int>();
            foreach (int id in allowed)
                if (id < 0 || id >= tokenizer.VocabSize || tokenizer.IsEos(id))
                    throw new ArgumentException("Grammar control-token allowlist contains an invalid or EOS token.", nameof(allowedControlTokens));
            string key = string.Join(",", allowed);
            var per = Cache.GetValue(tokenizer, _ => new PerTokenizer());
            lock (per)
            {
                if (!per.ByAllowedControls.TryGetValue(key, out var vocabulary))
                    per.ByAllowedControls[key] = vocabulary = Build(tokenizer, allowed);
                return vocabulary;
            }
        }

        private static GrammarTokenVocabulary Build(ITokenizer tokenizer, IReadOnlyCollection<int> allowedControlTokens)
        {
            int vocabSize = tokenizer.VocabSize;

            var special = new HashSet<int>();
            if (tokenizer is ISpecialTokenVocabulary sv)
            {
                foreach (int id in sv.SpecialTokenIds) special.Add(id);
            }
            else
            {
                foreach (int id in tokenizer.EosTokenIds) special.Add(id);
            }
            // A protocol grammar may deliberately consume a printable control
            // marker. This vocabulary is cached separately: JSON and other
            // grammars must continue to exclude it and every other control ID.
            special.ExceptWith(allowedControlTokens);

            // Mutable build representation; flattened below.
            var childMaps = new List<Dictionary<byte, int>> { new() };
            var tokenAt = new List<int> { -1 };
            var aliases = new Dictionary<int, List<int>>();

            var buffer = new List<byte>(64);
            for (int id = 0; id < vocabSize; id++)
            {
                if (special.Contains(id)) continue;

                buffer.Clear();
                try
                {
                    tokenizer.AppendTokenBytes(id, buffer);
                }
                catch
                {
                    // A vocabulary entry we cannot render as bytes cannot be
                    // matched against a grammar either; leaving it out of the
                    // trie means it is never unmasked, which is the safe side.
                    continue;
                }
                if (buffer.Count == 0) continue;

                int node = 0;
                for (int k = 0; k < buffer.Count; k++)
                {
                    byte b = buffer[k];
                    if (!childMaps[node].TryGetValue(b, out int next))
                    {
                        next = childMaps.Count;
                        childMaps.Add(new Dictionary<byte, int>());
                        tokenAt.Add(-1);
                        childMaps[node][b] = next;
                    }
                    node = next;
                }
                // Keep every ID with identical bytes. In particular a dormant
                // grammar must mask every alias of an invalid trigger suffix.
                if (tokenAt[node] < 0) tokenAt[node] = id;
                else
                {
                    if (!aliases.TryGetValue(node, out var ids))
                        aliases[node] = ids = new List<int> { tokenAt[node] };
                    ids.Add(id);
                }
            }

            int nodeCount = childMaps.Count;
            var childStart = new int[nodeCount + 1];
            int totalEdges = 0;
            for (int i = 0; i < nodeCount; i++)
            {
                childStart[i] = totalEdges;
                totalEdges += childMaps[i].Count;
            }
            childStart[nodeCount] = totalEdges;

            var childByte = new byte[totalEdges];
            var childNode = new int[totalEdges];
            var scratch = new List<byte>(256);
            for (int i = 0; i < nodeCount; i++)
            {
                scratch.Clear();
                foreach (byte b in childMaps[i].Keys) scratch.Add(b);
                scratch.Sort();
                int at = childStart[i];
                foreach (byte b in scratch)
                {
                    childByte[at] = b;
                    childNode[at] = childMaps[i][b];
                    at++;
                }
            }

            return new GrammarTokenVocabulary(
                childStart, childByte, childNode, tokenAt.ToArray(), vocabSize, special,
                aliases.ToDictionary(p => p.Key, p => p.Value.ToArray()));
        }

        internal ReadOnlySpan<int> TokensAt(int node)
            => _tokenAliases.TryGetValue(node, out var ids) ? ids
                : _nodeToken[node] < 0 ? ReadOnlySpan<int>.Empty : _nodeToken.AsSpan(node, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int ChildStart(int node) => _childStart[node];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int ChildEnd(int node) => _childStart[node + 1];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal byte EdgeByte(int edge) => _childByte[edge];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int EdgeNode(int edge) => _childNode[edge];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int TokenAt(int node) => _nodeToken[node];
    }
}
