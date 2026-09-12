// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;

namespace InferenceWeb.Tests;

/// <summary>
/// The Engram sidecar decides which of 384 million rows each token reads. A
/// single wrong multiplier, prime or offset selects a different row and changes
/// every embedding silently, so the parser and the hashing are pinned here
/// against an independently written expectation rather than against themselves.
/// </summary>
public class Dsv41EngramDataTests : IDisposable
{
    private readonly List<string> _temp = new();

    public void Dispose()
    {
        foreach (string p in _temp)
        {
            try { File.Delete(p); } catch { /* best effort */ }
        }
    }

    /// <summary>Builds a valid sidecar so the tests exercise the real parser.</summary>
    private string WriteSidecar(
        uint vocab = 8, uint compressed = 4, uint pad = 3,
        uint maxNgram = 3, uint heads = 2, uint headDim = 16,
        ulong tokenizerHash = 0xABCDEF0123456789UL,
        int[] layerIds = null, uint[][] primesPerLayer = null,
        ulong[][] multipliersPerLayer = null, int[] tokenMap = null,
        int candidateSource = -1, uint candidateTopk = 0, uint candidateBlock = 0)
    {
        layerIds ??= new[] { 1, 5 };
        uint columns = (maxNgram - 1) * heads;
        primesPerLayer ??= layerIds.Select((_, li) =>
            Enumerable.Range(0, (int)columns).Select(c => (uint)(11 + 2 * c + 4 * li)).ToArray()).ToArray();
        multipliersPerLayer ??= layerIds.Select((_, li) =>
            Enumerable.Range(0, (int)maxNgram).Select(j => (ulong)(2 * (j + 3 + li) + 1)).ToArray()).ToArray();
        // Every compressed id must appear at least once.
        tokenMap ??= Enumerable.Range(0, (int)vocab).Select(i => i % (int)compressed).ToArray();

        var bytes = new List<byte>();
        bytes.AddRange(System.Text.Encoding.ASCII.GetBytes("TSD41E01"));
        void U32(uint v) { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(b, v); bytes.AddRange(b.ToArray()); }
        void I32(int v) => U32(unchecked((uint)v));
        void U64(ulong v) { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(b, v); bytes.AddRange(b.ToArray()); }

        U32(vocab); U32(compressed); U32(pad);
        U32((uint)layerIds.Length); U32(maxNgram); U32(heads); U32(headDim);
        U64(tokenizerHash);
        I32(candidateSource); U32(candidateTopk); U32(candidateBlock);
        U32(1); U32(1);                       // one kv source, one index source
        I32(0);                               // kv source layer ids
        I32(0);                               // index source layer ids
        foreach (int v in tokenMap) I32(v);
        for (int li = 0; li < layerIds.Length; li++)
        {
            I32(layerIds[li]);
            U64((ulong)primesPerLayer[li].Select(p => (long)p).Sum());
            foreach (ulong m in multipliersPerLayer[li]) U64(m);
            foreach (uint p in primesPerLayer[li]) U32(p);
            long running = 0;
            foreach (uint p in primesPerLayer[li]) { U64((ulong)running); running += p; }
        }

        string path = Path.Combine(Path.GetTempPath(), $"dsv41-engram-{Guid.NewGuid():N}.bin");
        File.WriteAllBytes(path, bytes.ToArray());
        _temp.Add(path);
        return path;
    }

    [Fact]
    public void Load_ReadsEveryFieldOfAWellFormedSidecar()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, expectedVocab: 8, expectedTokenizerHash: 0xABCDEF0123456789UL);

        Assert.Equal(8u, data.VocabSize);
        Assert.Equal(4u, data.CompressedVocabSize);
        Assert.Equal(3u, data.PadTokenId);
        Assert.Equal(3u, data.MaxNgramSize);
        Assert.Equal(2u, data.HeadCount);
        Assert.Equal(16u, data.HeadDim);
        Assert.Equal(4u, data.HashColumns);              // (3-1) lookbacks x 2 heads
        Assert.Equal(new[] { 1, 5 }, data.Layers.Select(l => l.Id));
        // Offsets must be the running sum of the bucket sizes, and the last
        // bucket must land exactly on the row count.
        foreach (var layer in data.Layers)
        {
            long running = 0;
            for (int c = 0; c < layer.Primes.Length; c++)
            {
                Assert.Equal(running, layer.Offsets[c]);
                running += layer.Primes[c];
            }
            Assert.Equal(layer.Rows, running);
        }
    }

    [Theory]
    [InlineData("vocab")]
    [InlineData("hash")]
    public void Load_RefusesASidecarFromADifferentCheckpoint(string mismatch)
    {
        string path = WriteSidecar();
        uint vocab = mismatch == "vocab" ? 9u : 8u;
        ulong hash = mismatch == "hash" ? 1UL : 0xABCDEF0123456789UL;
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(path, vocab, hash));
    }

    [Fact]
    public void Load_RefusesBucketsThatDoNotTileTheTable()
    {
        // Offsets are written as the running sum; corrupt one and the layout is
        // no longer a partition of the rows.
        string path = WriteSidecar();
        byte[] raw = File.ReadAllBytes(path);
        int firstOffset = raw.Length - (8 * 4) - (4 * 4);   // last layer's first offset
        BinaryPrimitives.WriteUInt64LittleEndian(raw.AsSpan(firstOffset), 999);
        File.WriteAllBytes(path, raw);
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL));
    }

    /// <summary>
    /// The hashing recurrence, checked against the formula written out longhand:
    /// a rolling XOR of (compressed token * multiplier) over the last
    /// MaxNgramSize tokens, reduced per column by that column's prime and shifted
    /// by its offset. Every head at one lookback shares the rolling value.
    /// </summary>
    [Fact]
    public void HashTokens_MatchesTheRecurrenceWrittenOutLonghand()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL);

        int[] tokens = { 5, 2, 7, 0 };
        int[] history = null; int historyLength = 0;
        int[] actual = data.HashTokens(tokens, 0, ref history, ref historyLength);

        uint columns = data.HashColumns;
        Assert.Equal(data.Layers.Length * tokens.Length * (int)columns, actual.Length);

        int[] compressed = tokens.Select(t => data.TokenMap[t]).ToArray();
        int padCompressed = data.TokenMap[data.PadTokenId];

        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < tokens.Length; i++)
            {
                ulong rolling = 0;
                for (uint shift = 0; shift < data.MaxNgramSize; shift++)
                {
                    // Before the start of the sequence the lookback is blocked and
                    // reads the pad token instead.
                    int token = (i < shift) ? padCompressed : compressed[i - (int)shift];
                    rolling ^= (ulong)token * layer.Multipliers[shift];
                    if (shift == 0) continue;
                    for (uint head = 0; head < data.HeadCount; head++)
                    {
                        uint column = (shift - 1) * data.HeadCount + head;
                        int expected = (int)(rolling % layer.Primes[column] + (ulong)layer.Offsets[column]);
                        Assert.Equal(expected, actual[(li * tokens.Length + i) * columns + column]);
                    }
                }
            }
        }
    }

    [Fact]
    public void HashTokens_EveryRowLandsInsideItsOwnColumnBucket()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL);
        int[] tokens = { 1, 6, 3, 4, 0, 7 };
        int[] history = null; int historyLength = 0;
        int[] rows = data.HashTokens(tokens, 0, ref history, ref historyLength);

        uint columns = data.HashColumns;
        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < tokens.Length; i++)
            {
                for (uint c = 0; c < columns; c++)
                {
                    int row = rows[(li * tokens.Length + i) * columns + c];
                    Assert.InRange(row, (int)layer.Offsets[c], (int)(layer.Offsets[c] + layer.Primes[c] - 1));
                    Assert.InRange(row, 0, (int)layer.Rows - 1);
                }
            }
        }
    }

    /// <summary>
    /// A lookback that reaches a visual position is blocked, and so is every
    /// longer one behind it. That is what keeps an image from leaking into the
    /// n-grams of the text that follows it.
    /// </summary>
    [Fact]
    public void HashTokens_AnImagePositionBlocksItselfAndEveryLookbackThroughIt()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL);
        uint columns = data.HashColumns;

        int[] history = null; int historyLength = 0;
        int[] withImage = data.HashTokens(new[] { 1, -1, 4 }, 0, ref history, ref historyLength);

        // Position 2 looks back one to the image and two past it, so with
        // MaxNgramSize 3 every lookback it has is blocked; only the token itself
        // and pad substitutions contribute.
        int[] h2 = null; int l2 = 0;
        int[] allBlocked = data.HashTokens(new[] { -1, -1, 4 }, 0, ref h2, ref l2);
        for (int li = 0; li < data.Layers.Length; li++)
        {
            for (uint c = 0; c < columns; c++)
            {
                int a = withImage[(li * 3 + 2) * columns + c];
                int b = allBlocked[(li * 3 + 2) * columns + c];
                Assert.Equal(b, a);
            }
        }
    }

    /// <summary>History spans calls, because a lookback reaches tokens the
    /// previous ubatch supplied.</summary>
    [Fact]
    public void HashTokens_LooksBackIntoAnEarlierCall()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL);
        uint columns = data.HashColumns;

        int[] whole = null; int wholeLen = 0;
        int[] oneShot = data.HashTokens(new[] { 5, 2, 7, 0 }, 0, ref whole, ref wholeLen);

        int[] split = null; int splitLen = 0;
        data.HashTokens(new[] { 5, 2 }, 0, ref split, ref splitLen);
        int[] tail = data.HashTokens(new[] { 7, 0 }, 2, ref split, ref splitLen);

        for (int li = 0; li < data.Layers.Length; li++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (uint c = 0; c < columns; c++)
                {
                    int expected = oneShot[(li * 4 + (2 + i)) * columns + c];
                    int actual = tail[(li * 2 + i) * columns + c];
                    Assert.Equal(expected, actual);
                }
            }
        }
    }

    [Fact]
    public void HashTokens_RefusesANonContiguousHistory()
    {
        string path = WriteSidecar();
        var data = Dsv41EngramData.Load(path, 8, 0xABCDEF0123456789UL);
        int[] history = null; int historyLength = 0;
        data.HashTokens(new[] { 1, 2 }, 0, ref history, ref historyLength);
        Assert.Throws<InvalidOperationException>(() =>
            data.HashTokens(new[] { 3 }, 5, ref history, ref historyLength));
    }

    /// <summary>
    /// The real sidecar from the DeepSeek-V4.1-Flash checkpoint, when it is
    /// present. The synthetic ones above prove the parser is self-consistent;
    /// only this one proves it agrees with what eng/dsv41-prepare.py writes.
    /// Set TS_DSV41_ENGRAM_SIDECAR to the file to enable it.
    /// </summary>
    [Fact]
    public void Load_ReadsTheRealCheckpointSidecar()
    {
        string path = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_SIDECAR");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;   // not staged on this host

        var data = Dsv41EngramData.Load(path, expectedVocab: 129280,
            expectedTokenizerHash: 1610858572546052822UL);

        Assert.Equal(99092u, data.CompressedVocabSize);
        Assert.Equal(4u, data.MaxNgramSize);
        Assert.Equal(8u, data.HeadCount);
        Assert.Equal(256u, data.HeadDim);
        Assert.Equal(24u, data.HashColumns);
        Assert.Equal(new[] { 1, 14 }, data.Layers.Select(l => l.Id));
        foreach (var layer in data.Layers)
        {
            // Both published tables are ~384 million rows.
            Assert.InRange(layer.Rows, 380_000_000L, 390_000_000L);
            Assert.Equal(24, layer.Primes.Length);
            Assert.Equal(layer.Rows, layer.Offsets[^1] + layer.Primes[^1]);
        }

        // Hashing the real map must land every row inside its own bucket.
        int[] history = null; int historyLength = 0;
        int[] rows = data.HashTokens(new[] { 100, 2000, 55, 129279, 0 }, 0, ref history, ref historyLength);
        for (int li = 0; li < data.Layers.Length; li++)
        {
            var layer = data.Layers[li];
            for (int i = 0; i < 5; i++)
            {
                for (int c = 0; c < 24; c++)
                {
                    int row = rows[(li * 5 + i) * 24 + c];
                    Assert.InRange(row, (int)layer.Offsets[c], (int)(layer.Offsets[c] + layer.Primes[c] - 1));
                }
            }
        }
    }

    /// <summary>
    /// The canonical hash oracle. These row ids were produced by a NumPy
    /// reference using the official tokenizer normalization, outside TensorSharp
    /// entirely, and are pinned in the native test at
    /// TensorSharp.GGML.Native/tests/dsv41_engram_test.cpp:92-116. Matching them
    /// is what proves the managed hashing agrees with the checkpoint, including
    /// the image position at index 4 and the lookbacks it blocks.
    /// Set TS_DSV41_ENGRAM_SIDECAR to the real sidecar to enable it.
    /// </summary>
    [Fact]
    public void HashTokens_MatchesTheCanonicalNumPyOracle()
    {
        string path = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_SIDECAR");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var data = Dsv41EngramData.Load(path, 129280, 1610858572546052822UL);
        Assert.Equal(new[] { 2, 8, 14, 20 }, data.KvSourceLayerIds);
        Assert.Equal(new[] { 2, 8, 14, 20, 24, 28, 32, 36 }, data.IndexSourceLayerIds);
        Assert.Equal(20, data.CandidateSourceLayerId);
        Assert.Equal(8u, data.CandidateBlockSize);
        Assert.Equal(2048u, data.CandidateTopkBlocks);

        int[] tokens = { 0, 100, 101, 102, -1, 103, 104 };
        int[][][] expected =
        {
            new[]
            {
                new[] { 5702652, 121476532, 131476717, 380066193 },
                new[] { 9357813, 120674457, 142523656, 371635201 },
                new[] { 14967493, 120087816, 140350287, 379577895 },
                new[] { 8085103, 120276459, 133334380, 369222544 },
                new[] { 4299726, 112312540, 129401291, 370096053 },
                new[] { 5558068, 126975926, 140551575, 383461561 },
                new[] { 13583392, 116008417, 137862774, 377607687 },
            },
            new[]
            {
                new[] { 14361964, 120604547, 132225184, 372375971 },
                new[] { 15410290, 112117398, 140943284, 383683101 },
                new[] { 3011271, 114792526, 137397375, 378010528 },
                new[] { 5098593, 122052292, 143543847, 368813437 },
                new[] { 6788701, 120694389, 131862336, 381853422 },
                new[] { 10655595, 119663028, 134680611, 369662410 },
                new[] { 4014438, 123436036, 130889979, 374480667 },
            },
        };
        int[] columns = { 0, 7, 8, 23 };

        int[] history = null; int historyLength = 0;
        int[] actual = data.HashTokens(tokens, 0, ref history, ref historyLength);
        for (int layer = 0; layer < 2; layer++)
            for (int token = 0; token < tokens.Length; token++)
                for (int c = 0; c < columns.Length; c++)
                    Assert.Equal(expected[layer][token][c],
                        actual[(layer * tokens.Length + token) * 24 + columns[c]]);
    }

    /// <summary>
    /// Chunking a prompt must not change a single row id: history is indexed by
    /// absolute position, so a lookback reaches back into an earlier ubatch. The
    /// C# executor splits every prompt longer than its microbatch, so this is the
    /// property that keeps long prompts correct.
    /// </summary>
    [Fact]
    public void HashTokens_ChunkingAPromptChangesNothing()
    {
        string path = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_SIDECAR");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var data = Dsv41EngramData.Load(path, 129280, 1610858572546052822UL);
        int[] tokens = { 0, 100, 101, 102, -1, 103, 104 };
        int columns = (int)data.HashColumns;

        int[] whole = null; int wholeLen = 0;
        int[] oneShot = data.HashTokens(tokens, 0, ref whole, ref wholeLen);

        foreach (int[] split in new[] { new[] { 1, 6 }, new[] { 3, 4 }, new[] { 1, 1, 1, 1, 1, 1, 1 } })
        {
            int[] history = null; int historyLength = 0;
            int at = 0;
            foreach (int take in split)
            {
                int[] chunk = data.HashTokens(tokens.AsSpan(at, take), at, ref history, ref historyLength);
                for (int li = 0; li < data.Layers.Length; li++)
                    for (int i = 0; i < take; i++)
                        for (int c = 0; c < columns; c++)
                            Assert.Equal(
                                oneShot[(li * tokens.Length + at + i) * columns + c],
                                chunk[(li * take + i) * columns + c]);
                at += take;
            }
        }
    }

    [Fact]
    public void FingerprintToken_FoldsLengthThenBytes()
    {
        // Independently: FNV-1a over the 8 little-endian length bytes, then the
        // UTF-8 bytes, starting from the supplied seed.
        const ulong seed = 1469598103934665603UL;
        ulong expected = seed;
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes("ab");
        for (int i = 0; i < 8; i++)
            expected = (expected ^ (byte)((ulong)utf8.Length >> (8 * i))) * 1099511628211UL;
        foreach (byte b in utf8)
            expected = (expected ^ b) * 1099511628211UL;

        Assert.Equal(expected, Dsv41EngramData.FingerprintToken(seed, "ab"));
    }
}
