// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The Engram GGUF metadata decides which of 384 million rows each token reads. A
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

    /// <summary>Creates embedded metadata with two distinct layer-major hash
    /// layouts. The GGUF parser reads the file; tests may mutate metadata to
    /// check rejection without allocating embedding-table payloads.</summary>
    private GgufFile EmbeddedFixture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"dsv41-engram-{Guid.NewGuid():N}.gguf");
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(0x46554747u); writer.Write(3u);
            writer.Write(0UL); writer.Write(0UL);
        }
        _temp.Add(path);
        var file = GgufFile.OpenWithoutSiblingShards(path);
        var m = file.Metadata;
        m["tokenizer.ggml.tokens"] = Enumerable.Range(0, 8).Select(i => $"token{i}").ToArray();
        m["deepseek41.block_count"] = 6u;
        m["deepseek41.engram.pad_id"] = 3u;
        m["deepseek41.engram.max_ngram_size"] = 3u;
        m["deepseek41.engram.head_count"] = 2u;
        m["deepseek41.engram.key_length"] = 16u;
        m["deepseek41.engram.layer_ids"] = new uint[] { 1, 5 };
        m["deepseek41.engram.multipliers"] = new ulong[] { 7, 9, 11, 9, 11, 13 };
        m["deepseek41.engram.primes"] = new uint[] { 11, 13, 15, 17, 15, 17, 19, 21 };
        m["deepseek41.engram.offsets"] = new ulong[] { 0, 11, 24, 39, 0, 15, 32, 51 };
        // The embedded pad ID is compressed already; TokenMap[3] != 3.
        m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 0, 3, 1, 2, 3 };
        m["deepseek41.attention.compress_ratios"] = new uint[] { 2, 2, 1, 1, 1, 1 };
        void Tensor(string name, params ulong[] shape) =>
            file.Tensors[name] = new GgufTensorInfo { Name = name, Shape = shape, Type = GgmlTensorType.F32 };
        Tensor("token_embd.weight", 16, 8);
        Tensor("blk.1.engram_embd.weight", 16, 56);
        Tensor("blk.5.engram_embd.weight", 16, 72);
        Tensor("blk.0.attn_compressor_kv.weight", 16, 16);
        Tensor("blk.2.attn_compressor_kv.weight", 16, 16);
        Tensor("blk.0.indexer.attn_q_b.weight", 16, 16);
        Tensor("blk.2.indexer.attn_q_b.weight", 16, 16);
        Tensor("blk.4.indexer.attn_q_b.weight", 16, 16);
        return file;
    }

    [Fact]
    public void Load_ReadsEveryFieldOfAWellFormedEmbeddedGguf()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);

        Assert.Equal(8u, data.VocabSize);
        Assert.Equal(4u, data.CompressedVocabSize);
        Assert.Equal(3u, data.PadId);
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
    [InlineData("deepseek41.engram.token_map")]
    [InlineData("deepseek41.engram.multipliers")]
    [InlineData("deepseek41.engram.primes")]
    [InlineData("deepseek41.engram.offsets")]
    public void Load_RequiresEmbeddedMetadata(string key)
    {
        using var file = EmbeddedFixture();
        file.Metadata.Remove(key);
        var error = Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file));
        Assert.Contains(key, error.Message);
    }

    [Theory]
    [InlineData("map-length")]
    [InlineData("map-gap")]
    [InlineData("map-negative")]
    [InlineData("map-overflow")]
    [InlineData("offset-gap")]
    [InlineData("multiplier-even")]
    [InlineData("multiplier-overflow")]
    [InlineData("layer-order")]
    [InlineData("table-rows")]
    [InlineData("table-width")]
    [InlineData("unlisted-table")]
    [InlineData("vocab")]
    [InlineData("pad")]
    [InlineData("candidate")]
    [InlineData("candidate-topk")]
    [InlineData("candidate-topk-overflow")]
    [InlineData("candidate-block-overflow")]
    public void Load_RejectsInvalidEmbeddedGeometry(string fault)
    {
        using var file = EmbeddedFixture();
        var m = file.Metadata;
        switch (fault)
        {
            case "map-length": m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 3 }; break;
            case "map-gap": m["deepseek41.engram.token_map"] = new int[] { 0, 1, 2, 0, 4, 1, 2, 4 }; break;
            case "map-negative": m["deepseek41.engram.token_map"] = new int[] { -1, 1, 2, 0, 3, 1, 2, 3 }; break;
            case "map-overflow": m["deepseek41.engram.token_map"] = new ulong[] { 0, 1, 2, 0, 3, 1, 2, ulong.MaxValue }; break;
            case "offset-gap": ((ulong[])m["deepseek41.engram.offsets"])[6]++; break;
            case "multiplier-even": ((ulong[])m["deepseek41.engram.multipliers"])[0] = 8; break;
            case "multiplier-overflow": ((ulong[])m["deepseek41.engram.multipliers"])[5] = ulong.MaxValue; break;
            case "layer-order": m["deepseek41.engram.layer_ids"] = new uint[] { 5, 1 }; break;
            case "table-rows": file.Tensors["blk.5.engram_embd.weight"].Shape[1]++; break;
            case "unlisted-table": file.Tensors["blk.3.engram_embd.weight"] = file.Tensors["blk.5.engram_embd.weight"]; break;
            case "table-width": file.Tensors["blk.5.engram_embd.weight"].Shape[0]++; break;
            case "vocab": file.Tensors["token_embd.weight"].Shape[1]++; break;
            case "pad": m["deepseek41.engram.pad_id"] = 4u; break;
            case "candidate": m["deepseek41.attention.indexer.candidate_source_layer"] = 1; break;
            case "candidate-topk-overflow": m["deepseek41.attention.indexer.candidate_top_k"] = 2147483648u; break;
            case "candidate-block-overflow": m["deepseek41.attention.indexer.candidate_block_size"] = 2147483648u; break;
            case "candidate-topk": m["deepseek41.attention.indexer.candidate_top_k"] = 0u; break;
        }
        Assert.Throws<InvalidDataException>(() => Dsv41EngramData.Load(file));
    }

    [Fact]
    public void Load_DerivesCacheOwnershipAndAppliesCandidateOverrides()
    {
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        Assert.Equal(new[] { 0, 2 }, data.KvSourceLayerIds);
        Assert.Equal(new[] { 0, 2, 4 }, data.IndexSourceLayerIds);
        Assert.Equal(2, data.CandidateSourceLayerId);
        Assert.Equal(2048u, data.CandidateTopkBlocks);
        Assert.Equal(8u, data.CandidateBlockSize);
        file.Metadata["deepseek41.attention.indexer.candidate_source_layer"] = 4;
        file.Metadata["deepseek41.attention.indexer.candidate_top_k"] = 2u;
        file.Metadata["deepseek41.attention.indexer.candidate_block_size"] = 2u;
        data = Dsv41EngramData.Load(file);
        Assert.Equal(4, data.CandidateSourceLayerId);
        Assert.Equal(2u, data.CandidateTopkBlocks);
        Assert.Equal(2u, data.CandidateBlockSize);
        file.Metadata["deepseek41.attention.indexer.candidate_source_layer"] = -1;
        Assert.Equal(-1, Dsv41EngramData.Load(file).CandidateSourceLayerId);
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
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);

        int[] tokens = { 5, 2, 7, 0 };
        int[] history = null; int historyLength = 0;
        int[] actual = data.HashTokens(tokens, 0, ref history, ref historyLength);

        uint columns = data.HashColumns;
        Assert.Equal(data.Layers.Length * tokens.Length * (int)columns, actual.Length);

        int[] compressed = tokens.Select(t => data.TokenMap[t]).ToArray();
        int padCompressed = (int)data.PadId;

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
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
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
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
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
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
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
        using var file = EmbeddedFixture();
        var data = Dsv41EngramData.Load(file);
        int[] history = null; int historyLength = 0;
        data.HashTokens(new[] { 1, 2 }, 0, ref history, ref historyLength);
        Assert.Throws<InvalidOperationException>(() =>
            data.HashTokens(new[] { 3 }, 5, ref history, ref historyLength));
    }

    /// <summary>Validates the embedded metadata in the published checkpoint.
    /// Unavailable checkpoints are reported as skipped, never passed.</summary>
    [Dsv41RealGgufFact]
    public void Load_ReadsTheRealCheckpointEmbeddedGguf()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);

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
    /// entirely. Matching them
    /// is what proves the managed hashing agrees with the checkpoint, including
    /// the image position at index 4 and the lookbacks it blocks.
    /// Set TS_TEST_DSV41_REAL_GGUF to the real GGUF metadata to enable it.
    /// </summary>
    [Dsv41RealGgufFact]
    public void HashTokens_MatchesTheCanonicalNumPyOracle()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);
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
    [Dsv41RealGgufFact]
    public void HashTokens_ChunkingAPromptChangesNothing()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");

        using var file = new GgufFile(path);
        var data = Dsv41EngramData.Load(file);
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

public sealed class Dsv41RealGgufFactAttribute : FactAttribute
{
    public Dsv41RealGgufFactAttribute()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_DSV41_REAL_GGUF");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            Skip = "Requires TS_TEST_DSV41_REAL_GGUF pointing to the published DeepSeek V4.1 Flash first GGUF shard.";
    }
}
