// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class EmbeddingTokenizerTests
{
    [Fact]
    public void WordPieceNormalizesUnicodeAndKeepsWordBoundaries()
    {
        using var fixture = Vocabulary("bert", ["[UNK]", "[CLS]", "[SEP]", "▁cafe", "▁!", "▁你", "▁好", "▁play", "ing"]);
        var tokenizer = EmbeddingTokenizer.Create(fixture.File);
        Assert.Equal(new[] { 1, 3, 4, 5, 6, 7, 8, 2 }, tokenizer.Encode("CAFÉ!你好 playing"));
        Assert.Equal(new[] { 1, 0, 2 }, tokenizer.Encode("playingunknown"));
        Assert.Equal(new[] { 1, 2 }, tokenizer.Encode(" \t\n\r"));
    }

    [Fact]
    public void WordPieceHonorsCaseSensitiveNormalizerMetadata()
    {
        using var fixture = Vocabulary("bert", ["[UNK]", "[CLS]", "[SEP]", "▁Café", "▁cafe"]);
        fixture.File.Metadata["tokenizer.ggml.normalizer.lowercase"] = false;
        var tokenizer = EmbeddingTokenizer.Create(fixture.File);
        Assert.Equal(new[] { 1, 3, 2 }, tokenizer.Encode("Café"));
        Assert.Equal(new[] { 1, 0, 2 }, tokenizer.Encode("CAFÉ"));
    }

    [Fact]
    public void WordPiecePreservesDecomposedHangulAndSpacingVowels()
    {
        // NFD expands a Korean syllable to all three Jamo; only Mn accent
        // marks are stripped. Hindi's Mc vowel remains part of the word.
        using var fixture = Vocabulary("bert", ["[UNK]", "[CLS]", "[SEP]", "▁ᄒ", "ᅡ", "ᆫ", "▁क", "ि"]);
        Assert.Equal(new[] { 1, 3, 4, 5, 6, 7, 2 }, EmbeddingTokenizer.Create(fixture.File).Encode("한 कि"));
    }

    [Fact]
    public void UnigramFindsGloballyBestPath()
    {
        using var fixture = Vocabulary("t5", ["<unk>", "<s>", "</s>", "a", "ab", "bc", "c"],
            [0, 0, 0, -2, -1, -1, -10]);
        Assert.Equal(new[] { 1, 3, 5, 2 }, EmbeddingTokenizer.Create(fixture.File).Encode("abc"));
    }

    [Fact]
    public void UnigramUnknownFallbackRemainsAvailableBesideLongerPieces()
    {
        using var fixture = Vocabulary("t5", ["<unk>", "<s>", "</s>", "abc", "b"], [0, 0, 0, -1, -1]);
        var tokenizer = EmbeddingTokenizer.Create(fixture.File);
        Assert.Equal(new[] { 1, 0, 4, 2 }, tokenizer.Encode("ab"));
        Assert.Equal(new[] { 1, 0, 4, 2 }, tokenizer.Encode("🙂🐻b"));
    }

    [Fact]
    public void UnigramUsesCompiledCharacterMapBeforeScoring()
    {
        using var fixture = Vocabulary("t5", ["<unk>", "<s>", "</s>", "a", "x"], [0, 0, 0, -1, -1]);
        // A tiny valid XCDA with one rule, 'a' -> 'x'. All integers are LE.
        byte[] charsmap = new byte[4 + 128 * 4 + 2];
        BinaryPrimitives.WriteUInt32LittleEndian(charsmap, 128 * 4);
        BinaryPrimitives.WriteUInt32LittleEndian(charsmap.AsSpan(4), 1U << 10);
        BinaryPrimitives.WriteUInt32LittleEndian(charsmap.AsSpan(4 + 96 * 4), (2U << 10) | 0x100 | 'a');
        BinaryPrimitives.WriteUInt32LittleEndian(charsmap.AsSpan(4 + 98 * 4), 0x80000000);
        charsmap[^2] = (byte)'x';
        fixture.File.Metadata["tokenizer.ggml.precompiled_charsmap"] = charsmap;
        Assert.Equal(new[] { 1, 4, 2 }, EmbeddingTokenizer.Create(fixture.File).Encode("a"));
    }

    [Fact]
    public void MalformedCharacterMapIsRejectedAtLoad()
    {
        using var fixture = Vocabulary("t5", ["<unk>", "<s>", "</s>", "a"]);
        fixture.File.Metadata["tokenizer.ggml.precompiled_charsmap"] = new byte[] { 1, 2, 3 };
        Assert.Throws<InvalidDataException>(() => EmbeddingTokenizer.Create(fixture.File));
    }

    [Fact]
    public void UnigramCollapsesSpacesAndPreservesSpecialTokens()
    {
        using var fixture = Vocabulary("t5", ["<unk>", "<s>", "</s>", "▁a", "▁b"]);
        fixture.File.Metadata["tokenizer.ggml.add_space_prefix"] = true;
        fixture.File.Metadata["tokenizer.ggml.remove_extra_whitespaces"] = true;
        var tokenizer = EmbeddingTokenizer.Create(fixture.File);
        Assert.Equal(new[] { 1, 3, 4, 2 }, tokenizer.Encode("  a   b  "));
        Assert.Equal(new[] { 1, 2 }, tokenizer.Encode("   "));
        Assert.Equal(new[] { 3, 2, 4 }, tokenizer.Encode("a</s>b", addSpecial: false));
        Assert.Equal("a b", tokenizer.Decode([1, 3, 4, 2]));
    }

    [ModelFact("TENSORSHARP_SNOWFLAKE_EMBED_MODEL")]
    public void SnowflakeTokensMatchLlamaCppAndSentencePieceOracles()
    {
        using var file = new GgufFile(Environment.GetEnvironmentVariable("TENSORSHARP_SNOWFLAKE_EMBED_MODEL")!);
        ITokenizer tokenizer = EmbeddingTokenizer.Create(file);
        using var reference = JsonDocument.Parse(System.IO.File.ReadAllText(FixtureFile("snowflake-tokenization.json")));
        foreach (var item in reference.RootElement.EnumerateArray())
            AssertTokens(tokenizer, item.GetProperty("text").GetString()!, item.GetProperty("tokens"));
        AssertHuggingFace(tokenizer, "snowflake");
    }

    [ModelFact("TENSORSHARP_MINILM_EMBED_MODEL")]
    public void MiniLmTokensMatchWordPieceOracles()
    {
        using var file = new GgufFile(Environment.GetEnvironmentVariable("TENSORSHARP_MINILM_EMBED_MODEL")!);
        AssertHuggingFace(EmbeddingTokenizer.Create(file), "minilm");
    }

    private static void AssertHuggingFace(ITokenizer tokenizer, string model)
    {
        using var reference = JsonDocument.Parse(System.IO.File.ReadAllText(FixtureFile("huggingface-tokenization.json")));
        foreach (var item in reference.RootElement.GetProperty(model).GetProperty("cases").EnumerateArray())
            // Exported GGUF semantics are authoritative where the upstream HF
            // tokenizer differs; the fixture retains both IDs and the reason.
            AssertTokens(tokenizer, item.GetProperty("text").GetString()!,
                item.TryGetProperty("gguf_ids", out var ggufIds) ? ggufIds : item.GetProperty("ids"));
    }

    private static void AssertTokens(ITokenizer tokenizer, string text, JsonElement expected)
    {
        int[] ids = expected.EnumerateArray().Select(e => e.GetInt32()).ToArray();
        List<int> actual = tokenizer.Encode(text);
        Assert.True(ids.SequenceEqual(actual), $"Text {JsonSerializer.Serialize(text)}\nExpected: [{string.Join(',', ids)}]\nActual: [{string.Join(',', actual)}]");
    }

    private static string FixtureFile(string name)
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
        {
            if (System.IO.File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                return Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures", "EmbeddingTokenizer", name);
        }
        throw new DirectoryNotFoundException("TensorSharp source tree is required for tokenizer reference fixtures.");
    }

    private static VocabularyFixture Vocabulary(string kind, string[] vocab, float[]? scores = null) => new(kind, vocab, scores);

    private sealed class VocabularyFixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), $"embedding-tokenizer-{Guid.NewGuid():N}.gguf");
        public GgufFile File { get; }
        public VocabularyFixture(string kind, string[] vocab, float[]? scores)
        {
            using (var writer = new BinaryWriter(System.IO.File.Create(_path)))
            {
                writer.Write(Encoding.ASCII.GetBytes("GGUF"));
                writer.Write(3U); writer.Write(0UL); writer.Write(0UL); writer.Write(0UL);
            }
            File = new GgufFile(_path);
            File.Metadata["tokenizer.ggml.model"] = kind;
            File.Metadata["tokenizer.ggml.tokens"] = vocab;
            File.Metadata["tokenizer.ggml.scores"] = scores ?? Enumerable.Repeat(-1f, vocab.Length).ToArray();
            File.Metadata["tokenizer.ggml.token_type"] = new[] { 2, 3, 3 }.Concat(Enumerable.Repeat(1, vocab.Length - 3)).ToArray();
            File.Metadata["tokenizer.ggml.unknown_token_id"] = 0U;
            File.Metadata["tokenizer.ggml.bos_token_id"] = 1U;
            File.Metadata["tokenizer.ggml.eos_token_id"] = 2U;
            File.Metadata["tokenizer.ggml.add_bos_token"] = true;
            File.Metadata["tokenizer.ggml.add_eos_token"] = true;
        }
        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
