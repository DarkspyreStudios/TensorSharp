using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

public class GrammarUtf8ValidationTests
{
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    [Fact]
    public void DecoderAcceptsEveryUnicodeScalarAcrossIndividualBytes()
    {
        Span<byte> bytes = stackalloc byte[4];
        for (int scalar = 0; scalar <= 0x10FFFF; ++scalar)
        {
            if (!Rune.IsValid(scalar)) continue;
            int count = new Rune(scalar).EncodeToUtf8(bytes);
            var partial = PartialUtf8.Empty;
            for (int i = 0; i < count; ++i)
            {
                if (!GrammarMatcher.TryFeedByte(ref partial, bytes[i], out uint decoded, out bool complete) ||
                    complete != (i == count - 1) || (complete && decoded != scalar))
                    Assert.Fail($"Valid U+{scalar:X} rejected or decoded incorrectly at byte {i}.");
            }
            Assert.Equal(0, partial.Remaining);
        }
    }

    [Theory]
    [InlineData("C0AF")]
    [InlineData("C1A1")]
    [InlineData("E080AF")]
    [InlineData("E09FBF")]
    [InlineData("EDA080")]
    [InlineData("EDBFBF")]
    [InlineData("F08080AF")]
    [InlineData("F08FBFBF")]
    [InlineData("F4908080")]
    [InlineData("F5808080")]
    [InlineData("80")]
    [InlineData("FF")]
    public void JsonMasksMalformedScalarInWholeTokenAndSplitByteTokens(string hex)
    {
        byte[] invalid = Convert.FromHexString(hex);
        byte[] prefix = Encoding.UTF8.GetBytes("{\"value\":\"");
        byte[] suffix = Encoding.UTF8.GetBytes("\"}");
        byte[] whole = prefix.Concat(invalid).Concat(suffix).ToArray();
        Assert.Throws<DecoderFallbackException>(() => StrictUtf8.GetString(whole));
        var tokenizer = new ByteTokenizer(new[] { whole, prefix }.Concat(invalid.Select(b => new[] { b })).Append(suffix).ToArray());
        var constraint = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        Assert.False(Allows(constraint, 1));
        constraint.Accept(2);
        bool rejected = false;
        for (int id = 3; id < tokenizer.VocabSize; ++id)
        {
            bool allowed = Allows(constraint, id);
            var logits = new float[tokenizer.VocabSize];
            constraint.ApplyMask(logits, allowEos: false);
            Assert.Equal(allowed, float.IsFinite(logits[id]));
            rejected |= !allowed;
            constraint.Accept(id);
        }
        Assert.True(rejected);
        Assert.True(constraint.IsDead);
        Assert.False(constraint.IsComplete);
    }

    [Theory]
    [InlineData("C280")]
    [InlineData("DFBF")]
    [InlineData("E0A080")]
    [InlineData("ED9FBF")]
    [InlineData("EE8080")]
    [InlineData("EFBFBF")]
    [InlineData("F0908080")]
    [InlineData("F09F9A80")]
    [InlineData("F48FBFBF")]
    public void JsonPermitsEveryByteOfValidBoundaryScalars(string hex)
    {
        byte[] scalar = Convert.FromHexString(hex);
        var pieces = new[] { Encoding.UTF8.GetBytes("{\"value\":\"") }
            .Concat(scalar.Select(b => new[] { b })).Append(Encoding.UTF8.GetBytes("\"}")).ToArray();
        var tokenizer = new ByteTokenizer(pieces);
        var constraint = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        for (int id = 1; id < tokenizer.VocabSize; ++id)
        {
            Assert.True(Allows(constraint, id));
            constraint.Accept(id);
            Assert.False(constraint.IsDead);
        }
        Assert.True(constraint.IsComplete);
        Assert.Contains(StrictUtf8.GetString(scalar), StrictUtf8.GetString(pieces.SelectMany(b => b).ToArray()));
    }

    private static bool Allows(GrammarConstraint constraint, int id) =>
        (constraint.CurrentMask()[id >> 6] & (1UL << (id & 63))) != 0;

    private sealed class ByteTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        private readonly byte[][] _bytes;
        public ByteTokenizer(params byte[][] pieces)
        {
            _bytes = new[] { Array.Empty<byte>() }.Concat(pieces).ToArray();
            Vocab = _bytes.Select(bytes => Encoding.UTF8.GetString(bytes)).ToArray();
        }
        public string[] Vocab { get; }
        public int VocabSize => _bytes.Length;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public IReadOnlyCollection<int> SpecialTokenIds => new[] { 0 };
        public List<int> Encode(string text, bool addSpecial = true) => throw new NotSupportedException();
        public string Decode(List<int> ids) => Encoding.UTF8.GetString(ids.SelectMany(id => _bytes[id]).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(_bytes[tokenId]);
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string tokenStr) => Array.IndexOf(Vocab, tokenStr);
    }
}
