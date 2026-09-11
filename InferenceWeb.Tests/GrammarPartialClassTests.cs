using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

public sealed class GrammarPartialClassTests
{
    [Theory]
    [InlineData(@"[^\u0080-\u07FF]", "C2", false)]
    [InlineData(@"[^\u0080-\u07FF]", "DF", false)]
    [InlineData(@"[^\u0080-\u07FF]", "E0", true)]
    [InlineData(@"[^A\u0080-\u07FF]", "C2", false)]
    [InlineData(@"[^\u0080-\u009F\u00A0-\u00BF]", "C2", false)]
    [InlineData(@"[^\u00A0-\u00BF\u0080-\u009F]", "C2", false)]
    [InlineData(@"[^\u0080-\u00B0\u0090-\u00BF]", "C2", false)]
    [InlineData(@"[^\u0080-\u00BE]", "C2", true)]
    [InlineData(@"[^\uD000-\uD7FF]", "ED", false)]
    [InlineData(@"[^\U00100000-\U0010FFFF]", "F4", false)]
    [InlineData(@"[\uD800-\uDFFF]", "ED", false)]
    [InlineData(@"[\U00110000-\U0013FFFF]", "F4", false)]
    [InlineData(@"[\uD7FF]", "ED", true)]
    [InlineData(@"[\U0010FFFF]", "F4", true)]
    public void APartialTokenMustHaveAtLeastOneValidScalarCompletion(string characterClass, string prefix, bool expected)
    {
        var tokenizer = new ByteTokenizer(Convert.FromHexString(prefix));
        var constraint = new GrammarConstraint(Grammar.Parse("root ::= " + characterClass), tokenizer);
        Assert.Equal(expected, Allows(constraint, 1));
        var logits = new float[tokenizer.VocabSize];
        constraint.ApplyMask(logits, false);
        Assert.Equal(expected, float.IsFinite(logits[1]));
    }

    [Fact]
    public void SurrogateHoleDoesNotProvideAnEscapeFromAClassForbiddingAllScalars()
    {
        var tokenizer = new ByteTokenizer(new byte[] { 0xC2 }, new byte[] { 0xE0 }, new byte[] { 0xED },
            new byte[] { 0xF0 }, new byte[] { 0xF4 });
        var constraint = new GrammarConstraint(Grammar.Parse(@"root ::= [^\u0000-\uD7FF\uE000-\U0010FFFF]"), tokenizer);
        for (int id = 1; id < tokenizer.VocabSize; ++id) Assert.False(Allows(constraint, id));
    }

    [Fact]
    public void AnUncoveredScalarIsReachableThroughTheNextByteMask()
    {
        var tokenizer = new ByteTokenizer(new byte[] { 0xC2 }, new byte[] { 0xBE }, new byte[] { 0xBF });
        var constraint = new GrammarConstraint(Grammar.Parse(@"root ::= [^\u0080-\u00BE]"), tokenizer);
        Assert.True(Allows(constraint, 1));
        constraint.Accept(1);
        Assert.False(Allows(constraint, 2));
        Assert.True(Allows(constraint, 3));
        constraint.Accept(3);
        Assert.True(constraint.IsComplete && !constraint.IsDead);
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
