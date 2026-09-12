// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

public class BpeUnicodePreTokenizerTests
{
    private static BpeTokenizer Tokenizer(string? pre = "joyai-llm") =>
        new([], [], [], -1, [], false, false, pre);

    [Theory]
    [InlineData("joyai-llm")]
    [InlineData("deepseek-v3")]
    [InlineData("hunyuan-dense")]
    public void NumericAndCjkSplitsAreSequential(string pre)
    {
        Assert.Equal(new[] { "!hello", "'s", "/word", "  ", "中文", "abc", "かな", "def" },
            Tokenizer(pre).SplitForBpe("!hello's/word  中文abcかなdef"));
        Assert.Equal(new[] { "123", "٤٥٦", "789" }, Tokenizer(pre).SplitForBpe("123٤٥٦789"));
    }

    [Fact]
    public void SupplementarySymbolsKeepReferenceBoundaries()
    {
        var tokenizer = Tokenizer();
        Assert.Equal(new[] { "😀", "hello" }, tokenizer.SplitForBpe("😀hello"));
        Assert.Equal(new[] { "hello", "😀", "world" }, tokenizer.SplitForBpe("hello😀world"));
        Assert.Equal(new[] { " 😀" }, tokenizer.SplitForBpe(" 😀"));
        Assert.Equal(new[] { "a𐐀𐐨b" }, tokenizer.SplitForBpe("a𐐀𐐨b"));
        Assert.Equal(new[] { "A\U0001d165B" }, tokenizer.SplitForBpe("A\U0001d165B"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("joyai-llm")]
    [InlineData("qwen35")]
    [InlineData("gpt-4o")]
    [InlineData("gemma4")]
    public void PretokenizationNeverDropsCodePoints(string? pre)
    {
        const string text = "🧑🏽‍💻 🚀\n\t🙂\0a𐐀𐐨b\u200B\uE000";
        Assert.Equal(text, string.Concat(Tokenizer(pre).SplitForBpe(text)));
    }

    [Fact]
    public void ByteLevelEncodingRoundTripsSupplementaryUnicode()
    {
        string[] vocab = Enumerable.Range(0, 256).Select(i =>
        {
            char c = (char)i;
            if (c == 0xad) c = '\u0143';
            else if (c <= 0x20) c = (char)(c + 0x100);
            else if (c is >= '\u007f' and <= '\u00a0') c = (char)(c + 0xa2);
            return c.ToString();
        }).ToArray();
        var tokenizer = new BpeTokenizer(vocab, new int[256], [], -1, [], false, false, "joyai-llm");
        const string text = "🧑🏽‍💻 🚀\n\t🙂 a𐐀𐐨b";
        List<int> ids = tokenizer.Encode(text, false);
        Assert.Equal(Encoding.UTF8.GetByteCount(text), ids.Count);
        Assert.Equal(text, tokenizer.Decode(ids));
    }
}
