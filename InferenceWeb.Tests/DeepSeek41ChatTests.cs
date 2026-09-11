// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

public partial class DeepSeek41ChatTests
{
    [Theory]
    [MemberData(nameof(ReferencePrompts))]
    public void TextPromptsMatchReference(string name, bool thinking, string messagesJson, string? toolsJson, string expected)
    {
        var messages = JsonSerializer.Deserialize<List<ChatMessage>>(messagesJson)!;
        var tools = toolsJson == null ? null : ToolFunction.ParseList(toolsJson);
        Assert.Equal(expected, ChatTemplate.RenderDeepSeek41(messages, enableThinking: thinking, tools: tools));
    }

    [Fact]
    public void RenderingDoesNotMutateMessages()
    {
        List<ChatMessage> messages = [new() { Role = "user", Content = "One" }, new() { Role = "tool", Content = "Two" }];
        string before = JsonSerializer.Serialize(messages);
        ChatTemplate.RenderDeepSeek41(messages);
        Assert.Equal(before, JsonSerializer.Serialize(messages));
    }

    [Fact]
    public void RegisteredProtocolUsesV41FramingAndParser()
    {
        var protocol = ChatProtocolRegistry.For("deepseek41");
        Assert.Same(protocol, ChatProtocolRegistry.For("deepseek_v41"));
        Assert.NotSame(protocol, ChatProtocolRegistry.For("deepseek4"));
        Assert.IsType<DeepSeek41OutputParser>(protocol.CreateOutputParser());
        Assert.True(protocol.OutputParserAlwaysRequired);
        Assert.Equal("<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>",
            protocol.Render(new ChatRenderRequest([new() { Role = "user", Content = "Hi" }], true, "deepseek41", null, false)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OmittingGenerationPromptKeepsHistoricalAssistantHeaders(bool thinking)
    {
        List<ChatMessage> messages = [new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Hello" }, new() { Role = "user", Content = "Next" }];
        string rendered = ChatTemplate.RenderDeepSeek41(messages, false, thinking);
        Assert.Contains("<｜Assistant｜></think>Hello", rendered);
        Assert.EndsWith("<｜User｜>Next", rendered);
    }

    [Fact]
    public void ReasoningEffortIsValidatedAndRendered()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ChatTemplate.RenderDeepSeek41([], reasoningEffort: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ChatTemplate.RenderDeepSeek41([], reasoningEffort: 101));
        Assert.StartsWith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 ",
            ChatTemplate.RenderDeepSeek41([new() { Role = "user", Content = "Hi" }], enableThinking: true, reasoningEffort: 75));
    }

    [Fact]
    public void JsonGrammarWaitsForThinkingOnlyWhenEnabled()
    {
        Assert.Null(OutputParserFactory.GrammarActivationTrigger("deepseek41", false));
        Assert.Equal("</think>", OutputParserFactory.GrammarActivationTrigger("deepseek41", true));
        // Existing families retain their unconditional framing trigger.
        Assert.Equal("final<|message|>", OutputParserFactory.GrammarActivationTrigger("gpt-oss", false));
        Assert.Equal("final<|message|>", OutputParserFactory.GrammarActivationTrigger("gpt-oss", true));
    }

    [Fact]
    public void CachedRawReasoningCannotOverrideTheReferenceHistoryPolicy()
    {
        var tokenizer = new CharacterTokenizer();
        List<ChatMessage> messages = [new() { Role = "user", Content = "First" },
            new() { Role = "assistant", Content = "Answer", Thinking = "Old reasoning",
                RawOutputTokens = tokenizer.Encode("Old reasoning</think>Answer") },
            new() { Role = "user", Content = "Next" }];
        string expected = ChatTemplate.RenderDeepSeek41(messages, enableThinking: true);
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "deepseek41", true, enableThinking: true);
        Assert.Equal(expected, tokenizer.Decode(tokens));
        Assert.DoesNotContain("Old reasoning", tokenizer.Decode(tokens));
    }

    [Fact]
    public void TruncatedThinkingStaysInReasoningChannelOnEveryStreamBoundary()
    {
        const string unfinished = "Check the tool result again. No final answer yet.</th";
        for (int split = 0; split <= unfinished.Length; split++)
        {
            var parser = new DeepSeek41OutputParser();
            parser.Init(true, null);
            var first = parser.Add(unfinished.Substring(0, split), false);
            var last = parser.Add(unfinished.Substring(split), true);
            Assert.Equal(unfinished, first.Thinking + last.Thinking);
            Assert.Empty(first.Content + last.Content);
            Assert.Null(first.ToolCalls);
            Assert.Null(last.ToolCalls);
        }
    }

    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    private const string Calls = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"run\">\n" +
        "<｜DSML｜ parameter name=\"code\" string=\"true\">  print(1)\n</｜DSML｜ parameter>\n" +
        "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"count\": 3, \"enabled\": true}</｜DSML｜ parameter>\n" +
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"finish\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    [Fact]
    public void EveryStreamingBoundaryParsesLosslessParallelCalls()
    {
        string input = "checking</think>Before\n" + Calls + "\nAfter";
        for (int split = 0; split <= input.Length; split++)
        {
            var parser = new DeepSeek41OutputParser();
            parser.Init(true, []);
            var first = parser.Add(input[..split], false);
            var last = parser.Add(input[split..], true);
            Assert.Equal("checking", first.Thinking + last.Thinking);
            Assert.Equal("Before\n\nAfter", first.Content + last.Content);
            var calls = (first.ToolCalls ?? []).Concat(last.ToolCalls ?? []).ToList();
            Assert.Equal(2, calls.Count);
            Assert.Equal("run", calls[0].Name);
            Assert.Equal("  print(1)\n", calls[0].Arguments["code"]);
            Assert.Equal(3L, Assert.IsType<Dictionary<string, object>>(calls[0].Arguments["options"])["count"]);
            Assert.Equal(0, calls[0].Index);
            Assert.Equal(1, calls[1].Index);
        }
    }

    [Fact]
    public void CharacterStreamingDoesNotLeakDsml()
    {
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var content = new StringBuilder();
        var calls = new List<ToolCall>();
        foreach (char c in Calls)
        {
            var part = parser.Add(c.ToString(), false);
            content.Append(part.Content);
            calls.AddRange(part.ToolCalls ?? []);
        }
        var final = parser.Add("", true);
        content.Append(final.Content);
        calls.AddRange(final.ToolCalls ?? []);
        Assert.Empty(content.ToString());
        Assert.Equal(2, calls.Count);
    }

    [Theory]
    [InlineData("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"code\" string=\"true\">partial")]
    [InlineData("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"code\" string=\"true\">partial</｜DSML｜ invoke></｜DSML｜ calls>")]
    public void TruncatedCallsAreNotDispatched(string input)
    {
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var result = parser.Add(input, true);
        Assert.Null(result.ToolCalls);
        Assert.Empty(result.Content);
    }

    [Fact]
    public void CompletedInvokeSurvivesMissingOuterClose()
    {
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var result = parser.Add("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">\n\n</｜DSML｜ invoke>", true);
        Assert.Equal("run", Assert.Single(result.ToolCalls!).Name);
    }
}
