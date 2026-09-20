using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public class Qwen4ExpOutputParserTests
{
    // Actual Qwen3.8 Flash Next response from the three-A40 release campaign.
    private const string WeatherCall = "\n\n<tool_call>\n<function=get_weather>\n" +
        "<parameter=city>\nParis\n</parameter>\n<parameter=units>\ncelsius\n</parameter>\n" +
        "</function>\n</tool_call>";

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(false, 7)]
    [InlineData(false, 1024)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    [InlineData(true, 7)]
    [InlineData(true, 1024)]
    public void ActualModelCall_BecomesStructuredAcrossChunkBoundaries(bool thinking, int chunk)
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(thinking, new List<ToolFunction>());
        string raw = (thinking ? "Choose the weather function.</think>" : "") + WeatherCall;
        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new List<ToolCall>();
        void Collect(ParsedOutput delta)
        {
            content.Append(delta.Content);
            reasoning.Append(delta.Thinking);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
        }
        for (int i = 0; i < raw.Length; i += chunk)
            Collect(parser.Add(raw.Substring(i, Math.Min(chunk, raw.Length - i)), false));
        Collect(parser.Add("", true));

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal("celsius", call.Arguments["units"]);
        Assert.Equal(0, call.Index);
        Assert.True(string.IsNullOrWhiteSpace(content.ToString()));
        Assert.Equal(thinking ? "Choose the weather function." : "", reasoning.ToString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonStreamingCollector_ExtractsTheSameCall(bool thinking)
    {
        var collector = new ChatStreamCollector();
        collector.Add(new ChatStreamUpdate { Piece = (thinking ? "Use weather.</think>" : "") + WeatherCall });
        var output = collector.Resolve("qwen4exp", thinking, new List<ToolFunction>());
        Assert.Equal("get_weather", Assert.Single(output.ToolCalls!).Name);
        Assert.True(string.IsNullOrWhiteSpace(output.Content));
    }

    [Fact]
    public void ReusedParser_ResetDoesNotCarryToolStateIntoTheNextAnswer()
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(false, null);
        Assert.Equal("get_weather", Assert.Single(parser.Add(WeatherCall, true).ToolCalls!).Name);
        parser.Init(false, null);
        var answer = parser.Add("Paris is 18 degrees Celsius.", true);
        Assert.Equal("Paris is 18 degrees Celsius.", answer.Content);
        Assert.Null(answer.ToolCalls);
    }

    [Theory]
    [InlineData("123")]
    [InlineData("true")]
    [InlineData("null")]
    [InlineData("{\"answer\":42}")]
    [InlineData("[1,2,3]")]
    public void DeclaredStringArguments_KeepTheirText(string value)
    {
        var tool = Tool("write_file", ("path", "string"), ("content", "string"));
        var parsed = Parse(XmlCall("write_file", ("path", "123"), ("content", value)), new() { tool });

        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("123", Assert.IsType<string>(call.Arguments["path"]));
        Assert.Equal(value, Assert.IsType<string>(call.Arguments["content"]));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(4096)]
    public void EditArguments_PreserveExactIndentationAndBlankLines(int chunk)
    {
        const string oldText = "    return value + 1\n";
        const string newText = "\n    return value + 2\n\n";
        var tool = Tool("edit_file", ("path", "string"), ("old_string", "string"),
            ("new_string", "string"), ("replace_all", "boolean"));
        string raw = XmlCall("edit_file", ("path", "main.py"), ("old_string", oldText),
            ("new_string", newText), ("replace_all", "false"));

        var call = Assert.Single(Parse(raw, new() { tool }, chunk).ToolCalls!);

        Assert.Equal(oldText, call.Arguments["old_string"]);
        Assert.Equal(newText, call.Arguments["new_string"]);
        Assert.Equal(false, call.Arguments["replace_all"]);
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 7)]
    [InlineData(false, 4096)]
    [InlineData(true, 1)]
    [InlineData(true, 7)]
    [InlineData(true, 4096)]
    public void GeneratedCodeContainingToolMarkup_DoesNotEndTheCall(bool json, int chunk)
    {
        const string code = "def markup():\n    return \"<tool_call></tool_call>\"\n";
        string raw = json
            ? "<tool_call>" + JsonSerializer.Serialize(new { name = "write_file", arguments = new { path = "main.py", content = code } },
                new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }) + "</tool_call>"
            : XmlCall("write_file", ("path", "main.py"), ("content", code));
        var tool = Tool("write_file", ("path", "string"), ("content", "string"));

        var parsed = Parse(raw, new() { tool }, chunk);

        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(code, call.Arguments["content"]);
        Assert.True(string.IsNullOrWhiteSpace(parsed.Content));
    }

    [Theory]
    [InlineData("<function=shell><parameter=command>python3 main.py")]
    [InlineData("<function=shell><parameter=command>python3 main.py</parameter>")]
    [InlineData("<function=shell><parameter=command>python3 main.py</function>")]
    [InlineData("<function=shell><parameter=command>python3 main.py</parameter></tool_call>")]
    public void TruncatedXmlCall_DoesNotBecomeAnExecutableCommand(string body)
    {
        var parsed = Parse("<tool_call>" + body, new() { Tool("shell", ("command", "string")) });
        Assert.Null(parsed.ToolCalls);
        Assert.Contains("python3 main.py", parsed.ToolCallText);
    }

    [Theory]
    [InlineData("", "")]
    [InlineData("\n\n", "")]
    [InlineData("  pass  ", "  pass  ")]
    [InlineData("\r\n    pass\r\n\r\n", "    pass\r\n")]
    public void StringFraming_RemovesOnlyTheTemplateNewlines(string rawValue, string expected)
    {
        string raw = "<tool_call><function=edit_file><parameter=new_string>" + rawValue
            + "</parameter></function></tool_call>";
        var call = Assert.Single(Parse(raw, new() { Tool("edit_file", ("new_string", "string")) }, 1).ToolCalls!);
        Assert.Equal(expected, call.Arguments["new_string"]);
    }

    [Fact]
    public void Reinitializing_ClearsArgumentSchemasAndCallIndices()
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        string raw = XmlCall("write_file", ("content", "123"));
        parser.Init(false, new() { Tool("write_file", ("content", "string")) });
        Assert.Equal("123", Assert.Single(parser.Add(raw, true).ToolCalls!).Arguments["content"]);

        parser.Init(false, null);
        var call = Assert.Single(parser.Add(raw, true).ToolCalls!);
        Assert.Equal(123L, call.Arguments["content"]);
        Assert.Equal(0, call.Index);
    }

    [Theory]
    [InlineData("<function=shell><parameter=command>python3 main.py</parameter></function>")]
    [InlineData("{\"name\":\"shell\",\"arguments\":{\"command\":\"python3 main.py\"}}")]
    public void EndOfStream_RecoversACompleteBodyWithoutTheOuterClosingTag(string body)
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(false, new() { Tool("shell", ("command", "string")) });
        Assert.Null(parser.Add("<tool_call>" + body, false).ToolCalls);

        // Preserve ChatML's EOS recovery: the full invocation is already closed,
        // unlike the half-written arguments rejected by the truncation tests.
        var call = Assert.Single(parser.Add("", true).ToolCalls!);
        Assert.Equal("python3 main.py", call.Arguments["command"]);
        Assert.Null(parser.Add("", true).ToolCalls);
    }

    [Fact]
    public void QwenPromptRendering_RoundTripsExactStringFraming()
    {
        const string source = "\n    print(42)\n\n";
        var tools = new List<ToolFunction> { Tool("write_file", ("content", "string")) };
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Write code." },
            new() { Role = "assistant", ToolCalls = new()
            {
                new() { Name = "write_file", Arguments = new() { ["content"] = source } },
            } },
        };
        string prompt = ChatTemplate.RenderQwen35(history, addGenerationPrompt: false);
        int start = prompt.IndexOf("<tool_call>", StringComparison.Ordinal);
        int end = prompt.IndexOf("</tool_call>", start, StringComparison.Ordinal) + "</tool_call>".Length;

        var call = Assert.Single(Parse(prompt[start..end], tools, 1).ToolCalls!);
        Assert.Equal(source, call.Arguments["content"]);
    }

    [Fact]
    public void AgenticCalls_KeepNamesIndicesAndTypedArguments()
    {
        var tools = new List<ToolFunction>
        {
            Tool("skills_read", ("skill", "string"), ("path", "string")),
            Tool("shell", ("command", "string"), ("timeout_ms", "integer")),
            Tool("configure", ("options", "object"), ("files", "array")),
        };
        string raw = XmlCall("skills_read", ("skill", "code-review"), ("path", "SKILL.md"))
            + XmlCall("shell", ("command", "python3 - <<'PY'\nprint(6 * 7)\nPY"), ("timeout_ms", "60000"))
            + XmlCall("configure", ("options", "{\"enabled\":true}"), ("files", "[\"main.py\"]"));
        var parsed = Parse(raw, tools, 1);

        Assert.Equal(new[] { "skills_read", "shell", "configure" }, parsed.ToolCalls!.Select(c => c.Name));
        Assert.Equal(new[] { 0, 1, 2 }, parsed.ToolCalls.Select(c => c.Index));
        Assert.Equal("SKILL.md", parsed.ToolCalls[0].Arguments["path"]);
        Assert.Equal(60000L, parsed.ToolCalls[1].Arguments["timeout_ms"]);
        Assert.Equal(true, Assert.IsType<Dictionary<string, object?>>(parsed.ToolCalls[2].Arguments["options"])["enabled"]);
        Assert.Equal("main.py", Assert.Single(Assert.IsType<List<object?>>(parsed.ToolCalls[2].Arguments["files"])));
        Assert.True(string.IsNullOrWhiteSpace(parsed.Content));
    }

    private static ToolFunction Tool(string name, params (string Name, string Type)[] parameters)
        => new() { Name = name, Parameters = parameters.ToDictionary(p => p.Name, p => new ToolParameter { Type = p.Type }) };

    private static string XmlCall(string name, params (string Name, string Value)[] parameters)
        => "<tool_call>\n<function=" + name + ">\n"
            + string.Concat(parameters.Select(p => "<parameter=" + p.Name + ">\n" + p.Value + "\n</parameter>\n"))
            + "</function>\n</tool_call>";

    private static ParsedOutput Parse(string raw, List<ToolFunction> tools, int chunk = 7)
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(false, tools);
        var output = new ParsedOutput();
        void Collect(ParsedOutput delta)
        {
            output.Content += delta.Content;
            output.Thinking += delta.Thinking;
            output.ToolCallText += delta.ToolCallText;
            if (delta.ToolCalls != null) (output.ToolCalls ??= new()).AddRange(delta.ToolCalls);
        }
        for (int i = 0; i < raw.Length; i += chunk)
            Collect(parser.Add(raw.Substring(i, Math.Min(chunk, raw.Length - i)), false));
        Collect(parser.Add("", true));
        return output;
    }
}
