// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public class DeepSeek41ToolHistoryTests
{
    [Fact]
    public void OpenAiParallelToolHistoryRetainsCallsReasoningAndResultsInCallOrder()
    {
        const string json = """
            [
              {"role":"user","content":"Compare weather."},
              {"role":"assistant","content":null,"reasoning_content":"Check both cities.","tool_calls":[
                {"id":"call_first","type":"function","function":{"name":"weather","arguments":"{\"city\":\" Paris \"}"}},
                {"id":"call_second","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Berlin\"}"}}
              ]},
              {"role":"tool","tool_call_id":"call_second","content":"Berlin: rain"},
              {"role":"user","content":"Also compare temperatures."},
              {"role":"tool","tool_call_id":"call_first","content":"Paris: sun"}
            ]
            """;
        List<ChatMessage> messages;
        using (var document = JsonDocument.Parse(json))
            messages = ChatMessageParser.ParseOpenAI(document.RootElement, null);
        // Parsing survives disposal of the request JSON document.
        Assert.Equal("Check both cities.", messages[1].Thinking);
        Assert.Equal("call_first", messages[1].ToolCalls![0].Id);
        Assert.Equal("call_second", messages[2].ToolCallId);
        var tools = ToolFunction.ParseList("""[{"name":"weather","parameters":{"city":{"type":"string"}},"required":["city"]}]""");
        string prompt = ChatTemplate.RenderDeepSeek41(messages, enableThinking: true, tools: tools);
        Assert.Contains("<think>Check both cities.</think>", prompt);
        Assert.Contains("<｜DSML｜ parameter name=\"city\" string=\"true\"> Paris </｜DSML｜ parameter>", prompt);
        Assert.Contains("<｜User｜><tool_result>Paris: sun</tool_result>\n\nAlso compare temperatures.\n\n<tool_result>Berlin: rain</tool_result>", prompt);
        Assert.EndsWith("<｜Assistant｜><think>", prompt);
        Assert.Equal("Berlin: rain", messages[2].Content);
    }

    [Theory]
    [InlineData("42")]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("\"not JSON\"")]
    public void MalformedToolArgumentsAreRejected(string arguments)
    {
        string json = "[{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"run\",\"arguments\":" + arguments + "}}]}]";
        using var document = JsonDocument.Parse(json);
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(document.RootElement, null));
    }
}
