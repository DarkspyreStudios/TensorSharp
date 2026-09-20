using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>Model-free wire-contract coverage through the request schema parser,
/// Qwen output parser/collector, and the serializers used by the OpenAI adapter.</summary>
public class Qwen4ExpToolResponseTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void GenericTools_ReturnStructuredOpenAiCallsAndToolFinishReason(bool streaming, bool thinking)
    {
        using var request = JsonDocument.Parse("""
            {"tools":[
              {"type":"function","function":{"name":"write_file","parameters":{
                "type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
                "required":["path","content"]}}},
              {"type":"function","function":{"name":"shell","parameters":{
                "type":"object","properties":{"command":{"type":"string"},"timeout_ms":{"type":"integer"}},
                "required":["command"]}}}
            ]}
            """);
        var tools = ToolFunctionParser.ParseOpenAI(request.RootElement);
        const string code = "    print(42)\n";
        string raw = (thinking ? "Write and run the program.</think>" : "")
            + "<tool_call><function=write_file><parameter=path>\n123\n</parameter>"
            + "<parameter=content>\n" + code + "\n</parameter></function></tool_call>"
            + "<tool_call><function=shell><parameter=command>\npython3 123\n</parameter>"
            + "<parameter=timeout_ms>\n60000\n</parameter></function></tool_call>";
        var wireCalls = new List<JsonElement>();
        object finalResponse;
        if (streaming)
        {
            var parser = OutputParserFactory.Create("qwen4exp");
            parser.Init(thinking, tools);
            void Collect(ParsedOutput parsed)
            {
                Assert.True(string.IsNullOrWhiteSpace(parsed.Content));
                if (parsed.ToolCalls == null) return;
                JsonElement chunk = JsonSerializer.SerializeToElement(
                    OpenAIResponseFactory.ToolCallsChunk("test", "qwen3.8-flash-next", parsed.ToolCalls));
                JsonElement choice = chunk.GetProperty("choices")[0];
                Assert.Equal(JsonValueKind.Null, choice.GetProperty("finish_reason").ValueKind);
                wireCalls.AddRange(choice.GetProperty("delta").GetProperty("tool_calls").EnumerateArray());
            }
            foreach (char value in raw) Collect(parser.Add(value.ToString(), false));
            Collect(parser.Add("", true));
            finalResponse = OpenAIResponseFactory.EndChunk("test", "qwen3.8-flash-next",
                FinishReasonMapper.ToOpenAIChat("eos", wireCalls.Count > 0), 10, 20, 0);
        }
        else
        {
            var collector = new ChatStreamCollector();
            collector.Add(ChatStreamUpdate.Text(raw));
            ParsedOutput parsed = collector.Resolve("qwen4exp", thinking, tools);
            Assert.True(string.IsNullOrWhiteSpace(parsed.Content));
            finalResponse = OpenAIResponseFactory.Completion("test", "qwen3.8-flash-next",
                OpenAIResponseFactory.ParsedAssistantMessage(parsed.Content, parsed.Thinking, parsed.ToolCalls),
                FinishReasonMapper.ToOpenAIChat("eos", parsed.ToolCalls?.Count > 0), 10, 20, 0);
            JsonElement response = JsonSerializer.SerializeToElement(finalResponse);
            wireCalls.AddRange(response.GetProperty("choices")[0].GetProperty("message").GetProperty("tool_calls").EnumerateArray());
        }

        Assert.Equal(2, wireCalls.Count);
        Assert.Equal(2, wireCalls.Select(call => call.GetProperty("id").GetString()).Distinct().Count());
        for (int i = 0; i < wireCalls.Count; i++)
        {
            Assert.Equal("function", wireCalls[i].GetProperty("type").GetString());
            Assert.StartsWith("call_", wireCalls[i].GetProperty("id").GetString());
            if (streaming) Assert.Equal(i, wireCalls[i].GetProperty("index").GetInt32());
        }
        JsonElement write = wireCalls[0].GetProperty("function");
        Assert.Equal("write_file", write.GetProperty("name").GetString());
        using var writeArgs = JsonDocument.Parse(write.GetProperty("arguments").GetString()!);
        Assert.Equal("123", writeArgs.RootElement.GetProperty("path").GetString());
        Assert.Equal(code, writeArgs.RootElement.GetProperty("content").GetString());
        JsonElement shell = wireCalls[1].GetProperty("function");
        Assert.Equal("shell", shell.GetProperty("name").GetString());
        using var shellArgs = JsonDocument.Parse(shell.GetProperty("arguments").GetString()!);
        Assert.Equal("python3 123", shellArgs.RootElement.GetProperty("command").GetString());
        Assert.Equal(60000, shellArgs.RootElement.GetProperty("timeout_ms").GetInt32());
        Assert.Equal("tool_calls", JsonSerializer.SerializeToElement(finalResponse)
            .GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }
}
