using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp.Runtime;

namespace TensorSharp.Server;

internal sealed record SharedPrefixRenderKey(
    string Architecture,
    bool EnableThinking,
    string ReasoningEffort,
    string Template,
    List<ChatMessage> Messages,
    List<ToolFunction> Tools,
    string[] ParameterSchemas);

[JsonSourceGenerationOptions(GenerationMode = JsonSourceGenerationMode.Metadata)]
[JsonSerializable(typeof(SharedPrefixRenderKey))]
[JsonSerializable(typeof(ToolFunction))]
[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(Dictionary<string, object>))]
[JsonSerializable(typeof(List<object>))]
[JsonSerializable(typeof(object[]))]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(bool))]
[JsonSerializable(typeof(byte))]
[JsonSerializable(typeof(sbyte))]
[JsonSerializable(typeof(short))]
[JsonSerializable(typeof(ushort))]
[JsonSerializable(typeof(int))]
[JsonSerializable(typeof(uint))]
[JsonSerializable(typeof(long))]
[JsonSerializable(typeof(ulong))]
[JsonSerializable(typeof(float))]
[JsonSerializable(typeof(double))]
[JsonSerializable(typeof(decimal))]
internal sealed partial class SharedPrefixRenderJson : JsonSerializerContext;
