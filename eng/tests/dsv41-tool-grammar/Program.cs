// Independent integration probe using the checkpoint's real GGUF vocabulary.
// Input is the first GGUF shard (metadata reads only), or saved header JSON {metadata: {...}}.
// dotnet run --project eng/tests/dsv41-tool-grammar -c Release -p:EngineBin=/absolute/bin -- header.json report.json
using System.Diagnostics;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

if (args.Length != 2)
    throw new ArgumentException("Usage: dsv41-tool-grammar <first-shard.gguf|header.json> <report.json>");
JsonElement meta;
byte[] metadataPayload;
if (Path.GetExtension(args[0]).Equals(".json", StringComparison.OrdinalIgnoreCase))
{
    metadataPayload = File.ReadAllBytes(args[0]);
    using var document = JsonDocument.Parse(metadataPayload);
    meta = document.RootElement.GetProperty("metadata").Clone();
}
else
{
    using var gguf = new GgufFile(args[0]); // Parse tables only; never read tensor payloads.
    metadataPayload = JsonSerializer.SerializeToUtf8Bytes(gguf.Metadata);
    using var document = JsonDocument.Parse(metadataPayload);
    meta = document.RootElement.Clone();
}
T Read<T>(string key) => meta.GetProperty(key).Deserialize<T>()!;
var tokenizer = new BpeTokenizer(Read<string[]>("tokenizer.ggml.tokens"), Read<int[]>("tokenizer.ggml.token_type"),
    Read<string[]>("tokenizer.ggml.merges"), Read<int>("tokenizer.ggml.bos_token_id"),
    new[] { Read<int>("tokenizer.ggml.eos_token_id") }, false, false, Read<string>("tokenizer.ggml.pre"));
const string open = "<｜DSML｜ calls>", close = "</｜DSML｜ calls>";
const string invokeEnd = "</｜DSML｜ invoke>", parameterEnd = "</｜DSML｜ parameter>";
string Invoke(string name, string value) => "<｜DSML｜ invoke name=\"" + name + "\"><｜DSML｜ parameter name=\"city\" string=\"true\">" + value + parameterEnd + invokeEnd;
var tool = new ToolFunction { Name = "weather", ParametersSchemaJson = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"additionalProperties\":false}" };
var tools = new[] { tool };
string nestedText = "</parameter> </｜DSML｜ parameter> <parameter <｜DSML｜ parameter <｜DSML｜ invoke </｜DSML｜ invoke> </｜DSML｜ calls>";
string nestedJson = JsonSerializer.Serialize(new[] { nestedText });
string nestedCall = open + "<｜DSML｜ invoke name=\"weather\"><｜DSML｜ parameter name=\"city\" string=\"false\">" + nestedJson + parameterEnd + invokeEnd + close;
var nestedTool = new ToolFunction { Name = "weather", ParametersSchemaJson = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"city\"]}" };

string openMapJson = JsonSerializer.Serialize(new Dictionary<string, object>
{
    ["</parameter> 東京"] = new object[] { 42, true, new Dictionary<string, string> { ["任意の鍵"] = nestedText } },
    ["emoji"] = "🦊 café",
});
string openMapCall = open + "<｜DSML｜ invoke name=\"weather\"><｜DSML｜ parameter name=\"city\" string=\"false\">" + openMapJson + parameterEnd + invokeEnd + close;
ToolFunction OpenMap(bool? additional) => new() { Name = "weather", ParametersSchemaJson =
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"object\"" +
    (additional.HasValue ? ",\"additionalProperties\":" + additional.Value.ToString().ToLowerInvariant() : "") + "}},\"required\":[\"city\"]}" };

var cases = new[]
{
    ("json-literal-unicode", DeepSeek41ToolChoice.Auto, false, "{\"city\":\"東京\",\"rocket\":\"🚀\",\"note\":\"café 🦊\"}", true, (string?)null),
    ("json-nested-unicode", DeepSeek41ToolChoice.Auto, false, "{\"任意の鍵\":[1,true,{\"text\":\"<b>東京</b>\"}]}", true, (string?)null),
    ("open-map-default", DeepSeek41ToolChoice.Required, false, openMapCall, true, openMapJson),
    ("open-map-explicit-true", DeepSeek41ToolChoice.Required, false, openMapCall, true, openMapJson),
    ("open-map-false-rejects-extra", DeepSeek41ToolChoice.Required, false, openMapCall, false, (string?)null),
    ("nested-json-escaped-delimiters", DeepSeek41ToolChoice.Required, false, nestedCall, true, nestedJson),
    ("nested-json-reject-literal-delimiters", DeepSeek41ToolChoice.Required, false, nestedCall.Replace("\\u003C", "<"), false, (string?)null),
    ("required-ascii", DeepSeek41ToolChoice.Required, false, open + Invoke("weather", "Paris") + close, true, "Paris"),
    ("required-unicode", DeepSeek41ToolChoice.Required, false, open + Invoke("weather", "  東京\n\"café\" 🦊 <b>Paris</b>  ") + close, true, "  東京\n\"café\" 🦊 <b>Paris</b>  "),
    ("auto-think-quoted-marker", DeepSeek41ToolChoice.Auto, true, "A quoted " + open + " is only reasoning.</think>" + open + Invoke("weather", "Paris") + close, true, "Paris"),
    ("auto-plain-answer", DeepSeek41ToolChoice.Auto, false, "Paris is sunny. 東京", true, (string?)null),
    ("required-reject-plain-close", DeepSeek41ToolChoice.Required, false, open + Invoke("weather", "Paris").Replace(invokeEnd, "</invoke>") + close, false, (string?)null),
    ("required-reject-unknown-tool", DeepSeek41ToolChoice.Required, false, open + Invoke("send_email", "Paris") + close, false, (string?)null),
    ("required-reject-nested-close", DeepSeek41ToolChoice.Required, false, open + Invoke("weather", "Paris " + close) + close, false, (string?)null),
};
var reports = new List<object>();
int failures = 0;
foreach (var (name, choice, thinking, text, expected, expectedCity) in cases)
{
    var caseTools = name.StartsWith("open-map", StringComparison.Ordinal)
        ? new[] { OpenMap(name.Contains("explicit-true") ? true : name.Contains("false") ? false : null) }
        : name.StartsWith("nested-json", StringComparison.Ordinal) ? new[] { nestedTool } : tools;
    var plan = DeepSeek41ToolGrammar.Compile(caseTools, choice);
    var watch = Stopwatch.StartNew();
    bool genericJson = name.StartsWith("json-", StringComparison.Ordinal);
    var constraint = genericJson ? new GrammarConstraint(Grammar.JsonObject(), tokenizer) : plan.NewConstraint(tokenizer, thinking);
    double initializeMs = watch.Elapsed.TotalMilliseconds;
    int[] ids = tokenizer.Encode(text, false).ToArray();
    var maskMs = new List<double>();
    using var maskHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    int rejected = -1;
    for (int i = 0; i < ids.Length; i++)
    {
        long started = Stopwatch.GetTimestamp();
        ulong[] mask = constraint.CurrentMask();
        maskMs.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        maskHash.AppendData(MemoryMarshal.AsBytes(mask.AsSpan()));
        int id = ids[i];
        if ((mask[id >> 6] & (1UL << (id & 63))) == 0) { rejected = i; break; }
        constraint.Accept(id);
    }
    bool accepted = rejected < 0 && !constraint.IsDead && constraint.IsComplete;
    bool parsed = false;
    if (accepted)
    {
        var parser = new DeepSeek41OutputParser();
        parser.Init(thinking, caseTools.ToList());
        var calls = new List<ToolCall>();
        var decoder = new UTF8Encoding(false, true).GetDecoder();
        var reconstructed = new StringBuilder();
        var chars = new char[text.Length + 4];
        foreach (int id in ids)
        {
            var bytes = new List<byte>(); tokenizer.AppendTokenBytes(id, bytes);
            int count = decoder.GetChars(bytes.ToArray(), chars, false);
            string chunk = new(chars, 0, count);
            reconstructed.Append(chunk);
            var output = parser.Add(chunk, false);
            if (output.ToolCalls != null) calls.AddRange(output.ToolCalls);
        }
        int tail = decoder.GetChars(Array.Empty<byte>(), chars, true);
        reconstructed.Append(chars, 0, tail);
        var finished = parser.Add(new string(chars, 0, tail), true);
        if (finished.ToolCalls != null) calls.AddRange(finished.ToolCalls);
        parsed = reconstructed.ToString() == text && (expectedCity == null ? calls.Count == 0
            : calls.Count == 1 && (name.StartsWith("nested-json", StringComparison.Ordinal) || name.StartsWith("open-map", StringComparison.Ordinal) ? JsonSerializer.Serialize(calls[0].Arguments["city"]) : (string)calls[0].Arguments["city"]) == expectedCity);
    }
    if (accepted && genericJson)
    {
        using var jsonOutput = JsonDocument.Parse(text);
        parsed &= jsonOutput.RootElement.ValueKind == JsonValueKind.Object;
    }
    bool pass = accepted == expected && (!accepted || parsed);
    if (!pass) failures++;
    reports.Add(new { name, expected, accepted, parsed, pass, token_ids = ids, first_rejected_token_index = rejected,
        initialize_ms = initializeMs, mask_ms = maskMs, mask_sha256 = Convert.ToHexString(maskHash.GetHashAndReset()).ToLowerInvariant(), elapsed_ms = watch.Elapsed.TotalMilliseconds });
}
var cachedPlan = DeepSeek41ToolGrammar.Compile(tools, DeepSeek41ToolChoice.Auto);
var cached = cachedPlan.NewConstraint(tokenizer, true);
_ = cached.CurrentMask();
long begin = Stopwatch.GetTimestamp();
for (int i = 0; i < 10000; i++) _ = cached.CurrentMask();
double cachedUs = Stopwatch.GetElapsedTime(begin).TotalMicroseconds / 10000;
var provenance = new
{
    metadata_path = Path.GetFullPath(args[0]), metadata_sha256 = Convert.ToHexString(SHA256.HashData(metadataPayload)).ToLowerInvariant(),
    metadata_hash_scope = "Saved header JSON bytes, or serialized GGUF metadata dictionary; no tensor payload",
    runtime_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(TokenSampler).Assembly.Location))).ToLowerInvariant(),
    platform = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
    dsml_token_id = tokenizer.LookupToken("｜DSML｜"), reasoning_end_token_id = tokenizer.LookupToken("</think>"),
    vocab_size = tokenizer.VocabSize, cached_mask_microseconds = cachedUs, failures, reports,
    timing_scope = "Local tokenizer/grammar only, no inference; cold timings include JIT and cache construction."
};
File.WriteAllText(args[1], JsonSerializer.Serialize(provenance, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Actual vocabulary grammar probe: {reports.Count - failures}/{reports.Count} passed; cached mask {cachedUs:F3} us.");
return failures == 0 ? 0 : 1;
