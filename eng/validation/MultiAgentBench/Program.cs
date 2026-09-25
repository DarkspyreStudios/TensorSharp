using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

var settings = Settings.Parse(args);
if (settings.Help)
{
    Console.WriteLine("MultiAgentBench [--iterations 8] [--warmup 1] [--work-ms 80] [--distractor-lines 0] [--out artifacts/multi-agent/benchmark.json] [--endpoint http://localhost:8000/v1/chat/completions --model MODEL --tensorsharp]");
    return;
}
string fixtureRoot = Path.Combine(AppContext.BaseDirectory, "Fixtures");
if (settings.DistractorLines > 0)
{
    string generatedRoot = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(settings.Output))!, "fixtures", Guid.NewGuid().ToString("N"));
    string target = Path.Combine(generatedRoot, "benchmark");
    Directory.CreateDirectory(target);
    foreach (string source in Directory.GetFiles(Path.Combine(fixtureRoot, "benchmark"), "*.md"))
    {
        string text = File.ReadAllText(source);
        if (Path.GetFileName(source) != "SKILL.md")
            text = "# Historical audit records (superseded; do not report as current facts)\n\n"
                + string.Join('\n', Enumerable.Range(1, settings.DistractorLines).Select(index =>
                    $"Archived review {index:D3}: migration considered {index + 10} minute leases, {index % 11 + 1} retries, and retention of {index + 24} hours. This proposal was superseded and is not the active service configuration."))
                + "\n\n# Current service evidence\n\n" + text;
        File.WriteAllText(Path.Combine(target, Path.GetFileName(source)), text);
    }
    fixtureRoot = generatedRoot;
}
var registry = new SkillRegistry(new SkillRegistryOptions { Roots = [fixtureRoot] });
if (registry.Skills.Count != 1) throw new InvalidOperationException("Benchmark skill fixture was not discovered.");
string[] expected = Directory.GetFiles(Path.Combine(fixtureRoot, "benchmark"), "*.md")
    .SelectMany(path => Facts(File.ReadAllText(path))).Order().ToArray();
long fixtureCharacters = Directory.GetFiles(Path.Combine(fixtureRoot, "benchmark"), "*.md")
    .Sum(path => (long)File.ReadAllText(path).Length);
var rows = new List<Measurement>();
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
string? apiKey = Environment.GetEnvironmentVariable("MULTI_AGENT_BENCH_API_KEY");
if (!string.IsNullOrWhiteSpace(apiKey)) http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

Console.WriteLine(settings.Endpoint is null
    ? "Mode: scripted scheduling fixture. Injected analysis delay; no real model or quality-improvement claim."
    : $"Mode: real OpenAI-compatible endpoint; model={settings.Model}. Autonomous delegation is measured, not forced.");
for (int iteration = -settings.Warmup; iteration < settings.Iterations; iteration++)
{
    // Alternate ordering to reduce systematic warm-cache and machine-load bias.
    bool[] order = iteration % 2 == 0 ? [false, true] : [true, false];
    foreach (bool multi in order)
    {
        Measurement row = await Run(multi, iteration);
        Console.WriteLine($"{(iteration < 0 ? "warmup" : "sample")} {iteration + 1,2} {row.Mode,-6}: {row.ElapsedMilliseconds,8:F1} ms; coverage={row.ContentCoverage:P0}; unsupported={row.UnsupportedFacts}; tool_errors={row.ToolErrors}; generations={row.Generations}; children={row.Children}");
        if (iteration >= 0) rows.Add(row);
    }
}
var summaries = rows.GroupBy(r => r.Mode).Select(group => new
{
    mode = group.Key,
    samples = group.Count(),
    wall_ms_p50 = Percentile(group.Select(r => r.ElapsedMilliseconds), .50),
    wall_ms_p95 = Percentile(group.Select(r => r.ElapsedMilliseconds), .95),
    content_coverage_mean = group.Average(r => r.ContentCoverage),
    unsupported_facts = group.Sum(r => r.UnsupportedFacts),
    tool_errors = group.Sum(r => r.ToolErrors),
    generations_mean = group.Average(r => r.Generations),
    child_agents_mean = group.Average(r => r.Children),
    prompt_tokens_mean = group.Average(r => r.PromptTokens),
    completion_tokens_mean = group.Average(r => r.CompletionTokens),
    round_limit_hits = group.Count(r => r.HitRoundLimit),
}).ToArray();
double singleP50 = summaries.Single(s => s.mode == "single").wall_ms_p50;
double multiP50 = summaries.Single(s => s.mode == "multi").wall_ms_p50;
var report = new
{
    schema_version = 1,
    measured_at_utc = DateTimeOffset.UtcNow,
    mode = settings.Endpoint is null ? "scripted_orchestration" : "real_endpoint",
    model = settings.Model,
    server_agents_explicitly_disabled = settings.TensorSharpEndpoint,
    runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
    os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
    processor_count = Environment.ProcessorCount,
    harness_assembly_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Settings).Assembly.Location))),
    agent_host_assembly_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(MultiAgentSession).Assembly.Location))),
    settings.Iterations,
    settings.Warmup,
    settings.DistractorLines,
    simulated_analysis_ms = settings.Endpoint is null ? settings.WorkMilliseconds : (int?)null,
    expected_fact_count = expected.Length,
    fixture_total_characters = fixtureCharacters,
    p50_speedup = singleP50 / multiP50,
    limitations = settings.Endpoint is null
        ? "Scripted generators and injected Task.Delay model independent analysis latency. This validates scheduler overlap, fixture evidence preservation, tool routing and synthesis, not real LLM speed, reasoning quality or device throughput."
        : "Small fixed extraction workload; exact FACT recall and unsupported FACT count are narrow quality proxies. No general quality or performance claim follows. Model/device identity and load must be recorded separately. Concurrent inference may be slower on a saturated single device.",
    summaries,
    rows,
};
string outputPath = Path.GetFullPath(settings.Output);
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
await File.WriteAllTextAsync(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"p50 speedup: {singleP50 / multiP50:F2}x; report: {outputPath}");
if (rows.Any(row => row.ContentCoverage < 1 || row.UnsupportedFacts != 0 || row.ToolErrors != 0 || row.HitRoundLimit))
    Environment.ExitCode = 1;

async Task<Measurement> Run(bool multi, int iteration)
{
    var counters = new Counters();
    var invocations = new ConcurrentQueue<SkillToolInvocation>();
    SkillTurnGenerator Factory(string agentId)
    {
        Interlocked.Increment(ref counters.Children);
        return settings.Endpoint is null ? ScriptedChild(agentId, counters) : Remote(counters);
    }
    var options = new SkillAgentLoopOptions
    {
        MaxRounds = 20,
        ClientTools = [],
        MultiAgent = new() { Enabled = multi, MaxConcurrentAgents = 3, MaxAgents = 8, MaxDepth = 1, MaxRoundsPerAgent = 8 },
        SubagentGeneratorFactory = multi ? Factory : null,
        OnInvocation = invocations.Enqueue,
    };
    List<ChatMessage> messages =
    [
        new() { Role = "system", Content = "Inspect the assigned service documents using skills_read. Return their current FACT lines verbatim, one per line, without obsolete values or invented facts. Preserve evidence from any delegated work. Use read-only tools." },
        new() { Role = "user", Content = "Read skill benchmark and its auth.md, billing.md and inventory.md files. Produce the consolidated current FACT lines for all three services. The service reviews are independent; choose your approach." },
    ];
    using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(10));
    Stopwatch stopwatch = Stopwatch.StartNew();
    SkillLoopResult result = await SkillAgentLoop.RunAsync(messages, SkillTools.BuiltIn(), new(registry.Skills),
        settings.Endpoint is null ? ScriptedParent(multi, counters) : Remote(counters), options, timeout.Token);
    stopwatch.Stop();
    string answer = result.Output.Parsed.Content ?? "";
    string[] actual = Facts(answer).Distinct().ToArray();
    return new(iteration + 1, multi ? "multi" : "single", stopwatch.Elapsed.TotalMilliseconds,
        (double)expected.Count(actual.Contains) / expected.Length, actual.Count(fact => !expected.Contains(fact)),
        invocations.Count(call => !call.Ok), counters.Generations, counters.Children,
        counters.PromptTokens, counters.CompletionTokens, result.HitRoundLimit, answer);
}

SkillTurnGenerator ScriptedChild(string agentId, Counters counters)
{
    bool read = false;
    return async (messages, _, ct) =>
    {
        Interlocked.Increment(ref counters.Generations);
        if (!read)
        {
            read = true;
            string service = new[] { "auth", "billing", "inventory" }.Single(name => agentId.EndsWith(name, StringComparison.Ordinal));
            return CallOutput(Tool("skills_read", ("skill", "benchmark"), ("path", service + ".md")));
        }
        await Task.Delay(settings.WorkMilliseconds, ct);
        return TextOutput(string.Join("\n", messages.Where(m => m.Role == "tool").SelectMany(m => Facts(m.Content ?? ""))));
    };
}

SkillTurnGenerator ScriptedParent(bool multi, Counters counters)
{
    string[] services = ["auth", "billing", "inventory"];
    int round = 0;
    var evidence = new HashSet<string>();
    var consumedResults = new HashSet<string>();
    return async (messages, _, ct) =>
    {
        Interlocked.Increment(ref counters.Generations);
        round++;
        foreach (ChatMessage message in messages.Where(m => m.Role is "tool" or "user"))
        {
            if (!consumedResults.Add(message.Content ?? "")) continue;
            string[] found = Facts(message.Content ?? "").ToArray();
            if (!multi && found.Length > 0) await Task.Delay(settings.WorkMilliseconds, ct);
            foreach (string fact in found) evidence.Add(fact);
            // wait_agent and automatically collected child results are JSON envelopes.
            if (multi) ExtractJsonFacts(message.Content ?? "", evidence);
        }
        if (multi && round == 1)
            return CallOutput(services.Select(service => Tool("spawn_agent", ("task_name", service),
                ("agent_type", "explorer"), ("task", $"Read benchmark/{service}.md using skills_read and return all its current FACT lines verbatim."))).ToArray());
        if (multi && round == 2) return CallOutput(Tool("wait_agent", ("timeout_ms", 10000)));
        if (!multi && round <= services.Length)
            return CallOutput(Tool("skills_read", ("skill", "benchmark"), ("path", services[round - 1] + ".md")));
        return TextOutput(string.Join("\n", evidence.Order()));
    };
}

SkillTurnGenerator Remote(Counters counters) => async (messages, tools, ct) =>
{
    Interlocked.Increment(ref counters.Generations);
    var requestMessages = messages.Select(message =>
    {
        var item = new Dictionary<string, object?> { ["role"] = message.Role, ["content"] = message.Content ?? "" };
        if (message.ToolCallId is not null) item["tool_call_id"] = message.ToolCallId;
        if (message.ToolCalls is { Count: > 0 }) item["tool_calls"] = message.ToolCalls.Select(call => new
        {
            id = call.Id,
            type = "function",
            function = new { name = call.Name, arguments = JsonSerializer.Serialize(call.Arguments) },
        }).ToArray();
        return item;
    }).ToArray();
    var body = new Dictionary<string, object?>
    {
        ["model"] = settings.Model, ["messages"] = requestMessages, ["stream"] = false,
        ["temperature"] = 0, ["max_tokens"] = 4096,
    };
    // This harness owns both arms' tools. Prevent TensorSharp's server-owned
    // delegation from silently turning the baseline into another agent tree.
    if (settings.TensorSharpEndpoint) body["multi_agent"] = false;
    if (tools is { Count: > 0 }) body["tools"] = tools.Select(tool => new
    {
        type = "function",
        function = new
        {
            name = tool.Name, description = tool.Description,
            parameters = new
            {
                type = "object",
                properties = tool.Parameters.ToDictionary(pair => pair.Key, pair => ParameterSchema(pair.Value)),
                required = tool.Required,
            },
        },
    }).ToArray();
    using HttpResponseMessage response = await http.PostAsJsonAsync(settings.Endpoint, body, ct);
    string json = await response.Content.ReadAsStringAsync(ct);
    if (!response.IsSuccessStatusCode) throw new InvalidOperationException($"Endpoint returned HTTP {(int)response.StatusCode}: {json[..Math.Min(json.Length, 1000)]}");
    using JsonDocument doc = JsonDocument.Parse(json);
    if (doc.RootElement.TryGetProperty("usage", out JsonElement usage))
    {
        if (usage.TryGetProperty("prompt_tokens", out JsonElement promptTokens) && promptTokens.TryGetInt64(out long promptCount))
            Interlocked.Add(ref counters.PromptTokens, promptCount);
        if (usage.TryGetProperty("completion_tokens", out JsonElement completionTokens) && completionTokens.TryGetInt64(out long completionCount))
            Interlocked.Add(ref counters.CompletionTokens, completionCount);
    }
    JsonElement answer = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
    var parsed = new ParsedOutput { Content = answer.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.String ? content.GetString() ?? "" : "" };
    if (answer.TryGetProperty("tool_calls", out JsonElement calls))
        parsed.ToolCalls = calls.EnumerateArray().Select(call => new ToolCall
        {
            Id = call.GetProperty("id").GetString(), Name = call.GetProperty("function").GetProperty("name").GetString() ?? "",
            Arguments = JsonSerializer.Deserialize<Dictionary<string, object?>>(call.GetProperty("function").GetProperty("arguments").GetString()!) ?? [],
        }).ToList();
    return new(parsed)
    {
        FinishReason = doc.RootElement.GetProperty("choices")[0].TryGetProperty("finish_reason", out JsonElement finish)
            && finish.ValueKind == JsonValueKind.String ? finish.GetString() : null,
    };
};

static IEnumerable<string> Facts(string text) => Regex.Matches(text, @"(?m)^FACT: [^\r\n]+").Select(match => match.Value.Trim());
static void ExtractJsonFacts(string text, HashSet<string> target)
{
    int start = text.IndexOf('{');
    if (start < 0) return;
    try
    {
        using JsonDocument document = JsonDocument.Parse(text[start..]);
        Visit(document.RootElement);
        void Visit(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object)
                foreach (JsonProperty property in element.EnumerateObject()) Visit(property.Value);
            else if (element.ValueKind == JsonValueKind.Array)
                foreach (JsonElement item in element.EnumerateArray()) Visit(item);
            else if (element.ValueKind == JsonValueKind.String)
                foreach (string fact in Facts(element.GetString()!)) target.Add(fact);
        }
    }
    catch (JsonException) { /* Ordinary user and tool text need not be JSON. */ }
}
static ToolCall Tool(string name, params (string Key, object Value)[] arguments) => new()
{
    Id = Guid.NewGuid().ToString("N"), Name = name, Arguments = arguments.ToDictionary(pair => pair.Key, pair => (object?)pair.Value),
};
static SkillTurnOutput CallOutput(params ToolCall[] calls) => new(new ParsedOutput { ToolCalls = calls.ToList() });
static SkillTurnOutput TextOutput(string text) => new(new ParsedOutput { Content = text });
static Dictionary<string, object> ParameterSchema(ToolParameter parameter)
{
    var schema = new Dictionary<string, object> { ["type"] = parameter.Type, ["description"] = parameter.Description };
    if (parameter.Enum.Count > 0) schema["enum"] = parameter.Enum;
    return schema;
}
static double Percentile(IEnumerable<double> values, double p)
{
    double[] sorted = values.Order().ToArray();
    return sorted[Math.Clamp((int)Math.Ceiling(sorted.Length * p) - 1, 0, sorted.Length - 1)];
}
sealed class Counters { public int Generations; public int Children; public long PromptTokens; public long CompletionTokens; }
sealed record Measurement(int Iteration, string Mode, double ElapsedMilliseconds, double ContentCoverage,
    int UnsupportedFacts, int ToolErrors, int Generations, int Children, long PromptTokens, long CompletionTokens, bool HitRoundLimit, string Answer);
sealed record Settings(int Iterations, int Warmup, int WorkMilliseconds, string Output, string? Endpoint, string? Model, bool Help, bool TensorSharpEndpoint, int DistractorLines)
{
    public static Settings Parse(string[] args)
    {
        var values = new Dictionary<string, string>();
        bool tensorSharpEndpoint = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--help" or "-h") return new(1, 0, 0, "", null, null, true, false, 0);
            if (args[i] == "--tensorsharp") { tensorSharpEndpoint = true; continue; }
            if (!new[] { "--iterations", "--warmup", "--work-ms", "--distractor-lines", "--out", "--endpoint", "--model" }.Contains(args[i]) || i + 1 >= args.Length)
                throw new ArgumentException("Unknown or incomplete option: " + args[i]);
            values[args[i]] = args[++i];
        }
        int Number(string name, int fallback, int minimum) => values.TryGetValue(name, out string? value)
            ? int.TryParse(value, out int n) && n >= minimum ? n : throw new ArgumentException(name + " is out of range") : fallback;
        string? endpoint = values.GetValueOrDefault("--endpoint");
        string? model = values.GetValueOrDefault("--model");
        if (endpoint is not null && (string.IsNullOrWhiteSpace(model) || !Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? uri) || uri.Scheme is not ("http" or "https")))
            throw new ArgumentException("Endpoint mode requires an HTTP(S) --endpoint and --model.");
        return new(Number("--iterations", 8, 1), Number("--warmup", 1, 0), Number("--work-ms", 80, 0),
            values.GetValueOrDefault("--out") ?? "artifacts/multi-agent/benchmark.json", endpoint, model, false, tensorSharpEndpoint,
            Number("--distractor-lines", 0, 0));
    }
}
