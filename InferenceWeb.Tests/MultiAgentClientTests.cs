using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class MultiAgentClientTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LocalClientDelegatesThroughHttp_WithIsolatedMessagesAndCombinedUsage(bool serverSkillsDisabled)
    {
        using var handler = new DelegatingEndpoint(serverSkillsDisabled);
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var history = new List<ChatMessage> { new() { Role = "user", Content = "PARENT_PRIVATE_TASK" } };
        SkillsChatResponse result = await client.CompleteAsync(new() { Messages = history, Tools = [] });
        Assert.Equal("combined alpha beta", result.Content);
        Assert.Empty(result.ToolCalls);
        Assert.Equal("stop", result.FinishReason);
        Assert.Equal(50, result.PromptTokens);
        Assert.Equal(10, result.CompletionTokens);
        Assert.Equal(2, handler.Children);
        Assert.Equal(3, result.Rounds);
        Assert.Single(history);
        Assert.All(handler.Payloads, payload =>
        {
            using var doc = JsonDocument.Parse(payload);
            Assert.False(doc.RootElement.GetProperty("multi_agent").GetBoolean());
            Assert.False(doc.RootElement.GetProperty("skills_discovery").GetBoolean());
        });
    }

    [Theory]
    [InlineData(SkillDelivery.Local)]
    [InlineData(SkillDelivery.Server)]
    public async Task RequestOptOutPreservesSingleAgentPath(SkillDelivery delivery)
    {
        using var handler = new DelegatingEndpoint();
        using var http = new HttpClient(handler);
        using var client = Client(http, delivery);
        SkillsChatResponse result = await client.CompleteAsync(new()
        {
            Messages = [new() { Role = "user", Content = "simple" }], MultiAgent = false,
        });
        Assert.Equal("simple answer", result.Content);
        Assert.Equal(0, handler.Children);
        using var doc = JsonDocument.Parse(handler.Payloads.Single());
        Assert.False(doc.RootElement.GetProperty("multi_agent").GetBoolean());
        if (doc.RootElement.TryGetProperty("tools", out var tools))
            Assert.DoesNotContain(tools.EnumerateArray(), tool => tool.GetProperty("function").GetProperty("name").GetString() == "spawn_agent");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("system")]
    [InlineData("developer")]
    public async Task ReusingResponseHistoryDoesNotAccumulateHostPolicies(string? preambleRole)
    {
        using var handler = new DelegatingEndpoint();
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var history = new List<ChatMessage>();
        if (preambleRole != null)
            history.Add(new() { Role = preambleRole, Content = "Caller policy" });
        history.Add(new() { Role = "user", Content = "simple" });

        SkillsChatResponse first = await client.CompleteAsync(new() { Messages = history });
        var secondHistory = first.Messages.ToList();
        secondHistory.Add(new() { Role = "user", Content = "Continue the simple task" });
        SkillsChatResponse second = await client.CompleteAsync(new() { Messages = secondHistory });

        Assert.Equal(history.Count + 3, second.Messages.Count);
        Assert.Equal(2, second.Messages.Count(m => m.Role == "assistant" && m.Content == "simple answer"));
        Assert.All(second.Messages, m => Assert.DoesNotContain("[TensorSharp multi-agent coordination]", m.Content ?? ""));
        if (preambleRole != null)
        {
            Assert.Equal(preambleRole, second.Messages[0].Role);
            Assert.Equal("Caller policy", second.Messages[0].Content);
        }
        Assert.Equal(2, handler.Payloads.Count);
        Assert.All(handler.Payloads, payload =>
        {
            using var doc = JsonDocument.Parse(payload);
            string content = doc.RootElement.GetProperty("messages")[0].GetProperty("content").GetString()!;
            Assert.Equal(1, content.Split("[TensorSharp multi-agent coordination]", StringSplitOptions.None).Length - 1);
        });
    }

    private static SkillsChatClient Client(HttpClient http, SkillDelivery delivery = SkillDelivery.Local) => new(new()
    {
        Endpoint = "http://fixture/v1", DefaultModel = "fixture", Delivery = delivery,
        Registry = new SkillRegistry(new SkillRegistryOptions { Roots = [] }),
    }, http);

    private sealed class DelegatingEndpoint(bool serverSkillsDisabled = false) : HttpMessageHandler
    {
        public ConcurrentQueue<string> Payloads { get; } = new();
        private readonly TaskCompletionSource _both = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _children;
        public int Children => Volatile.Read(ref _children);

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            if (request.Method == HttpMethod.Get)
            {
                if (request.RequestUri!.AbsolutePath.EndsWith("/health", StringComparison.Ordinal))
                    return Response("\"TensorSharp.Server is running\"");
                return serverSkillsDisabled ? new HttpResponseMessage(HttpStatusCode.NotFound) : Response("{}");
            }
            string payload = await request.Content!.ReadAsStringAsync(ct);
            Payloads.Enqueue(payload);
            using var doc = JsonDocument.Parse(payload);
            var messages = doc.RootElement.GetProperty("messages").EnumerateArray().ToArray();
            string user = messages.First(m => m.GetProperty("role").GetString() == "user").GetProperty("content").GetString()!;
            if (user == "simple") return Reply("simple answer");
            if (user.StartsWith("CHILD_", StringComparison.Ordinal))
            {
                Assert.DoesNotContain("PARENT_PRIVATE_TASK", payload);
                if (Interlocked.Increment(ref _children) == 2) _both.TrySetResult();
                await _both.Task.WaitAsync(TimeSpan.FromSeconds(5), ct);
                return Reply(user == "CHILD_alpha" ? "alpha" : "beta");
            }
            int results = messages.Count(m => m.GetProperty("role").GetString() == "tool");
            if (results == 0)
                return Reply("", new[] { Spawn("alpha"), Spawn("beta") });
            if (results == 2)
                return Reply("", new[] { Call("wait_agent", new { timeout_ms = 10000 }) });
            Assert.Contains("alpha", messages[^1].GetProperty("content").GetString());
            Assert.Contains("beta", messages[^1].GetProperty("content").GetString());
            return Reply("combined alpha beta");
        }

        private static object Spawn(string name) => Call("spawn_agent", new { task_name = name, task = "CHILD_" + name });
        private static object Call(string name, object arguments) => new
        {
            id = Guid.NewGuid().ToString("N"), type = "function",
            function = new { name, arguments = JsonSerializer.Serialize(arguments) },
        };
        private static HttpResponseMessage Reply(string content, object[]? tools = null) => Response(JsonSerializer.Serialize(new
        {
            choices = new[] { new { finish_reason = tools == null ? "stop" : "tool_calls",
                message = new { role = "assistant", content, tool_calls = tools ?? [] } } },
            usage = new { prompt_tokens = 10, completion_tokens = 2 },
        }));
        private static HttpResponseMessage Response(string body) => new(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json"),
        };
    }
}
