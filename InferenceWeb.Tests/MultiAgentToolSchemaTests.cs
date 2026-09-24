// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TensorSharp.AgentHost.Agents;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MultiAgentToolSchemaTests
{
    [Theory]
    [InlineData("system")]
    [InlineData("developer")]
    public void CoordinationPrompt_PreservesStableCacheBoundaryAndHistoryMetadata(string role)
    {
        const string preamble = "Stable governing instructions.";
        var first = new ChatMessage
        {
            Role = role,
            Content = preamble + "  ",
            CacheControl = new CacheControlMarker(),
            ContentCacheBreakpoints = new List<int> { 6 },
        };
        var prior = new ChatMessage
        {
            Role = "assistant", Content = "Earlier answer.",
            RawOutputTokens = new List<int> { 11, 22, 33 },
            RawPromptTrailingWhitespace = "\n",
            RawGenerationSuffix = "<think>\n",
            AttachmentPaths = new List<string> { "/uploads/source.pdf" },
            AttachmentNames = new List<string> { "report.pdf" },
            TextFilePaths = new List<string> { "/uploads/notes.txt" },
            ToolCallId = "call-17",
        };

        List<ChatMessage> injected = MultiAgentPrompt.Apply(
            new List<ChatMessage> { first, prior }, new MultiAgentOptions());

        Assert.Equal(2, injected.Count);
        Assert.Equal(role, injected[0].Role);
        Assert.StartsWith(preamble + "\n\n", injected[0].Content);
        Assert.Contains("[TensorSharp multi-agent coordination]", injected[0].Content);
        Assert.Null(injected[0].CacheControl);
        Assert.Equal(new[] { 6, preamble.Length }, injected[0].ContentCacheBreakpoints);
        Assert.NotNull(first.CacheControl);
        Assert.Equal(preamble + "  ", first.Content);
        Assert.Equal(new[] { 6 }, first.ContentCacheBreakpoints);

        Assert.NotSame(prior, injected[1]);
        Assert.Equal(prior.RawOutputTokens, injected[1].RawOutputTokens);
        Assert.Equal(prior.RawPromptTrailingWhitespace, injected[1].RawPromptTrailingWhitespace);
        Assert.Equal(prior.RawGenerationSuffix, injected[1].RawGenerationSuffix);
        Assert.Equal(prior.ToolCallId, injected[1].ToolCallId);
        Assert.Equal(prior.AttachmentPaths, injected[1].AttachmentPaths);
        Assert.Equal(prior.AttachmentNames, injected[1].AttachmentNames);
        Assert.Equal(prior.TextFilePaths, injected[1].TextFilePaths);
        injected[1].AttachmentPaths!.Clear();
        injected[1].AttachmentNames!.Clear();
        injected[1].RawOutputTokens!.Clear();
        Assert.Single(prior.AttachmentPaths);
        Assert.Single(prior.AttachmentNames);
        Assert.Equal(3, prior.RawOutputTokens.Count);
    }

    [Fact]
    public void JinjaToolSchema_PreservesOptionalWaitAndRequiredSpawnParameters()
    {
        // Inspect the actual model-visible Jinja context, not only the C# schema.
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            "{{ tools | tojson }}", new List<ChatMessage>(),
            addGenerationPrompt: false, tools: MultiAgentTools.Create());
        using JsonDocument document = JsonDocument.Parse(rendered);
        JsonElement Function(string name) => document.RootElement.EnumerateArray()
            .Select(tool => tool.GetProperty("function"))
            .Single(function => function.GetProperty("name").GetString() == name);

        JsonElement wait = Function(MultiAgentTools.Wait).GetProperty("parameters");
        Assert.Equal(0, wait.GetProperty("required").GetArrayLength());
        Assert.Equal("string", wait.GetProperty("properties").GetProperty("agent_id")
            .GetProperty("type").GetString());
        Assert.Equal("integer", wait.GetProperty("properties").GetProperty("timeout_ms")
            .GetProperty("type").GetString());

        JsonElement spawn = Function(MultiAgentTools.Spawn).GetProperty("parameters");
        Assert.Equal(new[] { "task_name", "task" }, spawn.GetProperty("required")
            .EnumerateArray().Select(parameter => parameter.GetString()).ToArray());
        Assert.True(spawn.GetProperty("properties").TryGetProperty("agent_type", out _));
    }
}
