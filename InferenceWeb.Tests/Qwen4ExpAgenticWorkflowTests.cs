// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// TensorSharp is licensed under the BSD-3-Clause license in the repository root.

using System.Text;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Script the model's raw Qwen3.8 output, then use the production parser, request
/// planner, tool dispatcher, file tools and shell. No structured calls are fabricated:
/// losing a call in the parser stops the workflow before its result exists on disk.
/// Real-model generation is covered separately by the opt-in validation campaign.
/// </summary>
public sealed class Qwen4ExpAgenticWorkflowTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-qwen38-agent-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _workspaces;

    public Qwen4ExpAgenticWorkflowTests()
    {
        Directory.CreateDirectory(_root);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_root, "sessions"));
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 4096)]
    [InlineData(true, 1)]
    [InlineData(true, 4096)]
    public async Task SkillRead_Create_Read_Patch_AndShellExecute_CompleteTheSameWorkspace(bool thinking, int chunkSize)
    {
        string skills = Path.Combine(_root, "skills");
        string skill = Path.Combine(skills, "arithmetic");
        Directory.CreateDirectory(skill);
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: arithmetic\ndescription: Create and verify an arithmetic program.\n---\n"
            + "Read the source, change 6 * 6 to 6 * 7 with apply_patch, then run it. The result must be 42.\n");

        using var runner = CreateRunner();
        Assert.True(runner.CanRun, runner.UnavailableReason);
        SessionWorkspace workspace = _workspaces.GetOrCreate("workflow");
        var completions = new List<CodeExecResult>();
        var codeRunner = new CodeRunnerAdapter(runner, onCompleted: completions.Add);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skills } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "qwen3.8-flash-next", "--skills-dir", skills }, _root);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, new[] { "arithmetic" }, discovery: false, clientTools: null,
            architecture: "qwen4exp", contextTokens: 32768, options,
            out IReadOnlyList<string> unknown, codeRunner: codeRunner, workspace: workspace);

        Assert.Empty(unknown);
        Assert.NotNull(plan);
        Assert.True(plan.ToolsOffered);
        Assert.True(plan.LoopOptions.ToolResultsAreRendered);
        foreach (string tool in new[] { "skills_read", "write_file", "read_file", "apply_patch", "shell" })
            Assert.Contains(plan.Tools, declaration => declaration.Name == tool);

        bool posix = runner.Shell!.Kind == ShellKind.Posix;
        string path = posix ? "answer.sh" : "answer.ps1";
        string source = posix
            ? "total() {\n    printf '%s\\n' \"$((6 * 6))\"\n}\ntotal\n"
            : "function Get-Total {\n    return (6 * 6)\n}\nGet-Total\n";
        string oldLine = source.Split('\n')[1];
        string newLine = oldLine.Replace("6 * 6", "6 * 7", StringComparison.Ordinal);
        string patch = $"*** Begin Patch\n*** Update File: {path}\n@@\n-{oldLine}\n+{newLine}\n*** End Patch";
        string command = posix
            ? "sh answer.sh | tee result.txt"
            : "./answer.ps1 | Tee-Object -FilePath result.txt";
        string[] turns =
        {
            Call("skills_read", ("skill", "arithmetic"), ("path", "SKILL.md")),
            Call("write_file", ("path", path), ("content", source)),
            Call("read_file", ("path", path)),
            Call("apply_patch", ("patch", patch)),
            Call("shell", ("command", command)),
            "The corrected program ran successfully and printed 42.",
        };
        int turn = 0;
        Task<SkillTurnOutput> Generate(List<ChatMessage> history, List<ToolFunction>? tools, CancellationToken ct)
        {
            Assert.True(turn < turns.Length, "The workflow requested an unexpected additional generation.");
            Assert.Same(plan.Tools, tools);
            if (turn > 0)
            {
                ChatMessage toolResult = history[^1];
                Assert.Equal("tool", toolResult.Role);
                Assert.Equal(Assert.Single(history[^2].ToolCalls!).Id, toolResult.ToolCallId);
            }
            if (turn == 1)
                Assert.Contains("The result must be 42", history[^1].Content, StringComparison.Ordinal);
            if (turn == 2)
                Assert.Equal(source, File.ReadAllText(Path.Combine(workspace.WorkDirectory, path)));
            if (turn == 3)
                Assert.Contains(path, history[^1].Content, StringComparison.Ordinal);
            if (turn == 4)
                Assert.Contains("6 * 7", File.ReadAllText(Path.Combine(workspace.WorkDirectory, path)), StringComparison.Ordinal);
            if (turn == 5)
                Assert.Contains("42", history[^1].Content, StringComparison.Ordinal);

            string raw = (thinking ? "I will perform the next workflow step.</think>\n" : "") + turns[turn++];
            return Task.FromResult(new SkillTurnOutput(Parse(raw, thinking, tools, chunkSize)));
        }

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Use the arithmetic skill to create, correct, and run the program." },
        };
        SkillPrompt.Apply(messages, plan.Prompt);
        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            messages, plan.Tools, plan.ToolContext, Generate,
            plan.LoopOptions.WithClientTools(plan.ClientTools));

        Assert.Equal(turns.Length, result.Rounds);
        Assert.False(result.HitRoundLimit);
        Assert.Empty(result.PendingClientToolCalls);
        Assert.Equal(new[] { "skills_read", "write_file", "read_file", "apply_patch", "shell" },
            result.Invocations.Select(invocation => invocation.Tool));
        Assert.All(result.Invocations, invocation => Assert.True(invocation.Ok));
        Assert.Equal(4, completions.Count);
        Assert.All(completions, completion => Assert.True(completion.Ok, completion.Content));
        Assert.Equal("42", File.ReadAllText(Path.Combine(workspace.WorkDirectory, "result.txt")).Trim());
        Assert.Equal(turns[^1], result.Output.Parsed.Content.Trim());
    }

    [Fact]
    public async Task GenericClientTool_IsReturnedAsAStructuredCallWithoutHostExecution()
    {
        using var runner = CreateRunner();
        Assert.True(runner.CanRun, runner.UnavailableReason);
        SessionWorkspace workspace = _workspaces.GetOrCreate("client-tool");
        var completions = new List<CodeExecResult>();
        var clientTools = new List<ToolFunction>
        {
            new()
            {
                Name = "lookup_build",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["build"] = new() { Type = "string" },
                },
                Required = new List<string> { "build" },
            },
        };
        SkillRequestPlan plan = SkillRequestPlan.Create(
            null!, null, discovery: false, clientTools, architecture: "qwen4exp", contextTokens: 32768,
            options: null!, out _, codeRunner: new CodeRunnerAdapter(runner, onCompleted: completions.Add),
            workspace: workspace);
        Assert.NotNull(plan);
        Assert.True(plan.ToolsOffered);

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            new List<ChatMessage> { new() { Role = "user", Content = "Look up build 123." } },
            plan.Tools, plan.ToolContext,
            (_, tools, _) => Task.FromResult(new SkillTurnOutput(
                Parse(Call("lookup_build", ("build", "123")), false, tools, chunkSize: 1))),
            plan.LoopOptions.WithClientTools(plan.ClientTools));

        ToolCall pending = Assert.Single(result.PendingClientToolCalls);
        Assert.Equal("lookup_build", pending.Name);
        Assert.Equal("123", pending.Arguments["build"]);
        Assert.Empty(result.Invocations);
        Assert.Empty(completions);
        Assert.Equal(1, result.Rounds);
    }

    private ShellRunner CreateRunner() => new(new CodeExecOptions
    {
        Enabled = true,
        Sandbox = SkillSandboxMode.Off,
        ScratchDirectory = _root,
        Timeout = TimeSpan.FromSeconds(15),
    });

    private static string Call(string name, params (string Name, string Value)[] arguments)
    {
        var raw = new StringBuilder($"<tool_call>\n<function={name}>\n");
        foreach (var argument in arguments)
            raw.Append($"<parameter={argument.Name}>\n{argument.Value}\n</parameter>\n");
        return raw.Append("</function>\n</tool_call>").ToString();
    }

    private static ParsedOutput Parse(string raw, bool thinking, List<ToolFunction>? tools, int chunkSize)
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(thinking, tools);
        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new List<ToolCall>();
        void Collect(ParsedOutput delta)
        {
            content.Append(delta.Content);
            reasoning.Append(delta.Thinking);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
        }
        for (int i = 0; i < raw.Length; i += chunkSize)
            Collect(parser.Add(raw.Substring(i, Math.Min(chunkSize, raw.Length - i)), done: false));
        Collect(parser.Add("", done: true));
        return new ParsedOutput { Content = content.ToString(), Thinking = reasoning.ToString(), ToolCalls = calls };
    }

    public void Dispose()
    {
        _workspaces.Release("workflow");
        _workspaces.Release("client-tool");
        Directory.Delete(_root, recursive: true);
    }
}
