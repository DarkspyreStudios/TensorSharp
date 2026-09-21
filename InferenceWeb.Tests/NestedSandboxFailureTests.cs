// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.IO;
using System.Linq;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

public sealed class NestedSandboxFailureTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-nested-sandbox-" + Guid.NewGuid().ToString("N"));

    public NestedSandboxFailureTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        Directory.Delete(_root, recursive: true);
        GC.SuppressFinalize(this);
    }

    private static ConfinedResult Failure(string stderr = "Sandbox: worker deny(1) forbidden-sandbox-reinit") =>
        new(true, false, 1, string.Empty, stderr, TimeSpan.Zero, "sandbox-exec", null);

    [Fact]
    public void ExplicitNestedSandboxFailurePreservesHostIsolationAndDoesNotInventAnApplicationFlag()
    {
        string? hint = CodeRepairHint.NestedSandboxFailure(Failure(), hostSandboxActive: true);

        Assert.NotNull(hint);
        Assert.Contains("application's documented launch or configuration options", hint, StringComparison.Ordinal);
        Assert.Contains("Keep the host OS sandbox enabled", hint, StringComparison.Ordinal);
        Assert.DoesNotContain("--no-sandbox", hint, StringComparison.Ordinal);
        Assert.DoesNotContain("playwright", hint, StringComparison.OrdinalIgnoreCase);
        Assert.True(hint.Length < 500);
    }

    [Theory]
    [InlineData("Operation not permitted")]
    [InlineData("Permission denied")]
    [InlineData("sandbox initialization failed")]
    [InlineData("")]
    public void OrdinaryPermissionFailuresDoNotPrescribeSandboxConfiguration(string stderr)
    {
        Assert.Null(CodeRepairHint.NestedSandboxFailure(Failure(stderr), hostSandboxActive: true));
    }

    [Fact]
    public void AbsentConfinementSuccessAndTimeoutDoNotGetRecoveryAdvice()
    {
        ConfinedResult failed = Failure();
        Assert.Null(CodeRepairHint.NestedSandboxFailure(failed, hostSandboxActive: false));
        Assert.Null(CodeRepairHint.NestedSandboxFailure(failed with { SandboxName = "none" }, hostSandboxActive: true));
        Assert.Null(CodeRepairHint.NestedSandboxFailure(failed with { ExitCode = 0 }, hostSandboxActive: true));
        Assert.Null(CodeRepairHint.NestedSandboxFailure(failed with { TimedOut = true }, hostSandboxActive: true));
        Assert.Null(CodeRepairHint.NestedSandboxFailure(failed with { Started = false }, hostSandboxActive: true));
    }

    [Fact]
    public void BoundedDiagnosticInspectionStillRecognizesTheTailOfALongLog()
    {
        Assert.NotNull(CodeRepairHint.NestedSandboxFailure(
            Failure(new string('x', 1024 * 1024) + "\nforbidden-sandbox-reinit"), hostSandboxActive: true));
    }

    [Fact]
    public void ShellAndSkillResultsCarryTheSameRecoveryAdvice()
    {
        var backend = new FakeShellBackend
        {
            Sandbox = new InProcessSandbox(new SkillSandboxCapabilities(true, true, true, true)),
            Answer = _ => Failure(),
        };
        SessionWorkspace workspace = new SessionWorkspaceManager(Path.Combine(_root, "sessions")).GetOrCreate("test");
        var shell = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Required,
            ScratchDirectory = _root,
        }, backend: backend);

        CodeExecResult shellResult = shell.Run(new ShellRequest("application --open"), workspace);

        string skillDirectory = Path.Combine(_root, "automation");
        Directory.CreateDirectory(skillDirectory);
        File.WriteAllText(Path.Combine(skillDirectory, "SKILL.md"), "---\nname: automation\ndescription: Test application.\n---\nUse run.js.\n");
        File.WriteAllText(Path.Combine(skillDirectory, "run.js"), "process.exit(1);\n");
        Skill skill = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skillDirectory } }).Skills.Single();
        var scripts = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Backend = backend,
            Workspace = workspace,
            Sandbox = SkillSandboxMode.Required,
        });

        SkillToolResult scriptResult = scripts.Run(skill, "run.js", Array.Empty<string>());

        string hint = CodeRepairHint.NestedSandboxFailure(Failure(), hostSandboxActive: true)!;
        Assert.False(shellResult.Ok);
        Assert.Contains(hint, shellResult.Content, StringComparison.Ordinal);
        Assert.False(scriptResult.Ok);
        Assert.Contains(hint, scriptResult.Content, StringComparison.Ordinal);
    }
}
