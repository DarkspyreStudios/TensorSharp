// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

public sealed class MacClangTheoryAttribute : TheoryAttribute
{
    private static readonly Lazy<string?> UnavailableReason = new(() =>
    {
        if (!OperatingSystem.IsMacOS() || !File.Exists("/usr/bin/sandbox-exec"))
            return "Requires macOS Seatbelt and IOSurface.";
        if (!File.Exists("/usr/bin/xcrun"))
            return "Requires Apple's command line tools (xcrun and clang).";
        try
        {
            var start = new ProcessStartInfo("/usr/bin/xcrun")
            {
                UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true,
            };
            start.ArgumentList.Add("--find");
            start.ArgumentList.Add("clang");
            using Process? process = Process.Start(start);
            if (process == null) return "Could not start the clang availability probe.";
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(10_000))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
                return "The clang availability probe timed out.";
            }
            return process.ExitCode == 0 && File.Exists(stdout.GetAwaiter().GetResult().Trim())
                ? null : "Requires an available Apple clang compiler.";
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            return "Could not start the Apple clang compiler probe.";
        }
    });

    public MacClangTheoryAttribute() => Skip = UnavailableReason.Value;
}

public sealed class DesktopSurfaceSandboxTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-desktop-surface-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _workspaces;

    public DesktopSurfaceSandboxTests() => _workspaces = new SessionWorkspaceManager(_root);

    public void Dispose()
    {
        _workspaces.Release("surface");
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [MacClangTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConfinedNativeApplicationCanAllocateItsPresentationSurface(bool allowNetwork)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("surface");
        using (Stream fixture = typeof(DesktopSurfaceSandboxTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.DesktopSurface.create_surface.c")!)
        using (FileStream source = File.Create(Path.Combine(workspace.WorkDirectory, "create_surface.c")))
        {
            Assert.NotNull(fixture);
            fixture.CopyTo(source);
        }

        // Compile only this trusted fixture outside the sandbox; the regression is
        // whether the resulting native application can allocate a surface while confined.
        var compile = new ProcessStartInfo("/usr/bin/xcrun")
        {
            WorkingDirectory = workspace.WorkDirectory,
            UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (string argument in new[]
        {
            "clang", "-Wall", "-Wextra", "-Werror", "create_surface.c", "-o", "create_surface",
            "-framework", "CoreFoundation", "-framework", "IOSurface",
        }) compile.ArgumentList.Add(argument);
        using Process compiler = Process.Start(compile)!;
        var stdout = compiler.StandardOutput.ReadToEndAsync();
        var stderr = compiler.StandardError.ReadToEndAsync();
        bool compiled = compiler.WaitForExit(30_000);
        if (!compiled) { compiler.Kill(entireProcessTree: true); compiler.WaitForExit(); }
        Assert.True(compiled, "The trusted IOSurface fixture compilation timed out.");
        Assert.True(compiler.ExitCode == 0, stdout.GetAwaiter().GetResult() + stderr.GetAwaiter().GetResult());

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true, Sandbox = SkillSandboxMode.Required, AllowNetwork = allowNetwork,
            ScratchDirectory = _root, Timeout = TimeSpan.FromSeconds(20), MaxAutoInstalls = 0,
        });
        Assert.True(runner.CanRun, runner.UnavailableReason);
        CodeExecResult result = runner.Run(new ShellRequest("./create_surface"), workspace);
        Assert.True(result.Ok, result.Content);
        Assert.Contains("sandbox: sandbox-exec", result.Content, StringComparison.Ordinal);
        Assert.Contains("surface-created-and-writable:16x16", result.Content, StringComparison.Ordinal);
    }
}
