using System;
using System.IO;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NativePosixFactAttribute : FactAttribute
{
    public NativePosixFactAttribute()
    {
        if (!ShellProgram.TryResolve(null, out var shell, out _) || shell?.Kind != ShellKind.Posix)
            Skip = "Requires a native POSIX shell.";
    }
}

public sealed class MacNodeFactAttribute : FactAttribute
{
    public MacNodeFactAttribute()
    {
        if (!OperatingSystem.IsMacOS() || !File.Exists("/usr/bin/sandbox-exec"))
            Skip = "Requires macOS Seatbelt.";
        else if (CodeEnvironment.Which("node") == null)
            Skip = "Requires Node.js on PATH.";
    }
}

public sealed class ShellNativeRuntimeTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-native-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _manager;

    public ShellNativeRuntimeTests() => _manager = new SessionWorkspaceManager(_root);

    public void Dispose()
    {
        _manager.Release("runtime");
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private ShellRunner Runner(SkillSandboxMode mode = SkillSandboxMode.Required) => new(new CodeExecOptions
    {
        Enabled = true, Sandbox = mode, Timeout = TimeSpan.FromSeconds(20), ScratchDirectory = _root,
    });

    [NativePosixFact]
    public void WorkspaceInstalledExecutableIsAvailableOnTheNextCall()
    {
        using var runner = Runner(SkillSandboxMode.Off);
        var workspace = _manager.GetOrCreate("runtime");
        var install = runner.Run(new ShellRequest("mkdir -p .local/bin; printf '#!/bin/sh\\nprintf runtime-installed' > .local/bin/runtime-probe; chmod +x .local/bin/runtime-probe"), workspace);
        Assert.True(install.Ok, install.Content);
        var run = runner.Run(new ShellRequest("runtime-probe"), workspace);
        Assert.True(run.Ok, run.Content);
        Assert.Contains("runtime-installed", run.Content, StringComparison.Ordinal);
    }

    [MacNodeFact]
    public void ShortTemporaryAliasKeepsFilesInWorkspaceAndIsRemovedOnRelease()
    {
        var workspace = _manager.GetOrCreate("runtime");
        string alias = workspace.RuntimeTempDirectory;
        Assert.True(alias.Length < 60, alias);
        File.WriteAllText(Path.Combine(alias, "scratch.txt"), "session scratch");
        Assert.Equal("session scratch", File.ReadAllText(Path.Combine(workspace.TempDirectory, "scratch.txt")));
        _manager.Release("runtime");
        Assert.Null(new DirectoryInfo(alias).LinkTarget);
        Assert.False(Directory.Exists(alias));
    }

    [MacNodeFact]
    public void ConfinedNodeCanUseShortUnixSocketsAndStopItsChild()
    {
        using var runner = Runner();
        var workspace = _manager.GetOrCreate("runtime");
        const string script = "const net=require('node:net'),path=require('node:path'),cp=require('node:child_process');"
            + "const socket=path.join(process.env.TMPDIR,'runtime-012345678901234567890123456789.sock');"
            + "const server=net.createServer(c=>c.end('socket-ok'));server.listen(socket,()=>{"
            + "const client=net.connect(socket);client.on('data',d=>console.log(d.toString()));"
            + "client.on('end',()=>server.close());});"
            + "const child=cp.spawn(process.execPath,['-e','setTimeout(()=>{},30000)'],{stdio:'ignore'});"
            + "child.on('spawn',()=>child.kill('SIGTERM'));child.on('exit',()=>console.log('child-stopped'));";
        var result = runner.Run(new ShellRequest("node -e " + ShellCommand.QuotePosix(script)), workspace);
        Assert.True(result.Ok, result.Content);
        Assert.Contains("socket-ok", result.Content, StringComparison.Ordinal);
        Assert.Contains("child-stopped", result.Content, StringComparison.Ordinal);
    }

    [MacNodeFact]
    public void NativeDarwinTemporaryStorageWorksButHostHomeFilesStayDenied()
    {
        using var runner = Runner();
        var workspace = _manager.GetOrCreate("runtime");
        const string script = "const fs=require('node:fs'),cp=require('node:child_process'),path=require('node:path');"
            + "const temp=cp.execFileSync('/usr/bin/getconf',['DARWIN_USER_TEMP_DIR'],{encoding:'utf8'}).trim();"
            + "const dir=fs.mkdtempSync(path.join(temp,'ts-native-test-'));fs.writeFileSync(path.join(dir,'probe'),'ok');"
            + "fs.rmSync(dir,{recursive:true});console.log('native-temp-ok');";
        var result = runner.Run(new ShellRequest("node -e " + ShellCommand.QuotePosix(script)), workspace);
        Assert.True(result.Ok, result.Content);
        Assert.Contains("native-temp-ok", result.Content, StringComparison.Ordinal);

        string secret = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ts-runtime-probe-" + Guid.NewGuid().ToString("N"));
        try
        {
            File.WriteAllText(secret, "must-not-be-readable");
            var denied = runner.Run(new ShellRequest("node -e " + ShellCommand.QuotePosix(
                "console.log(require('node:fs').readFileSync(" + JsonSerializer.Serialize(secret) + ",'utf8'))")), workspace);
            Assert.False(denied.Ok, denied.Content);
            Assert.DoesNotContain("must-not-be-readable", denied.Content, StringComparison.Ordinal);
        }
        finally { File.Delete(secret); }
    }

    [MacNodeFact]
    public void SkillScriptsShareShellRuntimePathAndTemporaryDirectory()
    {
        using var runner = Runner();
        var workspace = _manager.GetOrCreate("runtime");
        var install = runner.Run(new ShellRequest("mkdir -p .local/bin; printf '#!/bin/sh\\nprintf shared-runtime' > .local/bin/runtime-probe; chmod +x .local/bin/runtime-probe"), workspace);
        Assert.True(install.Ok, install.Content);
        string skillRoot = Path.Combine(_root, "skills", "runtime");
        Directory.CreateDirectory(Path.Combine(skillRoot, "scripts"));
        File.WriteAllText(Path.Combine(skillRoot, "SKILL.md"), "---\nname: runtime\ndescription: Runtime integration fixture.\n---\nRun the script.\n");
        File.WriteAllText(Path.Combine(skillRoot, "scripts", "probe.sh"),
            "#!/usr/bin/env bash\nset -euo pipefail\nargs=(node -e)\n\"${args[@]}\" "
            + ShellCommand.QuotePosix("require('node:fs').writeFileSync('script-runtime.json',JSON.stringify({tmp:process.env.TMPDIR,probe:require('node:child_process').execFileSync('runtime-probe').toString()}))") + "\n");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { Path.GetDirectoryName(skillRoot)! } });
        var scriptRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Workspace = workspace, Sandbox = SkillSandboxMode.Required, Timeout = TimeSpan.FromSeconds(20),
        });
        var result = scriptRunner.Run(registry.Skills[0], "scripts/probe.sh", Array.Empty<string>());
        Assert.True(result.Ok, result.Content);
        using var state = JsonDocument.Parse(File.ReadAllText(Path.Combine(workspace.WorkDirectory, "script-runtime.json")));
        Assert.Equal(workspace.RuntimeTempDirectory, state.RootElement.GetProperty("tmp").GetString());
        Assert.Equal("shared-runtime", state.RootElement.GetProperty("probe").GetString());
    }

    [MacNodeFact]
    public void ReplacedTemporaryAliasCannotGrantHomeAccessToShellOrSkill()
    {
        using var runner = Runner();
        var workspace = _manager.GetOrCreate("runtime");
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string secret = Path.Combine(home, ".ts-alias-probe-" + Guid.NewGuid().ToString("N"));
        string marker = "private-value-" + Guid.NewGuid().ToString("N");
        string skillRoot = Path.Combine(_root, "skills", "alias-probe");
        Directory.CreateDirectory(Path.Combine(skillRoot, "scripts"));
        File.WriteAllText(Path.Combine(skillRoot, "SKILL.md"), "---\nname: alias-probe\ndescription: Alias regression fixture.\n---\nRead an argument.\n");
        File.WriteAllText(Path.Combine(skillRoot, "scripts", "probe.sh"), "#!/usr/bin/env bash\ncat \"$1\"\n");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { Path.GetDirectoryName(skillRoot)! } });
        try
        {
            File.WriteAllText(secret, marker);
            string oldAlias = ReplaceAlias();
            var shell = runner.Run(new ShellRequest("cat " + ShellCommand.QuotePosix(secret)), workspace);
            Assert.False(shell.Ok, shell.Content);
            Assert.DoesNotContain(marker, shell.Content, StringComparison.Ordinal);
            Assert.NotEqual(oldAlias, workspace.RuntimeTempDirectory);

            oldAlias = ReplaceAlias();
            var scriptRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Workspace = workspace, Sandbox = SkillSandboxMode.Required,
            });
            var script = scriptRunner.Run(registry.Skills[0], "scripts/probe.sh", new[] { secret });
            Assert.False(script.Ok, script.Content);
            Assert.DoesNotContain(marker, script.Content, StringComparison.Ordinal);
            Assert.NotEqual(oldAlias, workspace.RuntimeTempDirectory);
        }
        finally { File.Delete(secret); }

        string ReplaceAlias()
        {
            string alias = workspace.RuntimeTempDirectory;
            Directory.Delete(alias);
            Directory.CreateSymbolicLink(alias, home);
            return alias;
        }
    }

    [MacNodeFact]
    public void WritableTemporaryRootCannotBeRenamedByConfinedCode()
    {
        using var runner = Runner();
        var workspace = _manager.GetOrCreate("runtime");
        var result = runner.Run(new ShellRequest("node -e " + ShellCommand.QuotePosix(
            "require('node:fs').renameSync(" + JsonSerializer.Serialize(workspace.TempDirectory) + ","
            + JsonSerializer.Serialize(workspace.TempDirectory + "-moved") + ")")), workspace);
        Assert.False(result.Ok, result.Content);
        Assert.True(Directory.Exists(workspace.TempDirectory));
    }

    [NativePosixFact]
    public void SessionInterpreterPrefersLocalRuntimeAndRejectsEscapingLinks()
    {
        var workspace = _manager.GetOrCreate("runtime");
        string bin = Path.Combine(workspace.WorkDirectory, ".local", "bin");
        Directory.CreateDirectory(bin);
        string node = Path.Combine(bin, "node");
        File.WriteAllText(node, "#!/bin/sh\nprintf local-runtime\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(node, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        Assert.True(CodeEnvironment.TryResolveSessionInterpreter(workspace, CodeLanguage.JavaScript, out string? resolved, out _));
        Assert.Equal(node, resolved);
        File.Delete(node);
        File.CreateSymbolicLink(node, "/bin/sh");
        CodeEnvironment.TryResolveSessionInterpreter(workspace, CodeLanguage.JavaScript, out resolved, out _);
        Assert.NotEqual("/bin/sh", resolved);
        Assert.NotEqual(node, resolved);
    }
}
