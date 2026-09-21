// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Text.Json;
using System.Net;
using System.Net.Sockets;
using System.Text;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

if (args.Length == 0 || args.Contains("--help", StringComparer.Ordinal))
{
    Console.WriteLine("""
        DesktopBrowserProbe --sandbox required|off --output <fresh evidence directory> [--hold]

        Launches a separate headed persistent Chrome test session through the current
        TensorSharp ShellRunner, with identical settings except the sandbox mode.
        Run from the repository root. Output must be under artifacts/ or docs/validation/.
        Requires npx on PATH and installed Chrome. Uses @playwright/cli@0.1.21.

        --hold: wait until <output>/stop exists, or Ctrl+C; close only this test session.
        Otherwise stdin accepts JSON arrays of CLI arguments, e.g. ["tab-list"],
        ["run-code", "async page => ({ title: await page.title() })"], or the line exit.
        Commands and results are saved locally; use only the harmless test page.
        The reported browser launch success does not assert desktop visibility.
        """);
    return 0;
}

string? mode = null;
string? outputArgument = null;
bool hold = false;
try
{
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--sandbox" when i + 1 < args.Length: mode = args[++i]; break;
            case "--output" when i + 1 < args.Length: outputArgument = args[++i]; break;
            case "--hold": hold = true; break;
            default: throw new ArgumentException("Unknown or incomplete option: " + args[i]);
        }
    }
    if (mode is not ("required" or "off") || string.IsNullOrWhiteSpace(outputArgument))
        throw new ArgumentException("Specify --sandbox required|off and --output <fresh evidence directory>.");
    if (!OperatingSystem.IsMacOS())
        throw new PlatformNotSupportedException("This desktop visibility probe currently targets macOS.");
}
catch (Exception ex) when (ex is ArgumentException or PlatformNotSupportedException)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

string repo = Directory.GetCurrentDirectory();
while (!Directory.Exists(Path.Combine(repo, "TensorSharp.AgentHost")))
{
    string? parent = Path.GetDirectoryName(repo);
    if (parent == null)
    {
        Console.Error.WriteLine("Run this probe from the TensorSharp repository.");
        return 2;
    }
    repo = parent;
}
string output = Path.GetFullPath(outputArgument);
bool allowedOutput = new[] { "artifacts", Path.Combine("docs", "validation") }
    .Any(root => output.StartsWith(Path.Combine(repo, root) + Path.DirectorySeparatorChar, StringComparison.Ordinal));
if (!allowedOutput || File.Exists(output) || Directory.Exists(output))
{
    Console.Error.WriteLine("Output must be a new directory under repository artifacts/ or docs/validation/.");
    return 2;
}
Directory.CreateDirectory(output);

string scratch = Path.Combine(output, "scratch");
string session = "visibility-" + mode + "-" + Guid.NewGuid().ToString("N");
var workspaces = new SessionWorkspaceManager(scratch);
SessionWorkspace workspace = workspaces.GetOrCreate(session);
using var runner = new ShellRunner(new CodeExecOptions
{
    Enabled = true,
    Sandbox = mode == "required" ? SkillSandboxMode.Required : SkillSandboxMode.Off,
    AllowNetwork = true,
    AllowInstall = true,
    ScratchDirectory = scratch,
    Timeout = TimeSpan.FromMinutes(2),
    MaxAutoInstalls = 0,
    MaxOutputBytes = 16 * 1024,
});
using var stopping = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; stopping.Cancel(); };
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
string title = "TensorSharp browser visibility test — " + mode;
string configDirectory = Path.Combine(workspace.WorkDirectory, ".playwright");
Directory.CreateDirectory(configDirectory);
File.WriteAllText(Path.Combine(configDirectory, "cli.config.json"),
    "{\"browser\":{\"launchOptions\":{\"chromiumSandbox\":false}}}\n");
string pagePath = Path.Combine(workspace.WorkDirectory, "visibility.html");
File.WriteAllText(pagePath, $$"""
    <!doctype html><html lang="en"><meta charset="utf-8"><title>{{title}}</title>
    <style>body{font:26px system-ui;background:#ecf4ff;color:#102443;margin:64px}h1{font-size:38px}code{font-size:18px}</style>
    <h1>{{title}}</h1><p>This is a temporary TensorSharp desktop visibility test.</p>
    <p>Sandbox mode: <strong>{{mode}}</strong></p><p><code>{{session}}</code></p>
    <p>No account or sign-in is involved.</p></html>
    """);
using HttpListener fixture = StartFixture(out int fixturePort);
byte[] pageBytes = Encoding.UTF8.GetBytes(File.ReadAllText(pagePath));
Task fixtureTask = Task.Run(async () =>
{
    try
    {
        while (!stopping.IsCancellationRequested)
        {
            HttpListenerContext context = await fixture.GetContextAsync().WaitAsync(stopping.Token);
            context.Response.ContentType = "text/html; charset=utf-8";
            context.Response.Headers["Cache-Control"] = "no-store";
            context.Response.ContentLength64 = pageBytes.Length;
            await context.Response.OutputStream.WriteAsync(pageBytes);
            context.Response.Close();
        }
    }
    catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException or OperationCanceledException) { }
});
string fixtureUrl = $"http://127.0.0.1:{fixturePort}/";
string prefix = "DEBUG=pw:browser npx --yes --prefer-offline --package @playwright/cli@0.1.21 playwright-cli --session " + Quote(session);
string stopPath = Path.Combine(output, "stop");
int step = 0;
bool attemptedOpen = false;
var identity = new
{
    mode, session, title, output, workspace.WorkDirectory, fixtureUrl,
    daemonSessionDirectory = Path.Combine(workspace.WorkDirectory, "Library", "Caches", "ms-playwright", "daemon"),
    sandbox = runner.Sandbox?.Name ?? "none",
    stopPath,
    processId = Environment.ProcessId,
    launchIsNotVisibilityProof = true,
};
File.WriteAllText(Path.Combine(output, "identity.json"), JsonSerializer.Serialize(identity, jsonOptions));
Console.WriteLine(JsonSerializer.Serialize(identity, jsonOptions));

CodeExecResult Run(string name, params string[] commandArgs)
{
    string command = prefix + " " + string.Join(" ", commandArgs.Select(Quote));
    DateTime started = DateTime.UtcNow;
    CodeExecResult result = runner.Run(new ShellRequest(command), workspace);
    string evidence = Path.Combine(output, $"{++step:D2}-{name}.json");
    File.WriteAllText(evidence, JsonSerializer.Serialize(new
    {
        commandArgs, startedUtc = started, elapsedSeconds = (DateTime.UtcNow - started).TotalSeconds,
        result.Ok, result.Content,
    }, jsonOptions));
    Console.WriteLine($"{name}: {(result.Ok ? "ok" : "failed")}; evidence: {evidence}");
    Console.WriteLine(result.Content);
    return result;
}

int exitCode = 0;
var cleanupFailures = new List<string>();
void RecordCleanupFailure(string stage, Exception error)
{
    exitCode = 1;
    string detail = stage + ": " + error;
    cleanupFailures.Add(detail);
    Console.Error.WriteLine(detail);
}
try
{
    if (!runner.CanRun)
        throw new InvalidOperationException(runner.UnavailableReason);
    attemptedOpen = true;
    CodeExecResult opened = Run("open", "open", fixtureUrl, "--headed", "--persistent");
    // The CLI can render a command error with exit status zero; demand its explicit launch marker too.
    if (!opened.Ok || !opened.Content.Contains("opened with pid", StringComparison.Ordinal))
        throw new InvalidOperationException("The CLI did not confirm a browser launch; inspect open evidence.");
    Run("tabs", "tab-list");
    Console.WriteLine(hold ? "Waiting for stop file: " + stopPath : "Ready for JSON CLI argument arrays, or exit.");
    if (hold)
    {
        while (!File.Exists(stopPath))
            await Task.Delay(250, stopping.Token);
    }
    else
    {
        while (await Console.In.ReadLineAsync(stopping.Token) is { } line && line != "exit")
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                string[]? commandArgs = JsonSerializer.Deserialize<string[]>(line);
                if (commandArgs is not { Length: > 0 } || commandArgs.Any(x => x is null))
                    throw new ArgumentException("Expected a nonempty JSON array of strings.");
                if (commandArgs[0] is "kill-all" or "close-all" or "attach" or "delete-data"
                    || commandArgs.Any(x => x is "-s" or "--session"
                        || x.StartsWith("-s=", StringComparison.Ordinal) || x.StartsWith("--session=", StringComparison.Ordinal)))
                    throw new ArgumentException("This probe only controls its own session.");
                Run("command", commandArgs);
            }
            catch (Exception ex) when (ex is JsonException or ArgumentException)
            {
                Console.Error.WriteLine(ex.Message);
            }
        }
    }
}
catch (OperationCanceledException) when (stopping.IsCancellationRequested) { }
catch (Exception ex)
{
    Console.Error.WriteLine(ex.Message);
    File.WriteAllText(Path.Combine(output, "failure.txt"), ex.ToString());
    exitCode = 1;
}
finally
{
    try
    {
        try
        {
            if (attemptedOpen && !Run("close", "close").Ok)
                RecordCleanupFailure("close", new InvalidOperationException("The test browser close command failed."));
        }
        catch (Exception ex) { RecordCleanupFailure("close", ex); }
        try
        {
            stopping.Cancel();
            fixture.Stop();
            await fixtureTask;
        }
        catch (Exception ex) { RecordCleanupFailure("fixture shutdown", ex); }
        try
        {
            // Preserve native launch diagnostics before this probe's workspace is released.
            // These roots belong only to the synthetic probe, never a user's browser profile.
            string runtimeLogs = Path.Combine(output, "runtime-logs");
            foreach (string root in new[]
            {
                Path.Combine(workspace.WorkDirectory, ".playwright-cli"),
                Path.Combine(workspace.WorkDirectory, "Library", "Caches", "ms-playwright", "daemon"),
            })
            {
                if (!Directory.Exists(root)) continue;
                foreach (string log in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                    .Where(p => p.EndsWith(".err", StringComparison.Ordinal) || p.EndsWith(".log", StringComparison.Ordinal)))
                {
                    string target = Path.Combine(runtimeLogs, Path.GetRelativePath(workspace.WorkDirectory, log));
                    Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                    File.Copy(log, target, overwrite: true);
                }
            }
        }
        catch (Exception ex) { RecordCleanupFailure("preserve runtime logs", ex); }
    }
    finally
    {
        bool workspaceReleased = false;
        try
        {
            workspaces.Release(session);
            workspaceReleased = true;
        }
        catch (Exception ex) { RecordCleanupFailure("release workspace", ex); }
        try
        {
            File.WriteAllText(Path.Combine(output, "finished.json"), JsonSerializer.Serialize(new
            {
                session, exitCode, finishedUtc = DateTime.UtcNow, workspaceReleased, cleanupFailures,
            }, jsonOptions));
        }
        catch (Exception ex) { RecordCleanupFailure("write completion evidence", ex); }
    }
}
return exitCode;

static string Quote(string value) => "'" + value.Replace("'", "'\"'\"'", StringComparison.Ordinal) + "'";

static HttpListener StartFixture(out int port)
{
    // HttpListener cannot request port zero. Retry the small reserve/rebind race.
    for (int attempt = 0; attempt < 5; attempt++)
    {
        using var reservation = new TcpListener(IPAddress.Loopback, 0);
        reservation.Start();
        port = ((IPEndPoint)reservation.LocalEndpoint).Port;
        reservation.Stop();
        var listener = new HttpListener();
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        try { listener.Start(); return listener; }
        catch (HttpListenerException) { listener.Close(); }
    }
    throw new IOException("Could not bind the local visibility fixture.");
}
