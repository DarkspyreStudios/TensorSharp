using System.Net;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// The link a model hands you for a file its own code produced has to resolve.
///
/// <para>
/// The runner emits <c>/api/code/artifacts/{runId}/{path}</c>, and the app mapped no
/// such route: TensorSharp.Server had the endpoint, the phone did not. The model
/// correctly said "here is your PDF", the link rendered, and tapping it produced
/// "error: not found" -- for every generated document, PDF and PPTX alike.
/// </para>
/// </summary>
public sealed class CodeArtifactRouteTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ta-artifacts-" + Guid.NewGuid().ToString("n"));
    private readonly LoopbackServer _server;
    private readonly HttpClient _client;
    private readonly CodeArtifactStore _artifacts;

    public CodeArtifactRouteTests()
    {
        Directory.CreateDirectory(_root);
        _artifacts = new CodeArtifactStore(Path.Combine(_root, "artifacts"));
        _server = new LoopbackServer(NullLogger.Instance);
        _server.MapCodeArtifacts(_artifacts);
        _server.Start();
        _client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
    }

    public void Dispose()
    {
        _client.Dispose();
        _server.Dispose();
        try { Directory.Delete(_root, true); } catch { }
    }

    /// <summary>Capture a file the way a finished code run does, and return its URL.</summary>
    private string CaptureOne(string runId, string name, byte[] bytes)
    {
        string work = Path.Combine(_root, "work-" + runId);
        Directory.CreateDirectory(work);
        File.WriteAllBytes(Path.Combine(work, name), bytes);
        IReadOnlyList<CodeArtifact> captured = _artifacts.Capture(
            runId, work, (id, rel, _) => $"/api/code/artifacts/{id}/{rel}", out _);
        return captured.Single(a => a.Path.EndsWith(name, StringComparison.Ordinal)).Pointer;
    }

    [Fact]
    public async Task AGeneratedPdfIsDownloadableFromTheLinkTheModelShows()
    {
        // A real PDF header, so the content type is decided by extension rather than by
        // sniffing bytes a model's program chose.
        byte[] pdf = "%PDF-1.7\n%âãÏÓ\ntrailer\n"u8.ToArray();
        string url = CaptureOne("runpdf", "photo.pdf", pdf);

        HttpResponseMessage response = await _client.GetAsync(url);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(pdf, await response.Content.ReadAsByteArrayAsync());
        Assert.Equal("application/pdf", response.Content.Headers.ContentType?.MediaType);
    }

    [Fact]
    public async Task AGeneratedPptxIsDownloadableToo()
    {
        // The other half of the report: "search the news and make me a deck" produces
        // this, and it was unreachable for exactly the same reason.
        byte[] pptx = new byte[] { 0x50, 0x4B, 0x03, 0x04, 0, 0, 0, 0 };
        string url = CaptureOne("rundeck", "news.pptx", pptx);

        HttpResponseMessage response = await _client.GetAsync(url);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(pptx, await response.Content.ReadAsByteArrayAsync());
    }

    [Fact]
    public async Task TheFileArrivesAsAnAttachmentSoAWebViewSavesRatherThanRenders()
    {
        // Served as an attachment on purpose: this content came out of a program a
        // model wrote, and a WebView must never render it inline.
        string url = CaptureOne("runattach", "report.pdf", "%PDF-1.7\n"u8.ToArray());

        HttpResponseMessage response = await _client.GetAsync(url);

        Assert.Equal("attachment", response.Content.Headers.ContentDisposition?.DispositionType);
        Assert.Contains("report.pdf", response.Content.Headers.ContentDisposition?.FileName ?? string.Empty);
    }

    [Fact]
    public async Task AnUnknownRunIsNotFoundRatherThanAServerError()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/code/artifacts/nope/missing.pdf");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task EscapingTheStoreIsRefused()
    {
        // The path segment comes from a program a model wrote. Confinement is
        // re-checked by the store, not trusted from the route.
        CaptureOne("runesc", "ok.pdf", "%PDF-1.7\n"u8.ToArray());
        HttpResponseMessage response = await _client.GetAsync("/api/code/artifacts/runesc/..%2f..%2fsettings.json");
        Assert.NotEqual(HttpStatusCode.OK, response.StatusCode);
    }

    /// <summary>
    /// The link a real command hands back has to be the link the route answers.
    ///
    /// <para>
    /// It was not, and mapping the route did not fix it: the app never set
    /// <c>ArtifactUriPrefix</c>, so <see cref="ShellRunner"/> fell through to its
    /// no-server branch and handed back the artifact's ABSOLUTE PATH ON DISK. The model
    /// was told "tell the user where they are on disk", the page rendered
    /// <c>/var/mobile/Containers/Data/.../report.pdf</c> as an ordinary link, and tapping
    /// it asked this server for a path it serves nothing at -- <c>{"error":"not found"}</c>,
    /// which is precisely the reported symptom and survived the route being added.
    /// </para>
    /// </summary>
    [Fact]
    public async Task ACommandThatWritesAFileHandsBackALinkThisServerAnswers()
    {
        string root = Path.Combine(_root, "app");
        using var host = new AgentAppHost(new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"))
        {
            ExecutionMode = AgentExecutionMode.InProcess,
        });
        host.Start();
        using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");

        SessionWorkspace workspace = host.Workspaces.GetOrCreate("session-artifacts");
        var call = new ToolCall
        {
            Name = "shell",
            Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["command"] = "printf '%PDF-1.7\\n' > report.pdf",
            },
        };
        SkillToolResult result = host.CodeRunner!.Execute(call, workspace: workspace);

        SkillProducedFile file = Assert.Single(result.Files);
        Assert.StartsWith("/api/code/artifacts/", file.Url, StringComparison.Ordinal);
        // And it is a LINK in what the model is told, not a path it has to describe.
        Assert.Contains("](" + file.Url + ")", result.Content, StringComparison.Ordinal);

        HttpResponseMessage response = await client.GetAsync(file.Url);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/pdf", response.Content.Headers.ContentType?.MediaType);
    }

    /// <summary>
    /// A name with a space in it. The two producers built this URL separately and only
    /// one of them escaped anything, so the same file was reachable through a shell
    /// command's link and a 404 through a skill script's.
    /// </summary>
    [Fact]
    public async Task ANameThatNeedsEscapingResolvesFromTheEscapedLink()
    {
        string work = Path.Combine(_root, "work-spaces");
        Directory.CreateDirectory(Path.Combine(work, "out dir"));
        File.WriteAllBytes(Path.Combine(work, "out dir", "my report #2.pdf"), "%PDF-1.7\n"u8.ToArray());
        IReadOnlyList<CodeArtifact> captured = _artifacts.Capture(
            "runescape", work,
            (id, rel, _) => CodeArtifactStore.UrlFor("/api/code/artifacts", id, rel), out _);

        string url = Assert.Single(captured).Pointer;
        Assert.Equal("/api/code/artifacts/runescape/out%20dir/my%20report%20%232.pdf", url);
        Assert.Equal(HttpStatusCode.OK, (await _client.GetAsync(url)).StatusCode);
    }

    [Fact]
    public async Task ListingARunNamesEveryFileItLeft()
    {
        CaptureOne("runlist", "a.pdf", "%PDF-1.7\n"u8.ToArray());
        HttpResponseMessage response = await _client.GetAsync("/api/code/artifacts/runlist");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains("a.pdf", await response.Content.ReadAsStringAsync());
    }
}
