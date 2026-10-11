// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Knowledge;

namespace InferenceWeb.Tests;

public sealed class KnowledgeCitationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-citations-" + Guid.NewGuid().ToString("N"));
    private KnowledgeStore? _store;
    public void Dispose() { _store?.Dispose(); if (Directory.Exists(_root)) Directory.Delete(_root, true); }

    private async Task<(KnowledgeToolScope Scope, KnowledgeSearchResult Hit)> Prepare(string fileName = "source.txt")
    {
        _store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
        var collection = await _store.CreateCollectionAsync("citations");
        await _store.ImportAsync(collection.Id, fileName, new MemoryStream(Encoding.UTF8.GetBytes("Unique cobalt evidence.")));
        var scope = new KnowledgeToolScope(_store, new[] { collection.Id });
        var hits = await _store.SearchAsync("cobalt", new() { CollectionIds = scope.CollectionIds });
        scope.RecordSources(hits);
        return (scope, Assert.Single(hits));
    }

    [Fact]
    public async Task OnlyExactRetrievedFilenameUrlsBindToAnAuthoritativeSource()
    {
        var (scope, hit) = await Prepare("星图 notes (1).txt");
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        string right = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        string text = $"[notebook]({wrong}) and <{wrong}>.";
        var repaired = KnowledgeTools.NormalizeCitationUrls(text, scope);
        Assert.Equal(text.Replace(wrong, right, StringComparison.Ordinal), repaired.Text);
        Assert.Equal(2, repaired.RepairCount);
        foreach (string unchanged in new[]
        {
            right, wrong.Replace("version=1", "version=2"), wrong.Replace(hit.CollectionId, new string('a', 32)),
            KnowledgeTools.SourceUrl(hit.CollectionId, Guid.NewGuid().ToString("N"), 1),
            KnowledgeTools.SourceUrl(hit.CollectionId, "unknown.txt", 1),
            "https://example.com" + wrong, "/prefix" + wrong, wrong + "&other=true", wrong + "#page=2", wrong + "0",
            wrong + ".5", wrong.Replace("notes", "NOTES"),
        })
        {
            Assert.Equal((unchanged, 0), KnowledgeTools.NormalizeCitationUrls(unchanged, scope));
        }
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, null));
        var unretrieved = new KnowledgeToolScope(scope.Store, scope.CollectionIds);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, unretrieved));
        await scope.Store.ReindexAsync(hit.CollectionId, hit.DocumentId);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, scope));
    }

    [Fact]
    public async Task AmbiguousEvidenceOrDeletedSourcesAreNotRepaired()
    {
        var (scope, hit) = await Prepare();
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        // A duplicated filename from an external retriever must not choose one ID.
        var second = await scope.Store.ImportAsync(hit.CollectionId, "other.txt", new MemoryStream(Encoding.UTF8.GetBytes("Other cobalt evidence.")));
        scope.RecordSources(new[] { hit with { Reference = "kb://ambiguous", DocumentId = second.Id } });
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, scope));
        var single = new KnowledgeToolScope(scope.Store, scope.CollectionIds);
        single.RecordSources(new[] { hit, hit with { Reference = hit.Reference + "neighbor", ChunkIndex = 1 } });
        Assert.Equal(1, KnowledgeTools.NormalizeCitationUrls(wrong, single).RepairCount);
        await scope.Store.DeleteDocumentAsync(hit.CollectionId, hit.DocumentId);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, single));
    }

    [Fact]
    public async Task FinalParserFlushIsNormalizedBeforeDoneAndCarriesExplicitRepairCount()
    {
        var (scope, hit) = await Prepare();
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        string right = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        using var session = new ChatSession();
        var shown = new StringBuilder("Evidence: ");
        var frames = WebUiChatService.FinalFrames(new FlushParser($"[source]({wrong})"), false, null,
            session, Stopwatch.StartNew(), 1, 0, 0, false, shown, knowledge: scope)
            .Select(frame => JsonSerializer.SerializeToElement(frame)).ToArray();
        Assert.Equal($"Evidence: [source]({right})", shown.ToString());
        Assert.Contains(wrong, frames[0].GetProperty("token").GetString());
        Assert.Equal(shown.ToString(), frames[1].GetProperty("replace").GetString());
        Assert.Equal(1, frames[2].GetProperty("knowledgeCitationRepairs").GetInt32());
        Assert.True(frames[^1].GetProperty("done").GetBoolean());
    }

    private sealed class FlushParser(string final) : IOutputParser
    {
        public bool HasThinkingSupport => false;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => false;
        public void Init(bool enableThinking, List<ToolFunction>? tools) { }
        public ParsedOutput Add(string text, bool done) => new() { Content = done ? final : "" };
    }
}
