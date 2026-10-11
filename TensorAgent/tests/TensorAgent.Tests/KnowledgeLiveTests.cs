// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

[Collection(LiveModelCollection.Name)]
public sealed class KnowledgeLiveTests : LiveModelHarness
{
    [LiveModelFact]
    public async Task UploadedPrivateEvidenceAnswersThroughRealChatAndSurvivesConversationPersistence()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        AgentAppHost host = Start(model, weights);
        await LoadAsync(model);
        using var create = await Client.PostAsJsonAsync("/api/knowledge/collections", new { name = "Field notebook" });
        create.EnsureSuccessStatusCode();
        var collection = (await create.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("collection");
        string id = collection.GetProperty("id").GetString()!;
        const string fact = "The Larkspur expedition's agreed rendezvous code is AMBER-KIWI-731. The code was generated for this private field notebook.";
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes(fact)), "files", "field-notebook.txt");
        using var imported = await Client.PostAsync($"/api/knowledge/collections/{id}/documents", form);
        imported.EnsureSuccessStatusCode();
        Assert.Empty((await imported.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("errors").EnumerateArray());
        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        var frames = await StreamAsync(new
        {
            sessionId, knowledgeCollections = new[] { id }, skills = Array.Empty<string>(), skills_discovery = false,
            messages = new[] { new { role = "user", content = "What is the Larkspur expedition rendezvous code? Use the selected field notebook and cite the source link." } },
            maxTokens = 256, think = false,
        });
        string answer = TextOf(frames);
        Assert.Contains("AMBER-KIWI-731", answer, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("/api/knowledge/collections/", answer);
        JsonElement cited = frames.Last(frame => frame.TryGetProperty("knowledgeSources", out _)).GetProperty("knowledgeSources")[0];
        using var original = await Client.GetAsync(cited.GetProperty("sourceUrl").GetString());
        original.EnsureSuccessStatusCode();
        Assert.Equal(fact, await original.Content.ReadAsStringAsync());
        // The final replacement, when present, must be what users and saved history
        // receive. Resolve every local source URL in the final visible prose as well.
        foreach (Match match in Regex.Matches(answer, @"/api/knowledge/collections/[a-f0-9]{32}/documents/[^\s)<>""`]+/source\?version=[0-9]+"))
        {
            using var proseSource = await Client.GetAsync(match.Value);
            proseSource.EnsureSuccessStatusCode();
            Assert.Equal(fact, await proseSource.Content.ReadAsStringAsync());
        }
        Assert.Contains(frames, frame => frame.TryGetProperty("done", out _));
        var saved = host.Conversations.Load(session.GetProperty("conversationId").GetString()!)!;
        Assert.Equal(new[] { id }, saved.KnowledgeCollections);
        Assert.NotEmpty(saved.Messages[^1].KnowledgeSources!);
        Assert.Equal(answer, saved.Messages[^1].Content);
        Assert.DoesNotContain(saved.Messages, message => message.Content.Contains("untrustedSourceData", StringComparison.Ordinal));
    }
}
