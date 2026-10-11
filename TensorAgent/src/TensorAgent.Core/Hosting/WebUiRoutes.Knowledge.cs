// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Knowledge;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Core.Hosting;

public static partial class WebUiRoutes
{
    /// <summary>Personal, authenticated loopback library. Original sources outlive the upload cache.</summary>
    public static void MapKnowledge(this LoopbackServer server, KnowledgeStore store)
    {
        const string root = "/api/knowledge";
        const long maxBatchBytes = 128L * 1024 * 1024;
        server.MapGet(root + "/collections", (_, _) => KnowledgeRoute(() =>
            Task.FromResult<LoopbackResponse?>(KnowledgeJson(new { collections = store.ListCollections() }))));
        server.MapPost(root + "/collections", (request, ct) => KnowledgeRoute(async () =>
        {
            JsonElement body = await ReadKnowledgeJson(request, ct);
            string name = RequiredString(body, "name");
            string? description = body.TryGetProperty("description", out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
            return KnowledgeJson(new { collection = await store.CreateCollectionAsync(name, description, ct) }, 201);
        }));
        server.MapDelete(root + "/collections/{collection}", (request, ct) => KnowledgeRoute(async () =>
            await store.DeleteCollectionAsync(request.RouteValues["collection"], ct)
                ? KnowledgeJson(new { deleted = true }) : LoopbackResponse.NotFound(new { error = "Collection not found." })));
        server.MapGet(root + "/collections/{collection}/documents", (request, _) => KnowledgeRoute(() =>
        {
            int offset = QueryInteger(request, "offset", 0, 0, int.MaxValue);
            int limit = QueryInteger(request, "limit", 100, 1, 200);
            var all = store.ListDocuments(request.RouteValues["collection"]);
            var documents = all.Skip(offset).Take(limit).ToArray();
            return Task.FromResult<LoopbackResponse?>(KnowledgeJson(new
            {
                documents, total = all.Count,
                nextOffset = offset < all.Count - documents.Length ? offset + documents.Length : (int?)null,
            }));
        }));
        server.MapPost(root + "/collections/{collection}/documents", (request, ct) => KnowledgeRoute(async () =>
        {
            if (!request.HasFormContentType) throw new ArgumentException("A multipart file upload is required.");
            if (request.Raw.ContentLength64 > maxBatchBytes) throw new KnowledgeLimitException("Upload batch exceeds 128 MiB.");
            // Validate the destination before accepting or spooling a large request.
            string collection = request.RouteValues["collection"];
            _ = store.ListDocuments(collection);
            using var form = await MultipartFormReader.ReadAsync(request.Raw.InputStream,
                request.Raw.ContentType ?? "", ct, maxBatchBytes, 16);
            if (form.Files.Count == 0) throw new ArgumentException("Select at least one source file.");
            var documents = new List<KnowledgeDocument>();
            var errors = new List<object>();
            foreach (var file in form.Files)
            {
                try
                {
                    await using Stream input = file.OpenRead();
                    documents.Add(await store.ImportAsync(collection, UploadNaming.ResolveFileName(file), input, file.ContentType, ct));
                }
                catch (Exception ex) when (ex is ArgumentException or InvalidDataException or KnowledgeLimitException)
                {
                    // A batch may contain several independent formats. Report each
                    // failure explicitly while retaining the successfully imported sources.
                    errors.Add(new { fileName = file.FileName, error = ex.Message });
                }
            }
            return KnowledgeJson(new { documents, errors });
        }));
        server.MapDelete(root + "/collections/{collection}/documents/{document}", (request, ct) => KnowledgeRoute(async () =>
            await store.DeleteDocumentAsync(request.RouteValues["collection"], request.RouteValues["document"], ct)
                ? KnowledgeJson(new { deleted = true }) : LoopbackResponse.NotFound(new { error = "Source not found." })));
        server.MapPost(root + "/collections/{collection}/documents/{document}/reindex", (request, ct) => KnowledgeRoute(async () =>
            KnowledgeJson(new { document = await store.ReindexAsync(request.RouteValues["collection"], request.RouteValues["document"], ct) })));
        server.MapGet(root + "/collections/{collection}/documents/{document}/source", (request, _) => KnowledgeRoute(() =>
        {
            if (!int.TryParse(request.Query("version"), out int version) || version < 1)
                throw new ArgumentException("A positive source version is required.");
            string collection = request.RouteValues["collection"], id = request.RouteValues["document"];
            var document = store.ListDocuments(collection).FirstOrDefault(d => d.Id == id)
                ?? throw new KeyNotFoundException("Source not found.");
            return Task.FromResult<LoopbackResponse?>(new KnowledgeSourceResponse(
                store.OpenOriginal(collection, id, version), document.FileName));
        }));
        server.MapPost(root + "/search", (request, ct) => KnowledgeRoute(async () =>
        {
            JsonElement body = await ReadKnowledgeJson(request, ct);
            var results = await store.SearchAsync(RequiredString(body, "query"), new KnowledgeSearchOptions
            {
                CollectionIds = KnowledgeSelection(body),
                Limit = OptionalInteger(body, "limit", 8, 1, 50),
                MaxCharacters = OptionalInteger(body, "maxCharacters", 12000, 256, 32000),
            }, ct);
            return KnowledgeJson(new { results = results.Select(KnowledgeTools.Describe) });
        }));
        server.MapPost(root + "/read", (request, ct) => KnowledgeRoute(async () =>
        {
            JsonElement body = await ReadKnowledgeJson(request, ct);
            var results = store.Read(RequiredString(body, "reference"),
                OptionalInteger(body, "maxCharacters", 12000, 256, 32000),
                OptionalInteger(body, "adjacentChunks", 1, 0, 2), KnowledgeSelection(body));
            return KnowledgeJson(new { results = results.Select(KnowledgeTools.Describe) });
        }));
    }

    private static readonly JsonSerializerOptions KnowledgeJsonOptions = new(JsonSerializerDefaults.Web);
    private static LoopbackResponse KnowledgeJson(object value, int status = 200) =>
        LoopbackResponse.Text(JsonSerializer.Serialize(value, KnowledgeJsonOptions), status, "application/json; charset=utf-8");

    private static string[] KnowledgeSelection(JsonElement body)
    {
        if (!body.TryGetProperty("collectionIds", out var ids) || ids.ValueKind != JsonValueKind.Array || ids.GetArrayLength() > 32)
            throw new ArgumentException("collectionIds must be an explicit array of at most 32 IDs.");
        return ids.EnumerateArray().Select(id => id.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(id.GetString())
            ? id.GetString()! : throw new ArgumentException("Collection IDs must be nonempty strings.")).Distinct(StringComparer.Ordinal).ToArray();
    }

    private static string RequiredString(JsonElement body, string name) =>
        body.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString())
            ? value.GetString()! : throw new ArgumentException(name + " is required.");

    private static int OptionalInteger(JsonElement body, string name, int fallback, int min, int max)
    {
        if (!body.TryGetProperty(name, out var value)) return fallback;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int number) || number < min || number > max)
            throw new ArgumentException($"{name} must be between {min} and {max}.");
        return number;
    }

    private static int QueryInteger(LoopbackRequest request, string name, int fallback, int min, int max)
    {
        string? supplied = request.Query(name);
        if (supplied == null) return fallback;
        if (!int.TryParse(supplied, out int value) || value < min || value > max)
            throw new ArgumentException($"{name} must be between {min} and {max}.");
        return value;
    }

    private static async Task<JsonElement> ReadKnowledgeJson(LoopbackRequest request, CancellationToken ct)
    {
        const int limit = 64 * 1024;
        using var output = new MemoryStream();
        byte[] buffer = new byte[4096];
        int read;
        while ((read = await request.Raw.InputStream.ReadAsync(buffer, ct)) != 0)
        {
            if (output.Length + read > limit) throw new KnowledgeLimitException("Knowledge request exceeds 64 KiB.");
            output.Write(buffer, 0, read);
        }
        using var json = JsonDocument.Parse(output.ToArray());
        if (json.RootElement.ValueKind != JsonValueKind.Object) throw new ArgumentException("A JSON object is required.");
        return json.RootElement.Clone();
    }

    private static async Task<LoopbackResponse?> KnowledgeRoute(Func<Task<LoopbackResponse?>> action)
    {
        try { return await action(); }
        catch (KnowledgeLimitException ex) { return KnowledgeJson(new { error = ex.Message }, 413); }
        catch (KnowledgeStaleReferenceException ex) { return KnowledgeJson(new { error = ex.Message }, 409); }
        catch (KeyNotFoundException ex) { return KnowledgeJson(new { error = ex.Message }, 404); }
        catch (UnauthorizedAccessException) { return KnowledgeJson(new { error = "This source is outside the selected collections." }, 403); }
        catch (UploadLimitExceededException ex) { return KnowledgeJson(new { error = ex.Message }, ex.StatusCode); }
        catch (Exception ex) when (ex is ArgumentException or InvalidDataException or JsonException or FormatException)
        { return KnowledgeJson(new { error = ex.Message }, 400); }
    }

    private sealed class KnowledgeSourceResponse(Stream source, string fileName) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            await using (source)
            {
                response.StatusCode = 200;
                response.ContentType = "application/octet-stream";
                response.Headers["X-Content-Type-Options"] = "nosniff";
                response.Headers["Content-Disposition"] = "attachment; filename*=UTF-8''" + Uri.EscapeDataString(fileName);
                response.Headers["Cache-Control"] = "no-store";
                if (source.CanSeek) response.ContentLength64 = source.Length;
                await source.CopyToAsync(response.OutputStream, ct);
            }
        }
    }
}
