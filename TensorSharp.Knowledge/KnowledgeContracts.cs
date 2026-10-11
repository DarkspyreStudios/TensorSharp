using System.Text.Json.Serialization;

namespace TensorSharp.Knowledge;

public sealed class KnowledgeOptions
{
    public long MaxFileBytes { get; init; } = 64L * 1024 * 1024;
    public long MaxStoredBytes { get; init; } = 2L * 1024 * 1024 * 1024;
    public int MaxDocuments { get; init; } = 10000;
    public int MaxCollections { get; init; } = 100;
    public int MaxExtractedCharacters { get; init; } = 2_000_000;
    public int MaxSegments { get; init; } = 10000;
    public int MaxChunksPerDocument { get; init; } = 10000;
    public int MaxTotalChunks { get; init; } = 100000;
    public int ChunkCharacters { get; init; } = 1200;
    public int ChunkOverlapCharacters { get; init; } = 120;
    public int MaxSearchResults { get; init; } = 50;
    public int MaxContextCharacters { get; init; } = 32000;
    public int MaxQueryCharacters { get; init; } = 4000;
    public int EmbeddingBatchSize { get; init; } = 32;
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum KnowledgeDocumentStatus { Ready, MetadataOnly, Partial }

public sealed record KnowledgeLocation
{
    public int? Page { get; init; }
    public string? Sheet { get; init; }
    public int? Row { get; init; }
    public int? EndRow { get; init; }
    public int? StartLine { get; init; }
    public int? EndLine { get; init; }
    public string? Section { get; init; }
    public long? StartMilliseconds { get; init; }
    public long? EndMilliseconds { get; init; }
    public double? StartSeconds { get; init; }
    public double? EndSeconds { get; init; }
    public string? Label { get; init; }
}

public sealed record KnowledgeSegment(string Text, KnowledgeLocation? Location = null, string? Modality = null);
public sealed record KnowledgeExtractionContext(string FilePath, string FileName, string? ContentType);
public sealed record KnowledgeExtraction
{
    public string? ExtractorId { get; init; }
    public IReadOnlyList<KnowledgeSegment> Segments { get; init; } = Array.Empty<KnowledgeSegment>();
    public KnowledgeDocumentStatus Status { get; init; } = KnowledgeDocumentStatus.Ready;
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

public interface IKnowledgeExtractor
{
    Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default);
}

public interface IKnowledgeEmbeddingProvider
{
    string ModelId { get; }
    int Dimensions { get; }
    Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default);
}

public sealed record KnowledgeCollection(string Id, string Name, string? Description, DateTimeOffset CreatedAt);

public sealed record KnowledgeDocument
{
    public required string Id { get; init; }
    public required string CollectionId { get; init; }
    public required string FileName { get; init; }
    public string? ContentType { get; init; }
    public required string ContentHash { get; init; }
    public int Version { get; init; }
    public long SizeBytes { get; init; }
    public int ChunkCount { get; init; }
    public KnowledgeDocumentStatus Status { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
    public string? EmbeddingModelId { get; init; }
    public string? ExtractorId { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
    public DateTimeOffset UpdatedAt { get; init; }
}

public sealed record KnowledgeSearchOptions
{
    /// <summary>Required scope. An empty list searches nothing; callers explicitly select collections.</summary>
    public IReadOnlyList<string> CollectionIds { get; init; } = Array.Empty<string>();
    public int Limit { get; init; } = 8;
    public int MaxCharacters { get; init; } = 12000;
}

public sealed record KnowledgeSearchResult
{
    public required string Reference { get; init; }
    public required string CollectionId { get; init; }
    public required string DocumentId { get; init; }
    public required string FileName { get; init; }
    public int Version { get; init; }
    public int ChunkIndex { get; init; }
    public required string Text { get; init; }
    public KnowledgeLocation? Location { get; init; }
    public string? Modality { get; init; }
    public double Score { get; init; }
    public KnowledgeDocumentStatus Status { get; init; }
    public bool Truncated { get; init; }
}

public sealed class KnowledgeLimitException(string message) : InvalidOperationException(message);
public sealed class KnowledgeStaleReferenceException(string message) : InvalidOperationException(message);
