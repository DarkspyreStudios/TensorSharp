using System;
using TensorSharp.Knowledge;

namespace TensorSharp.Server.Knowledge;

/// <summary>Optional host configuration. No model downloads or external services are used implicitly.</summary>
public static class KnowledgeRuntime
{
    public static KnowledgeStore CreateStore(string root, ModelService modelService = null)
    {
        var options = new KnowledgeOptions();
        GgufKnowledgeEmbeddingProvider embeddings = null;
        try
        {
            string model = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_EMBEDDING_MODEL");
            if (!string.IsNullOrWhiteSpace(model))
                embeddings = new GgufKnowledgeEmbeddingProvider(model,
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_EMBEDDING_BACKEND") ?? "CPU",
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_QUERY_PREFIX") ?? "",
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_DOCUMENT_PREFIX") ?? "");
            return new KnowledgeStore(root, new BuiltInKnowledgeExtractor(options,
                modelService == null ? null : new ModelKnowledgeMediaExtractor(modelService)), embeddings, options);
        }
        catch { embeddings?.Dispose(); throw; }
    }
}
