using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

namespace InferenceWeb.Tests;

public sealed class Qwen3EmbeddingRetirementTests
{
    [Fact]
    public void RepeatedDisposalPreservesFailedModelReleaseAndFencesRequests()
    {
        var failure = new InvalidOperationException("Native model release failed.");
        // Seed the actual owner's post-failure state without acquiring native resources.
        var model = (Qwen3Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen3Model));
        SetField(typeof(ModelBase), model, "_ownershipCleanupFailed", true);
        SetField(typeof(ModelBase), model, "_ownershipCleanupFailure", failure);

        var embedding = (Qwen3EmbeddingModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen3EmbeddingModel));
        using var gate = new SemaphoreSlim(1, 1);
        SetField(typeof(Qwen3EmbeddingModel), embedding, "_gate", gate);
        SetField(typeof(Qwen3EmbeddingModel), embedding, "_model", model);

        var first = Assert.Throws<InvalidOperationException>(embedding.Dispose);
        Assert.Same(failure, first.InnerException);
        var second = Assert.Throws<InvalidOperationException>(embedding.Dispose);
        Assert.Same(failure, second.InnerException);
        Assert.Throws<ObjectDisposedException>(() => embedding.Tokenize("No new execution"));
    }

    private static void SetField(Type type, object owner, string name, object value) =>
        (type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field {type.Name}.{name} not found.")).SetValue(owner, value);
}
