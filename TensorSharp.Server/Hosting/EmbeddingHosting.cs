using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using TensorSharp.Models.Embeddings;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Hosting;

/// <summary>Registers a resident embedding model without creating a chat inference engine.</summary>
public static class EmbeddingHosting
{
    public static IServiceCollection AddTensorSharpEmbeddings(this IServiceCollection services, ServerHostingOptions options,
        string? requestedBackend = null)
    {
        if (!options.EmbeddingsEnabled)
            throw new ArgumentException("Embedding hosting requires EmbeddingsEnabled.", nameof(options));
        string selectedBackend = options.DefaultBackend;
        if (!string.IsNullOrWhiteSpace(requestedBackend)
            && !BackendSelector.TryResolveSupportedBackend(options, requestedBackend, out selectedBackend, out string error))
            throw new ArgumentException(error, nameof(requestedBackend));
        string backend = ResolveModelBackend(selectedBackend);
        services.TryAddSingleton(options);
        services.AddSingleton<IEmbeddingModel>(_ => EmbeddingModel.Load(options.StartupModelPath,
            new EmbeddingModelOptions { Backend = backend, Threads = options.EmbeddingThreads,
                MaxTokens = options.EmbeddingContextSize }));
        services.AddSingleton<EmbeddingAdapter>();
        return services;
    }

    internal static string ResolveModelBackend(string backend) => BackendCatalog.Canonicalize(backend) switch
    {
        "cpu" => "CPU",
        "ggml_cpu" => "GGML_CPU",
        "ggml_metal" => "GGML_METAL",
        "ggml_cuda" => "GGML_CUDA",
        _ => throw new ArgumentException($"Embedding hosting requires cpu (pure C#), ggml_cpu, ggml_metal, or ggml_cuda; got '{backend}'."),
    };

    /// <summary>Rejects generation requests before any adapter attempts to load an encoder as a chat model.</summary>
    public static IApplicationBuilder UseEmbeddingModelGuard(this IApplicationBuilder app) => app.Use(async (context, next) =>
    {
        if (HttpMethods.IsPost(context.Request.Method) && IsGenerationPath(context.Request.Path)
            && context.RequestServices.GetService<ServerHostingOptions>()?.EmbeddingsEnabled == true)
        {
            await EmbeddingAdapter.WriteErrorAsync(context,
                "This server hosts an embedding model. Use /v1/embeddings or /api/embed.").ConfigureAwait(false);
            return;
        }
        await next(context).ConfigureAwait(false);
    });

    internal static bool IsGenerationPath(PathString path) => path.Value?.TrimEnd('/').ToLowerInvariant() is
        "/v1/chat/completions" or "/v1/completions" or "/v1/responses" or "/v1/videos/generations"
        or "/api/generate" or "/api/chat" or "/api/chat/ollama" or "/api/models/load";

    internal static Task InvokeAsync(HttpContext context, Func<EmbeddingAdapter, HttpContext, Task> action)
    {
        var adapter = context.RequestServices.GetService<EmbeddingAdapter>();
        return adapter != null ? action(adapter, context) : EmbeddingAdapter.WriteErrorAsync(context,
            "This server does not host an embedding model. Start it with --model <encoder.gguf> --embeddings.");
    }
}
