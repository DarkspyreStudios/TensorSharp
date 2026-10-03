// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>
/// A Qwen3-Embedding decoder: each input runs as a fresh sequence ending in <c>&lt;|endoftext|&gt;</c>, and the
/// embedding is the L2-normalised final-normed hidden state of that last token. The server does not add query
/// instructions; callers send <c>Instruct: {task}\nQuery:{text}</c> for queries and raw text for documents.
/// </summary>
public sealed class Qwen3EmbeddingModel : IEmbeddingModel
{
    public const string EndOfTextToken = "<|endoftext|>";
    private const int DefaultMaxTokens = 8192;

    private readonly Qwen3Model _model;
    private readonly int _endOfText;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public string ModelName { get; }
    public string Architecture => "qwen3";
    public int Dimensions { get; }
    public int MaxTokens { get; }
    public int VocabularySize => _model.Tokenizer.VocabSize;

    private Qwen3EmbeddingModel(string path, EmbeddingModelOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (options.MaxTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The context limit must be nonnegative.");
        BackendType backend = (options.Backend ?? "CPU").ToUpperInvariant() switch
        {
            "CPU" => BackendType.Cpu,
            "GGML_CPU" => BackendType.GgmlCpu,
            "METAL" or "GGML_METAL" => BackendType.GgmlMetal,
            "CUDA" or "GGML_CUDA" => BackendType.GgmlCuda,
            _ => throw new NotSupportedException($"Unsupported embedding backend '{options.Backend}'."),
        };
        _model = new Qwen3Model(path, backend);
        try
        {
            _endOfText = _model.Tokenizer.LookupToken(EndOfTextToken);
            if (_endOfText < 0)
                throw new InvalidDataException($"The tokenizer has no {EndOfTextToken} token.");
            Dimensions = _model.Config.HiddenSize;
            int limit = options.MaxTokens == 0 ? DefaultMaxTokens : options.MaxTokens;
            MaxTokens = Math.Min(limit, _model.MaxContextLength);
            ModelName = string.IsNullOrWhiteSpace(options.ModelName) ? Path.GetFileNameWithoutExtension(path) : options.ModelName;
        }
        catch
        {
            _model.Dispose();
            throw;
        }
    }

    public static Qwen3EmbeddingModel Load(string path, EmbeddingModelOptions options = null) => new(path, options ?? new());

    public int[] Tokenize(string text, bool truncate = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);
        List<int> tokens = _model.Tokenizer.Encode(text, addSpecial: false);
        if (tokens.Count + 1 > MaxTokens)
        {
            if (!truncate)
                throw new ArgumentException($"Input has {tokens.Count + 1} tokens and exceeds the model context limit of {MaxTokens}.", nameof(text));
            tokens.RemoveRange(MaxTokens - 1, tokens.Count - (MaxTokens - 1));
        }
        tokens.Add(_endOfText);
        return tokens.ToArray();
    }

    public async Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        if (inputs.Count == 0) return new(Array.Empty<float[]>(), 0);
        var owned = new int[inputs.Count][];
        int totalTokens = 0;
        for (int i = 0; i < inputs.Count; ++i)
        {
            var row = inputs[i] ?? throw new ArgumentException("Input sequences must not be null.", nameof(inputs));
            if (row.Length == 0 || row.Length > MaxTokens)
                throw new ArgumentException($"Every input must have between 1 and {MaxTokens} tokens.", nameof(inputs));
            foreach (int token in row)
                if ((uint)token >= (uint)VocabularySize) throw new ArgumentException($"Token {token} is outside the model vocabulary.", nameof(inputs));
            owned[i] = (int[])row.Clone();
            totalTokens = checked(totalTokens + row.Length);
        }
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return await Task.Run(() =>
            {
                var results = new float[owned.Length][];
                for (int i = 0; i < owned.Length; ++i)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    results[i] = Normalize(_model.ForwardLastHiddenState(owned[i]));
                }
                return new EmbeddingBatchResult(results, totalTokens);
            }, CancellationToken.None).ConfigureAwait(false);
        }
        finally { _gate.Release(); }
    }

    internal static float[] Normalize(float[] vector)
    {
        double sum = 0;
        foreach (float value in vector) sum += (double)value * value;
        if (sum <= 0) throw new InvalidOperationException("The model produced a zero embedding.");
        float scale = (float)(1.0 / Math.Sqrt(sum));
        for (int i = 0; i < vector.Length; ++i) vector[i] *= scale;
        return vector;
    }

    public void Dispose()
    {
        _gate.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            _model.Dispose();
        }
        finally { _gate.Release(); }
    }
}

/// <summary>Opens the embedding model a GGUF's <c>general.architecture</c> names.</summary>
public static class EmbeddingModelFactory
{
    public static IEmbeddingModel Load(string path, EmbeddingModelOptions options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string architecture;
        using (var file = new GgufFile(path))
            architecture = file.GetString("general.architecture") ?? string.Empty;
        return architecture switch
        {
            "qwen3" => Qwen3EmbeddingModel.Load(path, options),
            _ => EmbeddingModel.Load(path, options),
        };
    }
}
