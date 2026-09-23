// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Server.Jev;

/// <summary>What the binder needs from the loaded model: where prepared spans are kept, how the
/// retained set is released, and the structured read itself. <see cref="JevDiffusionVisionTarget"/>
/// is the production implementation; the interface exists so the span-switching rules can be
/// exercised without a checkpoint and a vision tower.</summary>
internal interface IJevVisionTarget
{
    IMultimodalInjector Injector { get; }
    void ClearVisionEmbeddings();
    float[][] ReadStructured(int[] prompt, int[] canvas, int[] positions, int[][] tokenIds,
        CancellationToken cancellationToken);
}

/// <summary>The DiffusionGemma model behind <see cref="IJevVisionTarget"/>.</summary>
internal sealed class JevDiffusionVisionTarget(DiffusionGemmaModel model) : IJevVisionTarget
{
    public IMultimodalInjector Injector => model.MultimodalInjector;
    public void ClearVisionEmbeddings() => model.ClearVisionEmbeddings();
    public float[][] ReadStructured(int[] prompt, int[] canvas, int[] positions, int[][] tokenIds,
        CancellationToken cancellationToken)
        => model.ReadStructured(prompt, canvas, positions, tokenIds, cancellationToken);
}

/// <summary>
/// Carries one request's images from the rendered prompt into the structured read.
///
/// <para>Two things have to line up. The prompt's <c>&lt;|image&gt;</c> markers must be expanded
/// into [BOI] + N soft rows + [EOI] before the context check measures the prompt, and the encoded
/// embeddings must be installed on the model before the forward that prefills that prompt, or the
/// filler rows are read as ordinary tokens and the answer is a confident probability about pixels
/// the model never received.</para>
///
/// <para>A schema wider than the canvas is split into chunks, and each chunk renders its OWN
/// prompt: the same image sits at a different token offset in each one. So spans are prepared per
/// chunk prompt and installed when a read for that prompt arrives. Installing bumps the model's
/// span version, which invalidates its retained prompt K/V, so this switches only when the chunk
/// actually changes — repeated reads of one chunk (fixed or adaptive samples) keep the cache.</para>
///
/// <para>The retained span set is model-global state, shared with ordinary diffusion chat. The
/// caller holds <c>GpuComputeLock</c> across the whole request, and <see cref="Dispose"/> must run
/// before releasing it: a span left installed would be spliced into the next request's prompt.</para>
/// </summary>
internal sealed class JevVisionBinder : IDisposable
{
    private readonly IJevVisionTarget _target;
    private readonly string[] _imagePaths;
    // Keyed by array identity: JevInference hands the very array a render returned back to the
    // read, so identity — not content — is what names the chunk whose spans belong to it.
    private readonly Dictionary<int[], string> _promptRequests =
        new((IEqualityComparer<int[]>)ReferenceEqualityComparer.Instance);
    private readonly List<string> _prepared = new();
    private string _installed;

    internal JevVisionBinder(IJevVisionTarget target, string[] imagePaths)
    {
        _target = target;
        _imagePaths = imagePaths;
    }

    internal bool HasImages => _imagePaths.Length != 0;

    /// <summary>The image paths for one rendered turn, or null for a text-only request (the
    /// renderer emits no media placeholders for a message without them).</summary>
    internal List<string> ImagePathsOrNull() => HasImages ? new List<string>(_imagePaths) : null;

    /// <summary>
    /// Expand this chunk prompt's image placeholders and prepare its spans. Returns
    /// <paramref name="promptTokens"/> unchanged for a text-only request.
    /// </summary>
    internal int[] Expand(List<ChatMessage> history, int[] promptTokens)
    {
        if (!HasImages) return promptTokens;
        var injector = Injector();
        string requestId = "jev-" + Guid.NewGuid().ToString("N");
        _prepared.Add(requestId);
        int[] expanded;
        try
        {
            expanded = injector.ProcessPromptTokens(history, new List<int>(promptTokens), requestId).ToArray();
        }
        catch (InvalidDataException error)
        {
            // Every image codec in this tree reports "these bytes are not a usable image" as
            // InvalidDataException, and a request body that carries one is the client's mistake:
            // answer 422 with the decoder's reason instead of a bare 500.
            throw new JevValidationException($"images: {error.Message}");
        }
        if (!injector.HasPendingEmbeddings(requestId))
        {
            // Placeholders that reach the read unexpanded produce an answer about filler rows.
            throw new JevModelUnavailableException(
                "The request declares images but the model prepared no image embeddings for this prompt.");
        }
        _promptRequests[expanded] = requestId;
        return expanded;
    }

    /// <summary>One structured read, with this prompt's images installed first.</summary>
    internal float[][] Read(int[] prompt, int[] canvas, int[] positions, int[][] tokenIds,
        CancellationToken cancellationToken)
    {
        Activate(prompt);
        return _target.ReadStructured(prompt, canvas, positions, tokenIds, cancellationToken);
    }

    private void Activate(int[] prompt)
    {
        if (!HasImages) return;
        if (!_promptRequests.TryGetValue(prompt, out string requestId))
            throw new InvalidOperationException("Structured read received a prompt that was never prepared.");
        if (ReferenceEquals(requestId, _installed)) return;

        _target.ClearVisionEmbeddings();
        _installed = null;
        if (!Injector().QueuePromptEmbeddings(0, requestId))
            throw new JevModelUnavailableException("The prepared image embeddings for this request are missing.");
        _installed = requestId;
    }

    private IMultimodalInjector Injector() => _target.Injector
        ?? throw new JevModelUnavailableException("The loaded model has no multimodal injector for image input.");

    public void Dispose()
    {
        if (_installed != null)
        {
            _target.ClearVisionEmbeddings();
            _installed = null;
        }
        var injector = _target.Injector;
        if (injector != null)
            foreach (string requestId in _prepared) injector.ClearPreparedPromptState(requestId);
        _prepared.Clear();
        _promptRequests.Clear();
    }
}
