// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Server.Jev;

namespace TensorSharp.Server;

public partial class ModelService
{
    private readonly JevExecutionGate _jevExecution = new(ReadJevLimit("TS_JEV_MAX_PENDING", 32, 1, 1024));

    /// <summary>Evaluate Jev noul, choice and score questions using one denoise read per sample.</summary>
    public Task<object> JevAsync(JevRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return _jevExecution.ExecuteAsync(ct =>
        {
            if (_lifecycle.Model is not DiffusionGemmaModel model)
                throw new JevModelUnavailableException("Jev inference requires a loaded DiffusionGemma model.");
            if (request.Model != null && request.Model is not ("jev-latest" or "jev-preview") &&
                !string.Equals(request.Model, LoadedModelName, StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(request.Model, Path.GetFileNameWithoutExtension(LoadedModelName), StringComparison.OrdinalIgnoreCase))
                throw new JevModelNotFoundException("model must name the loaded model, jev-latest, or jev-preview");

            bool entered = false;
            try
            {
                // The existing chat scheduler owns this same gate for each entire
                // denoising block. Structured reads cannot race its shared native buffers.
                while (!(entered = Monitor.TryEnter(model.GpuComputeLock, 100))) ct.ThrowIfCancellationRequested();
                ct.ThrowIfCancellationRequested();
                var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
                int[] Render(string system, string state) => renderer.RenderToTokens(model.Tokenizer,
                    model.Config.ChatTemplate, new List<ChatMessage>
                    {
                        new() { Role = "system", Content = system },
                        new() { Role = "user", Content = state },
                    }, model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();
                int eos = model.Tokenizer.LookupToken("<turn|>");
                if (eos < 0) throw new JevModelUnavailableException("DiffusionGemma tokenizer is missing the <turn|> token.");
                int pad = model.Tokenizer.LookupToken("<pad>");
                if (pad < 0) pad = model.MaskTokenId;
                return JevInference.Run(request, LoadedModelName, text => model.Tokenizer.Encode(text, false).ToArray(),
                    Render, model.ReadStructured, Math.Min(model.CanvasLength, ReadJevLimit("TS_JEV_MAX_CANVAS", 64, 8, 4096)),
                    model.MaxContextLength, eos, pad, model.Tokenizer.VocabSize, ct);
            }
            finally { if (entered) Monitor.Exit(model.GpuComputeLock); }
        }, cancellationToken);
    }

    private static int ReadJevLimit(string variable, int fallback, int minimum, int maximum)
    {
        string? value = Environment.GetEnvironmentVariable(variable);
        if (string.IsNullOrEmpty(value)) return fallback;
        if (!int.TryParse(value, out int result) || result < minimum || result > maximum)
            throw new ArgumentException($"{variable} must be an integer from {minimum} to {maximum}.");
        return result;
    }
}
