// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Globalization;
using TensorSharp.Models;
using TensorSharp.Runtime.Logging;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Host.Hosting
{
    /// <summary>
    /// Emits the structured "what is this server doing?" banner to the logger.
    /// Centralised here so we can iterate on the operator-facing summary
    /// without touching <c>Program.cs</c>.
    /// </summary>
    public static class StartupBanner
    {
        private static readonly string[] EndpointSummary =
        {
            "GET  /health                    - Health check",
            "GET  /api/tags                  - List hosted models (Ollama)",
            "POST /api/show                  - Show model details (Ollama)",
            "POST /api/generate              - Generate text (Ollama)",
            "POST /api/chat/ollama           - Chat completion (Ollama)",
            "POST /v1/chat/completions       - Chat completion (OpenAI)",
            "POST /v1/systemone             - Typed Jev decisions (DiffusionGemma)",
            "GET  /v1/models                 - List hosted models (OpenAI)",
            "POST /api/chat                  - Chat (Web UI SSE)",
            "POST /api/models/load           - Reload hosted model (Web UI)",
            "GET  /api/models                - Show hosted model state (Web UI)",
        };

        private static readonly string[] EmbeddingEndpointSummary =
        {
            "GET  /health                    - Health check",
            "POST /v1/embeddings             - Embeddings (OpenAI; float or base64)",
            "POST /api/embed                 - Embeddings (Ollama; string or batch)",
            "POST /api/embeddings            - Embeddings (legacy Ollama)",
            "GET  /v1/models                 - List embedding model (OpenAI)",
            "GET  /api/tags                  - List embedding model (Ollama)",
            "POST /api/show                  - Embedding model details (Ollama)",
            "GET  /api/models                - Show embedding model state",
        };

        internal static IEnumerable<string> DescribeEndpoints(ServerHostingOptions options)
        {
            yield return options.WebUiEnabled
                ? "GET  /                          - Web UI (index.html)"
                : "GET  /                          - Health check";
            foreach (string endpoint in options.EmbeddingsEnabled ? EmbeddingEndpointSummary : EndpointSummary)
                yield return endpoint;
        }

        private static void EmitEndpoints(ILogger logger, ServerHostingOptions options, string listenAddress)
        {
            logger.LogInformation(LogEventIds.HostStarting,
                "Starting TensorSharp.Server on {ListenAddress}", listenAddress);
            foreach (string endpoint in DescribeEndpoints(options))
                logger.LogInformation(LogEventIds.HostConfiguration, "Endpoint: {Endpoint}", endpoint);
        }

        public static void Emit(ILogger logger, ServerHostingOptions options, string listenAddress)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (options == null) throw new ArgumentNullException(nameof(options));

            if (options.SupportedBackends.Count == 0)
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "No supported backends detected on this machine.");
            }
            else
            {
                logger.LogInformation(LogEventIds.BackendDetected,
                    "Supported backends: {SupportedBackends}",
                    string.Join(", ", options.SupportedBackends.Select(b => b.Value)));
            }

            // Why a probed backend is missing from the list above: the probe threw and
            // the exception was swallowed into "unavailable" during discovery.
            foreach (string probeFailure in options.UsesManagedEmbeddingBackend
                ? Array.Empty<string>() : BackendCatalogProbes.DescribeProbeFailures())
            {
                logger.LogInformation(LogEventIds.BackendUnavailable,
                    "Backend probe failed, so that backend is not offered: {ProbeFailure}", probeFailure);
            }

            if (options.EmbeddingsEnabled)
            {
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "Embedding server configuration: hostedModel={HostedModel} backend={Backend} execution={Execution} threads={Threads} contextLimit={ContextLimit} listen={ListenAddress}",
                    options.StartupModelPath, options.DefaultBackend, options.UsesManagedEmbeddingBackend ? "pure-csharp" : "native-ggml",
                    options.EmbeddingThreads > 0 ? options.EmbeddingThreads.ToString(CultureInfo.InvariantCulture) : "backend-default",
                    options.EmbeddingContextSize > 0 ? options.EmbeddingContextSize.ToString(CultureInfo.InvariantCulture) : "model-default",
                    listenAddress);
                EmitEndpoints(logger, options, listenAddress);
                return;
            }

            logger.LogInformation(LogEventIds.HostConfiguration,
                "Server configuration: hostedModel={HostedModel} hostedMmProj={HostedMmProj} defaultMaxTokens={DefaultMaxTokens}{MaxTokensPinned} videoFrames={VideoFrames} videoFps={VideoFps} videoSize={VideoSize} videoSteps={VideoSteps} videoMode={VideoMode} videoSampleFps={VideoSampleFps} videoMaxFrames={VideoMaxFrames} listen={ListenAddress}",
                options.StartupModelPath ?? "(none)",
                options.StartupMmProjPath ?? "(none)",
                options.DefaultMaxTokens,
                options.MaxTokensPinned ? " (server cap)" : string.Empty,
                options.DefaultVideoFrames > 0
                    ? options.DefaultVideoFrames.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                options.DefaultVideoFps > 0
                    ? options.DefaultVideoFps.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                options.DefaultVideoWidth > 0 || options.DefaultVideoHeight > 0
                    ? $"{options.DefaultVideoWidth}x{options.DefaultVideoHeight}"
                    : "model-default",
                options.DefaultVideoSteps > 0
                    ? options.DefaultVideoSteps.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                string.IsNullOrWhiteSpace(options.DefaultVideoMode) ? "auto" : options.DefaultVideoMode,
                MediaHelper.GetConfiguredVideoSampleFps().ToString("0.###", CultureInfo.InvariantCulture),
                MediaHelper.GetConfiguredMaxVideoFrames(),
                listenAddress);

            // Surface the resolved sampling defaults so operators can confirm
            // the CLI flags / env vars they passed actually took effect.
            // We log the structured fields (rather than just one big string)
            // so log scrapers can pull individual values.
            var sampling = options.DefaultSamplingConfig;
            logger.LogInformation(LogEventIds.HostConfiguration,
                "Default sampling: temperature={Temperature} topK={TopK} topP={TopP} minP={MinP} repeatPenalty={RepeatPenalty} repeatLastN={RepeatLastN} presencePenalty={PresencePenalty} frequencyPenalty={FrequencyPenalty} seed={Seed} stopSequences={StopSequences}",
                sampling.Temperature.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.TopK,
                sampling.TopP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.MinP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.RepetitionPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.PenaltyLastN,
                sampling.PresencePenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.FrequencyPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.Seed,
                sampling.StopSequences != null && sampling.StopSequences.Count > 0
                    ? "[" + string.Join(", ", sampling.StopSequences.Select(s => "\"" + s + "\"")) + "]"
                    : "(none)");

            // Which of those defaults a request can talk the server out of.
            // Without this line an operator whose client hardcodes temperature
            // (VS Code Copilot Chat does) has no way to tell whether their
            // config is in charge — see issue #113.
            logger.LogInformation(LogEventIds.HostConfiguration,
                "Sampling precedence: {SamplingPrecedence}",
                options.SamplingDefaults.DescribePolicy());

            EmitEndpoints(logger, options, listenAddress);
        }

        public static void EmitBackendFallback(ILogger logger, ServerHostingOptions options, string? requestedBackendInput)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (options == null) throw new ArgumentNullException(nameof(options));

            string? canonicalRequested = BackendCatalog.Canonicalize(requestedBackendInput);
            if (!string.Equals(options.DefaultBackend, canonicalRequested, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(options.DefaultBackend))
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "Requested default backend '{RequestedBackend}' is unavailable. Falling back to '{ResolvedBackend}'.",
                    requestedBackendInput, options.DefaultBackend);
            }
        }
    }
}
