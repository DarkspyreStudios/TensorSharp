// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server;
using TensorSharp.Studio;

namespace TensorSharp.Chat.Studio;

/// <summary>Uses the host's active model and existing chat video pipeline. Never loads or substitutes a model.</summary>
public sealed class NativeStudioGenerationAdapter : IStudioGenerationAdapter
{
    private static readonly ConcurrentDictionary<string, string> ModelHashes = new(StringComparer.Ordinal);
    private readonly ModelService _models;
    private readonly WebUiChatService _chat;
    private readonly Func<CancellationToken, Task> _prepare;
    private readonly TensorSharp.Runtime.Scheduling.ComputeGate _compute;
    public NativeStudioGenerationAdapter(ModelService models, WebUiChatService chat, Func<CancellationToken, Task> prepare = null, TensorSharp.Runtime.Scheduling.ComputeGate compute = null)
    { _models = models; _chat = chat; _prepare = prepare; _compute = compute; }

    public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_models.Model is not IVideoGenerationModel video)
            return Task.FromResult<IReadOnlyList<StudioModelCapability>>(Array.Empty<StudioModelCapability>());
        return Task.FromResult<IReadOnlyList<StudioModelCapability>>(new[] { CreateCapability(video, _models.LoadedModelName, _chat.VideoDefaults) });
    }

    internal static StudioModelCapability CreateCapability(IVideoGenerationModel video, string modelId, VideoGenerationParams defaults)
    {
        var capability = new StudioModelCapability
        {
            ModelId = modelId, DisplayName = modelId,
            Family = video.VideoModelFamily, Available = true, TextToVideo = true,
            ImageConditioning = video.SupportsImageConditioning, LastFrameConditioning = video.SupportsEndImageConditioning,
            AudioOutput = video.SupportsAudio, DefaultFrameCount = video.DefaultPreviewFrames,
            TemporalGuidance = video.SupportsKeyframeConditioning, IdentityReferences = video.SupportsReferenceConditioning, MaxReferences = video.MaxReferenceImages,
            DeferredCapabilities = new[] { "LoRA selection, native retake and extension are not exposed by the Studio generation adapter. Existing model APIs remain available." }
        };
        switch (video)
        {
            case Ltx25Model:
                capability.SpatialMultiple = 64; capability.TemporalMultiple = 8;
                capability.DefaultWidth = capability.DefaultHeight = Ltx25GenerationSettings.DefaultSize;
                capability.DefaultSteps = 8; capability.FixedSteps = 8;
                capability.DefaultCfgScale = 1; capability.FixedCfgScale = 1;
                break;
            case MiniMaxH3Model h3:
                capability.SpatialMultiple = 32; capability.TemporalMultiple = 17; capability.TemporalOffset = 5;
                capability.DefaultWidth = 640; capability.DefaultHeight = 384;
                capability.ImageConditioning = h3.Partition != MiniMaxH3Partition.Reference;
                capability.FixedFps = 24; capability.DefaultCfgScale = 1; capability.FixedCfgScale = 1;
                break;
            case WanVideoModel wan:
                capability.SpatialMultiple = wan.Variant == WanVariant.TI2V ? 32 : 16;
                capability.DefaultWidth = wan.Variant == WanVariant.TI2V ? 1280 : 832;
                capability.DefaultHeight = wan.Variant == WanVariant.TI2V ? 704 : 480;
                capability.DefaultFps = wan.Variant == WanVariant.TI2V ? 24 : 16;
                capability.ImageConditioning = wan.SupportsImage; capability.TextToVideo = !wan.RequiresImage;
                capability.DefaultSteps = wan.DistilledSteps > 0 ? wan.DistilledSteps : wan.Variant == WanVariant.TI2V ? 50 : wan.Variant == WanVariant.A14B ? 40 : 30;
                capability.DefaultCfgScale = wan.DistilledSteps > 0 ? 1 : 5;
                break;
            default:
                capability.Available = false;
                capability.UnavailableReason = "This model family has no validated Studio parameter adapter yet.";
                break;
        }
        // Studio must inherit the same host-configured recipe as chat. In particular,
        // overriding LTX's 1024 output with a generic 512 canvas reduces its first
        // denoising pass to 256 pixels, destroying fine face and hand detail.
        if (defaults.Width > 0) capability.DefaultWidth = defaults.Width;
        if (defaults.Height > 0) capability.DefaultHeight = defaults.Height;
        // These families accept arbitrary host dimensions and align them inside
        // native generation. Advertise that effective canvas so Studio's stricter
        // admission accepts its own defaults. LTX deliberately rejects off-grid sizes.
        if (video is MiniMaxH3Model)
        {
            // Without a conditioning image, H3 uses a square for a one-sided
            // size override. An image's aspect cannot be known at capability time.
            if (defaults.Width > 0 && defaults.Height <= 0) capability.DefaultHeight = defaults.Width;
            else if (defaults.Height > 0 && defaults.Width <= 0) capability.DefaultWidth = defaults.Height;
            capability.DefaultWidth = MiniMaxH3Geometry.AlignSpatial(capability.DefaultWidth);
            capability.DefaultHeight = MiniMaxH3Geometry.AlignSpatial(capability.DefaultHeight);
        }
        else if (video is WanVideoModel)
        {
            int grid = capability.SpatialMultiple;
            capability.DefaultWidth = Math.Max(grid, (capability.DefaultWidth + grid / 2) / grid * grid);
            capability.DefaultHeight = Math.Max(grid, (capability.DefaultHeight + grid / 2) / grid * grid);
        }
        if (defaults.Fps > 0) capability.DefaultFps = defaults.Fps;
        if (capability.FixedFps is int fixedFps) capability.DefaultFps = fixedFps;
        if (defaults.Steps > 0) capability.DefaultSteps = defaults.Steps;
        if (capability.FixedSteps is int fixedSteps) capability.DefaultSteps = fixedSteps;
        capability.DefaultGenerateAudio = capability.AudioOutput && defaults.GenerateAudio;
        if (defaults.Frames > 0)
        {
            // Chat's composer snaps duration to the model grid before generation.
            // Expose that effective count, so e.g. --video-frames 144 is usable as 145.
            int frames = defaults.Frames;
            capability.DefaultFrameCount = video switch
            {
                MiniMaxH3Model => MiniMaxH3Geometry.AlignFrames(frames),
                WanVideoModel => Math.Max(1, (frames + 2) / 4 * 4 + 1),
                _ => capability.TemporalOffset + Math.Max(0, checked(frames - capability.TemporalOffset + capability.TemporalMultiple - 1)) / capability.TemporalMultiple * capability.TemporalMultiple
            };
        }
        return capability;
    }

    public async Task ValidateAsync(StudioGenerationRequest request, IReadOnlyDictionary<string, string> conditioningAssets, CancellationToken cancellationToken)
    {
        string conditioningPath = Resolve(conditioningAssets, request.ConditioningAssetId);
        string endConditioningPath = Resolve(conditioningAssets, request.EndConditioningAssetId);
        var capabilities = await GetCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        var capability = capabilities.SingleOrDefault(c => c.ModelId == request.ModelId);
        if (capability == null || !capability.Available)
            throw new StudioException("model_unavailable", capability?.UnavailableReason ?? "The requested video model is no longer loaded. Select and load it explicitly before retrying.");
        if (request.Width == 0) request.Width = capability.DefaultWidth;
        if (request.Height == 0) request.Height = capability.DefaultHeight;
        if (request.Fps == 0) request.Fps = capability.DefaultFps;
        request.GenerateAudio ??= capability.DefaultGenerateAudio;
        if (request.Steps == 0) request.Steps = capability.DefaultSteps;
        if (request.CfgScale == 0) request.CfgScale = capability.DefaultCfgScale;
        if (request.FrameCount == 0) request.FrameCount = capability.DefaultFrameCount;
        if (capability.FixedSteps is int fixedSteps && request.Steps != fixedSteps || capability.FixedCfgScale is float fixedCfg && request.CfgScale != fixedCfg)
            throw new StudioException("invalid_generation", "This checkpoint requires its fixed sampling steps and guidance. Refresh the model's capability settings.");
        if (request.Width <= 0 || request.Height <= 0 || request.Width % capability.SpatialMultiple != 0 || request.Height % capability.SpatialMultiple != 0 || request.Width > capability.MaxDimension || request.Height > capability.MaxDimension)
            throw new StudioException("invalid_generation", $"Dimensions must be positive multiples of {capability.SpatialMultiple}, at most {capability.MaxDimension}.");
        if (request.FrameCount < capability.TemporalOffset || (request.FrameCount - capability.TemporalOffset) % capability.TemporalMultiple != 0 || request.FrameCount > capability.MaxFrames)
            throw new StudioException("invalid_generation", $"Frames must follow {capability.TemporalMultiple}n+{capability.TemporalOffset}, at most {capability.MaxFrames}.");
        if (request.Fps is < 1 or > 240 || capability.FixedFps is int fps && request.Fps != fps || request.Steps is < 1 or > 100 || !float.IsFinite(request.CfgScale) || request.CfgScale is < 1 or > 30)
            throw new StudioException("invalid_generation", "Unsupported frame rate, sampling steps, or guidance.");
        if (conditioningPath != null && !capability.ImageConditioning || endConditioningPath != null && !capability.LastFrameConditioning || request.GenerateAudio == true && !capability.AudioOutput)
            throw new StudioException("unsupported_conditioning", "The loaded checkpoint does not support the requested conditioning or audio output.");
        if (!capability.TextToVideo && conditioningPath == null)
            throw new StudioException("conditioning_required", "This checkpoint requires a first-frame image.");
        if (endConditioningPath != null && conditioningPath == null)
            throw new StudioException("conditioning_required", "Last-frame conditioning requires a first-frame image.");
        if (request.CfgScale == 1 && !string.IsNullOrWhiteSpace(request.NegativePrompt))
            throw new StudioException("unsupported_negative_prompt", "Negative prompts do not run at CFG 1. Clear the negative prompt rather than silently ignoring it.");
        if (request.TemporalConditions.Count > 0 && !capability.TemporalGuidance || request.ReferenceAssetIds.Count > 0 && !capability.IdentityReferences)
            throw new StudioException("unsupported_conditioning", "The loaded checkpoint does not support these temporal guides or identity references.");
        if (request.TemporalConditions.Count + request.ReferenceAssetIds.Count > Math.Max(0, capability.MaxReferences))
            throw new StudioException("invalid_conditioning", $"This model accepts at most {capability.MaxReferences} guides/references.");
        if (request.ReferenceAssetIds.Count > 0 && (request.TemporalConditions.Count > 0 || conditioningPath != null || endConditioningPath != null))
            throw new StudioException("invalid_conditioning", "Identity references cannot be combined with temporal image guidance.");
        foreach (var guide in request.TemporalConditions)
            if (guide.FrameIndex < -1 || guide.FrameIndex >= request.FrameCount || !float.IsFinite(guide.Strength) || guide.Strength is < 0 or > 1)
                throw new StudioException("invalid_conditioning", "Temporal guide position or strength is invalid.");
        if (_models.Model is Ltx25Model ltx)
            _ = Ltx25GenerationSettings.Resolve(Parameters(request, conditioningAssets, null, cancellationToken), ltx.SupportsReferenceConditioning);
        if (_models.Model is MiniMaxH3Model h3)
            _ = MiniMaxH3ModeResolver.Resolve(Parameters(request, conditioningAssets, null, cancellationToken), h3.Partition);
    }

    public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest request, string outputDirectory,
        IReadOnlyDictionary<string, string> conditioningAssets, IProgress<StudioProgress> progress, CancellationToken cancellationToken)
    {
        if (_prepare != null) await _prepare(cancellationToken).ConfigureAwait(false);
        var expectedModel = _models.Model as IVideoGenerationModel;
        string expectedName = _models.LoadedModelName;
        string expectedPath = _models.LoadedModelPath;
        string expectedBackend = _models.LoadedBackend;
        await ValidateAsync(request, conditioningAssets, cancellationToken).ConfigureAwait(false);
        if (expectedModel == null || expectedName != request.ModelId || !ReferenceEquals(expectedModel, _models.Model))
            throw new StudioException("model_changed", "The selected model changed during admission. Select it explicitly and retry.");
        using var interrupted = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        void ForegroundChanged(bool open) { if (!open) interrupted.Cancel(); }
        if (_compute != null) _compute.Changed += ForegroundChanged;
        try
        {
            if (_compute != null && !_compute.IsOpen) interrupted.Cancel();
            cancellationToken = interrupted.Token;
            string identity = expectedName + "; backend=" + expectedBackend;
            var source = new FileInfo(expectedPath);
            string cacheKey = source.FullName + "|" + source.Length + "|" + source.LastWriteTimeUtc.Ticks;
            if (!ModelHashes.TryGetValue(cacheKey, out string checksum))
            {
                progress.Report(new StudioProgress("model-identity", 0, "Computing the installed checkpoint SHA-256 once for generation provenance."));
                await using var modelFile = new FileStream(source.FullName, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
                checksum = Convert.ToHexStringLower(await SHA256.HashDataAsync(modelFile, cancellationToken).ConfigureAwait(false));
                ModelHashes.TryAdd(cacheKey, checksum);
            }
            identity += "; sha256=" + checksum + "; bytes=" + source.Length.ToString(CultureInfo.InvariantCulture);
            if (expectedModel is MiniMaxH3Model h3 && h3.Partition == MiniMaxH3Partition.Reference)
                identity += "; conditioning=identity references";
            var parameters = Parameters(request, conditioningAssets, progress, cancellationToken);
            var result = await _chat.GenerateVideoAsync(request.Prompt, parameters, expectedModel).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            if (result.Width != request.Width || result.Height != request.Height || result.Frames != request.FrameCount || result.Fps != request.Fps)
                throw new StudioException("unexpected_output_geometry", "Native generation returned different dimensions, frame count, or frame rate. The output was not accepted into the project; inspect model capabilities before retrying.");
            if (request.GenerateAudio == true && !result.AudioMuxed)
                throw new StudioException("audio_not_muxed", "The native encoder produced separate audio. Studio requires muxed audio to preserve preview/export parity; use an available local muxing encoder.");
            Directory.CreateDirectory(outputDirectory);
            string path = Path.Combine(outputDirectory, "generated.mp4");
            File.Copy(result.OutputPath, path, overwrite: false);
            return new StudioGenerationResult(path, identity, new Dictionary<string, string>
            {
                ["width"] = result.Width.ToString(CultureInfo.InvariantCulture), ["height"] = result.Height.ToString(CultureInfo.InvariantCulture),
                ["frames"] = result.Frames.ToString(CultureInfo.InvariantCulture), ["fps"] = result.Fps.ToString(CultureInfo.InvariantCulture),
                ["seed"] = result.Seed.ToString(CultureInfo.InvariantCulture), ["steps"] = request.Steps.ToString(CultureInfo.InvariantCulture),
                ["cfgScale"] = request.CfgScale.ToString(CultureInfo.InvariantCulture), ["codec"] = result.Codec,
                ["audioMuxed"] = result.AudioMuxed.ToString(), ["mode"] = parameters.Mode,
                ["recipe"] = expectedModel is Ltx25Model ? request.ReferenceAssetIds.Count > 0 ? "LTX Ingredients: 8 full-size steps; euler_ancestral" : "LTX distilled: 8 half-size steps + 3 full-size refinement steps; euler_ancestral" : "native model sampling"
            });
        }
        finally { if (_compute != null) _compute.Changed -= ForegroundChanged; }
    }

    private static string Resolve(IReadOnlyDictionary<string, string> assets, string id) => id == null ? null :
        assets.TryGetValue(id, out var path) ? path : throw new StudioException("conditioning_not_found", "Conditioning asset is not available in this project.");

    internal static VideoGenerationParams Parameters(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct) => new()
    {
        Width = r.Width, Height = r.Height, Frames = r.FrameCount, Fps = r.Fps, Steps = r.Steps,
        CfgScale = r.CfgScale, Seed = r.Seed, NegativePrompt = r.NegativePrompt,
        ImagePath = Resolve(assets, r.ConditioningAssetId), EndImagePath = Resolve(assets, r.EndConditioningAssetId),
        Mode = r.ReferenceAssetIds.Count > 0 ? "ref" : r.TemporalConditions.Count > 0 ? "keyframes" : r.EndConditioningAssetId != null ? "fl2v" : r.ConditioningAssetId != null ? "i2v" : "t2v",
        ReferenceImagePaths = r.ReferenceAssetIds.Select(id => Resolve(assets, id)).ToList(),
        ImageConditions = r.TemporalConditions.Select(c => new VideoImageCondition { ImagePath = Resolve(assets, c.AssetId), FrameIndex = c.FrameIndex, Strength = c.Strength }).ToList(),
        GenerateAudio = r.GenerateAudio ?? true, CancellationToken = ct,
        OnProgress = p => progress?.Report(new StudioProgress(p.Phase ?? "inference", p.TotalSteps > 0 ? Math.Min(.95, .1 + .8 * p.Step / p.TotalSteps) : 0, p.Detail))
    };
}
