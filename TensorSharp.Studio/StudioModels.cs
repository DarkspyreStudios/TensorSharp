// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorSharp.Studio;

public static class StudioJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };
    public static T Clone<T>(T value) => JsonSerializer.Deserialize<T>(JsonSerializer.Serialize(value, Options), Options)!;
    public static string NewId() => Guid.NewGuid().ToString("N");
}

public sealed class StudioException(string code, string message) : Exception(message)
{
    public string Code { get; } = code;
}

/// <summary>Timeline timestamps are integer frames. Seconds convert using midpoint away from zero.
/// Media render sampling uses presentation timestamps and the project's rational output frame rate.</summary>
public sealed record StudioFrameRate(int Numerator = 24, int Denominator = 1)
{
    public double Value => (double)Numerator / Denominator;
    public double Seconds(long frame) => (double)frame * Denominator / Numerator;
    public long Frames(double seconds) => checked((long)Math.Round(seconds * Numerator / Denominator, MidpointRounding.AwayFromZero));
    public void Validate()
    {
        if (Numerator is < 1 or > 240000 || Denominator is < 1 or > 10000 || Value is < 1 or > 240)
            throw new StudioException("invalid_frame_rate", "Frame rate must be between 1 and 240 fps with positive rational terms.");
    }
}

public sealed class StudioProject
{
    public int SchemaVersion { get; set; } = 1;
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Untitled video";
    public long Revision { get; set; }
    public int Width { get; set; } = 1280;
    public int Height { get; set; } = 720;
    public StudioFrameRate FrameRate { get; set; } = new();
    public string? ConversationId { get; set; }
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
    public List<StudioAsset> Assets { get; set; } = [];
    public List<StudioTrack> Tracks { get; set; } = [];
    public List<StudioClip> Clips { get; set; } = [];
    public List<StudioCaption> Captions { get; set; } = [];
    public long DurationFrames => Math.Max(Clips.Select(c => checked(c.StartFrame + c.DurationFrames)).DefaultIfEmpty(0).Max(),
        Captions.Select(c => checked(c.StartFrame + c.DurationFrames)).DefaultIfEmpty(0).Max());
}

public sealed class StudioAsset
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Media";
    public string FileName { get; set; } = "";
    public string Kind { get; set; } = "video";
    public string ContentType { get; set; } = "video/mp4";
    public long Bytes { get; set; }
    public string Sha256 { get; set; } = "";
    public double DurationSeconds { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public bool HasAudio { get; set; }
    public string? SourceFrameRate { get; set; }
    public bool VariableFrameRate { get; set; }
    public bool Missing { get; set; }
    public StudioGenerationProvenance? Generation { get; set; }
}

public sealed class StudioTrack
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Video";
    public string Kind { get; set; } = "video";
    public bool Muted { get; set; }
    public bool Visible { get; set; } = true;
    public double Volume { get; set; } = 1;
}

public sealed class StudioClip
{
    public string Id { get; set; } = StudioJson.NewId();
    public string AssetId { get; set; } = "";
    public string TrackId { get; set; } = "";
    public long StartFrame { get; set; }
    public long SourceStartFrame { get; set; }
    public long DurationFrames { get; set; }
    public double Volume { get; set; } = 1;
    public long FadeInFrames { get; set; }
    public long FadeOutFrames { get; set; }
    /// <summary>Additional audio-only fade-out, used by a dissolve while the outgoing picture remains opaque.</summary>
    public long AudioFadeOutFrames { get; set; }
    /// <summary>Clips with the same non-empty group move, trim, and split together.</summary>
    public string? LinkGroup { get; set; }
    public string Fit { get; set; } = "contain";
    public string? ShotId { get; set; }
}

public sealed class StudioCaption
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Text { get; set; } = "";
    public long StartFrame { get; set; }
    public long DurationFrames { get; set; }
}

/// <summary>One command shape for UI and agents. Only fields used by the selected Type are accepted semantically.</summary>
public sealed class StudioCommand
{
    public string Type { get; set; } = "";
    public string? ConversationId { get; set; }
    public string? TrackId { get; set; }
    public string? ClipId { get; set; }
    public string? OtherClipId { get; set; }
    public string? AssetId { get; set; }
    public string? CaptionId { get; set; }
    public string? Name { get; set; }
    public string? Kind { get; set; }
    public string? Text { get; set; }
    public string? LinkGroup { get; set; }
    public string? ShotId { get; set; }
    public string? Fit { get; set; }
    public long? StartFrame { get; set; }
    public long? SourceStartFrame { get; set; }
    public long? DurationFrames { get; set; }
    public long? AtFrame { get; set; }
    public long? FadeInFrames { get; set; }
    public long? FadeOutFrames { get; set; }
    public long? AudioFadeOutFrames { get; set; }
    public double? Volume { get; set; }
    public bool? Muted { get; set; }
    public bool? Visible { get; set; }
    /// <summary>Explicit confirmation to remove timeline uses when deleting an asset from the library.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? RemoveTimelineClips { get; set; }
    public int? Index { get; set; }
    public int? Width { get; set; }
    public int? Height { get; set; }
    public bool ApplyToLinked { get; set; } = true;
}

public sealed class StudioProposal
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public long ExpectedRevision { get; set; }
    public string Summary { get; set; } = "";
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public List<StudioCommand> Commands { get; set; } = [];
}

public sealed class StudioGenerationRequest
{
    public string ProjectId { get; set; } = "";
    public string RequestId { get; set; } = StudioJson.NewId();
    public string ModelId { get; set; } = "";
    public string Prompt { get; set; } = "";
    public string? NegativePrompt { get; set; }
    /// <summary>Zero uses the selected model's host-configured generation size.</summary>
    public int Width { get; set; }
    public int Height { get; set; }
    public int FrameCount { get; set; }
    public int Fps { get; set; }
    public int Steps { get; set; }
    public float CfgScale { get; set; }
    public long Seed { get; set; } = -1;
    /// <summary>Null follows the model default; explicit false retains silent generation.</summary>
    public bool? GenerateAudio { get; set; }
    public string? ConditioningAssetId { get; set; }
    public string? EndConditioningAssetId { get; set; }
    public string? ShotId { get; set; }
    public List<StudioTemporalCondition> TemporalConditions { get; set; } = [];
    public List<string> ReferenceAssetIds { get; set; } = [];
}

public sealed record StudioTemporalCondition(string AssetId, int FrameIndex, float Strength = 1);

public sealed class StudioGenerationProvenance
{
    public StudioGenerationRequest Request { get; set; } = new();
    public string ModelVersion { get; set; } = "";
    public Dictionary<string, string> EffectiveParameters { get; set; } = [];
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}

public sealed class StudioModelCapability
{
    public string ModelId { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public string Family { get; set; } = "";
    public bool Available { get; set; }
    public string? UnavailableReason { get; set; }
    public bool TextToVideo { get; set; }
    public bool ImageConditioning { get; set; }
    public bool LastFrameConditioning { get; set; }
    public bool AudioOutput { get; set; }
    public bool TemporalGuidance { get; set; }
    public bool IdentityReferences { get; set; }
    public int MaxReferences { get; set; }
    public int DefaultWidth { get; set; } = 512;
    public int DefaultHeight { get; set; } = 512;
    public int DefaultFps { get; set; } = 24;
    public bool DefaultGenerateAudio { get; set; }
    public int DefaultFrameCount { get; set; } = 17;
    public int DefaultSteps { get; set; } = 20;
    public int? FixedSteps { get; set; }
    public float DefaultCfgScale { get; set; } = 5;
    public float? FixedCfgScale { get; set; }
    public string[] DeferredCapabilities { get; set; } = [];
    public int SpatialMultiple { get; set; } = 16;
    public int TemporalMultiple { get; set; } = 4;
    public int TemporalOffset { get; set; } = 1;
    public int? FixedFps { get; set; }
    public int MaxFrames { get; set; } = 257;
    public int MaxDimension { get; set; } = 2048;
}

public sealed record StudioGenerationResult(string FilePath, string ModelVersion, Dictionary<string, string> EffectiveParameters);
public sealed record StudioProgress(string Stage, double Fraction, string? Detail = null);

/// <summary>Host adapter must use its existing model registry/resource leases. Output must be inside outputDirectory.
/// Cancellation must not return until inference and subprocesses have stopped using files or buffers.</summary>
public interface IStudioGenerationAdapter
{
    Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken cancellationToken);
    Task ValidateAsync(StudioGenerationRequest request, IReadOnlyDictionary<string, string> conditioningAssets, CancellationToken cancellationToken);
    Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest request, string outputDirectory,
        IReadOnlyDictionary<string, string> conditioningAssets, IProgress<StudioProgress> progress, CancellationToken cancellationToken);
}

public sealed class StudioJob
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public string RequestId { get; set; } = "";
    public string Kind { get; set; } = "render";
    public string State { get; set; } = "queued";
    public string Stage { get; set; } = "queued";
    public double Progress { get; set; }
    public long ProjectRevision { get; set; }
    public string? Error { get; set; }
    public string? OutputAssetId { get; set; }
    public string? OutputFileName { get; set; }
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
}

public sealed class StudioOptions
{
    public string FfmpegPath { get; set; } = "ffmpeg";
    public string FfprobePath { get; set; } = "ffprobe";
    public long MaxImportBytes { get; set; } = 2L * 1024 * 1024 * 1024;
    public int MaxQueuedJobs { get; set; } = 8;
    public int MaxHistory { get; set; } = 40;
    public long MaxTimelineFrames { get; set; } = 24 * 60 * 60;
    public long MinimumFreeDiskBytes { get; set; } = 256L * 1024 * 1024;
    public bool EnableExternalProcesses { get; set; } = !OperatingSystem.IsIOS() && !OperatingSystem.IsTvOS() && !OperatingSystem.IsAndroid();
    public string? CaptionFontPath { get; set; }
}
