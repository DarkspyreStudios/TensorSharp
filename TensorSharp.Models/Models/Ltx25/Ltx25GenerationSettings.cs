using System;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Ltx25;

/// <summary>The single-stage distilled text/image-to-video/audio recipe.</summary>
public sealed record Ltx25GenerationSettings(int Width, int Height, int Frames, int Fps,
    int LatentWidth, int LatentHeight, int LatentFrames, int AudioTokens, long Seed)
{
    public const int Steps = 8;
    public int VideoTokens => checked(LatentWidth * LatentHeight * LatentFrames);

    public static Ltx25GenerationSettings Resolve(VideoGenerationParams parameters, bool supportsIdentityReferences = false)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (parameters.ReferenceVideoPaths is { Count: > 0 } || parameters.ReferenceAudioPaths is { Count: > 0 } ||
            parameters.ReferenceVideoAudioPaths is { Count: > 0 })
            throw new NotSupportedException("LTX video/audio reference workflows are not implemented by this distilled image preset; use temporal image guides or Ingredients identity images.");
        bool identity = parameters.Mode == "ref";
        if (identity && !supportsIdentityReferences)
            throw new NotSupportedException("LTX identity/appearance references require the separate Ingredients IC-LoRA. This preset accepts temporal image guides in i2v, fl2v or keyframes mode.");
        if (!string.IsNullOrWhiteSpace(parameters.Mode) && parameters.Mode is not ("t2v" or "t2va" or "i2v" or "fl2v" or "keyframes" or "ref"))
            throw new NotSupportedException("LTX 2.5 generation mode must be t2v, t2va, i2v, fl2v, keyframes or Ingredients ref.");
        if (parameters.Steps is not (0 or Steps))
            throw new ArgumentException("LTX 2.5 Distilled requires its published eight-step schedule.");
        if (parameters.CfgScale is not (0 or 1))
            throw new ArgumentException("LTX 2.5 Distilled uses CFG 1; classifier-free guidance is not enabled.");
        if (!string.IsNullOrWhiteSpace(parameters.Sampler) && !parameters.Sampler.Equals("euler", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("LTX 2.5 Distilled uses the Euler sampler.");
        if (parameters.FlowShift != 0)
            throw new ArgumentException("LTX 2.5 Distilled uses fixed sigmas and does not accept a flow-shift override.");
        int width = parameters.Width == 0 ? identity ? 768 : 512 : parameters.Width;
        int height = parameters.Height == 0 ? identity ? 448 : 512 : parameters.Height;
        int frames = parameters.Frames == 0 ? identity ? 121 : 17 : parameters.Frames;
        int fps = parameters.Fps == 0 ? 24 : parameters.Fps;
        if (fps <= 0 || fps > 240) throw new ArgumentOutOfRangeException(nameof(parameters.Fps));
        var latent = Ltx25Geometry.VideoLatentShape(frames, height, width);
        if (identity)
        {
            if (frames < Ltx25ReferenceConditioning.MinimumFrames)
                throw new ArgumentException("Ingredients identity generation requires at least 121 frames on the 8n+1 temporal grid.");
            _ = Ltx25ReferenceConditioning.ResolveSources(parameters);
        }
        else _ = Ltx25ImageConditioning.Resolve(parameters, frames);
        // Original AudioLatentShape.from_duration uses Python's ties-to-even
        // round, rather than SD.cpp's ceil. Keep the original latent geometry.
        int audioTokens = Math.Max(1, checked((int)Math.Round(frames / (double)fps * 25, MidpointRounding.ToEven)));
        long seed = parameters.Seed < 0 ? Random.Shared.NextInt64(long.MaxValue) : parameters.Seed;
        return new(width, height, frames, fps, latent.Width, latent.Height, latent.Frames, audioTokens, seed);
    }

    /// <summary>CUDA Philox noise in the baseline's packed channel-major storage.
    /// One complete padded AV draw preserves its counter offsets, then the two
    /// streams are rearranged into the denoiser's token-major layout.</summary>
    public (float[] Video, float[] Audio) InitialNoise(int extraVideoTokens = 0)
    {
        if (extraVideoTokens < 0) throw new ArgumentOutOfRangeException(nameof(extraVideoTokens));
        int tokens = checked(VideoTokens + extraVideoTokens);
        const int videoChannels = 128, audioChannels = 8, audioBins = 16;
        int videoLength = checked(tokens * videoChannels);
        int audioLength = checked(AudioTokens * audioChannels * audioBins);
        int extraChannels = checked((audioLength + tokens - 1) / tokens);
        float[] packed = QwenImage21Sampling.Noise(checked(tokens * (videoChannels + extraChannels)), Seed);
        float[] video = Ltx25Geometry.PackVideo(packed.AsSpan(0, videoLength), tokens, videoChannels);
        float[] audio = Ltx25Geometry.PackAudio(packed.AsSpan(videoLength, audioLength), AudioTokens, audioChannels, audioBins);
        return (video, audio);
    }
}
