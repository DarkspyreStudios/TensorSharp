// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;

namespace TensorSharp.Models.Ltx25;

/// <summary>Token-major Float32 AV latents after the specified Euler schedule.
/// This is a denoiser-component result; no text encoding or VAE decode has run.
/// BlocksPerStep records diagnostic partial execution as well as full execution.</summary>
public sealed record Ltx25SamplingOutput(float[] Video, float[] Audio, int Steps, int BlocksPerStep);

/// <summary>Batch-one, positive-only AV Euler sampling over already
/// projected contexts and caller-supplied initial latents. Follows the original
/// X0Model and EulerDiffusionStep Float32 arithmetic. This does not implement a
/// video-generation pipeline, noise RNG, CFG/STG or VAE decoding. Optional video
/// conditioning follows X0Model, post_process_latent and EulerDiffusionStep.</summary>
public static class Ltx25Sampling
{
    /// <summary>Run the complete denoiser at each sigma. The schedule replaces
    /// the template's video/audio sigma together. Input arrays remain unchanged.</summary>
    public static Ltx25SamplingOutput RunEuler(Ltx25Denoiser denoiser, Ltx25DenoiserInput initial,
        ReadOnlySpan<float> sigmas, Action<int, int> onStep = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(denoiser);
        return RunEuler(input => denoiser.Forward(input), initial, sigmas, onStep, cancellationToken);
    }

    /// <summary>The callback receives both streams before either is advanced.
    /// Its output reports the actual block count, which must stay constant across
    /// the schedule. A diagnostic callback may run a partial checkpoint; the result
    /// reports that count and cannot establish whole-model generation quality.</summary>
    public static Ltx25SamplingOutput RunEuler(Func<Ltx25DenoiserInput, Ltx25DenoiserOutput> predict,
        Ltx25DenoiserInput initial, ReadOnlySpan<float> sigmas, Action<int, int> onStep = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(predict);
        ArgumentNullException.ThrowIfNull(initial);
        if (initial.VideoTimesteps != null || initial.AudioTimesteps != null)
            throw new NotSupportedException("LTX Euler assigns timesteps at each sigma; supply VideoDenoiseMask and CleanVideo for image conditioning instead of fixed timestep overrides.");
        ValidateRows(initial.Video, initial.VideoTokens);
        ValidateRows(initial.Audio, initial.AudioTokens);
        ValidateSchedule(sigmas);
        if (initial.KeyframeMask != null && (initial.KeyframeMask.Length != initial.VideoTokens ||
            Array.Exists(initial.KeyframeMask, value => !float.IsFinite(value))))
            throw new ArgumentException("LTX keyframe markers must match video tokens and be finite.");
        float[] mask = initial.VideoDenoiseMask, clean = initial.CleanVideo;
        if ((mask == null) != (clean == null) || mask != null &&
            (mask.Length != initial.VideoTokens || clean.Length != initial.Video.Length ||
             Array.Exists(mask, value => !float.IsFinite(value) || value < 0 || value > 1) ||
             Array.Exists(clean, value => !float.IsFinite(value))))
            throw new ArgumentException("LTX conditioning requires matching clean video latents and a [0,1] mask per token.");

        var video = (float[])initial.Video.Clone();
        var audio = (float[])initial.Audio.Clone();
        int steps = sigmas.Length - 1, blocks = 0;
        for (int step = 0; step < steps; ++step)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float currentSigma = sigmas[step];
            var input = new Ltx25DenoiserInput
            {
                Video = video, VideoTokens = initial.VideoTokens,
                Audio = audio, AudioTokens = initial.AudioTokens,
                VideoContext = initial.VideoContext, VideoContextTokens = initial.VideoContextTokens,
                AudioContext = initial.AudioContext, AudioContextTokens = initial.AudioContextTokens,
                VideoPositions = initial.VideoPositions, AudioPositions = initial.AudioPositions,
                FrequencyPrecision = initial.FrequencyPrecision,
                VideoSigma = sigmas[step], AudioSigma = sigmas[step],
                VideoTimesteps = mask == null ? null : Array.ConvertAll(mask, value => value * currentSigma),
                KeyframeMask = initial.KeyframeMask,
                UseReferenceLora = initial.UseReferenceLora,
                CancellationToken = cancellationToken,
            };
            Ltx25DenoiserOutput prediction = predict(input)
                ?? throw new InvalidOperationException("LTX sampling callback returned no prediction.");
            cancellationToken.ThrowIfCancellationRequested();
            if (prediction.ExecutedBlocks <= 0 || blocks != 0 && blocks != prediction.ExecutedBlocks)
                throw new InvalidOperationException("LTX sampling requires a consistent positive executed-block count.");
            blocks = prediction.ExecutedBlocks;
            float[] nextVideo = Step(video, prediction.VideoVelocity, sigmas[step], sigmas[step + 1], mask, clean, initial.VideoTokens);
            float[] nextAudio = Step(audio, prediction.AudioVelocity, sigmas[step], sigmas[step + 1]);
            video = nextVideo;
            audio = nextAudio;
            onStep?.Invoke(step + 1, steps);
        }
        return new(video, audio, steps, blocks);
    }

    private static float[] Step(float[] sample, float[] velocity, float sigma, float nextSigma,
        float[] mask = null, float[] clean = null, int tokens = 0)
    {
        if (velocity == null || velocity.Length != sample.Length)
            throw new InvalidOperationException("LTX sampling velocity dimensions differ from the latent stream.");
        var output = new float[sample.Length];
        for (int i = 0; i < output.Length; ++i)
        {
            if (!float.IsFinite(velocity[i]))
                throw new InvalidOperationException("LTX sampling velocity must be finite.");
            // Preserve the original Float32 X0Model -> EulerDiffusionStep
            // round-trip instead of silently fusing away its rounding behavior.
            float denoise = mask == null ? 1 : mask[i / (sample.Length / tokens)];
            float denoised = sample[i] - velocity[i] * (sigma * denoise);
            if (mask != null) denoised = denoised * denoise + clean[i] * (1 - denoise);
            float derivative = (sample[i] - denoised) / sigma;
            output[i] = sample[i] + derivative * (nextSigma - sigma);
            if (!float.IsFinite(output[i]))
                throw new InvalidOperationException("LTX sampling produced a nonfinite latent.");
        }
        return output;
    }

    private static void ValidateRows(float[] values, int tokens)
    {
        if (values == null || tokens <= 0 || values.Length == 0 || values.Length % tokens != 0)
            throw new ArgumentException("LTX sampling requires nonempty token-major AV latents.");
        foreach (float value in values)
            if (!float.IsFinite(value)) throw new ArgumentException("LTX initial latents must be finite.");
    }

    private static void ValidateSchedule(ReadOnlySpan<float> sigmas)
    {
        if (sigmas.Length < 2 || sigmas[^1] != 0)
            throw new ArgumentException("LTX Euler schedule must contain at least one positive sigma and finish at zero.");
        for (int i = 0; i < sigmas.Length - 1; ++i)
            if (!float.IsFinite(sigmas[i]) || sigmas[i] <= 0 || sigmas[i] > 1 || sigmas[i + 1] > sigmas[i])
                throw new ArgumentException("LTX Euler sigmas must be finite, positive and nonincreasing before the terminal zero.");
    }
}
