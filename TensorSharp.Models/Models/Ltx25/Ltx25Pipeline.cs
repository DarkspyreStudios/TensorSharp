using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25Pipeline
{
    internal static GeneratedVideo Generate(Ltx25Model model, string prompt, VideoGenerationParams parameters)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var settings = Ltx25GenerationSettings.Resolve(parameters, model.SupportsReferenceConditioning);
        var timer = Stopwatch.StartNew();
        var phase = Stopwatch.StartNew();
        int completed = 0;
        void Report(string name, string detail)
        {
            parameters.CancellationToken.ThrowIfCancellationRequested();
            Console.WriteLine($"  [ltx25] {name}: {detail}");
            parameters.OnProgress?.Invoke(new VideoGenerationProgress
            {
                Phase = name, Detail = detail, Step = completed, TotalSteps = Ltx25GenerationSettings.Steps,
                ElapsedSeconds = timer.Elapsed.TotalSeconds,
            });
        }
        void EndPhase(string name)
        {
            Console.WriteLine($"  [ltx25-timing] {name}: {phase.Elapsed.TotalMilliseconds:F3}ms");
            phase.Restart();
        }
        void Trim()
        {
            if (model.DirectAllocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
            else if (model.DirectAllocator is MlxAllocator) MlxBackend.ClearCache();
        }
        try
        {
            bool identity = parameters.Mode == "ref";
            var images = identity ? Array.Empty<VideoImageCondition>() : Ltx25ImageConditioning.Resolve(parameters, settings.Frames);
            var encodedImages = new List<Ltx25EncodedImage>(images.Count);
            float[] encodedReference = null;
            int identitySourceCount = 0;
            if (identity)
            {
                var sources = Ltx25ReferenceConditioning.ResolveSources(parameters);
                identitySourceCount = sources.Count;
                Report("reference-encode", $"encoding {sources.Count} identity images as a {settings.Frames}-frame Ingredients reference sheet");
                RgbImage sheet = Ltx25ReferenceConditioning.BuildSheet(sources.Select(source => source.ResolveImage(preserveAlpha: true)).ToArray(), settings.Width, settings.Height);
                encodedReference = model.EncodeStaticReferenceVideo(sheet.Pixels, settings.Frames, settings.Height, settings.Width, parameters.CancellationToken);
                Trim(); EndPhase("reference-encode");
            }
            else if (images.Count > 0)
            {
                Report("image-encode", $"encoding {images.Count} temporal image guides");
                foreach (var image in images)
                {
                    parameters.CancellationToken.ThrowIfCancellationRequested();
                    RgbImage resized = TensorSharp.Models.QwenImage.ImageIO.ResizeCover(image.ResolveImage(), settings.Width, settings.Height);
                    encodedImages.Add(new(model.EncodeImage(resized.Pixels, settings.Height, settings.Width, parameters.CancellationToken),
                        image.FrameIndex, image.Strength));
                }
                Trim();
                EndPhase("image-encode");
            }
            Report("text-encode", "Gemma 4 and learned AV conditioning");
            Ltx25PromptEncoding conditioning = model.Conditioning(identity
                ? Ltx25ReferencePrompt.Format(prompt, identitySourceCount) : prompt, parameters.CancellationToken);
            Trim();
            EndPhase("text-encode");

            int guideTokens = identity ? settings.VideoTokens : checked(encodedImages.Count(image => image.FrameIndex != 0) * settings.LatentHeight * settings.LatentWidth);
            var noise = settings.InitialNoise(guideTokens);
            var input = new Ltx25DenoiserInput
            {
                Video = noise.Video, VideoTokens = checked(settings.VideoTokens + guideTokens),
                Audio = noise.Audio, AudioTokens = settings.AudioTokens,
                VideoContext = conditioning.VideoContext, VideoContextTokens = conditioning.Tokens,
                AudioContext = conditioning.AudioContext, AudioContextTokens = conditioning.Tokens,
                VideoPositions = Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps),
                AudioPositions = Ltx25Geometry.AudioPositions(settings.AudioTokens),
                VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
                KeyframeMask = Enumerable.Range(0, checked(settings.VideoTokens + guideTokens))
                    .Select(token => token < settings.LatentHeight * settings.LatentWidth ? 1f : 0f).ToArray(),
            };
            input = identity ? Ltx25ReferenceConditioning.Apply(input, settings, encodedReference,
                settings.LatentFrames, settings.LatentHeight, settings.LatentWidth) : Ltx25ImageConditioning.Apply(input, settings, encodedImages);
            Report("denoise", $"{settings.VideoTokens} video + {guideTokens} guide, {settings.AudioTokens} audio, {conditioning.Tokens} prompt tokens");
            Ltx25SamplingOutput latents = model.Sample(input, (step, total) =>
                {
                    completed = step;
                    parameters.OnStep?.Invoke(step, total);
                    Report("denoise", $"step {step}/{total}");
                }, parameters.CancellationToken);
            if (latents.BlocksPerStep != 48) throw new InvalidOperationException("LTX generation did not execute all transformer blocks.");
            Trim();
            EndPhase("denoise");

            Report("vae-decode", "convolutional video decoder");
            float[] pixels = model.DecodeVideo(Ltx25Geometry.UnpackVideo(latents.Video.AsSpan(0, checked(settings.VideoTokens * 128)), settings.VideoTokens, 128),
                settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, parameters.CancellationToken);
            Trim();
            int frameSize = checked(settings.Width * settings.Height * 3);
            if (pixels.Length != checked(frameSize * settings.Frames)) throw new InvalidOperationException("LTX video decoder returned an unexpected frame geometry.");
            var frames = new RgbImage[settings.Frames];
            for (int frame = 0; frame < frames.Length; ++frame)
            {
                var rgb = new float[frameSize];
                Array.Copy(pixels, frame * frameSize, rgb, 0, frameSize);
                foreach (float value in rgb)
                    if (!float.IsFinite(value) || value < 0 || value > 1) throw new InvalidOperationException("LTX video decoder returned invalid RGB pixels.");
                frames[frame] = new RgbImage(settings.Width, settings.Height, rgb);
            }
            EndPhase("vae-decode");

            GeneratedVideoAudio audio = null;
            if (parameters.GenerateAudio)
            {
                Report("audio-decode", "audio VAE, vocoder and 48 kHz bandwidth extension");
                audio = model.DecodeAudio(latents.Audio, settings.AudioTokens,
                    checked((int)Math.Round(settings.Frames / (double)settings.Fps * 48000, MidpointRounding.ToEven)),
                    parameters.CancellationToken);
                Trim();
                EndPhase("audio-decode");
            }
            Report("done", $"{settings.Width}x{settings.Height}, {settings.Frames} frames at {settings.Fps} fps");
            return new GeneratedVideo { Frames = frames, Fps = settings.Fps, Seed = settings.Seed, Audio = audio };
        }
        finally { Trim(); }
    }
}
