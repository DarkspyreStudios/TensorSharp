using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Native LTX 2.5 Distilled text/image-to-video with joint stereo audio.</summary>
public sealed class Ltx25Model : ModelBase, IVideoGenerationModel
{
    public const string ArchitectureId = "ltx-2.5";
    internal string CheckpointPath { get; }
    internal string TextEncoderPath { get; }
    internal string VideoVaePath { get; }
    internal string AudioVaePath { get; }
    internal string ReferenceLoraPath { get; }
    internal IAllocator DirectAllocator => _allocator;
    private readonly object _generationLock = new();
    private readonly Dictionary<string, (Ltx25PromptEncoding Encoding, LinkedListNode<string> Node)> _promptCache = new(StringComparer.Ordinal);
    private readonly LinkedList<string> _promptOrder = new();
    private Ltx25Denoiser _denoiser;
    private Ltx25PromptEncoder _promptEncoder;
    private Ltx25VideoVaeDecoder _videoDecoder;
    private Ltx25VideoVaeEncoder _videoEncoder;
    private Ltx25AudioDecoder _audioDecoder;
    public long PromptCacheHits { get; private set; }
    private bool _disposed;

    public Ltx25Model(string checkpoint, BackendType backend) : base(checkpoint, DirectBackend(backend))
    {
        try
        {
            var geometry = Ltx25Config.FromGguf(_gguf);
            if (geometry.VideoDim != 4096 || geometry.AudioDim != 2048 || geometry.Layers != 48 ||
                geometry.VideoChannels != 128 || geometry.AudioChannels != 128)
                throw new NotSupportedException("LTX generation requires the authentic 48-layer LTX 2.5 AV checkpoint.");
            CheckpointPath = Path.GetFullPath(checkpoint);
            string directory = Path.GetDirectoryName(CheckpointPath);
            TextEncoderPath = Companion(directory, "TS_VIDEO_TEXT_ENCODER", "text_encoders", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
            VideoVaePath = Companion(directory, "TS_VIDEO_VAE", "vae", "ltx-2.5-video-vae-conv-bf16.safetensors");
            AudioVaePath = Companion(directory, "TS_VIDEO_AUDIO_VAE", "vae", "ltx-2.5-audio-vae-bf16.safetensors");
            string configuredReference = Environment.GetEnvironmentVariable("TS_LTX25_INGREDIENTS_LORA");
            if (!string.IsNullOrWhiteSpace(configuredReference))
            {
                if (!File.Exists(configuredReference)) throw new FileNotFoundException("The configured LTX Ingredients adapter does not exist.", configuredReference);
                ReferenceLoraPath = Path.GetFullPath(configuredReference);
            }
            else
            {
                string reference = Path.Combine(directory, "loras", "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors");
                if (File.Exists(reference)) ReferenceLoraPath = reference;
            }
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };
        }
        catch { base.Dispose(); throw; }
    }

    // GGML option names remain usable by hosts whose device picker uses them;
    // this family executes its owned direct CUDA/MLX kernels, not a GGML graph.
    internal static BackendType DirectBackend(BackendType backend) => backend switch
    {
        BackendType.GgmlCuda or BackendType.Cuda => BackendType.Cuda,
        BackendType.GgmlMetal or BackendType.Mlx => BackendType.Mlx,
        BackendType.GgmlCpu or BackendType.Cpu => BackendType.Cpu,
        _ => throw new NotSupportedException("LTX 2.5 supports CUDA, MLX and CPU execution; Vulkan is not implemented."),
    };

    private static string Companion(string directory, string variable, string subdirectory, string filename)
    {
        string configured = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            if (File.Exists(configured)) return Path.GetFullPath(configured);
            throw new FileNotFoundException($"The LTX companion specified by {variable} does not exist.", configured);
        }
        foreach (string path in new[] { Path.Combine(directory, subdirectory, filename), Path.Combine(directory, filename) })
            if (File.Exists(path)) return path;
        throw new FileNotFoundException($"LTX 2.5 requires {filename}; configure its companion path or install it beside the denoiser.", filename);
    }

    public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams parameters = null)
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
            return MlxWorker.Shared.Invoke(() => GenerateVideo(prompt, parameters));
        lock (_generationLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Ltx25Pipeline.Generate(this, prompt, parameters ?? new VideoGenerationParams());
        }
    }

    internal Ltx25PromptEncoding Conditioning(string prompt, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_promptCache.TryGetValue(prompt, out var cached))
        {
            _promptOrder.Remove(cached.Node);
            _promptOrder.AddLast(cached.Node);
            ++PromptCacheHits;
            return cached.Encoding;
        }
        Ltx25PromptEncoding encoding;
        // Keep the large mapped files and tokenizer across distinct prompts, but
        // release the connector's large device weights before denoising.
        var encoder = _promptEncoder ??= new Ltx25PromptEncoder(TextEncoderPath, CheckpointPath, _allocator);
        long mlxBf16 = encoder.NativeBf16Projections, cudaMapped = encoder.CudaMappedProjections,
            cudaBf16 = encoder.CudaBf16Projections, connectorProjections = encoder.CudaConnectorProjections,
            connectorAttentions = encoder.CudaConnectorAttentions;
        try
        {
            try
            {
                encoding = encoder.Encode(prompt, cancellationToken);
                Console.WriteLine($"  [ltx25-conditioning] gemma={encoder.LastGemmaMilliseconds:F3}ms aggregate={encoder.LastAggregationMilliseconds:F3}ms " +
                    $"connectors={encoder.LastConnectorMilliseconds:F3}ms mlx_bf16={encoder.NativeBf16Projections - mlxBf16} " +
                    $"cuda_mapped={encoder.CudaMappedProjections - cudaMapped} cuda_bf16={encoder.CudaBf16Projections - cudaBf16} " +
                    $"cuda_connector_projections={encoder.CudaConnectorProjections - connectorProjections} " +
                    $"cuda_connector_attentions={encoder.CudaConnectorAttentions - connectorAttentions}");
            }
            finally { encoder.ReleaseDeviceResidency(); }
        }
        catch { DisposeOwned(ref _promptEncoder); throw; }
        cancellationToken.ThrowIfCancellationRequested();
        _promptCache.Add(prompt, (encoding, _promptOrder.AddLast(prompt)));
        if (_promptCache.Count > 4)
        {
            _promptCache.Remove(_promptOrder.First.Value);
            _promptOrder.RemoveFirst();
        }
        return encoding;
    }

    internal Ltx25SamplingOutput Sample(Ltx25DenoiserInput input, Action<int, int> onStep,
        CancellationToken cancellationToken = default)
    {
        // Preserve mapped weight/cache identities across requests. In particular,
        // rebuilding MLX's Q4_K affine representation for each video is expensive.
        // The serialized generation lock also protects this single device model.
        _denoiser ??= new Ltx25Denoiser(CheckpointPath, _allocator, ReferenceLoraPath);
        try
        {
            return Ltx25Sampling.RunEuler(_denoiser, input, Ltx25Scheduler.DistilledSigmas(), (step, total) =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                onStep?.Invoke(step, total);
            }, cancellationToken);
        }
        catch { DisposeOwned(ref _denoiser); throw; }
    }

    internal float[] DecodeVideo(float[] latents, int frames, int height, int width,
        CancellationToken cancellationToken = default)
    {
        var decoder = _videoDecoder ??= new Ltx25VideoVaeDecoder(VideoVaePath, _allocator);
        try { return decoder.Decode(latents, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoDecoder); throw; }
    }

    internal float[] EncodeImage(float[] rgb, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.EncodeImage(rgb, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal float[] EncodeReferenceVideo(float[] rgb, int frames, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.Encode(rgb, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal float[] EncodeStaticReferenceVideo(float[] rgbFrame, int frames, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.EncodeStaticVideo(rgbFrame, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal GeneratedVideoAudio DecodeAudio(float[] latents, int tokens, int samples,
        CancellationToken cancellationToken = default)
    {
        var decoder = _audioDecoder ??= new Ltx25AudioDecoder(AudioVaePath, _allocator);
        try { return decoder.Decode(latents, tokens, samples, cancellationToken); }
        catch { DisposeOwned(ref _audioDecoder); throw; }
    }

    private static void DisposeOwned<T>(ref T component) where T : class, IDisposable
    {
        var previous = component;
        component = null;
        previous?.Dispose();
    }

    public override void TrimIdleMemory()
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
        {
            MlxWorker.Shared.Invoke(TrimIdleMemory);
            return;
        }
        lock (_generationLock)
        {
            _promptCache.Clear(); _promptOrder.Clear();
            DisposeOwned(ref _promptEncoder);
            DisposeOwned(ref _videoDecoder);
            DisposeOwned(ref _videoEncoder);
            DisposeOwned(ref _audioDecoder);
            DisposeOwned(ref _denoiser);
            base.TrimIdleMemory();
            if (_allocator is MlxAllocator) MlxBackend.ClearCache();
            else if (_allocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
        }
    }

    public string VideoModelFamily => ArchitectureId;
    public int DefaultPreviewFrames => 17;
    public bool SupportsAudio => true;
    public bool SupportsImageConditioning => true;
    public bool SupportsEndImageConditioning => true;
    public bool SupportsKeyframeConditioning => true;
    public bool SupportsReferenceConditioning => ReferenceLoraPath != null;
    public int MaxReferenceImages => Ltx25ImageConditioning.MaxImages;
    protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException("LTX is a video-generation model; use GenerateVideo().");
    protected override void ResetKVCacheCore() { }
    public override void WarmUpKernels() { }
    public override void Dispose()
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
        {
            MlxWorker.Shared.Invoke(Dispose);
            return;
        }
        lock (_generationLock)
        {
            if (_disposed) return;
            _disposed = true;
            _promptCache.Clear();
            _promptOrder.Clear();
            DisposeOwned(ref _promptEncoder);
            DisposeOwned(ref _videoDecoder);
            DisposeOwned(ref _videoEncoder);
            DisposeOwned(ref _audioDecoder);
            DisposeOwned(ref _denoiser);
            base.Dispose();
        }
    }
}
