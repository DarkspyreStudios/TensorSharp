// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Core;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Qwen-Image-Edit-2511 and Qwen-Image-2.1 generation/editing pipelines.
    /// The diffusion-transformer GGUF selects the architecture and requires
    /// matching VAE and text/vision companions:
    /// <list type="bullet">
    ///   <item>2511: 60-layer, 3072-wide double-stream DiT, 16-channel /8 VAE,
    ///   and Qwen2.5-VL-7B with 3584-dimensional conditioning.</item>
    ///   <item>2.1: 32-layer, 4096-wide single-stream DiT, 64-channel /16 RGBA VAE,
    ///   and Qwen3-VL-8B with 4096-dimensional conditioning.</item>
    /// </list>
    /// The companion GGUFs are resolved next to the DiT GGUF (or via the
    /// <c>TS_QWEN_IMAGE_VAE</c> / <c>TS_QWEN_IMAGE_TE</c> / <c>TS_QWEN_IMAGE_MMPROJ</c>
    /// environment variables). This class is not an <see cref="IModelArchitecture"/>
    /// text generator; use <see cref="GenerateImage"/> or <see cref="EditImage"/>.
    /// </summary>
    public sealed partial class QwenImageModel : ModelBase
    {
        // Legacy Qwen-Image/2511 constants. QwenImage21DiT owns the 2.1 dimensions.
        public const int DitHiddenSize = 3072;
        public const int DitNumLayers = 60;
        public const int DitNumHeads = 24;
        public const int DitHeadDim = 128;            // 24 * 128 = 3072
        public const int DitInChannels = 64;          // 16 latent channels * 2x2 patch
        public const int DitTextDim = 3584;           // Qwen2.5-VL hidden
        public const int VaeLatentChannels = 16;
        public const int VaeScaleFactor = 8;          // spatial 8x downsample
        public static readonly int[] AxesDimsRope = { 16, 56, 56 };  // sums to head_dim 128
        public const float DitEps = 1e-6f;

        private readonly string _vaePath;
        private readonly string _tePath;
        private readonly string _mmprojPath;   // optional (vision grounding, Stage 4)

        private GgufFile _vaeGguf;
        private GgufFile _teGguf;
        private SafetensorsModel _vaeSafetensors;
        private IFloatTensorStore _vaeWeights;

        private readonly string _ditPath;
        /// <summary>True for the 2.1 single-stream transformer and its Qwen3-VL/64-channel VAE companions.</summary>
        public bool IsVersion21 { get; }
        /// <summary>The diffusion-transformer GGUF (alias of the base <c>_gguf</c>).</summary>
        internal GgufFile DitGguf => _gguf;
        internal string DitGgufPath => _ditPath;
        internal GgufFile VaeGguf => _vaeGguf ??= new GgufFile(_vaePath);
        internal GgufFile TeGguf => _teGguf ??= new GgufFile(_tePath);
        internal string MmprojPath => _mmprojPath;
        internal string TePath => _tePath;
        internal string VaePath => _vaePath;
        internal BackendType Backend => _backend;

        internal static bool IsSafetensorsPath(string p) =>
            p != null && (p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                       || p.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase));

        /// <summary>
        /// The VAE weight source, loaded directly from a <c>.safetensors</c> file (BF16 upcast to F32)
        /// when the resolved companion is safetensors, or from the converted F32 VAE GGUF otherwise.
        /// Both yield bit-identical floats.
        /// </summary>
        internal IFloatTensorStore VaeWeightSource => _vaeWeights ??= OpenVaeWeightSource();

        private IFloatTensorStore OpenVaeWeightSource()
        {
            if (IsSafetensorsPath(_vaePath))
                return _vaeSafetensors = SafetensorsModel.Open(_vaePath);
            return new GgufFloatTensorStore(VaeGguf);
        }

        public QwenImageModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            try
            {
                _ditPath = ggufPath;
                IsVersion21 = _gguf.Tensors.Keys.Any(n => n == "txt_in.text_norm.weight" ||
                    n.EndsWith(".txt_in.text_norm.weight", StringComparison.Ordinal));
                if (IsVersion21 && !IsGgmlBackend)
                    throw new NotSupportedException("Qwen-Image-2.1 requires a GGML backend (ggml_metal, ggml_cuda, ggml_vulkan or ggml_cpu).");
                Config = new ModelConfig
                {
                    Architecture = "qwen_image",
                    HiddenSize = IsVersion21 ? 4096 : DitHiddenSize,
                    NumLayers = IsVersion21 ? 32 : DitNumLayers,
                    NumHeads = IsVersion21 ? 32 : DitNumHeads,
                    NumKVHeads = IsVersion21 ? 32 : DitNumHeads,
                    Eps = DitEps,
                    VocabSize = 0,
                };

                string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath)) ?? ".";
                _vaePath = IsVersion21 ? ResolveVersion21Companion("TS_QWEN_IMAGE_VAE", dir,
                    n => n.Contains("qwen_image_2.1_vae") && n.EndsWith(".safetensors")) : ResolveVaeCompanion(dir);
                _tePath = IsVersion21 ? ResolveVersion21Companion("TS_QWEN_IMAGE_TE", dir,
                    n => (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj") && n.EndsWith(".gguf")) : ResolveTeCompanion(dir);
                _mmprojPath = IsVersion21 ? ResolveVersion21Companion("TS_QWEN_IMAGE_MMPROJ", dir,
                    n => n.Contains("mmproj") && (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && n.EndsWith(".gguf")) :
                    ResolveCompanionOptional("TS_QWEN_IMAGE_MMPROJ", dir,
                        n => n.Contains("mmproj") && (n.Contains("qwen2") || n.Contains("qwen-image")) && n.EndsWith(".gguf"));

                Console.WriteLine($"{(IsVersion21 ? "Qwen-Image-2.1" : "Qwen-Image-Edit")}: DiT={Path.GetFileName(ggufPath)}");
                Console.WriteLine($"  VAE          = {_vaePath ?? "<missing>"}");
                Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
                Console.WriteLine($"  mmproj       = {_mmprojPath ?? "<none> (text-only grounding)"}");

                if (_vaePath == null || _tePath == null)
                    throw new FileNotFoundException(
                        (IsVersion21 ? "Qwen-Image-2.1 needs its 2.1 VAE and Qwen3-VL-8B text-encoder GGUF. " :
                        "Qwen-Image-Edit needs companion VAE + Qwen2.5-VL text-encoder GGUFs. ") +
                        "Place them next to the DiT GGUF or set TS_QWEN_IMAGE_VAE / TS_QWEN_IMAGE_TE.");

                if (IsVersion21)
                {
                    // Header-only checks fail before any expensive generation work.
                    QwenImage21CompanionValidation.ValidateVae(VaeWeightSource);
                    QwenImage21CompanionValidation.ValidateText(TeGguf);
                    if (_mmprojPath != null)
                    {
                        using var vision = new GgufFile(_mmprojPath);
                        QwenImage21CompanionValidation.ValidateVision(vision);
                    }
                }
            }
            catch { Dispose(); throw; }
        }

        private static string ResolveVersion21Companion(string envVar, string dir, Func<string, bool> match)
        {
            string explicitPath = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(explicitPath))
            {
                if (!File.Exists(explicitPath))
                    throw new FileNotFoundException($"Qwen-Image-2.1 companion specified by {envVar} does not exist.", explicitPath);
                return Path.GetFullPath(explicitPath);
            }
            return Directory.EnumerateFiles(dir).Where(f => match(Path.GetFileName(f).ToLowerInvariant()))
                .OrderByDescending(f => new FileInfo(f).Length).ThenBy(f => f, StringComparer.Ordinal).FirstOrDefault();
        }

        /// <summary>
        /// Resolve the companion VAE, accepting either the original <c>.safetensors</c> (loaded directly)
        /// or the converted <c>.gguf</c>. Order: env var override (any extension) &gt; preferred filenames
        /// (gguf then safetensors) &gt; any <c>*vae*</c> file with either extension in the directory.
        /// </summary>
        private static string ResolveVaeCompanion(string dir)
        {
            string env = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_VAE");
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

            foreach (var preferred in new[] { "qwen_image_vae.gguf", "qwen_image_vae.safetensors", "Qwen_Image-VAE.safetensors" })
            {
                string p = Path.Combine(dir, preferred);
                if (File.Exists(p)) return p;
            }
            if (Directory.Exists(dir))
            {
                foreach (var f in Directory.EnumerateFiles(dir))
                {
                    string n = Path.GetFileName(f).ToLowerInvariant();
                    if (n.Contains("vae") && (n.EndsWith(".gguf") || n.EndsWith(".safetensors")))
                        return f;
                }
            }
            return null;
        }

        /// <summary>
        /// Resolve the Qwen2.5-VL text encoder. Unlike the DiT, the text/vision encoders are
        /// FREED before the denoise loop (see <see cref="QwenImagePipeline"/>), so they never
        /// compete with the DiT for VRAM — a higher-quality (larger) TE quant costs nothing at
        /// denoise time but materially improves the prompt/image conditioning that drives the
        /// whole edit. A ~2-bit TE (IQ2_XXS) produces markedly softer faces and lower overall
        /// fidelity than Q4_K_M+ for the same DiT/steps/seed. So, absent an explicit
        /// <c>TS_QWEN_IMAGE_TE</c>, prefer the HIGHEST-quality VL GGUF present (largest file =
        /// higher bpw for the same 7B model) rather than the smallest, and warn on a very-low-bit
        /// pick so the user knows a better one would help at no denoise-VRAM cost.
        /// </summary>
        private static string ResolveTeCompanion(string dir)
        {
            string env = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE");
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) { WarnLowBitTe(env); return env; }
            if (!Directory.Exists(dir)) return null;

            string best = null; long bestSize = -1;
            foreach (var f in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                string n = Path.GetFileName(f).ToLowerInvariant();
                bool isTe = (n.Contains("qwen2.5-vl") || n.Contains("qwen2_5_vl") || n.Contains("qwen-image-te"))
                            && !n.Contains("mmproj");
                if (!isTe) continue;
                long size = new FileInfo(f).Length;
                if (size > bestSize) { bestSize = size; best = f; }   // largest = best quant (TE is freed pre-denoise)
            }
            if (best != null) WarnLowBitTe(best);
            return best;
        }

        // Warn when the chosen text encoder is an extremely low-bit quant (IQ1/IQ2 ~1-2 bpw):
        // these gut the Qwen2.5-VL conditioning and are the single biggest quality regression for
        // Qwen-Image-Edit. The encoders are freed before the DiT denoise, so a bigger quant is
        // effectively free VRAM-wise — steer users toward Q4_K_M+.
        private static void WarnLowBitTe(string tePath)
        {
            string n = Path.GetFileName(tePath).ToLowerInvariant();
            if (n.Contains("iq1") || n.Contains("iq2") || n.Contains("q2_k"))
                Console.WriteLine(
                    $"  [warn] text encoder '{Path.GetFileName(tePath)}' is a very low-bit quant — it severely " +
                    "degrades prompt/image conditioning (soft faces, low fidelity). The text encoder is freed " +
                    "before the denoise loop, so a Q4_K_M (or higher) VL GGUF improves quality at NO extra " +
                    "denoise VRAM. Pass --qwen-image-vl <better>.gguf or set TS_QWEN_IMAGE_TE.");
        }

        private static string ResolveCompanion(string envVar, string dir, string preferred, Func<string, bool> match)
        {
            string env = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            string pref = Path.Combine(dir, preferred);
            if (File.Exists(pref)) return pref;
            return ResolveCompanionOptional(null, dir, match);
        }

        private static string ResolveCompanionOptional(string envVar, string dir, Func<string, bool> match)
        {
            if (envVar != null)
            {
                string env = Environment.GetEnvironmentVariable(envVar);
                if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            }
            if (!Directory.Exists(dir)) return null;
            foreach (var f in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                if (match(Path.GetFileName(f).ToLowerInvariant())) return f;
            }
            return null;
        }

        /// <summary>
        /// Edit <paramref name="input"/> per the natural-language <paramref name="prompt"/>,
        /// returning the modified image. The heavy lifting lives in
        /// <see cref="QwenImagePipeline"/> (added incrementally across the VAE / text-encoder
        /// / DiT stages).
        /// </summary>
        public RgbImage EditImage(string prompt, RgbImage input, QwenImageParams p)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (IsVersion21) return GetPipeline21().Run(prompt, new[] { input }, p ?? new QwenImageParams());
            var pipeline = GetPipeline();
            return pipeline.Edit(prompt, input, p ?? new QwenImageParams());
        }

        /// <summary>
        /// Multi-image edit ("Picture 1..N" grounding + one reference-latent stream per input;
        /// diffusers <c>QwenImageEditPlusPipeline</c> semantics). <paramref name="inputs"/>[0]
        /// drives the output geometry; e.g. inputs = [model photo, garment photo] with a
        /// "dress the model in the garment" prompt.
        /// </summary>
        public RgbImage EditImage(string prompt, System.Collections.Generic.IReadOnlyList<RgbImage> inputs, QwenImageParams p)
        {
            if (inputs == null || inputs.Count == 0)
                throw new ArgumentException("At least one input image is required.", nameof(inputs));
            if (IsVersion21) return GetPipeline21().Run(prompt, inputs.ToArray(), p ?? new QwenImageParams());
            var pipeline = GetPipeline();
            return pipeline.Edit(prompt, System.Linq.Enumerable.ToArray(inputs), p ?? new QwenImageParams());
        }

        private QwenImagePipeline _pipeline;
        private QwenImagePipeline GetPipeline() => _pipeline ??= new QwenImagePipeline(this);
        private QwenImage21Pipeline _pipeline21;
        private QwenImage21Pipeline GetPipeline21() => _pipeline21 ??= new QwenImage21Pipeline(this);

        /// <summary>Generate an image from text using Qwen-Image-2.1.</summary>
        public RgbImage GenerateImage(string prompt, QwenImageParams p = null)
        {
            if (!IsVersion21)
                throw new NotSupportedException("Text-to-image generation requires Qwen-Image-2.1 weights.");
            return GetPipeline21().Run(prompt, Array.Empty<RgbImage>(), p ?? new QwenImageParams());
        }

        // ---- IModelArchitecture autoregressive surface: not applicable to an image model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("QwenImageModel is an image model; use GenerateImage() or EditImage().");

        protected override void ResetKVCacheCore() { /* no autoregressive KV cache */ }

        // The base warmup runs a dummy autoregressive Forward(); skip it for the image model
        // (the diffusion nets are loaded lazily on the first EditImage call).
        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _pipeline?.Dispose();
            _pipeline21?.Dispose();
            _vaeSafetensors?.Dispose();
            _vaeGguf?.Dispose();
            _teGguf?.Dispose();
            base.Dispose();
        }
    }
}
