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
    /// Qwen-Image-2.1 text-to-image generation and image editing: a 32-layer,
    /// 4096-wide single-stream diffusion transformer, a 64-channel /16 RGBA VAE
    /// and Qwen3-VL-8B with 4096-dimensional conditioning. The diffusion-transformer
    /// GGUF must be a Qwen-Image-2.1 checkpoint; any other Qwen-Image DiT is refused
    /// with a <see cref="ModelLoadRefusedException"/>.
    /// The companion files are resolved next to the DiT GGUF (or via the
    /// <c>TS_QWEN_IMAGE_VAE</c> / <c>TS_QWEN_IMAGE_TE</c> / <c>TS_QWEN_IMAGE_MMPROJ</c>
    /// environment variables). This class is not an <see cref="IModelArchitecture"/>
    /// text generator; use <see cref="GenerateImage"/> or <see cref="EditImage(string, RgbImage, QwenImageParams)"/>.
    /// </summary>
    public sealed class QwenImageModel : ModelBase
    {
        private readonly string _vaePath;
        private readonly string _tePath;
        private readonly string _mmprojPath;   // optional; required only for editing

        private GgufFile _vaeGguf;
        private GgufFile _teGguf;
        private SafetensorsModel _vaeSafetensors;
        private IFloatTensorStore _vaeWeights;

        private readonly string _ditPath;
        internal string DitGgufPath => _ditPath;
        internal GgufFile VaeGguf => _vaeGguf ??= new GgufFile(_vaePath);
        internal GgufFile TeGguf => _teGguf ??= new GgufFile(_tePath);
        internal string MmprojPath => _mmprojPath;
        internal string TePath => _tePath;
        internal BackendType Backend => _backend;

        internal static bool IsSafetensorsPath(string p) =>
            p != null && (p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                       || p.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase));

        /// <summary>
        /// The VAE weight source, loaded directly from a <c>.safetensors</c> file (BF16 upcast to F32)
        /// when the resolved companion is safetensors, or from a converted F32 VAE GGUF otherwise.
        /// Both yield bit-identical floats.
        /// </summary>
        internal IFloatTensorStore VaeWeightSource => _vaeWeights ??= OpenVaeWeightSource();

        private IFloatTensorStore OpenVaeWeightSource()
        {
            IFloatTensorStore source = IsSafetensorsPath(_vaePath)
                ? _vaeSafetensors = SafetensorsModel.Open(_vaePath)
                : new GgufFloatTensorStore(VaeGguf);
            return new QwenImage21VaeTensorStore(source);
        }

        public QwenImageModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            try
            {
                _ditPath = ggufPath;
                bool version21 = _gguf.Tensors.Keys.Any(n => n == "txt_in.text_norm.weight" ||
                    n.EndsWith(".txt_in.text_norm.weight", StringComparison.Ordinal));
                if (!version21)
                    throw new ModelLoadRefusedException(
                        $"'{Path.GetFileName(ggufPath)}' is not a Qwen-Image-2.1 diffusion transformer (no txt_in.text_norm.weight tensor). " +
                        "Earlier Qwen-Image / Qwen-Image-Edit checkpoints such as Qwen-Image-Edit-2511 are no longer supported; " +
                        "use a Qwen-Image-2.1 GGUF (see docs/models/qwenimage21.md).");
                if (!IsGgmlBackend)
                    throw new NotSupportedException("Qwen-Image-2.1 requires a GGML backend (ggml_metal, ggml_cuda, ggml_vulkan or ggml_cpu).");
                Config = new ModelConfig
                {
                    Architecture = "qwen_image",
                    HiddenSize = 4096,
                    NumLayers = 32,
                    NumHeads = 32,
                    NumKVHeads = 32,
                    Eps = 1e-6f,
                    VocabSize = 0,
                };

                string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath)) ?? ".";
                _vaePath = ResolveVersion21Companion("TS_QWEN_IMAGE_VAE", dir,
                    n => n.Contains("qwen_image_2.1_vae") && n.EndsWith(".safetensors"));
                _tePath = ResolveVersion21Companion("TS_QWEN_IMAGE_TE", dir,
                    n => (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj") && n.EndsWith(".gguf"));
                _mmprojPath = ResolveVersion21Companion("TS_QWEN_IMAGE_MMPROJ", dir,
                    n => n.Contains("mmproj") && (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && n.EndsWith(".gguf"));

                Console.WriteLine($"Qwen-Image-2.1: DiT={Path.GetFileName(ggufPath)}");
                Console.WriteLine($"  VAE          = {_vaePath ?? "<missing>"}");
                Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
                Console.WriteLine($"  mmproj       = {_mmprojPath ?? "<none> (text-only grounding)"}");

                if (_vaePath == null || _tePath == null)
                    throw new FileNotFoundException(
                        "Qwen-Image-2.1 needs its 2.1 VAE and Qwen3-VL-8B text-encoder GGUF. " +
                        "Place them next to the DiT GGUF or set TS_QWEN_IMAGE_VAE / TS_QWEN_IMAGE_TE.");

                // Header-only checks fail before any expensive generation work.
                QwenImage21CompanionValidation.ValidateVae(VaeWeightSource);
                QwenImage21CompanionValidation.ValidateText(TeGguf);
                if (_mmprojPath != null)
                {
                    using var vision = new GgufFile(_mmprojPath);
                    QwenImage21CompanionValidation.ValidateVision(vision);
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
        /// Edit <paramref name="input"/> per the natural-language <paramref name="prompt"/>,
        /// returning the modified image (runs <see cref="QwenImage21Pipeline"/>).
        /// </summary>
        public RgbImage EditImage(string prompt, RgbImage input, QwenImageParams p)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            return GetPipeline21().Run(prompt, new[] { input }, p ?? new QwenImageParams());
        }

        /// <summary>
        /// Multi-image edit: every entry of <paramref name="inputs"/> conditions the
        /// transformer as its own reference, and <paramref name="inputs"/>[0] drives the
        /// output geometry; e.g. inputs = [model photo, garment photo] with a
        /// "dress the model in the garment" prompt.
        /// </summary>
        public RgbImage EditImage(string prompt, System.Collections.Generic.IReadOnlyList<RgbImage> inputs, QwenImageParams p)
        {
            if (inputs == null || inputs.Count == 0)
                throw new ArgumentException("At least one input image is required.", nameof(inputs));
            return GetPipeline21().Run(prompt, inputs.ToArray(), p ?? new QwenImageParams());
        }

        private QwenImage21Pipeline _pipeline21;
        private QwenImage21Pipeline GetPipeline21() => _pipeline21 ??= new QwenImage21Pipeline(this);

        /// <summary>Generate an image from text using Qwen-Image-2.1.</summary>
        public RgbImage GenerateImage(string prompt, QwenImageParams p = null)
        {
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
            _pipeline21?.Dispose();
            _vaeSafetensors?.Dispose();
            _vaeGguf?.Dispose();
            _teGguf?.Dispose();
            base.Dispose();
        }
    }
}
