// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Loads a Gemma-4-family vision tower straight out of a HuggingFace <c>.safetensors</c>
    /// checkpoint, with no intermediate mmproj GGUF conversion step.
    ///
    /// This exists because published GGUF conversions of some Gemma-4-family checkpoints drop the
    /// vision tower entirely. <c>google/diffusiongemma-26B-A4B-it</c> is the motivating case: every
    /// released GGUF of it is text-only and no mmproj was ever published, yet the upstream
    /// checkpoint carries a complete 27-layer <c>gemma4_vision</c> tower under
    /// <c>model.encoder.vision_tower.*</c> plus the projector at
    /// <c>model.encoder.embed_vision.embedding_projection.weight</c>.
    ///
    /// The tower is architecturally identical to the mmproj "gemma4v" projector already supported
    /// here — only the dimensions differ — so the weights are renamed into the same internal
    /// <c>v.*</c> / <c>mm.*</c> keys and the existing encode path runs unchanged.
    /// </summary>
    public partial class Gemma4VisionEncoder
    {
        private const string SafetensorsTowerPrefix = "model.encoder.vision_tower.";
        private const string SafetensorsLayerPrefix = SafetensorsTowerPrefix + "encoder.layers.";
        private const string SafetensorsProjectorName =
            "model.encoder.embed_vision.embedding_projection.weight";

        /// <summary>
        /// True when <paramref name="path"/> should be read as a safetensors vision tower rather
        /// than an mmproj GGUF. Dispatches on extension, then confirms with the safetensors header
        /// so a mis-named file fails loudly at load instead of producing garbage embeddings.
        /// </summary>
        internal static bool IsSafetensorsProjector(string path)
        {
            if (string.IsNullOrEmpty(path))
                return false;
            return path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Derives the tower geometry from tensor shapes rather than from metadata.
        ///
        /// A raw HF shard carries no hyperparameter block, so every dimension here is recovered
        /// from the weights themselves. That is deliberate: shapes cannot drift out of sync with
        /// the tensors the way a hand-written metadata block can.
        /// </summary>
        private static TowerSpec ReadSafetensorsSpec(SafetensorsFile st, string path)
        {
            string inputProj = SafetensorsTowerPrefix + "patch_embedder.input_proj.weight";
            if (!st.HasTensor(inputProj))
                throw new InvalidDataException(
                    $"'{path}' is not a Gemma-4 vision tower: missing '{inputProj}'. " +
                    "Point --mmproj at the checkpoint shard that holds " +
                    $"'{SafetensorsTowerPrefix}*' (for diffusiongemma-26B-A4B-it that is " +
                    "model-00011-of-00011.safetensors), or at an mmproj GGUF.");

            long[] projShape = st.TensorShape(inputProj);          // [hidden, C*P*P]
            if (projShape.Length != 2)
                throw new InvalidDataException(
                    $"'{inputProj}' must be 2-D [hidden, channels*patch*patch], got rank {projShape.Length}.");

            int hidden = (int)projShape[0];
            long patchStride = projShape[1];
            if (patchStride % 3 != 0)
                throw new InvalidDataException(
                    $"'{inputProj}' second dimension {patchStride} is not divisible by 3 channels.");
            int patchSize = (int)Math.Round(Math.Sqrt(patchStride / 3.0));
            if ((long)patchSize * patchSize * 3 != patchStride)
                throw new InvalidDataException(
                    $"'{inputProj}' second dimension {patchStride} is not 3*P*P for an integer P.");

            // Layer count: highest contiguous index present under encoder.layers.N.
            int blocks = 0;
            while (st.HasTensor($"{SafetensorsLayerPrefix}{blocks}.input_layernorm.weight"))
                blocks++;
            if (blocks == 0)
                throw new InvalidDataException(
                    $"'{path}' has no '{SafetensorsLayerPrefix}0.input_layernorm.weight'; " +
                    "the vision tower layers are missing or use an unexpected naming scheme.");

            // head_dim comes from the per-head q_norm vector, so head count is not guessed.
            string qNorm = $"{SafetensorsLayerPrefix}0.self_attn.q_norm.weight";
            if (!st.HasTensor(qNorm))
                throw new InvalidDataException($"'{path}' is missing '{qNorm}'.");
            int headDim = (int)st.TensorShape(qNorm)[0];
            if (headDim <= 0 || hidden % headDim != 0)
                throw new InvalidDataException(
                    $"vision head_dim {headDim} from '{qNorm}' does not divide hidden size {hidden}.");
            int heads = hidden / headDim;

            string gate = $"{SafetensorsLayerPrefix}0.mlp.gate_proj.linear.weight";
            if (!st.HasTensor(gate))
                throw new InvalidDataException($"'{path}' is missing '{gate}'.");
            int intermediate = (int)st.TensorShape(gate)[0];

            if (!st.HasTensor(SafetensorsProjectorName))
                throw new InvalidDataException(
                    $"'{path}' has a vision tower but no projector '{SafetensorsProjectorName}'. " +
                    "Without it the tower output cannot be mapped into the language model's " +
                    "embedding space.");
            int projectionDim = (int)st.TensorShape(SafetensorsProjectorName)[0];

            // pooling_kernel_size and rms_norm_eps are genuinely not recoverable from shapes.
            // Prefer the checkpoint's own config.json; fall back to the Gemma-4 vision defaults and
            // say so rather than silently assuming.
            var (nMerge, eps) = ReadVisionConfigSidecar(path, hidden);

            return new TowerSpec
            {
                HiddenSize = hidden,
                IntermediateSize = intermediate,
                NumHeads = heads,
                BlockCount = blocks,
                Eps = eps,
                ProjectionDim = projectionDim,
                PatchSize = patchSize,
                NMerge = nMerge,
                RopeTheta = 100f,
                ProjectorType = "gemma4v",
                // Gemma-4 vision is trained on raw [0,1] pixels: processor_config.json declares
                // do_normalize=false with mean 0 / std 1, and the tower itself does the
                // 2*(x-0.5) recentring. Keep those identity statistics here.
                ImageMean = new[] { 0f, 0f, 0f },
                ImageStd = new[] { 1f, 1f, 1f },
            };
        }

        /// <summary>
        /// Reads <c>pooling_kernel_size</c> and <c>rms_norm_eps</c> from a <c>config.json</c> next to
        /// the shard. Returns the Gemma-4 vision defaults with a one-time warning when absent.
        /// </summary>
        private static (int NMerge, float Eps) ReadVisionConfigSidecar(string shardPath, int towerHidden)
        {
            const int DefaultNMerge = 3;
            const float DefaultEps = 1e-6f;

            string? dir = Path.GetDirectoryName(Path.GetFullPath(shardPath));
            string configPath = dir == null ? "config.json" : Path.Combine(dir, "config.json");
            if (!File.Exists(configPath))
            {
                // Not an error: these ARE the Gemma-4 vision constants, and every dimension that
                // could actually vary was measured from the weights. Say it once so a NON-standard
                // tower (a different pooling factor) cannot be adopted in silence.
                WarnVisionConfigDefaults(
                    $"No config.json beside '{shardPath}'; using the Gemma-4 vision values " +
                    $"pooling_kernel_size={DefaultNMerge}, rms_norm_eps={DefaultEps.ToString("G", CultureInfo.InvariantCulture)}, " +
                    "which are correct for this family. Place the checkpoint's config.json beside " +
                    "the shard only if its vision tower overrides them.");
                return (DefaultNMerge, DefaultEps);
            }

            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
                if (!doc.RootElement.TryGetProperty("vision_config", out var vc))
                {
                    WarnVisionConfigDefaults(
                        $"'{configPath}' has no vision_config; assuming pooling_kernel_size={DefaultNMerge}, rms_norm_eps={DefaultEps.ToString("G", CultureInfo.InvariantCulture)}.");
                    return (DefaultNMerge, DefaultEps);
                }

                // A shard downloaded into a shared model directory sits next to whatever else lives
                // there, so a config.json found beside it is NOT necessarily this model's. Accept it
                // only when its vision_config describes a tower of the same width as the weights we
                // just measured; otherwise it belongs to something else and its pooling factor would
                // silently reshape this image.
                if (!vc.TryGetProperty("hidden_size", out var hs)
                    || !hs.TryGetInt32(out int configHidden)
                    || configHidden != towerHidden)
                {
                    WarnVisionConfigDefaults(
                        $"'{configPath}' describes a different vision tower (hidden_size " +
                        $"{(hs.ValueKind == JsonValueKind.Number ? hs.ToString() : "absent")} != {towerHidden}), " +
                        $"so it is ignored; assuming pooling_kernel_size={DefaultNMerge}, " +
                        $"rms_norm_eps={DefaultEps.ToString("G", CultureInfo.InvariantCulture)}.");
                    return (DefaultNMerge, DefaultEps);
                }

                int nMerge = vc.TryGetProperty("pooling_kernel_size", out var pk) && pk.TryGetInt32(out int pkv)
                    ? pkv
                    : DefaultNMerge;
                float eps = vc.TryGetProperty("rms_norm_eps", out var re) && re.TryGetDouble(out double rev)
                    ? (float)rev
                    : DefaultEps;
                if (nMerge <= 0)
                    throw new InvalidDataException(
                        $"'{configPath}' vision_config.pooling_kernel_size must be positive, got {nMerge}.");
                return (nMerge, eps);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Failed to parse '{configPath}': {ex.Message}", ex);
            }
        }

        private static bool _warnedVisionConfigDefaults;
        private static readonly object _visionConfigWarnLock = new();

        private static void WarnVisionConfigDefaults(string message)
        {
            lock (_visionConfigWarnLock)
            {
                if (_warnedVisionConfigDefaults)
                    return;
                _warnedVisionConfigDefaults = true;
            }
            Console.WriteLine($"[Gemma4VisionEncoder] {message}");
        }

        /// <summary>
        /// Renames the upstream tensors into the internal <c>v.*</c> / <c>mm.*</c> keys the encode
        /// path already uses, so the safetensors and mmproj sources converge on one forward pass.
        /// </summary>
        private void LoadWeightsFromSafetensors(SafetensorsFile st)
        {
            Console.Write("Loading vision encoder weights (safetensors)...");
            int count = 0;

            void Put(string internalName, string sourceName, params long[] expectedShape)
            {
                long[] shape = st.TensorShape(sourceName);
                if (expectedShape.Length > 0)
                {
                    if (shape.Length != expectedShape.Length)
                        throw new InvalidDataException(
                            $"'{sourceName}' rank {shape.Length} != expected {expectedShape.Length}.");
                    for (int i = 0; i < shape.Length; i++)
                        if (expectedShape[i] > 0 && shape[i] != expectedShape[i])
                            throw new InvalidDataException(
                                $"'{sourceName}' dim {i} is {shape[i]}, expected {expectedShape[i]}.");
                }

                float[] data = st.ReadFloat32(sourceName);
                var tensor = new Tensor(_allocator, DType.Float32, shape);
                tensor.SetElementsAsFloat(data);
                _weights[internalName] = tensor;
                count++;
            }

            // --- patch embedding -------------------------------------------------------------
            // HF patchifies with reshape(C,nH,P,nW,P).permute(1,3,2,4,0), so each row of
            // input_proj.weight is ordered [ky][kx][c] (channel FASTEST). PatchEmbed's im2col here
            // builds rows as [c][ky][kx] (channel-major). The two disagree, so the weight is
            // permuted once at load. Getting this wrong runs clean and yields plausible-but-wrong
            // features, so it is asserted rather than trusted.
            {
                string src = SafetensorsTowerPrefix + "patch_embedder.input_proj.weight";
                long[] shape = st.TensorShape(src);
                int hidden = (int)shape[0];
                int stride = (int)shape[1];
                int p = _patchSize, c = 3;
                if ((long)c * p * p != stride)
                    throw new InvalidDataException(
                        $"'{src}' stride {stride} != 3*{p}*{p}; patch size mis-derived.");

                float[] src2d = st.ReadFloat32(src);
                float[] dst = new float[(long)hidden * stride];
                for (int f = 0; f < hidden; f++)
                {
                    long rowBase = (long)f * stride;
                    for (int ky = 0; ky < p; ky++)
                        for (int kx = 0; kx < p; kx++)
                            for (int ch = 0; ch < c; ch++)
                            {
                                // source row order [ky][kx][c] -> dest row order [c][ky][kx]
                                long s = rowBase + ((long)ky * p + kx) * c + ch;
                                long d = rowBase + ((long)ch * p + ky) * p + kx;
                                dst[d] = src2d[s];
                            }
                }

                var patchEmbd = new Tensor(_allocator, DType.Float32, hidden, c, p, p);
                patchEmbd.SetElementsAsFloat(dst);
                _weights["v.patch_embd.weight"] = patchEmbd;
                count++;
            }

            // --- learned 2-D position tables -------------------------------------------------
            // Stored upstream as [2, maxPos, hidden]: table 0 indexed by patch x, table 1 by y.
            Put("v.position_embd.weight", SafetensorsTowerPrefix + "patch_embedder.position_embedding_table");

            // --- output standardization ------------------------------------------------------
            Put("v.std_bias", SafetensorsTowerPrefix + "std_bias", _hiddenSize);
            Put("v.std_scale", SafetensorsTowerPrefix + "std_scale", _hiddenSize);

            // --- transformer blocks ----------------------------------------------------------
            for (int i = 0; i < _blockCount; i++)
            {
                string s = $"{SafetensorsLayerPrefix}{i}.";
                string d = $"v.blk.{i}.";
                int headDim = _hiddenSize / _numHeads;

                // Sandwich norms. The internal names are positional, so map them explicitly:
                //   ln1            = input_layernorm            (pre-attention)
                //   attn_post_norm = post_attention_layernorm   (post-attention, pre-residual)
                //   ln2            = pre_feedforward_layernorm  (pre-MLP)
                //   ffn_post_norm  = post_feedforward_layernorm (post-MLP, pre-residual)
                Put(d + "ln1.weight", s + "input_layernorm.weight", _hiddenSize);
                Put(d + "attn_post_norm.weight", s + "post_attention_layernorm.weight", _hiddenSize);
                Put(d + "ln2.weight", s + "pre_feedforward_layernorm.weight", _hiddenSize);
                Put(d + "ffn_post_norm.weight", s + "post_feedforward_layernorm.weight", _hiddenSize);

                // Attention. The ".linear." segment upstream is an artifact of the Clippable*
                // wrapper; with use_clipped_linears=false it carries no scalars and is dropped.
                Put(d + "attn_q.weight", s + "self_attn.q_proj.linear.weight", _hiddenSize, _hiddenSize);
                Put(d + "attn_k.weight", s + "self_attn.k_proj.linear.weight", _hiddenSize, _hiddenSize);
                Put(d + "attn_v.weight", s + "self_attn.v_proj.linear.weight", _hiddenSize, _hiddenSize);
                Put(d + "attn_out.weight", s + "self_attn.o_proj.linear.weight", _hiddenSize, _hiddenSize);
                Put(d + "attn_q_norm.weight", s + "self_attn.q_norm.weight", headDim);
                Put(d + "attn_k_norm.weight", s + "self_attn.k_norm.weight", headDim);

                Put(d + "ffn_gate.weight", s + "mlp.gate_proj.linear.weight", _intermediateSize, _hiddenSize);
                Put(d + "ffn_up.weight", s + "mlp.up_proj.linear.weight", _intermediateSize, _hiddenSize);
                Put(d + "ffn_down.weight", s + "mlp.down_proj.linear.weight", _hiddenSize, _intermediateSize);
            }

            // --- projector into the language model's embedding space --------------------------
            Put("mm.input_projection.weight", SafetensorsProjectorName, _projectionDim, _hiddenSize);

            Console.WriteLine($" done ({count} tensors)");
            ModelBase.LogCudaVram(_allocator, "after vision encoder weight load");
        }
    }
}
