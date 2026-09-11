// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Globalization;
using System.IO;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>V4.1 selects its own native graph; it cannot run the V4 graph.</summary>
    internal static class DeepSeek41Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "deepseek41",
            DisplayName = "DeepSeek V4.1 Flash",
            Aliases = new[] { "deepseek41" },
            MultiGpu = MultiGpuMode.LayerSplit,
            MultiGpuLimitation = "DeepSeek V4.1 (deepseek41) uses a single-process native executor with layer placement and optional routed-MoE tensor parallelism; full attention tensor parallelism and distributed groups are not implemented.",
            DescribeMultiGpuPlacement = DescribePlacement,
            ApplyNativeTunables = c => ValidateLoad(c.GgufPath, c.Backend, c.DraftModelPath, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "deepseek41.vision.gguf" },
            Factory = c => new DeepSeek41Model(c.GgufPath, c.Backend,
                Math.Max(c.TpDegree, c.LayerSplitDegree), c.TpGroup, c.DraftModelPath),
        };

        internal static void ValidateLoad(string ggufPath, BackendType backend, string draftModelPath,
            int requestedGpuCount = 1, ITensorParallelGroup tpGroup = null)
        {
            if (backend != BackendType.GgmlCuda)
                throw new NotSupportedException(
                    "DeepSeek V4.1 Flash requires --backend ggml_cuda. The managed CPU and direct CUDA executors implement V4, not V4.1.");
            if (tpGroup != null)
                throw new NotSupportedException(
                    "DeepSeek V4.1 uses a single-process native executor and does not support distributed tensor-parallel groups. " +
                    "Start without --tp-node-id/--tp-peers.");
            if (!string.IsNullOrWhiteSpace(draftModelPath) ||
                !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_DSV4_DSPARK")))
                throw new NotSupportedException("DeepSeek V4.1 DSpark speculative decoding is not implemented; omit the draft model.");

            ResolveRoutedMoeTensorParallelRanks(requestedGpuCount);

            string sidecar = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(ggufPath))!, "deepseek41.engram.bin");
            if (!File.Exists(sidecar))
                throw new FileNotFoundException(
                    "DeepSeek V4.1 requires its tokenizer-derived Engram lookup sidecar. " +
                    "Run python eng/dsv41-prepare.py --help for preparation instructions.", sidecar);
        }

        internal static int ResolveRoutedMoeTensorParallelRanks(int requestedGpuCount)
            => ParseRoutedMoeTensorParallelRanks(Environment.GetEnvironmentVariable("TS_DSV41_TP"),
                ResolveSelectedGpuCount(requestedGpuCount));

        internal static int ParseRoutedMoeTensorParallelRanks(string raw, int selectedGpuCount)
        {
            if (raw == null)
                return 0;

            // Match the native whole-value strtol check, including its rejection of
            // empty values and trailing characters. Native remains authoritative
            // when GPU count is automatic and after visible devices are enumerated.
            if (!int.TryParse(raw, NumberStyles.AllowLeadingWhite | NumberStyles.AllowLeadingSign,
                    CultureInfo.InvariantCulture, out int ranks) || ranks < 0 || ranks == 1 || ranks > 8)
                throw new ArgumentException("TS_DSV41_TP must be 0 (disabled) or an integer from 2 through 8.");

            if (ranks > 0 && selectedGpuCount > 0 && ranks != selectedGpuCount)
                throw new ArgumentException(
                    $"TS_DSV41_TP={ranks} must equal the selected GPU count ({selectedGpuCount}); " +
                    "set --tp and TS_DSV4_NGPU consistently. TS_DSV4_NGPU overrides --tp.");
            return ranks;
        }

        private static int ResolveSelectedGpuCount(int requestedGpuCount)
            => int.TryParse(Environment.GetEnvironmentVariable("TS_DSV4_NGPU"), out int count)
                ? Math.Max(0, count) : requestedGpuCount > 1 ? requestedGpuCount : 0;

        private static string DescribePlacement(int requestedGpuCount)
        {
            int ranks = ResolveRoutedMoeTensorParallelRanks(requestedGpuCount);
            int count = ResolveSelectedGpuCount(requestedGpuCount);
            if (ranks == 0)
                return $"  Multi-GPU: DeepSeek V4.1 uses {(count > 0 ? count + " GPUs" : "automatically selected visible GPUs")} by LAYER SPLIT (whole-layer placement). " +
                    "TS_DSV41_TP=2..8 enables routed-MoE tensor parallelism with a matching GPU count.";

            return $"  Multi-GPU: DeepSeek V4.1 routed-MoE tensor parallelism across {ranks} GPUs: " +
                "gate/up/down expert dimensions are sharded, with host-staged F32 reduction. " +
                "Attention and shared experts retain layer placement; CPU-offloaded layers retain whole CPU experts. " +
                "Full attention tensor parallelism and distributed groups are not supported.";
        }
    }
}
