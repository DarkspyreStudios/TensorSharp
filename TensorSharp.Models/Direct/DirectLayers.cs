// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Threading.Tasks;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Direct
{
    /// <summary>
    /// A row-major token embedding table that remains on the selected direct
    /// backend. Input IDs are gathered without routing through a framework
    /// runtime.
    /// </summary>
    public sealed class DirectEmbedding : IDisposable
    {
        private readonly DirectContext _ctx;
        private readonly Tensor _weight;

        public long VocabularySize { get; }
        public long EmbeddingSize { get; }

        private DirectEmbedding(DirectContext ctx, Tensor weight, long vocabularySize, long embeddingSize)
        {
            _ctx = ctx;
            _weight = weight;
            VocabularySize = vocabularySize;
            EmbeddingSize = embeddingSize;
        }

        /// <summary>Load a row-major <c>[vocabulary, embedding]</c> table.</summary>
        public static DirectEmbedding FromFloats(
            DirectContext ctx,
            float[] weight,
            long vocabularySize,
            long embeddingSize)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            ArgumentNullException.ThrowIfNull(weight);
            if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
            if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            if (weight.LongLength != checked(vocabularySize * embeddingSize))
                throw new ArgumentException(
                    $"Embedding weight length {weight.LongLength} does not match " +
                    $"[{vocabularySize}, {embeddingSize}].",
                    nameof(weight));

            return new DirectEmbedding(
                ctx,
                ctx.FromFloats(weight, vocabularySize, embeddingSize),
                vocabularySize,
                embeddingSize);
        }

        /// <summary>Load a row-major <c>[vocabulary, embedding]</c> tensor.</summary>
        public static DirectEmbedding FromTensorStore(
            DirectContext ctx,
            IFloatTensorStore store,
            string weightName)
        {
            ArgumentNullException.ThrowIfNull(store);
            ArgumentException.ThrowIfNullOrEmpty(weightName);
            long[] shape = store.TensorShape(weightName);
            if (shape.Length != 2 || shape[0] <= 0 || shape[1] <= 0)
                throw new InvalidOperationException(
                    $"Direct embedding weight '{weightName}' must have shape [vocabulary, embedding].");
            return FromFloats(ctx, store.ReadFloat32(weightName), shape[0], shape[1]);
        }

        /// <summary>Gather token IDs into a new <c>[tokens, embedding]</c> F32 tensor.</summary>
        public Tensor Forward(ReadOnlySpan<int> tokenIds)
        {
            if (tokenIds.Length == 0)
                throw new ArgumentException("At least one token ID is required.", nameof(tokenIds));

            int[] ids = tokenIds.ToArray();
            for (int i = 0; i < ids.Length; i++)
            {
                if ((uint)ids[i] >= (ulong)VocabularySize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds),
                        $"Token ID {ids[i]} at index {i} is outside [0, {VocabularySize}).");
            }

            using var indices = new Tensor(_ctx.Allocator, DType.Int32, ids.LongLength);
            indices.SetElementsAsInt(ids);
            var result = _ctx.NewF32(ids.LongLength, EmbeddingSize);
            try
            {
                Ops.IndexSelect(result, _weight, indices);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        public void Dispose() => _weight.Dispose();
    }

    /// <summary>
    /// A channels-last 2D convolution for direct model graphs. Inputs and outputs
    /// use <c>[batch, height, width, channels]</c>; weights use the native
    /// PyTorch/safetensors layout <c>[out, in, kernelHeight, kernelWidth]</c>.
    /// </summary>
    public sealed class DirectConv2D : IDisposable
    {
        private const long DefaultWorkspaceBytes = 256L << 20;

        private readonly DirectContext _ctx;
        private readonly DirectLinear _linear;

        public int InputChannels { get; }
        public int OutputChannels { get; }
        public int KernelHeight { get; }
        public int KernelWidth { get; }

        private DirectConv2D(
            DirectContext ctx,
            DirectLinear linear,
            int inputChannels,
            int outputChannels,
            int kernelHeight,
            int kernelWidth)
        {
            _ctx = ctx;
            _linear = linear;
            InputChannels = inputChannels;
            OutputChannels = outputChannels;
            KernelHeight = kernelHeight;
            KernelWidth = kernelWidth;
        }

        /// <summary>
        /// Create a convolution from a flattened row-major
        /// <c>[out, in, kernelHeight, kernelWidth]</c> weight.
        /// </summary>
        public static DirectConv2D FromFloats(
            DirectContext ctx,
            float[] weight,
            int outputChannels,
            int inputChannels,
            int kernelHeight,
            int kernelWidth,
            float[] bias = null)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            ArgumentNullException.ThrowIfNull(weight);
            if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
            if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
            if (kernelHeight <= 0) throw new ArgumentOutOfRangeException(nameof(kernelHeight));
            if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth));

            long flattened = checked((long)inputChannels * kernelHeight * kernelWidth);
            if (weight.LongLength != checked((long)outputChannels * flattened))
                throw new ArgumentException(
                    "Convolution weight length does not match " +
                    $"[{outputChannels}, {inputChannels}, {kernelHeight}, {kernelWidth}].",
                    nameof(weight));
            if (bias != null && bias.LongLength != outputChannels)
                throw new ArgumentException(
                    $"Convolution bias must have {outputChannels} values.",
                    nameof(bias));

            return new DirectConv2D(
                ctx,
                DirectLinear.FromFloats(ctx, weight, flattened, outputChannels, bias),
                inputChannels,
                outputChannels,
                kernelHeight,
                kernelWidth);
        }

        /// <summary>Load a PyTorch-layout convolution weight and optional bias.</summary>
        public static DirectConv2D FromTensorStore(
            DirectContext ctx,
            IFloatTensorStore store,
            string weightName,
            string biasName = null)
        {
            ArgumentNullException.ThrowIfNull(store);
            ArgumentException.ThrowIfNullOrEmpty(weightName);
            long[] shape = store.TensorShape(weightName);
            if (shape.Length != 4 || Array.Exists(shape, value => value <= 0))
                throw new InvalidOperationException(
                    $"Direct convolution weight '{weightName}' must have shape " +
                    "[out, in, kernelHeight, kernelWidth].");
            if (Array.Exists(shape, value => value > int.MaxValue))
                throw new NotSupportedException(
                    $"Direct convolution weight '{weightName}' has a dimension larger than Int32.MaxValue.");

            float[] bias = null;
            if (biasName != null && store.HasTensor(biasName))
            {
                long[] biasShape = store.TensorShape(biasName);
                if (biasShape.Length != 1 || biasShape[0] != shape[0])
                    throw new InvalidOperationException(
                        $"Direct convolution bias '{biasName}' must have shape [{shape[0]}].");
                bias = store.ReadFloat32(biasName);
            }

            return FromFloats(
                ctx,
                store.ReadFloat32(weightName),
                checked((int)shape[0]),
                checked((int)shape[1]),
                checked((int)shape[2]),
                checked((int)shape[3]),
                bias);
        }

        /// <summary>
        /// Apply the convolution. Padding and stride are symmetric per spatial
        /// axis. The im2col workspace is banded to <paramref name="workspaceBytes"/>.
        /// </summary>
        public Tensor Forward(
            Tensor input,
            int paddingHeight = 0,
            int paddingWidth = 0,
            int strideHeight = 1,
            int strideWidth = 1,
            long workspaceBytes = DefaultWorkspaceBytes)
        {
            ArgumentNullException.ThrowIfNull(input);
            if (input.DimensionCount != 4)
                throw new ArgumentException(
                    "DirectConv2D input must have shape [batch, height, width, channels].",
                    nameof(input));
            if (input.ElementType != DType.Float32 || !input.IsContiguous())
                throw new ArgumentException("DirectConv2D input must be contiguous F32.", nameof(input));
            if (input.Sizes[3] != InputChannels)
                throw new ArgumentException(
                    $"DirectConv2D expected {InputChannels} channels but received {input.Sizes[3]}.",
                    nameof(input));
            if (!ReferenceEquals(input.Allocator, _ctx.Allocator))
                throw new ArgumentException("Input must use the DirectContext allocator.", nameof(input));
            if (paddingHeight < 0) throw new ArgumentOutOfRangeException(nameof(paddingHeight));
            if (paddingWidth < 0) throw new ArgumentOutOfRangeException(nameof(paddingWidth));
            if (strideHeight <= 0) throw new ArgumentOutOfRangeException(nameof(strideHeight));
            if (strideWidth <= 0) throw new ArgumentOutOfRangeException(nameof(strideWidth));
            if (workspaceBytes <= 0) throw new ArgumentOutOfRangeException(nameof(workspaceBytes));

            long batch = input.Sizes[0];
            long height = input.Sizes[1];
            long width = input.Sizes[2];
            long outputHeight = (height + 2L * paddingHeight - KernelHeight) / strideHeight + 1;
            long outputWidth = (width + 2L * paddingWidth - KernelWidth) / strideWidth + 1;
            if (outputHeight <= 0 || outputWidth <= 0)
                throw new ArgumentException("Convolution kernel is larger than the padded input.", nameof(input));

            long columns = checked((long)InputChannels * KernelHeight * KernelWidth);
            long bandHeight = Math.Max(
                1,
                Math.Min(
                    outputHeight,
                    workspaceBytes / Math.Max(1, checked(sizeof(float) * outputWidth * columns))));
            var output = _ctx.NewF32(batch, outputHeight, outputWidth, OutputChannels);
            try
            {
                for (long batchIndex = 0; batchIndex < batch; batchIndex++)
                {
                    using var inputFrame = input.Select(0, batchIndex);
                    using var outputFrame = output.Select(0, batchIndex);
                    using var outputRows = outputFrame.View(outputHeight * outputWidth, OutputChannels);
                    for (long outputY = 0; outputY < outputHeight; outputY += bandHeight)
                    {
                        long rows = Math.Min(bandHeight, outputHeight - outputY);
                        using var columnRows = _ctx.NewF32(rows * outputWidth, columns);
                        Im2Col(
                            columnRows,
                            inputFrame,
                            height,
                            width,
                            outputHeight,
                            outputWidth,
                            paddingHeight,
                            paddingWidth,
                            strideHeight,
                            strideWidth,
                            outputY,
                            rows);
                        using var projected = _linear.Forward(columnRows);
                        using var target = outputRows.Narrow(0, outputY * outputWidth, rows * outputWidth);
                        Ops.Copy(target, projected);
                    }
                }

                return output;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }

        private void Im2Col(
            Tensor columns,
            Tensor input,
            long height,
            long width,
            long outputHeight,
            long outputWidth,
            int paddingHeight,
            int paddingWidth,
            int strideHeight,
            int strideWidth,
            long outputY,
            long bandHeight)
        {
            if (_ctx.IsCuda)
            {
                if (!CudaWanOps.TryIm2Col(
                        columns,
                        input,
                        InputChannels,
                        checked((int)height),
                        checked((int)width),
                        KernelHeight,
                        KernelWidth,
                        checked((int)outputHeight),
                        checked((int)outputWidth),
                        paddingHeight,
                        paddingWidth,
                        strideHeight,
                        strideWidth,
                        checked((int)outputY),
                        checked((int)bandHeight)))
                    throw new InvalidOperationException("CUDA im2col kernel unavailable (stale PTX?).");
                return;
            }

            Im2ColCpu(
                columns,
                input,
                height,
                width,
                outputWidth,
                paddingHeight,
                paddingWidth,
                strideHeight,
                strideWidth,
                outputY,
                bandHeight);
        }

        private unsafe void Im2ColCpu(
            Tensor columns,
            Tensor input,
            long height,
            long width,
            long outputWidth,
            int paddingHeight,
            int paddingWidth,
            int strideHeight,
            int strideWidth,
            long outputY,
            long bandHeight)
        {
            float* source = (float*)CpuNativeHelpers.GetBufferStart(input);
            float* destination = (float*)CpuNativeHelpers.GetBufferStart(columns);
            long flattened = (long)InputChannels * KernelHeight * KernelWidth;
            Parallel.For(0, checked((int)(bandHeight * outputWidth)), position =>
            {
                long oy = outputY + position / outputWidth;
                long ox = position % outputWidth;
                float* row = destination + position * flattened;
                long index = 0;
                for (int channel = 0; channel < InputChannels; channel++)
                {
                    for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                    {
                        long inputY = oy * strideHeight - paddingHeight + kernelY;
                        for (int kernelX = 0; kernelX < KernelWidth; kernelX++, index++)
                        {
                            long inputX = ox * strideWidth - paddingWidth + kernelX;
                            row[index] = inputY >= 0 && inputY < height && inputX >= 0 && inputX < width
                                ? source[(inputY * width + inputX) * InputChannels + channel]
                                : 0f;
                        }
                    }
                }
            });
        }

        public void Dispose() => _linear.Dispose();
    }

    /// <summary>Backend-neutral image operations for channels-last direct graphs.</summary>
    public static class DirectImageOps
    {
        /// <summary>
        /// Group normalization over contiguous F32
        /// <c>[batch, height, width, channels]</c> input.
        /// </summary>
        public static Tensor GroupNorm(
            DirectContext ctx,
            Tensor input,
            int groups,
            Tensor gain = null,
            Tensor bias = null,
            float epsilon = 1e-5f)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            ArgumentNullException.ThrowIfNull(input);
            ValidateImage(ctx, input);
            if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));
            if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));

            long batch = input.Sizes[0];
            long height = input.Sizes[1];
            long width = input.Sizes[2];
            long channels = input.Sizes[3];
            if (channels % groups != 0)
                throw new ArgumentException(
                    $"Channel count {channels} must be divisible by group count {groups}.",
                    nameof(groups));
            ValidateAffine(gain, channels, nameof(gain));
            ValidateAffine(bias, channels, nameof(bias));

            long channelsPerGroup = channels / groups;
            using var groupedView = input.View(batch, height, width, groups, channelsPerGroup);
            using var groupedPermutation = groupedView.Permute(0, 3, 1, 2, 4);
            using var grouped = Ops.NewContiguous(groupedPermutation);
            using var groupedRows = grouped.View(batch * groups, height * width * channelsPerGroup);
            using var normalizedRows = DirectOps.LayerNorm(ctx, groupedRows, null, null, epsilon);
            using var normalizedGroups = normalizedRows.View(batch, groups, height, width, channelsPerGroup);
            using var outputPermutation = normalizedGroups.Permute(0, 2, 3, 1, 4);
            using var outputGroups = Ops.NewContiguous(outputPermutation);
            Tensor output = outputGroups.View(batch, height, width, channels);
            try
            {
                using var outputRows = output.View(batch * height * width, channels);
                if (gain != null) DirectOps.MulColsRows(ctx, outputRows, gain);
                if (bias != null) DirectOps.AddBiasRows(ctx, outputRows, bias);
                return output;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Nearest-neighbour spatial upsample of contiguous F32
        /// <c>[batch, height, width, channels]</c> input by an integer scale.
        /// </summary>
        public static Tensor NearestUpsample2D(DirectContext ctx, Tensor input, int scale = 2)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            ArgumentNullException.ThrowIfNull(input);
            ValidateImage(ctx, input);
            if (scale <= 0) throw new ArgumentOutOfRangeException(nameof(scale));
            if (scale == 1) return Ops.NewContiguous(input);

            long batch = input.Sizes[0];
            long height = input.Sizes[1];
            long width = input.Sizes[2];
            long channels = input.Sizes[3];

            using var widthRows = input.View(batch * height * width, 1, channels);
            Tensor widthExpanded = null;
            try
            {
                widthExpanded = widthRows.CopyRef();
                for (int i = 1; i < scale; i++)
                {
                    Tensor previous = widthExpanded;
                    widthExpanded = Ops.Concat(null, 1, previous, widthRows);
                    previous.Dispose();
                }

                using var wide = widthExpanded.View(batch, height, width * scale, channels);
                using var heightRows = wide.View(batch * height, 1, width * scale * channels);
                Tensor heightExpanded = null;
                try
                {
                    heightExpanded = heightRows.CopyRef();
                    for (int i = 1; i < scale; i++)
                    {
                        Tensor previous = heightExpanded;
                        heightExpanded = Ops.Concat(null, 1, previous, heightRows);
                        previous.Dispose();
                    }

                    using var resultView = heightExpanded.View(
                        batch,
                        height * scale,
                        width * scale,
                        channels);
                    return Ops.NewContiguous(resultView);
                }
                finally
                {
                    heightExpanded?.Dispose();
                }
            }
            finally
            {
                widthExpanded?.Dispose();
            }
        }

        private static void ValidateImage(DirectContext ctx, Tensor input)
        {
            if (input.DimensionCount != 4)
                throw new ArgumentException(
                    "Image tensor must have shape [batch, height, width, channels].",
                    nameof(input));
            if (input.ElementType != DType.Float32 || !input.IsContiguous())
                throw new ArgumentException("Image tensor must be contiguous F32.", nameof(input));
            if (!ReferenceEquals(input.Allocator, ctx.Allocator))
                throw new ArgumentException("Image tensor must use the DirectContext allocator.", nameof(input));
        }

        private static void ValidateAffine(Tensor value, long channels, string name)
        {
            if (value == null) return;
            if (value.ElementType != DType.Float32 || !value.IsContiguous() ||
                value.DimensionCount != 1 || value.Sizes[0] != channels)
                throw new ArgumentException(
                    $"{name} must be a contiguous F32 tensor with shape [{channels}].",
                    name);
        }
    }
}
