// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;

namespace TensorSharp.Models
{
    /// <summary>Official V4.1 RGB pad/resize, BF16 normalization and patch ordering.</summary>
    public sealed class DeepSeek41ImageProcessor
    {
        public int PatchSize { get; }
        public int DownsampleRatio { get; }
        public int MinPixels { get; }
        public int MaxImageTokens { get; }
        public int MaxWidthHeightRatio { get; }

        public DeepSeek41ImageProcessor(int patchSize = 14, int downsampleRatio = 3,
            int minPixels = 295936, int maxImageTokens = 1024, int maxWidthHeightRatio = 0)
        {
            if (patchSize <= 0 || downsampleRatio <= 0 || minPixels <= 0 ||
                maxImageTokens < 4 || maxWidthHeightRatio < 0)
                throw new ArgumentOutOfRangeException(nameof(patchSize), "Invalid V4.1 image geometry.");
            PatchSize = patchSize;
            DownsampleRatio = downsampleRatio;
            MinPixels = minPixels;
            MaxImageTokens = maxImageTokens;
            MaxWidthHeightRatio = maxWidthHeightRatio;
        }

        public readonly record struct Grid(int Width, int Height, int PatchColumns, int PatchRows,
            int ImageColumns, int ImageRows)
        {
            public int TokenCount => checked(ImageRows * (ImageColumns + 1) + 2);
        }

        public Grid Plan(int width, int height)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Image dimensions must be positive.");
            if (MaxWidthHeightRatio > 0 && width > (long)height * MaxWidthHeightRatio)
                width = checked(height * MaxWidthHeightRatio);
            if ((long)width * height < MinPixels)
            {
                double ratio = Math.Sqrt((double)MinPixels / ((long)width * height));
                width = checked((int)(width * ratio));
                height = checked((int)(height * ratio));
            }
            int bestWidth = checked((int)(((long)width + PatchSize - 1) / PatchSize * PatchSize));
            int bestHeight = checked((int)(((long)height + PatchSize - 1) / PatchSize * PatchSize));
            if (TokenCount(bestWidth, bestHeight) > MaxImageTokens)
            {
                double aspect = (double)height / width;
                double maxWidth = Math.Sqrt((MaxImageTokens - 2) / aspect + 0.25) - 0.5;
                double maxHeight = maxWidth * aspect;
                int cell = checked(PatchSize * DownsampleRatio);
                if (maxWidth < 1)
                {
                    bestHeight = checked((MaxImageTokens - 2) / 2 * cell);
                    bestWidth = cell;
                }
                else if (maxHeight < 1)
                {
                    bestHeight = cell;
                    bestWidth = checked((MaxImageTokens - 3) * cell);
                }
                else
                {
                    double beta = Math.Min(Math.Floor(maxWidth) * cell / width, Math.Floor(maxHeight) * cell / height);
                    bestHeight = checked((int)Math.Floor(height * beta / PatchSize) * PatchSize);
                    bestWidth = checked((int)Math.Floor(width * beta / PatchSize) * PatchSize);
                }
            }
            int rows = bestHeight / PatchSize, cols = bestWidth / PatchSize;
            var result = new Grid(bestWidth, bestHeight, cols, rows,
                (cols + DownsampleRatio - 1) / DownsampleRatio, (rows + DownsampleRatio - 1) / DownsampleRatio);
            if (rows <= 0 || cols <= 0 || result.TokenCount > MaxImageTokens)
                throw new InvalidDataException("Image aspect ratio cannot fit the V4.1 image token budget.");
            return result;
        }

        private long TokenCount(int width, int height)
            => ((long)height / PatchSize + DownsampleRatio - 1) / DownsampleRatio *
                (((long)width / PatchSize + DownsampleRatio - 1) / DownsampleRatio + 1) + 2;

        public (float[] Patches, Grid Grid) ProcessImage(string path)
        {
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(File.ReadAllBytes(path), out int width, out int height);
            return ProcessRgba(rgba, width, height);
        }

        internal (float[] Patches, Grid Grid) ProcessRgba(byte[] rgba, int width, int height)
        {
            ArgumentNullException.ThrowIfNull(rgba);
            if (width <= 0 || height <= 0 || (long)width * height * 4 != rgba.Length)
                throw new ArgumentException("RGBA length does not match the image dimensions.");
            Grid grid = Plan(width, height);
            byte[] rgb = new byte[checked(width * height * 3)];
            // PIL convert("RGB") discards alpha; it does not composite over a background.
            for (int i = 0, j = 0; i < rgba.Length; i += 4, j += 3)
            {
                rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
            }

            int resizedWidth = grid.Width, resizedHeight = grid.Height;
            bool stretch = MaxWidthHeightRatio > 0 && width >= (long)MaxWidthHeightRatio * height;
            if (!stretch && (double)width / height != (double)grid.Width / grid.Height)
            {
                if ((double)width / height > (double)grid.Width / grid.Height)
                    resizedHeight = (int)Math.Round((double)height / width * grid.Width);
                else
                    resizedWidth = (int)Math.Round((double)width / height * grid.Height);
            }
            if (resizedWidth <= 0 || resizedHeight <= 0)
                throw new InvalidDataException("Image aspect ratio produces an empty resized image.");
            rgb = ResizeRgb(rgb, width, height, resizedWidth, resizedHeight);
            int offsetX = (int)Math.Round((grid.Width - resizedWidth) * 0.5);
            int offsetY = (int)Math.Round((grid.Height - resizedHeight) * 0.5);
            var patches = new float[checked(grid.Width * grid.Height * 3)];
            int dst = 0;
            // Patch rows/columns, then CHW inside each 14x14 patch.
            for (int py = 0; py < grid.PatchRows; py++)
                for (int px = 0; px < grid.PatchColumns; px++)
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < PatchSize; y++)
                            for (int x = 0; x < PatchSize; x++)
                            {
                                int sx = px * PatchSize + x - offsetX, sy = py * PatchSize + y - offsetY;
                                byte value = sx >= 0 && sx < resizedWidth && sy >= 0 && sy < resizedHeight
                                    ? rgb[(sy * resizedWidth + sx) * 3 + c] : (byte)127;
                                float normalized = (value / 255f - 0.5f) / 0.5f;
                                uint bits = BitConverter.SingleToUInt32Bits(normalized);
                                bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
                                patches[dst++] = BitConverter.UInt32BitsToSingle(bits);
                            }
            return (patches, grid);
        }

        // Match Pillow's default RGB BICUBIC path: pixel-center coordinates,
        // wider support when shrinking, 22-bit coefficients, clipped uint8 after
        // EACH separable pass. A single floating-point pass differs at edges.
        // Algorithm reference: python-pillow/Pillow src/libImaging/Resample.c.
        private static byte[] ResizeRgb(byte[] source, int width, int height, int targetWidth, int targetHeight)
        {
            if (width != targetWidth)
            {
                var filter = Coefficients(width, targetWidth);
                var result = new byte[checked(targetWidth * height * 3)];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < targetWidth; x++)
                        for (int c = 0; c < 3; c++)
                        {
                            var (start, weights) = filter[x];
                            int sum = 1 << 21;
                            for (int k = 0; k < weights.Length; k++)
                                sum += source[(y * width + start + k) * 3 + c] * weights[k];
                            result[(y * targetWidth + x) * 3 + c] = (byte)Math.Clamp(sum >> 22, 0, 255);
                        }
                source = result;
                width = targetWidth;
            }
            if (height != targetHeight)
            {
                var filter = Coefficients(height, targetHeight);
                var result = new byte[checked(width * targetHeight * 3)];
                for (int y = 0; y < targetHeight; y++)
                    for (int x = 0; x < width; x++)
                        for (int c = 0; c < 3; c++)
                        {
                            var (start, weights) = filter[y];
                            int sum = 1 << 21;
                            for (int k = 0; k < weights.Length; k++)
                                sum += source[((start + k) * width + x) * 3 + c] * weights[k];
                            result[(y * width + x) * 3 + c] = (byte)Math.Clamp(sum >> 22, 0, 255);
                        }
                source = result;
            }
            return source;
        }

        private static (int Start, int[] Weights)[] Coefficients(int input, int output)
        {
            var result = new (int, int[])[output];
            double scale = (double)input / output, filterScale = Math.Max(1, scale), support = 2 * filterScale;
            for (int i = 0; i < output; i++)
            {
                double center = (i + 0.5) * scale;
                int start = Math.Max(0, (int)(center - support + 0.5));
                int end = Math.Min(input, (int)(center + support + 0.5));
                var weights = new double[end - start];
                double sum = 0;
                for (int k = 0; k < weights.Length; k++)
                {
                    double x = Math.Abs((k + start - center + 0.5) / filterScale);
                    weights[k] = x < 1 ? (1.5 * x - 2.5) * x * x + 1
                        : x < 2 ? (((x - 5) * x + 8) * x - 4) * -0.5 : 0;
                    sum += weights[k];
                }
                var quantized = new int[weights.Length];
                for (int k = 0; k < weights.Length; k++)
                {
                    double value = sum == 0 ? weights[k] : weights[k] / sum;
                    quantized[k] = (int)(value * (1 << 22) + (value < 0 ? -0.5 : 0.5));
                }
                result[i] = (start, quantized);
            }
            return result;
        }
    }
}
