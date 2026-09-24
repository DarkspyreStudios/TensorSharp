// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace TensorSharp.Models.QwenImage
{
    /// <summary>A single-frame VAE latent, planar CHW (Qwen-Image-2.1: 64 channels at 1/16 spatial resolution).</summary>
    public sealed class VaeLatent
    {
        public int Channels { get; }
        public int Height { get; }
        public int Width { get; }
        /// <summary>Planar CHW, length <c>Channels*Height*Width</c>.</summary>
        public float[] Data { get; }

        public VaeLatent(int channels, int height, int width, float[] data)
        {
            Channels = channels; Height = height; Width = width; Data = data;
        }
    }
}
