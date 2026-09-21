// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the root of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class QwenImage21Conditioner : IDisposable
    {
        internal const string SystemPrompt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
        private readonly QwenImageTextEncoder _text;
        private readonly string _visionPath;
        private Qwen35VisionEncoder _vision;
        private readonly Dictionary<RgbImage, float[][]> _imageCache = new();

        public QwenImage21Conditioner(string textGguf, string visionGguf, BackendType backend)
        {
            _text = new QwenImageTextEncoder(textGguf, backend, qwenImage21: true);
            if (_text.HiddenSize != 4096)
                throw new ArgumentException("Qwen-Image-2.1 requires a Qwen3-VL-8B encoder with 4096 hidden channels.");
            _visionPath = visionGguf;
        }

        public (float[] Embeddings, int SequenceLength, int[] ImageSlots) EncodePrompt(string prompt, RgbImage[] refs = null)
        {
            int drop = _text.Tokenizer.Encode(SystemPrompt, addSpecial: false).Count;
            var template = new StringBuilder(SystemPrompt).Append("<|im_start|>user\n");
            var images = new List<ImageCond>();
            if (refs != null && refs.Length > 0)
            {
                if (string.IsNullOrWhiteSpace(_visionPath))
                    throw new InvalidOperationException("Qwen-Image-2.1 editing requires a Qwen3-VL-8B vision projector.");
                _vision ??= new Qwen35VisionEncoder(_visionPath, _text.ConditionerAllocator, qwenImage21: true);
                foreach (var image in refs)
                {
                    if (image.Width % 32 != 0 || image.Height % 32 != 0)
                        throw new ArgumentException("Qwen-Image-2.1 reference dimensions must be multiples of 32.");
                    if (!_imageCache.TryGetValue(image, out var features))
                    {
                        float[] pixels = image.ToPlanarChw();
                        int hw = image.Width * image.Height;
                        for (int j = 0; j < pixels.Length; j++)
                        {
                            float alpha = image.Alpha == null ? 1f : image.Alpha[j % hw];
                            pixels[j] = 2f * (pixels[j] * alpha + 1f - alpha) - 1f;
                        }
                        var tensors = _vision.EncodeWithDeepStack(pixels, image.Height, image.Width);
                        try { features = tensors.Select(t => t.GetElementsAsFloat((int)t.ElementCount())).ToArray(); }
                        finally { foreach (var tensor in tensors) tensor.Dispose(); }
                        if (features.Length != 4) throw new InvalidOperationException("Qwen3-VL vision projector must provide three DeepStack mergers.");
                        _imageCache[image] = features;
                    }
                    int index = images.Count + 1;
                    if (index > 1) template.Append(' ');
                    template.Append($"<image{index}><|vision_start|>");
                    int start = _text.Tokenizer.Encode(template.ToString(), addSpecial: false).Count;
                    int count = image.Width / 32 * (image.Height / 32);
                    if (features[0].Length != count * 4096) throw new InvalidOperationException("Vision embedding shape does not match reference geometry.");
                    images.Add(new ImageCond { Start = start, Count = count, GridH = image.Height / 16,
                        GridW = image.Width / 16, Embeds = features[0], DeepStack = features.Skip(1).ToArray() });
                    for (int j = 0; j < count; j++) template.Append("<|image_pad|>");
                    template.Append("<|vision_end|>");
                }
            }
            template.Append(string.IsNullOrEmpty(prompt) ? " " : prompt).Append("<|im_end|>\n<|im_start|>assistant\n");
            int[] tokens = _text.Tokenizer.Encode(template.ToString(), addSpecial: false).ToArray();
            float[] hidden = _text.EncodeHidden(tokens, images.ToArray());
            int seq = tokens.Length - drop;
            var embeddings = new float[seq * 4096];
            Array.Copy(hidden, drop * 4096, embeddings, 0, embeddings.Length);
            var slots = new int[seq];
            for (int i = 0; i < images.Count; i++)
                Array.Fill(slots, i + 1, images[i].Start - drop, images[i].Count);
            return (embeddings, seq, slots);
        }

        public void Dispose()
        {
            _vision?.Dispose();
            _text.Dispose();
            _imageCache.Clear();
        }
    }
}
