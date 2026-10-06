// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the root of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Text;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class QwenImage21Conditioner : IDisposable
    {
        internal const string SystemPrompt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
        private readonly QwenImageTextEncoder _text;
        private readonly string _visionPath;
        private readonly NativeConstructionCleanupHandle _cleanup;
        private NativeConstructionCleanupHandle _pendingVisionConstruction;
        private bool _retirementStarted;
        private bool _released;
        private Exception _unsafeFailure;
        private Qwen35VisionEncoder _vision;
        private readonly List<Qwen35VisionEncoder> _ownedVisionEncoders = new();
        private readonly Dictionary<RgbImage, float[][]> _imageCache = new();

        public QwenImage21Conditioner(string textGguf, string visionGguf, BackendType backend)
        {
            _cleanup = new NativeConstructionCleanupHandle(this, Release, () => _released,
                () => _unsafeFailure != null);
            try
            {
                _text = new QwenImageTextEncoder(textGguf, backend);
                _text.AttachConditioner(this);
                if (_text.HiddenSize != 4096)
                    throw new ArgumentException("Qwen-Image-2.1 requires a Qwen3-VL-8B encoder with 4096 hidden channels.");
                _visionPath = visionGguf;
            }
            catch (Exception loadError)
            {
                // No returned text model means its constructor already owns any recovery carrier.
                if (_text == null) throw;
                try { _cleanup.Dispose(); }
                catch (Exception cleanup) { throw new NativeConstructionCleanupException(loadError, cleanup, _cleanup); }
                throw;
            }
        }

        internal bool IsReleased => _released;

        public (float[] Embeddings, int SequenceLength, int[] ImageSlots) EncodePrompt(string prompt, RgbImage[] refs = null)
        {
            if (_retirementStarted)
                throw new ObjectDisposedException(nameof(QwenImage21Conditioner), "Conditioner retirement has fenced new execution.");
            _text.ThrowIfOwnershipCleanupFailed();
            int drop = _text.Tokenizer.Encode(SystemPrompt, addSpecial: false).Count;
            var template = new StringBuilder(SystemPrompt).Append("<|im_start|>user\n");
            var images = new List<ImageCond>();
            if (refs != null && refs.Length > 0)
            {
                if (string.IsNullOrWhiteSpace(_visionPath))
                    throw new InvalidOperationException("Qwen-Image-2.1 editing requires a Qwen3-VL-8B vision projector.");
                EnsureVisionEncoder();
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
                        features = _vision.EncodeWithDeepStack(pixels, image.Height, image.Width, _cleanup);
                        if (features.Length != 4) throw new InvalidOperationException("Qwen3-VL vision projector must provide three DeepStack mergers.");
                        _imageCache[image] = features;
                    }
                    int index = images.Count + 1;
                    if (index > 1) template.Append(' ');
                    template.Append($"<image{index}><|vision_start|>");
                    int start = _text.Tokenizer.Encode(template.ToString(), addSpecial: false).Count;
                    int count = image.Width / 32 * (image.Height / 32);
                    if (features[0].Length != count * 4096) throw new InvalidOperationException("Vision embedding shape does not match reference geometry.");
                    images.Add(new ImageCond
                    {
                        Start = start,
                        Count = count,
                        GridH = image.Height / 16,
                        GridW = image.Width / 16,
                        Embeds = features[0],
                        DeepStack = features.Skip(1).ToArray()
                    });
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
            if (_released) return;
            if (_retirementStarted)
                throw new InvalidOperationException("Conditioner cleanup is incomplete; use its retained cleanup handle for explicit release.");
            _retirementStarted = true;
            MediaConstructionCleanup.RequireReleasedConstruction(ref _pendingVisionConstruction);
            _cleanup.Dispose();
        }

        private void Release()
        {
            if (_released) return;
            if (_unsafeFailure != null) ExceptionDispatchInfo.Capture(_unsafeFailure).Throw();
            _retirementStarted = true;
            // Each child handle retains its exact checked progress. Text never releases first.
            foreach (var vision in _ownedVisionEncoders) vision.ConstructionCleanup.Dispose();
            try { _text.Dispose(); }
            catch (Exception cleanup)
            {
                // Restoration can fail after the text owner has already proven complete release.
                if (_text.OwnershipResourcesReleased) CompleteRelease();
                else if (_text.IsOwnershipCleanupUnsafe) _unsafeFailure = cleanup;
                throw;
            }
            CompleteRelease();
        }

        private void CompleteRelease()
        {
            _imageCache.Clear();
            _pendingVisionConstruction = null;
            _vision = null;
            _ownedVisionEncoders.Clear();
            _released = true;
            _cleanup.CompleteRelease(this);
        }

        internal void DisposeAfterFailure(Exception operationError)
        {
            // Keep the same composite carrier when the lazy construction already exposed it.
            if (operationError is NativeConstructionCleanupException failure
                && ReferenceEquals(failure.Cleanup, _cleanup)) return;
            try { Dispose(); }
            catch (Exception cleanup) { throw new NativeConstructionCleanupException(operationError, cleanup, _cleanup); }
        }

        private void EnsureVisionEncoder()
        {
            if (_vision != null) return;
            MediaConstructionCleanup.RequireReleasedConstruction(ref _pendingVisionConstruction);
            try
            {
                _ownedVisionEncoders.EnsureCapacity(checked(_ownedVisionEncoders.Count + 1));
                var vision = new Qwen35VisionEncoder(_visionPath, _text.ConditionerAllocator, true,
                    RetainVisionConstruction, RetainUnsafeVisionConstruction);
                _vision = vision;
                _pendingVisionConstruction = null;
            }
            catch (Exception loadError)
            {
                // Normal disposal refuses a retained failed child; only the composite handle retries it.
                if (_pendingVisionConstruction != null && !_pendingVisionConstruction.IsReleased)
                    DisposeAfterFailure(loadError);
                throw;
            }
        }

        private void RetainVisionConstruction(Qwen35VisionEncoder vision)
        {
            _ownedVisionEncoders.Add(vision);
            _pendingVisionConstruction = vision.ConstructionCleanup;
            vision.SetHostModel(_text);
        }

        private void RetainUnsafeVisionConstruction(Qwen35VisionEncoder vision, Exception error)
        {
            _unsafeFailure ??= error;
            _text.RetainConditionerCleanupFailure(vision, error);
        }

        internal void CollectDisposalOwnership(ICollection<Tensor> tensors)
        {
            foreach (var vision in _ownedVisionEncoders) vision.CollectDisposalOwnership(tensors);
        }

        internal void DisposeOwnedVision()
        {
            foreach (var vision in _ownedVisionEncoders) vision.DisposeOwned();
            _ownedVisionEncoders.Clear();
            _vision = null;
        }
    }
}
