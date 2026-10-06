// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// GLM-5.3-Flash (glm5next) vision support. The managed GlmNextVisionEncoder
// produces projected [nTokens, n_embd] embeddings; they reach the native
// executor as queued OVERRIDE rows that replace the token embeddings of the
// <|image|> placeholder positions in the next prompt forward (the text tower is
// NoPE, so image tokens simply occupy sequential cache positions - no MRoPE
// bookkeeping like the Qwen-VL family needs).
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        public GlmNextVisionEncoder VisionEncoder { get; private set; }
        private readonly List<GlmNextVisionEncoder> _ownedVisionEncoders = new();

        private NativeConstructionCleanupHandle _pendingVisionConstruction;

        public void LoadVisionEncoder(string mmProjPath)
        {
            ThrowIfOwnershipCleanupFailed();
            if (Config.Architecture != "glm5next")
            {
                Console.WriteLine($"Warning: {Config.Architecture} has no vision tower; ignoring mmproj {mmProjPath}.");
                return;
            }
            MediaConstructionCleanup.RequireReleasedConstruction(ref _pendingVisionConstruction);
            _ownedVisionEncoders.EnsureCapacity(checked(_ownedVisionEncoders.Count + 1));
            var encoder = new GlmNextVisionEncoder(mmProjPath, _allocator,
                RetainVisionConstruction, RetainUnsafeVisionConstruction);
            VisionEncoder = encoder;
            _pendingVisionConstruction = null;
        }

        private void RetainVisionConstruction(GlmNextVisionEncoder encoder)
        {
            _ownedVisionEncoders.Add(encoder);
            _pendingVisionConstruction = encoder.ConstructionCleanup;
            encoder.SetHostModel(this);
        }

        private void RetainUnsafeVisionConstruction(GlmNextVisionEncoder encoder, Exception error)
            => RetainFailedModelOwnership(encoder, error);

        /// <summary>
        /// Queue projected vision embeddings to replace the token embeddings of
        /// the placeholder span starting at <paramref name="startPosition"/>
        /// (an index into the token array of the NEXT Forward call). Takes
        /// ownership of <paramref name="visionEmbeddings"/>.
        /// </summary>
        public void SetVisionEmbeddings(Tensor visionEmbeddings, int startPosition)
        {
            ThrowIfOwnershipCleanupFailed();
            if (visionEmbeddings == null)
                return;
            try
            {
                if (startPosition < 0)
                    return;
                if (!UsesNativeExecutor)
                {
                    Console.WriteLine("Warning: glm5next vision requires the native executor; dropping image embeddings.");
                    return;
                }

                int rows = (int)visionEmbeddings.Sizes[0];
                int dim = (int)visionEmbeddings.Sizes[1];
                if (dim != Config.HiddenSize)
                {
                    Console.WriteLine($"Warning: vision embedding dim {dim} != hidden {Config.HiddenSize}; dropping.");
                    return;
                }

                Tensor src = visionEmbeddings.IsContiguous() ? visionEmbeddings : Ops.NewContiguous(visionEmbeddings);
                float[] data = src.GetElementsAsFloat(rows * dim);
                if (!ReferenceEquals(src, visionEmbeddings))
                    src.Dispose();

                lock (_nativeSync)
                {
                    GgmlGlmNative.QueueVisionRows(_native, data, rows, startPosition);
                }
            }
            finally
            {
                visionEmbeddings.Dispose();
            }
        }
    }
}
