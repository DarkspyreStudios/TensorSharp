// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Image spans must belong to a SEQUENCE, not to the model.
//
// They used to be a model-global list, which meant two image requests in flight would overwrite
// each other -- the scheduler worked around it by denoising every image turn alone, serializing
// all multimodal traffic. vLLM and SGLang both hang multimodal features off the request
// (MultiModalFeatureSpec on Request, MultimodalInputs on Req) and never off the runner; these
// tests pin that same property here.
//
// They exercise the state machine directly rather than through a server, so they need no weights
// and run in milliseconds.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class DiffusionGemmaVisionConcurrencyTests
    {
        private readonly ITestOutputHelper _output;
        public DiffusionGemmaVisionConcurrencyTests(ITestOutputHelper output) { _output = output; }

        private static Tensor Span(IAllocator allocator, int rows, int hidden, float fill)
        {
            var t = new Tensor(allocator, DType.Float32, rows, hidden);
            var data = new float[(long)rows * hidden];
            for (long i = 0; i < data.LongLength; i++) data[i] = fill;
            t.SetElementsAsFloat(data);
            return t;
        }

        /// <summary>
        /// Two sequences hold their own spans simultaneously, and neither sees the other's. This is
        /// the property that lets the scheduler batch image turns instead of running them alone.
        /// </summary>
        [Fact]
        public void TwoSequences_HoldIndependentSpans()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;

            var a = NewSeq(2);
            var b = NewSeq(2);

            AttachSpan(a, Span(allocator, 4, hidden, 1f), position: 3);
            AttachSpan(b, Span(allocator, 6, hidden, 2f), position: 11);

            Assert.Equal(new[] { (3, 4) }, SpansOf(a));
            Assert.Equal(new[] { (11, 6) }, SpansOf(b));

            // Re-attaching to one must not disturb the other.
            AttachSpan(a, Span(allocator, 5, hidden, 3f), position: 40);
            Assert.Equal(new[] { (3, 4), (40, 5) }, SpansOf(a));
            Assert.Equal(new[] { (11, 6) }, SpansOf(b));

            _output.WriteLine($"a={string.Join(",", SpansOf(a))}  b={string.Join(",", SpansOf(b))}");
        }

        /// <summary>
        /// Freeing a sequence drops its spans, so a finished request cannot leak its picture into
        /// whatever the scheduler admits next.
        /// </summary>
        [Fact]
        public void DisposingASequence_DropsOnlyItsOwnSpans()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;

            var a = NewSeq(2);
            var b = NewSeq(2);
            AttachSpan(a, Span(allocator, 4, hidden, 1f), position: 0);
            AttachSpan(b, Span(allocator, 4, hidden, 2f), position: 0);

            DisposeVision(a);

            Assert.Empty(SpansOf(a));
            Assert.Equal(new[] { (0, 4) }, SpansOf(b));
        }

        /// <summary>
        /// Overlapping spans on the SAME sequence are a caller bug and must throw, because silently
        /// accepting them writes one image's rows over another's.
        /// </summary>
        [Fact]
        public void OverlappingSpansOnOneSequence_Throw()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;
            var a = NewSeq(2);

            AttachSpan(a, Span(allocator, 10, hidden, 1f), position: 5);
            Assert.ThrowsAny<ArgumentException>(
                () => AttachSpan(a, Span(allocator, 4, hidden, 2f), position: 12));
        }

        // ---- reflection helpers -------------------------------------------------------------
        // DiffusionSeqState's vision members are internal to TensorSharp.Models; the test asserts
        // on the state machine without widening that surface just for testing.

        private static readonly Type SeqType =
            typeof(TensorSharp.Models.Gemma4VisionEncoder).Assembly
                .GetType("TensorSharp.Models.DiffusionSeqState", throwOnError: true);

        private static object NewSeq(int numLayers)
        {
            var ctor = SeqType.GetConstructor(
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance,
                binder: null, new[] { typeof(int) }, modifiers: null);
            Assert.NotNull(ctor);
            return ctor.Invoke(new object[] { numLayers });
        }

        private static List<(Tensor Embeddings, int Position)> ListOf(object seq)
        {
            var field = SeqType.GetField("VisionEmbeddings",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.NotNull(field);
            return (List<(Tensor Embeddings, int Position)>)field.GetValue(seq);
        }

        /// <summary>Mirrors DiffusionGemmaModel.SetSequenceVisionEmbeddings' bookkeeping without
        /// constructing a 26B model: same overlap rule, same per-sequence ownership.</summary>
        private static void AttachSpan(object seq, Tensor embeddings, int position)
        {
            var field = SeqType.GetField("VisionEmbeddings",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var list = (List<(Tensor Embeddings, int Position)>)field.GetValue(seq)
                       ?? new List<(Tensor Embeddings, int Position)>();

            int rows = (int)embeddings.Sizes[0];
            foreach (var (existing, pos) in list)
            {
                int existingRows = (int)existing.Sizes[0];
                if (position < pos + existingRows && pos < position + rows)
                {
                    embeddings.Dispose();
                    throw new ArgumentException(
                        $"Image span [{position},{position + rows}) overlaps [{pos},{pos + existingRows}).");
                }
            }
            list.Add((embeddings, position));
            list.Sort((x, y) => x.Position.CompareTo(y.Position));
            field.SetValue(seq, list);
        }

        private static void DisposeVision(object seq) =>
            SeqType.GetMethod("DisposeVision",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .Invoke(seq, null);

        private static (int Start, int Length)[] SpansOf(object seq)
        {
            var list = ListOf(seq);
            if (list == null) return Array.Empty<(int, int)>();
            return list.Select(x => (x.Position, (int)x.Embeddings.Sizes[0]))
                       .OrderBy(x => x.Position).ToArray();
        }
    }
}
