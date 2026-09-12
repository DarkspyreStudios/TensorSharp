// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4.1 attention for the pure C# CPU executor.
//
// V4.1 keeps V4's LoRA-factored Q, shared 512-dim K(=V) head, attention sinks
// and grouped LoRA output projection, and changes five things:
//
//   * The projected query heads are NOT re-normalized (V4 RMS-norms each head).
//   * Compression ratios are 1 and 2 instead of 4 and 128, the window is the
//     block itself rather than two overlapping blocks, and there is no absolute
//     positional embedding on the gate.
//   * The compressed latent feeds the lightning indexer's K projection, so
//     there is no second (indexer) compressor.
//   * The compressed and indexer caches are SHARED: one layer per ratio group
//     builds them, and one layer per group publishes the sparse selection that
//     every later layer in the group reuses. A "candidate" layer additionally
//     prunes the indexer's key space to the best few blocks.
//   * Every cache commit reproduces the checkpoint's trained quantization
//     (FP8 E4M3 for the raw rows, MXFP4 for the indexer, NVFP4 for the
//     compressed rows) instead of only rounding to F16.
//
// This is the managed counterpart of ggml_ops_deepseek41.inc plus
// dsv41_quant.h; the two have to agree because the quantization bins decide
// which rows the sparse selection keeps.
// ---------------------------------------------------------------------------
using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace TensorSharp.Models
{
    internal sealed unsafe partial class DeepSeek4CpuExecutor
    {
        // -------------------------------------------------------------------
        // Trained cache quantization (port of TensorSharp.GGML.Native/dsv41_quant.h)
        //
        // The reference checkpoint quantizes BF16 activations and dequantizes
        // straight back to BF16, so both rounding boundaries are reproduced here.
        // -------------------------------------------------------------------

        /// <summary>Round to BF16 and back, leaving NaN/Inf alone.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Bf16(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            return BitConverter.UInt32BitsToSingle(bits);
        }

        /// <summary>BF16-round a contiguous span and return its largest magnitude.</summary>
        private static float Bf16AndAbsMax(float* row, int n)
        {
            int width = Vector<float>.Count;
            float amax = 0f;
            int i = 0;
            if (width > 1 && n >= width)
            {
                var half = new Vector<uint>(0x7fffu);
                var one = new Vector<uint>(1u);
                var keep = new Vector<uint>(0xffff0000u);
                var expo = new Vector<uint>(0x7f800000u);
                var sign = new Vector<uint>(0x7fffffffu);
                var best = Vector<float>.Zero;
                for (; i + width <= n; i += width)
                {
                    var bits = Vector.AsVectorUInt32(Unsafe.ReadUnaligned<Vector<float>>(row + i));
                    var rounded = (bits + half + (Vector.ShiftRightLogical(bits, 16) & one)) & keep;
                    // Only finite values round; NaN/Inf keep their exact bits.
                    var finite = Vector.OnesComplement(Vector.Equals(bits & expo, expo));
                    var result = Vector.ConditionalSelect(finite, rounded, bits);
                    Unsafe.WriteUnaligned(row + i, Vector.AsVectorSingle(result));
                    best = Vector.Max(best, Vector.AsVectorSingle(result & sign));
                }
                for (int lane = 0; lane < width; lane++)
                    amax = MathF.Max(amax, best[lane]);
            }
            for (; i < n; i++)
            {
                float v = Bf16(row[i]);
                row[i] = v;
                amax = MathF.Max(amax, MathF.Abs(v));
            }
            return amax;
        }

        /// <summary>Round half to even, for a non-negative value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float RoundEvenPositive(float value)
        {
            float whole = MathF.Floor(value);
            float fraction = value - whole;
            return whole + (fraction > 0.5f || (fraction == 0.5f && ((int)whole & 1) != 0) ? 1f : 0f);
        }

        /// <summary>Finite E4M3 (FP8), round-to-nearest-even with saturation at 448.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float E4M3(float value)
        {
            float magnitude = MathF.Min(MathF.Abs(value), 448f);
            // frexp's exponent is ilogb + 1; the step is 2^(exponent - 4), which
            // leaves four significant bits (1 + 3 mantissa) and reproduces the
            // format's subnormal step below 2^-6.
            int exponent = MathF.ILogB(MathF.Max(magnitude, 0.015625f)) + 1;
            float step = MathF.ScaleB(1f, exponent - 4);
            return MathF.CopySign(RoundEvenPositive(magnitude / step) * step, value);
        }

        /// <summary>E2M1 (FP4). Adjacent magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6;
        /// the inclusive boundaries alternate so midpoint ties pick an even code.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float E2M1(float value)
        {
            float m = MathF.Abs(value);
            float rounded = m <= .25f ? 0f : m < .75f ? .5f :
                m <= 1.25f ? 1f : m < 1.75f ? 1.5f :
                m <= 2.5f ? 2f : m < 3.5f ? 3f :
                m <= 5.0f ? 4f : 6f;
            return MathF.CopySign(rounded, value);
        }

        /// <summary>Smallest power of two greater than or equal to a positive normal value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CeilPow2(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            int exponent = (int)((bits >> 23) & 255u) - 127 + ((bits & 0x7fffffu) != 0 ? 1 : 0);
            return MathF.ScaleB(1f, exponent);
        }

        /// <summary>Block scale. Mode 0 is FP8 E4M3, 1 is MXFP4, 2 is NVFP4.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float QuantScale(float amax, int mode)
        {
            if (mode == 0)
                return CeilPow2(MathF.Max(amax, 1e-4f) * (1f / 448f));
            if (mode == 1)
                return CeilPow2(MathF.Max(amax, 6f * MathF.ScaleB(1f, -126)) * (1f / 6f));
            return E4M3(MathF.Max(amax, 6f * MathF.ScaleB(1f, -9)) / 6f);
        }

        /// <summary>Number of values one block scale covers, per mode.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int QuantBlock(int mode) => mode == 2 ? 16 : 32;

        /// <summary>
        /// Quantize one contiguous row in place, the way the checkpoint's own
        /// cache does: BF16, per-block scale, FP8 or FP4 code, dequantize, BF16.
        /// </summary>
        private static void QuantizeRowV41(float* row, int n, int mode)
        {
            int block = QuantBlock(mode);
            for (int start = 0; start < n; start += block)
            {
                float* group = row + start;
                float amax = Bf16AndAbsMax(group, block);
                float scale = QuantScale(amax, mode);
                // Divide, never multiply by a reciprocal: the NVFP4 scale is an
                // E4M3 value rather than a power of two, so the reciprocal's
                // rounding moves exact FP4 midpoints across a bin.
                if (mode == 0)
                {
                    for (int i = 0; i < block; i++)
                        group[i] = Bf16(E4M3(group[i] / scale) * scale);
                }
                else
                {
                    for (int i = 0; i < block; i++)
                        group[i] = Bf16(E2M1(group[i] / scale) * scale);
                }
            }
        }

        /// <summary>Quantization harness entry point: runs the real kernel over a
        /// managed array so a test can hold it to the PyTorch reference.</summary>
        internal static void QuantizeForTest(float[] values, int mode)
        {
            int block = QuantBlock(mode);
            if (values.Length % block != 0)
                throw new ArgumentException($"V4.1 quantization mode {mode} needs a multiple of {block} values.");
            fixed (float* p = values)
                QuantizeRowV41(p, values.Length, mode);
        }

        /// <summary>Quantize and commit one row into an F16-rounded cache row.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CommitQuantized(float* src, float* dst, int n, int mode)
        {
            QuantizeRowV41(src, n, mode);
            for (int d = 0; d < n; d++)
                dst[d] = F16(src[d]);
        }

        // -------------------------------------------------------------------
        // Tracing
        //
        // TS_DSV4_CPU_TRACE_DIR writes the same per-tensor files, under the same
        // names, that eng/dsv41-reference.py writes with --output. Diffing the
        // two directories says which tensor of which layer first disagrees,
        // which is the only tractable way to debug a 5-layer 40-tensor graph.
        // -------------------------------------------------------------------

        private static readonly string TraceDir = Environment.GetEnvironmentVariable("TS_DSV4_CPU_TRACE_DIR");
        /// <summary>Only trace ubatches of this width. Several chunk sizes over
        /// one prompt write the same position, so without this the last runner
        /// wins and the files no longer describe the run being debugged.</summary>
        private static readonly int TraceNt = ParseEnvInt("TS_DSV4_CPU_TRACE_NT", 0);
        private int _traceP0, _traceNt;

        private void Trace(string name, float* data, long count)
        {
            if (TraceDir == null || (TraceNt > 0 && _traceNt != TraceNt))
                return;
            System.IO.Directory.CreateDirectory(TraceDir);
            var bytes = new byte[count * sizeof(float)];
            new ReadOnlySpan<byte>(data, bytes.Length).CopyTo(bytes);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(TraceDir, $"p{_traceP0:D6}_{name}.f32"), bytes);
        }

        private static string TraceLayer(int il, string what) => $"blk{il:D2}_{what}";

        // -------------------------------------------------------------------
        // Attention super-block
        // -------------------------------------------------------------------

        private void AttentionV41(int il, int nt, int p0)
        {
            Layer L = _layers[il];
            int E = _nEmbd, NH = _nHead, HD = _headDim, ROT = _nRot;
            int NOPE = HD - ROT;
            int ratio = L.Ratio;
            float* ropeCache = ratio != 0 ? _ropeCompCache : _ropeRawCache;

            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // q = wq_b(rms(wq_a(cur))), RoPE on the tail. No per-head norm: the
            // leading head_dim - n_rot components are MLA content dimensions and
            // V4.1 hands them to attention exactly as the projection produced them.
            MatMul(L.WqA, 0, _qLoraRank, _cur, E, nt, _qr, _qLoraRank);
            RmsNormRows(_qr, L.QANorm, nt, _qLoraRank);
            MatMul(L.WqB, 0, NH * HD, _qr, _qLoraRank, nt, _q, NH * HD);
            PFor(nt, t =>
            {
                float* cache = ropeCache + (long)t * ROT;
                float* qt = _q + (long)t * NH * HD;
                for (int h = 0; h < NH; h++)
                    RotatePairs(qt + (long)h * HD + NOPE, cache, ROT, 1f);
            });

            // kv: one shared head, RoPE'd, then quantized the way the trained
            // cache is before it is committed.
            MatMul(L.Wkv, 0, HD, _cur, E, nt, _kvRow, HD);
            RmsNormRows(_kvRow, L.KvNorm, nt, HD);
            PFor(nt, t =>
            {
                float* kv = _kvRow + (long)t * HD;
                RotatePairs(kv + NOPE, ropeCache + (long)t * ROT, ROT, 1f);
                CommitQuantized(kv, L.RawK + ((p0 + t) % _ringRaw) * HD, HD, 0);
            });
            Trace(TraceLayer(il, "q"), _q, (long)nt * NH * HD);
            Trace(TraceLayer(il, "raw_k"), _kvRow, (long)nt * HD);

            Tick(2, t0);
            t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            if (ratio != 0)
            {
                if (L.KvSource == il)
                    CompressV41(L, il, nt, p0, ratio);
                Tick(3, t0);
                t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                if (L.IndexSource == il)
                    BuildIndexerV41(L, il, nt, p0, ratio);
                Tick(4, t0);
            }
            else
            {
                Tick(3, t0);
            }

            t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            AttentionCoreV41(L, nt, p0, ratio);
            Tick(5, t0);
            t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // inverse RoPE on the rope slice, then the grouped LoRA output projection
            PFor(nt, t =>
            {
                float* cache = ropeCache + (long)t * ROT;
                float* ot = _attnO + (long)t * NH * HD;
                for (int h = 0; h < NH; h++)
                    RotatePairs(ot + (long)h * HD + NOPE, cache, ROT, -1f);
            });

            int oGroupDim = NH / _oGroups * HD;
            int oCat = _oGroups * _oLoraRank;
            if (!TryGroupedOutProjFused(L, nt, oGroupDim, oCat))
            {
                for (int g = 0; g < _oGroups; g++)
                {
                    MatMul(L.WoA, (long)g * _oLoraRank, _oLoraRank,
                        _attnO + (long)g * oGroupDim, NH * HD, nt,
                        _oG + (long)g * _oLoraRank, oCat);
                }
            }
            MatMul(L.WoB, 0, E, _oG, oCat, nt, _attnOut, E);
            Tick(6, t0);
        }

        // -------------------------------------------------------------------
        // Block compressor
        //
        // Ratio 1 compresses nothing: every token is its own block and the
        // latent is just the normalized projection. Ratio 2 reduces each
        // non-overlapping pair with a per-dimension softmax over the pair's
        // gate projection. Either way the latent is then committed twice: once
        // through the indexer's K projection into the indexer cache, and once
        // RoPE'd and quantized into the compressed cache.
        // -------------------------------------------------------------------

        private void CompressV41(Layer L, int il, int nt, int p0, int ratio)
        {
            int E = _nEmbd, HD = _headDim, ID = _idxHeadSize, ROT = _nRot;

            MatMul(L.CompWkv, 0, HD, _cur, E, nt, _stKv, HD);
            if (ratio > 1)
                MatMul(L.CompWgate, 0, HD, _cur, E, nt, _stScore, HD);

            // Which blocks complete inside this ubatch, and where each starts.
            int nBlocks;
            long firstBlockPos;      // absolute position of the first block's LAST token
            if (ratio == 1)
            {
                nBlocks = nt;
                firstBlockPos = p0;
            }
            else
            {
                long first = -1;
                for (long p = p0; p < p0 + nt; p++)
                {
                    if ((p + 1) % ratio == 0) { first = p; break; }
                }
                nBlocks = first < 0 ? 0 : (int)((p0 + nt - 1 - first) / ratio) + 1;
                firstBlockPos = first;
            }

            if (nBlocks > 0)
            {
                int r = ratio;
                long fb = firstBlockPos;
                PFor(nBlocks, bi =>
                {
                    long p = fb + (long)bi * r;
                    long start = p + 1 - r;
                    float* latent = _latent + (long)bi * HD;
                    if (r == 1)
                    {
                        Buffer.MemoryCopy(_stKv + (long)bi * HD, latent, HD * sizeof(float), HD * sizeof(float));
                    }
                    else
                    {
                        // Per dimension, softmax the window's gate values and take
                        // the matching weighted sum of the window's kv values.
                        // A window position before this ubatch lives in the state ring.
                        for (int d = 0; d < HD; d++)
                        {
                            float m = float.NegativeInfinity;
                            for (int w = 0; w < r; w++)
                            {
                                long tw = start + w;
                                float sc = tw >= p0
                                    ? _stScore[(tw - p0) * HD + d]
                                    : L.HistScore[(tw % r) * HD + d];
                                if (sc > m) m = sc;
                            }
                            float se = 0f, sv = 0f;
                            for (int w = 0; w < r; w++)
                            {
                                long tw = start + w;
                                long off = (tw >= p0 ? (tw - p0) : (tw % r)) * HD + d;
                                float sc = tw >= p0 ? _stScore[off] : L.HistScore[off];
                                float e = MathF.Exp(sc - m);
                                se += e;
                                sv += e * (tw >= p0 ? _stKv[off] : L.HistKv[off]);
                            }
                            latent[d] = sv / se;
                        }
                    }
                });

                // RMS + learned gain over the whole block batch at once.
                RmsNormRows(_latent, L.CompNorm, nBlocks, HD);
                Trace(TraceLayer(il, "compress_latent"), _latent, (long)nBlocks * HD);

                // Indexer K reads the latent BEFORE it is rotated or quantized.
                MatMul(L.IndexerK, 0, ID, _latent, HD, nBlocks, _latentK, ID);
                RmsNormRows(_latentK, L.IndexerKNorm, nBlocks, ID);

                int idxNope = ID - ROT;
                int r2 = ratio;
                long fb2 = firstBlockPos;
                PFor(nBlocks, bi =>
                {
                    long p = fb2 + (long)bi * r2;
                    long start = p + 1 - r2;
                    long blockRow = p / r2;

                    float* rc = stackalloc float[ROT];
                    RopeCacheInit(start, comp: true, rc);

                    float* ik = _latentK + (long)bi * ID;
                    RotatePairs(ik + idxNope, rc, ROT, 1f);
                    CommitQuantized(ik, L.LidK + blockRow * ID, ID, 1);

                    float* latent = _latent + (long)bi * HD;
                    RotatePairs(latent + (HD - ROT), rc, ROT, 1f);
                    CommitQuantized(latent, L.CompK + blockRow * HD, HD, 2);
                });
            }

            // Persist the tail of the ubatch so a block straddling the next
            // ubatch boundary still sees its earlier half.
            if (ratio > 1)
            {
                int persistStart = Math.Max(0, nt - ratio);
                for (int t = persistStart; t < nt; t++)
                {
                    long slot = (long)(p0 + t) % ratio;
                    Buffer.MemoryCopy(_stKv + (long)t * HD, L.HistKv + slot * HD, HD * sizeof(float), HD * sizeof(float));
                    Buffer.MemoryCopy(_stScore + (long)t * HD, L.HistScore + slot * HD, HD * sizeof(float), HD * sizeof(float));
                }
            }
        }

        // -------------------------------------------------------------------
        // Lightning indexer and candidate filtering
        //
        // Scores every visible compressed row with sum_h relu(q_h . k) * w_h and
        // publishes the top-k row ids for every later layer in the ratio group.
        // The candidate layer additionally pools those scores into fixed blocks,
        // keeps the best few, and masks everything else out of the later layers'
        // key space.
        // -------------------------------------------------------------------

        private void BuildIndexerV41(Layer L, int il, int nt, int p0, int ratio)
        {
            int E = _nEmbd;
            int IH = _idxNHead, ID = _idxHeadSize, ROT = _nRot;
            int idxNope = ID - ROT;
            Layer source = _layers[L.KvSource];
            int rows = _v41MaxCompRows;

            MatMul(L.IdxQB, 0, IH * ID, _qr, _qLoraRank, nt, _iq, IH * ID);
            MatMul(L.IdxProj, 0, IH, _cur, E, nt, _iw, IH);

            float iwScale = 1.0f / MathF.Sqrt((float)ID * IH);
            PFor(nt, t =>
            {
                float* cache = _ropeCompCache + (long)t * ROT;
                float* iqt = _iq + (long)t * IH * ID;
                for (int h = 0; h < IH; h++)
                {
                    float* head = iqt + (long)h * ID;
                    RotatePairs(head + idxNope, cache, ROT, 1f);
                    QuantizeRowV41(head, ID, 1);
                }
                float* iwt = _iw + (long)t * IH;
                for (int h = 0; h < IH; h++)
                    iwt[h] *= iwScale;
            });

            // Rows this layer may see at all: visibility, then (for layers after
            // the candidate layer) whatever the candidate pruning left.
            bool prune = _v41CandActive && il > _candidateSource;
            byte* candidate = _v41CandMask;

            const int RowChunk = 1024;
            int maxVis = (int)(((long)p0 + nt) / ratio);
            int nChunks = Math.Max(1, (maxVis + RowChunk - 1) / RowChunk);
            PFor(nt * nChunks, work =>
            {
                int t = work / nChunks;
                int ci = work % nChunks;
                long p = p0 + t;
                int nVis = (int)((p + 1) / ratio);
                int r0 = ci * RowChunk;
                if (r0 >= nVis) return;
                int r1 = Math.Min(r0 + RowChunk, nVis);

                float* iqt = _iq + (long)t * IH * ID;
                float* iwt = _iw + (long)t * IH;
                float* scores = _idxScores + (long)t * rows;
                byte* mine = prune ? candidate + (long)t * rows : null;
                for (int r = r0; r < r1; r++)
                {
                    if (mine != null && mine[r] == 0)
                    {
                        scores[r] = float.NegativeInfinity;
                        continue;
                    }
                    float* k = source.LidK + (long)r * ID;
                    var kSpan = new ReadOnlySpan<float>(k, ID);
                    float score = 0f;
                    for (int h = 0; h < IH; h++)
                    {
                        float qk = TensorPrimitives.Dot(new ReadOnlySpan<float>(iqt + (long)h * ID, ID), kSpan);
                        if (qk > 0f)
                            score += qk * iwt[h];
                    }
                    scores[r] = score;
                }
            });

            if (il == _candidateSource)
                BuildCandidateMask(nt, p0, ratio);
            if (TraceDir != null)
            {
                // The reference stores the score matrix at its own top-k width,
                // with invisible rows at -inf. Re-shape to match it.
                var flat = new float[(long)nt * rows];
                for (int t = 0; t < nt; t++)
                {
                    int nVis = (int)(((long)p0 + t + 1) / ratio);
                    for (int r = 0; r < rows; r++)
                        flat[(long)t * rows + r] = r < nVis ? _idxScores[(long)t * rows + r] : float.NegativeInfinity;
                }
                fixed (float* f = flat) Trace(TraceLayer(il, "index_scores_full"), f, flat.LongLength);
            }

            // Top-k over the rows that survived, per token.
            PFor(nt, t =>
            {
                long p = p0 + t;
                int nVis = (int)((p + 1) / ratio);
                int* outIdx = _topK + (long)t * _idxTopK;
                float* scores = _idxScores + (long)t * rows;

                // Top-k over EVERY visible row, pruned ones included: a row the
                // candidate layer masked scores -inf and so ranks last, but when
                // fewer than k rows survive pruning it is still selected and
                // still becomes a key. Both the reference and the native
                // executor behave this way, and a token near the start of a
                // sequence hits it.
                int k = Math.Min(_idxTopK, nVis);
                _topKCount[t] = k;
                if (k <= 0)
                    return;

                float* heapVal = stackalloc float[_idxTopK];
                for (int r = 0; r < k; r++) { heapVal[r] = scores[r]; outIdx[r] = r; }
                for (int i = k / 2 - 1; i >= 0; i--) HeapSiftDownTie(heapVal, outIdx, i, k);
                // Strict >, so an equal score never displaces an already-held
                // row -- and the heap's root is the tie-aware worst, so the row
                // evicted is the highest id among the lowest scores.
                for (int r = k; r < nVis; r++)
                {
                    if (scores[r] <= heapVal[0]) continue;
                    heapVal[0] = scores[r];
                    outIdx[0] = r;
                    HeapSiftDownTie(heapVal, outIdx, 0, k);
                }
            });
        }

        /// <summary>
        /// Sift-down for a "worst first" heap ordered by score, with ties broken
        /// so the HIGHEST id is the worst. Selection therefore keeps the lowest
        /// ids among equal scores, which is what the reference's stable
        /// descending argsort and ggml's top_k both do. Indexer scores are ReLU
        /// sums, so exact ties at zero are common rather than exotic.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapSiftDownTie(float* val, int* idx, int i, int n)
        {
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, s = i;
                if (l < n && Worse(val, idx, l, s)) s = l;
                if (r < n && Worse(val, idx, r, s)) s = r;
                if (s == i) return;
                (val[i], val[s]) = (val[s], val[i]);
                (idx[i], idx[s]) = (idx[s], idx[i]);
                i = s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Worse(float* val, int* idx, int a, int b)
            => val[a] < val[b] || (val[a] == val[b] && idx[a] > idx[b]);

        /// <summary>
        /// Pools the indexer scores into fixed row blocks, pins the block holding
        /// the query's own newest row, keeps the best <c>candidate_topk</c> of
        /// them and writes the surviving rows into the mask every later indexer
        /// layer intersects with visibility.
        /// </summary>
        private void BuildCandidateMask(int nt, int p0, int ratio)
        {
            int rows = _v41MaxCompRows;
            int block = _candidateBlock, topk = _candidateTopk;
            PFor(nt, t =>
            {
                long p = p0 + t;
                int nVis = (int)((p + 1) / ratio);
                int nBlocks = (nVis + block - 1) / block;
                byte* mine = _v41CandMask + (long)t * rows;
                new Span<byte>(mine, Math.Max(nVis, 0)).Clear();
                if (nBlocks <= 0)
                    return;

                float* scores = _idxScores + (long)t * rows;
                long pinned = p / block;

                // Keep the best `topk` blocks by max score, with the pinned block
                // always among them. A block whose rows are all masked stays out.
                int keep = Math.Min(topk, nBlocks);
                int* sel = stackalloc int[keep];
                float* val = stackalloc float[keep];
                int n = 0;
                for (int b = 0; b < nBlocks; b++)
                {
                    float best = float.NegativeInfinity;
                    int end = Math.Min(nVis, (b + 1) * block);
                    for (int r = b * block; r < end; r++)
                        best = MathF.Max(best, scores[r]);
                    if (float.IsNegativeInfinity(best))
                        continue;
                    if (b == pinned)
                        best = float.PositiveInfinity;
                    if (n < keep)
                    {
                        val[n] = best;
                        sel[n] = b;
                        n++;
                        if (n == keep)
                            for (int i = n / 2 - 1; i >= 0; i--) HeapSiftDownTie(val, sel, i, n);
                    }
                    else if (best > val[0])
                    {
                        val[0] = best;
                        sel[0] = b;
                        HeapSiftDownTie(val, sel, 0, n);
                    }
                }

                for (int i = 0; i < n; i++)
                {
                    int end = Math.Min(nVis, (sel[i] + 1) * block);
                    for (int r = sel[i] * block; r < end; r++)
                        mine[r] = 1;
                }
            });
            _v41CandActive = true;
        }

        // -------------------------------------------------------------------
        // Attention core: per (token, head) online softmax with sinks over the
        // raw sliding window plus the indexer-selected compressed rows of this
        // layer's shared cache. K doubles as V.
        // -------------------------------------------------------------------

        private void AttentionCoreV41(Layer L, int nt, int p0, int ratio)
        {
            int NH = _nHead, HD = _headDim;
            float kqScale = 1.0f / MathF.Sqrt(HD);
            float[] sinks = L.Sinks;
            int ring = _ringRaw;
            int nSwa = _nSwa;
            int maxRows = nSwa + _idxTopK + 8;
            float* compK = ratio != 0 ? _layers[L.KvSource].CompK : null;

            PFor(nt * NH, work =>
            {
                int t = work / NH;
                int h = work % NH;
                long p = p0 + t;

                float* q = _q + (long)t * NH * HD + (long)h * HD;
                var qSpan = new ReadOnlySpan<float>(q, HD);

                float** rowPtr = stackalloc float*[maxRows];
                float* score = stackalloc float[maxRows];
                int n = 0;

                long trStart = Math.Max(0, p - nSwa + 1);
                for (long tr = trStart; tr <= p; tr++)
                    rowPtr[n++] = L.RawK + (tr % ring) * HD;

                if (compK != null)
                {
                    int cnt = _topKCount[t];
                    int* sel = _topK + (long)t * _idxTopK;
                    for (int i = 0; i < cnt; i++)
                        rowPtr[n++] = compK + (long)sel[i] * HD;
                }

                float m = sinks[h];
                for (int i = 0; i < n; i++)
                {
                    float s = TensorPrimitives.Dot(qSpan, new ReadOnlySpan<float>(rowPtr[i], HD)) * kqScale;
                    score[i] = s;
                    if (s > m) m = s;
                }

                float sum = MathF.Exp(sinks[h] - m);
                float* acc = stackalloc float[HD];
                var accSpan = new Span<float>(acc, HD);
                accSpan.Clear();
                for (int i = 0; i < n; i++)
                {
                    float w = MathF.Exp(score[i] - m);
                    sum += w;
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(rowPtr[i], HD), w, accSpan, accSpan);
                }

                float invSum = 1.0f / sum;
                float* dst = _attnO + (long)t * NH * HD + (long)h * HD;
                for (int d = 0; d < HD; d++)
                    dst[d] = acc[d] * invSum;
            });
        }
    }
}
