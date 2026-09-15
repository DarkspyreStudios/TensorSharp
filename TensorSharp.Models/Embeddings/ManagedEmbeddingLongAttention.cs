// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics.Tensors;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

// Long bidirectional attention uses the same online softmax identity as Flash
// Attention: keep one maximum, exponential sum and weighted value accumulator
// per query while visiting bounded key tiles. No attention matrix is retained.
internal static class ManagedEmbeddingLongAttention
{
    internal const int QueryTile = 64;
    private const int KeyTile = 128;

    internal static int ScratchSize(int head) => checked((2 * QueryTile + 2 * KeyTile + 4) * head + 2 * QueryTile + 4 * KeyTile);

    internal static unsafe void Compute(float* queries, int queryStride, float* keys, int keyStride,
        float* values, int valueStride, int head, int length, int queryCount,
        float* output, int outputStride, float[] scratch, CancellationToken cancellationToken)
    {
        // The encoder selects this path only for heads divisible by ValueWidth.
        int width = ManagedEmbeddingMath.ValueWidth;
        float scale = 1.0f / MathF.Sqrt(head);
        fixed (float* storage = scratch)
        {
            float* packedQueries = storage;
            float* keyTile = packedQueries + QueryTile * head;
            float* valueTile = keyTile + KeyTile * head;
            float* accumulated = valueTile + KeyTile * head;
            float* valueResult = accumulated + QueryTile * head;
            float* maxima = valueResult + 4 * head;
            float* sums = maxima + QueryTile;
            float* scores = sums + QueryTile;
            new Span<float>(accumulated, queryCount * head).Clear();
            new Span<float>(sums, queryCount).Clear();
            new Span<float>(maxima, queryCount).Fill(float.NegativeInfinity);
            for (int query = 0; query < queryCount; ++query)
                new ReadOnlySpan<float>(queries + (long)query * queryStride, head)
                    .CopyTo(new Span<float>(packedQueries + query * head, head));

            for (int firstKey = 0; firstKey < length; firstKey += KeyTile)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int keyCount = Math.Min(KeyTile, length - firstKey);
                // The same small K/V block is reused by every query group in
                // this job. Compact strides also avoid long-context cache-set
                // conflicts in the global transposed tensors.
                for (int dim = 0; dim < head; ++dim)
                    new ReadOnlySpan<float>(keys + (long)dim * keyStride + firstKey, keyCount)
                        .CopyTo(new Span<float>(keyTile + dim * KeyTile, keyCount));
                for (int dim = 0; dim < head; dim += width)
                    new ReadOnlySpan<float>(values + (long)dim * valueStride + firstKey * width, keyCount * width)
                        .CopyTo(new Span<float>(valueTile + dim * KeyTile, keyCount * width));

                for (int firstQuery = 0; firstQuery < queryCount; firstQuery += 4)
                {
                    int active = Math.Min(4, queryCount - firstQuery);
                    float* query = packedQueries + firstQuery * head;
                    ManagedEmbeddingWideMath.Scores(query,
                        query + Math.Min(1, active - 1) * head,
                        query + Math.Min(2, active - 1) * head,
                        query + Math.Min(3, active - 1) * head,
                        keyTile, KeyTile, head, keyCount, scale, scores);
                    for (int lane = 0; lane < 4; ++lane)
                    {
                        int index = firstQuery + Math.Min(lane, active - 1);
                        var probabilities = new Span<float>(scores + lane * keyCount, keyCount);
                        float maximum = Math.Max(maxima[index], TensorPrimitives.Max(probabilities));
                        float tileSum = ManagedEmbeddingWideMath.ExpSum(scores + lane * keyCount, keyCount, maximum);
                        if (lane >= active) continue;
                        float rescale = maximum > maxima[index] ? MathF.Exp(maxima[index] - maximum) : 1.0f;
                        if (rescale != 1)
                            TensorPrimitives.Multiply(new Span<float>(accumulated + index * head, head), rescale,
                                new Span<float>(accumulated + index * head, head));
                        sums[index] = sums[index] * rescale + tileSum;
                        maxima[index] = maximum;
                    }
                    ManagedEmbeddingWideMath.Values(scores, keyCount, valueTile, KeyTile, head, valueResult, head, active);
                    for (int lane = 0; lane < active; ++lane)
                    {
                        var row = new Span<float>(accumulated + (firstQuery + lane) * head, head);
                        TensorPrimitives.Add(row, new ReadOnlySpan<float>(valueResult + lane * head, head), row);
                    }
                }
            }
            for (int query = 0; query < queryCount; ++query)
                TensorPrimitives.Multiply(new ReadOnlySpan<float>(accumulated + query * head, head), 1.0f / sums[query],
                    new Span<float>(output + (long)query * outputStride, head));
        }
    }
}
