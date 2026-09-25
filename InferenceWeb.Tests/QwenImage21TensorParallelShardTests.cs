// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>Weight sharding for Qwen-Image-2.1 tensor parallelism. The sharded forward itself
/// is compared with the unsharded graph natively on a loopback group
/// (tests/qwen_image21_test.cpp); these pin which bytes each rank receives.</summary>
public sealed unsafe class QwenImage21TensorParallelShardTests : IDisposable
{
    private readonly List<IntPtr> _allocations = new();
    private const int Q8Type = 8, Q8Block = 32, Q8Bytes = 34;

    public void Dispose()
    {
        foreach (var p in _allocations) Marshal.FreeHGlobal(p);
    }

    private QwenImage21Weight Q8(long ne0, long ne1)
    {
        long bytes = ne0 / Q8Block * Q8Bytes * ne1;
        IntPtr data = Marshal.AllocHGlobal((nint)bytes);
        _allocations.Add(data);
        var span = new Span<byte>((void*)data, (int)bytes);
        for (int i = 0; i < span.Length; i++) span[i] = (byte)(i * 31 + 7);
        return new QwenImage21Weight { Data = data, Type = Q8Type, Ne0 = ne0, Ne1 = ne1, Bytes = bytes };
    }

    private static byte[] Bytes(QwenImage21Weight w) => new ReadOnlySpan<byte>((void*)w.Data, (int)w.Bytes).ToArray();

    private QwenImage21Block Block(bool fused)
    {
        const int dim = 256, ff = 512;
        var gate = Q8(dim, fused ? ff * 2 : ff);
        return new QwenImage21Block
        {
            Q = Q8(dim, dim), K = Q8(dim, dim), V = Q8(dim, dim), Out = Q8(dim, dim),
            Gate = gate, Up = fused ? default : Q8(dim, ff), Down = Q8(ff, dim),
            NormQ = new IntPtr(0x1000), NormK = new IntPtr(0x2000),
        };
    }

    [Theory]
    [InlineData(true, 2)]
    [InlineData(false, 2)]
    [InlineData(true, 4)]
    public void RanksReceiveWholeHeadsAndMatchingMlpColumns(bool fused, int ranks)
    {
        var block = Block(fused);
        var owned = new List<IntPtr>();
        try
        {
            var shards = QwenImage21DiT.ShardBlocks(new[] { block }, ranks, owned);
            Assert.Equal(ranks, shards.Length);
            long rowBytes = 256 / Q8Block * Q8Bytes, local = 256 / ranks, ffLocal = 512 / ranks;
            for (int r = 0; r < ranks; r++)
            {
                var s = shards[r][0];
                // Column-parallel projections are views of consecutive output rows.
                Assert.Equal(block.Q.Data + (nint)(r * local * rowBytes), s.Q.Data);
                Assert.Equal((local, local * rowBytes), (s.K.Ne1, s.K.Bytes));
                Assert.Equal(block.V.Data + (nint)(r * local * rowBytes), s.V.Data);
                Assert.Equal(block.Gate.Data + (nint)(r * ffLocal * rowBytes), s.Gate.Data);
                // Up comes after all gate rows in a fused [gate; up] projection.
                var upSource = fused ? block.Gate.Data + (nint)(512 * rowBytes) : block.Up.Data;
                Assert.Equal(upSource + (nint)(r * ffLocal * rowBytes), s.Up.Data);
                Assert.Equal(ffLocal, s.Up.Ne1);
                // Row-parallel projections copy this rank's input columns of every row.
                AssertColumns(block.Out, s.Out, r, local);
                AssertColumns(block.Down, s.Down, r, ffLocal);
                Assert.Equal(block.NormQ, s.NormQ);
                Assert.Equal(block.NormK, s.NormK);
            }
            Assert.Equal(2 * ranks, owned.Count);
        }
        finally { foreach (var p in owned) QuantizedWeight.FreeBuffer(p); }
    }

    private static void AssertColumns(QwenImage21Weight source, QwenImage21Weight shard, int rank, long columns)
    {
        Assert.Equal((columns, source.Ne1), (shard.Ne0, shard.Ne1));
        long sourceRow = source.Ne0 / Q8Block * Q8Bytes, row = columns / Q8Block * Q8Bytes;
        Assert.Equal(row * source.Ne1, shard.Bytes);
        byte[] all = Bytes(source), part = Bytes(shard);
        for (long o = 0; o < source.Ne1; o++)
            Assert.True(all.AsSpan((int)(o * sourceRow + rank * row), (int)row).SequenceEqual(part.AsSpan((int)(o * row), (int)row)),
                $"row {o} of rank {rank}");
    }

    [Fact]
    public void UnsplittableGroupsAreRefused()
    {
        var owned = new List<IntPtr>();
        try
        {
            // 32 heads cannot be divided over 3 GPUs.
            Assert.Throws<NotSupportedException>(() => QwenImage21DiT.ShardBlocks(new[] { Block(true) }, 3, owned));
            // A 256-wide Q8_0 row has 8 blocks: 16 ranks would split a block.
            var narrow = Block(true);
            narrow.Out = Q8(256, 256);
            Assert.Throws<NotSupportedException>(() => QwenImage21DiT.ShardBlocks(new[] { narrow }, 16, owned));
        }
        finally { foreach (var p in owned) QuantizedWeight.FreeBuffer(p); }
    }
}
