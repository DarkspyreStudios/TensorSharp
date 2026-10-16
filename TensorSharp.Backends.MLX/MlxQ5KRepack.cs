// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TensorSharp.MLX;

/// <summary>Lossless GGML Q5_K to MLX group-32 affine packing. Every five-bit
/// quant is retained; F16 scale/bias rounding matches the original converter.</summary>
internal static unsafe class MlxQ5KRepack
{
    internal static void Pack(IntPtr source, IntPtr packed, IntPtr scales, IntPtr biases,
        int inner, int rows, bool parallel = true)
    {
        if (source == IntPtr.Zero || packed == IntPtr.Zero || scales == IntPtr.Zero || biases == IntPtr.Zero ||
            inner <= 0 || inner % 256 != 0 || rows <= 0)
            throw new ArgumentException("Q5_K packing requires valid buffers and positive 256-aligned rows.");
        int blocks = inner / 256;
        // Rows own disjoint packed bytes and affine parameters. Bound workers
        // to avoid monopolizing the host, and keep small projections serial.
        if (parallel && rows >= 64 && (long)inner * rows >= 1L << 20 && Environment.ProcessorCount > 1)
            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(8, Environment.ProcessorCount) },
                row => PackRow(source, packed, scales, biases, blocks, row));
        else
            for (int row = 0; row < rows; ++row) PackRow(source, packed, scales, biases, blocks, row);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PackRow(IntPtr source, IntPtr packed, IntPtr scales, IntPtr biases, int blocks, int row)
    {
        byte* src = (byte*)source + (long)row * blocks * 176;
        byte* destination = (byte*)packed + (long)row * blocks * 160;
        System.Half* scale = (System.Half*)scales + (long)row * blocks * 8;
        System.Half* bias = (System.Half*)biases + (long)row * blocks * 8;
        for (int block = 0; block < blocks; ++block)
        {
            byte* input = src + block * 176;
            byte* output = destination + block * 160;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(input[0] | input[1] << 8));
            float min = (float)BitConverter.UInt16BitsToHalf((ushort)(input[2] | input[3] << 8));
            byte* parameters = input + 4;
            byte* high = input + 16;
            byte* low = input + 48;
            for (int group = 0; group < 8; ++group)
            {
                int scaleByte = group < 4 ? parameters[group] & 63
                    : (parameters[group + 4] & 15) | ((parameters[group - 4] >> 6) << 4);
                int minByte = group < 4 ? parameters[group + 4] & 63
                    : (parameters[group + 4] >> 4) | ((parameters[group] >> 6) << 4);
                scale[block * 8 + group] = (System.Half)(d * scaleByte);
                bias[block * 8 + group] = (System.Half)(-min * minByte);
                byte* q = low + (group / 2) * 32;
                int nibbleShift = (group & 1) * 4;
                byte* target = output + group * 20;
                for (int start = 0; start < 32; start += 8)
                {
                    // Eight consecutive five-bit quants fill exactly five
                    // bytes. Assign every byte instead of OR-packing one
                    // quant at a time into a separately zeroed buffer.
                    ulong bits = 0;
                    for (int i = 0; i < 8; ++i)
                    {
                        int quant = ((q[start + i] >> nibbleShift) & 15) | (((high[start + i] >> group) & 1) << 4);
                        bits |= (ulong)quant << (i * 5);
                    }
                    target[0] = (byte)bits;
                    target[1] = (byte)(bits >> 8);
                    target[2] = (byte)(bits >> 16);
                    target[3] = (byte)(bits >> 24);
                    target[4] = (byte)(bits >> 32);
                    target += 5;
                }
            }
        }
    }
}
