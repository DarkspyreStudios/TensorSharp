// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>
/// ARM signed-dot matrix tiles. Each SIMD lane accumulates a different output
/// column, avoiding a horizontal reduction and half conversion for every dot.
/// Original Q8 values are retained exactly; only their storage order changes.
/// </summary>
internal sealed class ManagedEmbeddingQ8Matrix
{
    public static bool IsSupported => Dp.IsSupported && AdvSimd.Arm64.IsSupported;
    private readonly byte[] _weights;
    private readonly float[] _scales;
    private readonly int _input, _output, _blocks, _columnTiles;

    public ManagedEmbeddingQ8Matrix(byte[] source, int input, int output)
    {
        if (input <= 0 || input % 32 != 0 || output <= 0 || source.Length != checked(input / 32 * 34 * output))
            throw new ArgumentException("Invalid Q8_0 matrix shape or byte count.");
        _input = input; _output = output; _blocks = input / 32; _columnTiles = (output + 3) / 4;
        _weights = new byte[checked(_columnTiles * _blocks * 128)];
        _scales = new float[checked(_columnTiles * _blocks * 4)];
        for (int tile = 0; tile < _columnTiles; ++tile)
        for (int block = 0; block < _blocks; ++block)
        for (int column = 0; column < 4 && tile * 4 + column < output; ++column)
        {
            int sourceOffset = ((tile * 4 + column) * _blocks + block) * 34;
            ushort scaleBits = (ushort)(source[sourceOffset] | source[sourceOffset + 1] << 8);
            _scales[(tile * _blocks + block) * 4 + column] = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            for (int part = 0; part < 8; ++part)
                source.AsSpan(sourceOffset + 2 + part * 4, 4).CopyTo(
                    _weights.AsSpan((tile * _blocks + block) * 128 + part * 16 + column * 4, 4));
        }
    }

    public unsafe void Multiply(float[] input, int rows, float[] output, CpuWorkerPool pool, CancellationToken cancellationToken = default)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("ARM dot-product SIMD is required for packed Q8 matrices.");
        var activations = ArrayPool<byte>.Shared.Rent(checked(rows * _input));
        float[] scales = null;
        try
        {
            scales = ArrayPool<float>.Shared.Rent(checked(rows * _blocks));
            fixed (byte* activationPointer = activations)
            fixed (float* scalePointer = scales)
            fixed (float* inputPointer = input)
            fixed (byte* weightPointer = _weights)
            fixed (float* weightScalePointer = _scales)
            fixed (float* outputPointer = output)
            {
                nint activationAddress = (nint)activationPointer, scaleAddress = (nint)scalePointer;
                nint inputAddress = (nint)inputPointer, weightAddress = (nint)weightPointer;
                nint weightScaleAddress = (nint)weightScalePointer, outputAddress = (nint)outputPointer;
                pool.For(rows, row => Quantize((float*)inputAddress + (long)row * _input,
                    (sbyte*)activationAddress + (long)row * _input, (float*)scaleAddress + (long)row * _blocks, _blocks), cancellationToken);
                const int columnsPerTask = 8, rowsPerTask = 16;
                int columnGroups = (_columnTiles + columnsPerTask - 1) / columnsPerTask;
                int rowGroups = (rows + rowsPerTask - 1) / rowsPerTask;
                pool.For(checked(columnGroups * rowGroups), work =>
                {
                    int firstTile = work % columnGroups * columnsPerTask;
                    int endTile = Math.Min(firstTile + columnsPerTask, _columnTiles);
                    int firstRow = work / columnGroups * rowsPerTask;
                    int endRow = Math.Min(firstRow + rowsPerTask, rows);
                    for (int tile = firstTile; tile < endTile; ++tile)
                    for (int row = firstRow; row < endRow; row += 4)
                    {
                        if (rows - row == 1)
                            TileOneRow((sbyte*)weightAddress + (long)tile * _blocks * 128,
                                (float*)weightScaleAddress + (long)tile * _blocks * 4,
                                (sbyte*)activationAddress + (long)row * _input,
                                (float*)scaleAddress + (long)row * _blocks,
                                (float*)outputAddress + (long)row * _output + tile * 4,
                                _blocks, Math.Min(4, _output - tile * 4));
                        else
                            Tile((sbyte*)weightAddress + (long)tile * _blocks * 128,
                                (float*)weightScaleAddress + (long)tile * _blocks * 4,
                                (sbyte*)activationAddress + (long)row * _input,
                                (float*)scaleAddress + (long)row * _blocks,
                                (float*)outputAddress + (long)row * _output + tile * 4,
                                _blocks, _input, _output, Math.Min(4, rows - row), Math.Min(4, _output - tile * 4));
                    }
                }, cancellationToken);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(activations);
            if (scales != null) ArrayPool<float>.Shared.Return(scales);
        }
    }

    private static unsafe void Quantize(float* input, sbyte* output, float* scales, int blocks)
    {
        for (int block = 0; block < blocks; ++block)
        {
            float* x = input + block * 32;
            var x0 = Vector128.Load(x); var x1 = Vector128.Load(x + 4);
            var x2 = Vector128.Load(x + 8); var x3 = Vector128.Load(x + 12);
            var x4 = Vector128.Load(x + 16); var x5 = Vector128.Load(x + 20);
            var x6 = Vector128.Load(x + 24); var x7 = Vector128.Load(x + 28);
            var maximum = AdvSimd.Max(AdvSimd.Abs(x0), AdvSimd.Abs(x1));
            maximum = AdvSimd.Max(maximum, AdvSimd.Max(AdvSimd.Abs(x2), AdvSimd.Abs(x3)));
            maximum = AdvSimd.Max(maximum, AdvSimd.Max(AdvSimd.Abs(x4), AdvSimd.Abs(x5)));
            maximum = AdvSimd.Max(maximum, AdvSimd.Max(AdvSimd.Abs(x6), AdvSimd.Abs(x7)));
            float scale = AdvSimd.Arm64.MaxAcross(maximum).GetElement(0) / 127.0f;
            // GGUF Q8_0 quantizes with the original F32 scale but stores an F16
            // scale. Preserve that distinction, including round-to-even quants.
            scales[block] = (float)(System.Half)scale;
            var inverse = Vector128.Create(scale == 0 ? 0 : 1.0f / scale);
            var q0 = AdvSimd.ConvertToInt32RoundToEven(x0 * inverse);
            var q1 = AdvSimd.ConvertToInt32RoundToEven(x1 * inverse);
            var q2 = AdvSimd.ConvertToInt32RoundToEven(x2 * inverse);
            var q3 = AdvSimd.ConvertToInt32RoundToEven(x3 * inverse);
            var q4 = AdvSimd.ConvertToInt32RoundToEven(x4 * inverse);
            var q5 = AdvSimd.ConvertToInt32RoundToEven(x5 * inverse);
            var q6 = AdvSimd.ConvertToInt32RoundToEven(x6 * inverse);
            var q7 = AdvSimd.ConvertToInt32RoundToEven(x7 * inverse);
            var p0 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(q0), q1);
            var p1 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(q2), q3);
            var p2 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(q4), q5);
            var p3 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(q6), q7);
            AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(p0), p1).Store(output + block * 32);
            AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(p2), p3).Store(output + block * 32 + 16);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TileOneRow(sbyte* weights, float* weightScales, sbyte* input, float* inputScales,
        float* output, int blocks, int columns)
    {
        // A 17-token input should not run three extra projection rows in every
        // layer. Keep this remainder separate from the full four-row hot path.
        var sum = Vector128<float>.Zero;
        for (int block = 0; block < blocks; ++block)
        {
            sbyte* w = weights + block * 128;
            var first = Vector128.Load(input + block * 32);
            var last = Vector128.Load(input + block * 32 + 16);
            var dot = DotHalf(Vector128.Load(w), Vector128.Load(w + 16), Vector128.Load(w + 32), Vector128.Load(w + 48), first);
            dot = DotHalf(Vector128.Load(w + 64), Vector128.Load(w + 80), Vector128.Load(w + 96), Vector128.Load(w + 112), last, dot);
            sum = AdvSimd.FusedMultiplyAdd(sum, Vector128.Load(weightScales + block * 4) * inputScales[block], AdvSimd.ConvertToSingle(dot));
        }
        Store(sum, output, columns);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile(sbyte* weights, float* weightScales, sbyte* input, float* inputScales,
        float* output, int blocks, int inputStride, int outputStride, int rows, int columns)
    {
        var sum0 = Vector128<float>.Zero; var sum1 = Vector128<float>.Zero;
        var sum2 = Vector128<float>.Zero; var sum3 = Vector128<float>.Zero;
        int row1 = Math.Min(1, rows - 1), row2 = Math.Min(2, rows - 1), row3 = Math.Min(3, rows - 1);
        for (int block = 0; block < blocks; ++block)
        {
            var a0 = Vector128.Load(input + block * 32);
            var a1 = Vector128.Load(input + row1 * inputStride + block * 32);
            var a2 = Vector128.Load(input + row2 * inputStride + block * 32);
            var a3 = Vector128.Load(input + row3 * inputStride + block * 32);
            sbyte* w = weights + block * 128;
            var w0 = Vector128.Load(w); var w1 = Vector128.Load(w + 16);
            var w2 = Vector128.Load(w + 32); var w3 = Vector128.Load(w + 48);
            var d0 = DotHalf(w0, w1, w2, w3, a0);
            var d1 = DotHalf(w0, w1, w2, w3, a1);
            var d2 = DotHalf(w0, w1, w2, w3, a2);
            var d3 = DotHalf(w0, w1, w2, w3, a3);
            a0 = Vector128.Load(input + block * 32 + 16);
            a1 = Vector128.Load(input + row1 * inputStride + block * 32 + 16);
            a2 = Vector128.Load(input + row2 * inputStride + block * 32 + 16);
            a3 = Vector128.Load(input + row3 * inputStride + block * 32 + 16);
            w0 = Vector128.Load(w + 64); w1 = Vector128.Load(w + 80);
            w2 = Vector128.Load(w + 96); w3 = Vector128.Load(w + 112);
            d0 = DotHalf(w0, w1, w2, w3, a0, d0);
            d1 = DotHalf(w0, w1, w2, w3, a1, d1);
            d2 = DotHalf(w0, w1, w2, w3, a2, d2);
            d3 = DotHalf(w0, w1, w2, w3, a3, d3);
            var scales = Vector128.Load(weightScales + block * 4);
            sum0 = AdvSimd.FusedMultiplyAdd(sum0, scales * inputScales[block], AdvSimd.ConvertToSingle(d0));
            sum1 = AdvSimd.FusedMultiplyAdd(sum1, scales * inputScales[row1 * blocks + block], AdvSimd.ConvertToSingle(d1));
            sum2 = AdvSimd.FusedMultiplyAdd(sum2, scales * inputScales[row2 * blocks + block], AdvSimd.ConvertToSingle(d2));
            sum3 = AdvSimd.FusedMultiplyAdd(sum3, scales * inputScales[row3 * blocks + block], AdvSimd.ConvertToSingle(d3));
        }
        Store(sum0, output, columns);
        if (rows > 1) Store(sum1, output + outputStride, columns);
        if (rows > 2) Store(sum2, output + outputStride * 2, columns);
        if (rows > 3) Store(sum3, output + outputStride * 3, columns);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> DotHalf(Vector128<sbyte> w0, Vector128<sbyte> w1, Vector128<sbyte> w2, Vector128<sbyte> w3,
        Vector128<sbyte> input, Vector128<int> sum = default)
    {
        sum = Dp.DotProductBySelectedQuadruplet(sum, w0, input, 0);
        sum = Dp.DotProductBySelectedQuadruplet(sum, w1, input, 1);
        sum = Dp.DotProductBySelectedQuadruplet(sum, w2, input, 2);
        return Dp.DotProductBySelectedQuadruplet(sum, w3, input, 3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector128<float> values, float* output, int columns)
    {
        if (columns == 4) values.Store(output);
        else for (int column = 0; column < columns; ++column) output[column] = values.GetElement(column);
    }
}
