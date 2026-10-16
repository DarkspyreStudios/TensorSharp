using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3MlxWeightUploadTests
{
    private static readonly FieldInfo HostBuffer = typeof(MlxStorage).GetField("buffer", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly FieldInfo LinearWeight = typeof(DirectLinear).GetField("_wCpu", BindingFlags.NonPublic | BindingFlags.Instance)!;

    [MlxFact]
    public void UploadOwnsItsCopyAndViewsSurviveSourceTensorDisposal()
    {
        // Deliberately call the factory outside the MLX worker. It must dispatch
        // construction correctly without requiring a particular caller thread.
        using var allocator = new MlxAllocator();
        float[] values = [0f, -0f, float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue,
            1e-30f, -1e-30f, 1f, -1f, .125f, -.125f, 80000f, -80000f,
            float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(0x7fc12345),
            2f, -2f, 3f, -3f];
        float[] original = (float[])values.Clone();
        using var weight = MlxMiniMaxH3Ops.CreateFloat32Weight(allocator, values, 3, 7);
        var storage = (MlxStorage)weight.Storage;
        Assert.Equal(IntPtr.Zero, HostBuffer.GetValue(storage));
        EqualBits(original, values);
        using var alias = weight.CopyRef();
        using var transposed = weight.Transpose();
        using var narrowed = weight.Narrow(0, 1, 1);
        weight.Dispose();
        Array.Fill(values, 12345f); // Upload must not borrow this mutable array.

        MlxWorker.Shared.Invoke(() =>
        {
            using var contiguous = Ops.NewContiguous(transposed);
            float[] expectedTranspose = new float[original.Length];
            for (int row = 0; row < 3; row++) for (int column = 0; column < 7; column++)
                expectedTranspose[column * 3 + row] = original[row * 7 + column];
            EqualBits(expectedTranspose, contiguous.GetElementsAsFloat(original.Length));
            using var contiguousRow = Ops.NewContiguous(narrowed);
            EqualBits(original.AsSpan(7, 7).ToArray(), contiguousRow.GetElementsAsFloat(7));
        });
        Assert.Equal(IntPtr.Zero, HostBuffer.GetValue(storage));

        // Reading creates a host copy on demand, and a subsequent partial host
        // write must upload the whole preserved matrix correctly on device use.
        EqualBits(original, alias.GetElementsAsFloat(original.Length));
        Assert.NotEqual(IntPtr.Zero, HostBuffer.GetValue(storage));
        alias.SetElementsAsFloat([42.25f], 1, 2);
        original[9] = 42.25f;
        MlxWorker.Shared.Invoke(() =>
        {
            using var actual = Ops.NewContiguous(alias);
            EqualBits(original, actual.GetElementsAsFloat(original.Length));
        });
    }

    [MlxTheory]
    [InlineData(1, 3, 5)]
    [InlineData(17, 33, 65)]
    [InlineData(5, 2048, 16384)]
    [InlineData(3, 8192, 2048)]
    public void DeviceOnlyLinearExactlyMatchesPreviousF32WeightPath(int rows, int inputWidth, int outputWidth)
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator = new MlxAllocator();
            using var previousContext = new DirectContext(allocator);
            using var currentContext = new DirectContext(allocator);
            float[] weights = Values(checked(inputWidth * outputWidth), 137);
            float[] originalWeights = (float[])weights.Clone();
            float[] biases = Values(outputWidth, 263);
            float[] inputs = Values(rows * inputWidth, 397);
            var previous = DirectLinear.FromFloats(previousContext, weights, inputWidth, outputWidth, biases);
            var current = DirectLinear.FromFloats(currentContext, weights, inputWidth, outputWidth, biases, mlxDeviceOnly: true);
            var previousWeight = (Tensor)LinearWeight.GetValue(previous)!;
            var currentWeight = (Tensor)LinearWeight.GetValue(current)!;
            Assert.NotEqual(IntPtr.Zero, HostBuffer.GetValue(previousWeight.Storage));
            Assert.Equal(IntPtr.Zero, HostBuffer.GetValue(currentWeight.Storage));
            EqualBits(originalWeights, weights);
            Array.Fill(weights, 1000f);

            using var input = currentContext.FromFloats(inputs, rows, inputWidth);
            using var expected = previous.Forward(input);
            using var actual = current.Forward(input);
            float[] reference = DirectOps.ToArray(expected);
            EqualBits(reference, DirectOps.ToArray(actual));
            EqualBits(inputs, DirectOps.ToArray(input));
            Assert.Equal(IntPtr.Zero, HostBuffer.GetValue(currentWeight.Storage));

            // Disposing a peer context or requesting residency release must not
            // invalidate this context's owned MLX array or its transpose view.
            previousContext.Dispose();
            currentContext.ReleaseDeviceResidency();
            using var repeated = current.Forward(input);
            EqualBits(reference, DirectOps.ToArray(repeated));
            Assert.Equal(IntPtr.Zero, HostBuffer.GetValue(currentWeight.Storage));
        });

    [MlxFact]
    public void DeviceOnlyWeightRejectsMismatchedDimensions()
    {
        using var allocator = new MlxAllocator();
        Assert.Throws<ArgumentException>(() => MlxMiniMaxH3Ops.CreateFloat32Weight(allocator, [1, 2, 3], 2, 2));
        Assert.Throws<ArgumentException>(() => MlxMiniMaxH3Ops.CreateFloat32Weight(allocator, [], 0, 4));
        Assert.Throws<ArgumentException>(() => MlxMiniMaxH3Ops.CreateFloat32Weight(allocator, [1], long.MaxValue, 2));
    }

    [Fact]
    public void MlxOnlyOptInPreservesCpuLinearPath()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var context = new DirectContext(allocator);
        float[] weights = Values(7 * 11, 503), bias = Values(11, 607);
        using var input = context.FromFloats(Values(3 * 7, 709), 3, 7);
        var previous = DirectLinear.FromFloats(context, weights, 7, 11, bias);
        var current = DirectLinear.FromFloats(context, weights, 7, 11, bias, mlxDeviceOnly: true);
        using var expected = previous.Forward(input);
        using var actual = current.Forward(input);
        EqualBits(DirectOps.ToArray(expected), DirectOps.ToArray(actual));
    }

    private static float[] Values(int count, uint state)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            values[i] = ((state >> 8) * (1f / 16777216f) - .5f) * 2;
        }
        return values;
    }

    private static void EqualBits(float[] expected, float[] actual) => Assert.True(
        MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())),
        "MLX F32 weight upload changed float bits or array ownership.");
}
