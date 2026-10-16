using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed class CpuStorageBulkElementIoTests
{
    [Fact]
    public void FloatAndIntBulkReadsRespectNestedViewOffsets()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        float[] values = Enumerable.Range(0, 24).Select(i => i + .25f).ToArray();
        using var storage = Tensor.FromArray(allocator, values);
        using var matrix = storage.View(6, 4);
        using var middle = matrix.Narrow(0, 1, 4);
        using var last = middle.Narrow(0, 2, 1);
        Assert.Equal(values.AsSpan(12, 4).ToArray(), last.GetElementsAsFloat(4));
        Assert.Equal(values.AsSpan(4, 16).ToArray(), middle.GetElementsAsFloat(16));
        Assert.Equal(values, storage.GetElementsAsFloat(values.Length));

        using var ints = new Tensor(allocator, DType.Int32, 24);
        int[] integerValues = Enumerable.Range(0, 24).Select(i => i * 7 - 19).ToArray();
        ints.SetElementsAsInt(integerValues);
        using var intMatrix = ints.View(6, 4);
        using var intMiddle = intMatrix.Narrow(0, 1, 4);
        using var intLast = intMiddle.Narrow(0, 2, 1);
        Assert.Equal(integerValues.AsSpan(12, 4).ToArray(), intLast.GetElementsAsInt(4));
        Assert.Equal(integerValues.AsSpan(4, 16).ToArray(), intMiddle.GetElementsAsInt(16));
        Assert.Equal(integerValues, ints.GetElementsAsInt(integerValues.Length));
    }

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(0, -1)]
    [InlineData(4, 1)]
    [InlineData(3, 2)]
    [InlineData(long.MaxValue, 1)]
    [InlineData(0, int.MaxValue)]
    public void BulkReadsRejectRangesOutsideStorage(long index, int length)
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var floats = new Tensor(allocator, DType.Float32, 4);
        using var ints = new Tensor(allocator, DType.Int32, 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => floats.Storage.GetElementsAsFloat(index, length));
        Assert.Throws<ArgumentOutOfRangeException>(() => ints.Storage.GetElementsAsInt(index, length));
    }

    [Fact]
    public void EmptyReadAtStorageEndIsValid()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var floats = new Tensor(allocator, DType.Float32, 4);
        using var ints = new Tensor(allocator, DType.Int32, 4);
        Assert.Empty(floats.Storage.GetElementsAsFloat(4, 0));
        Assert.Empty(ints.Storage.GetElementsAsInt(4, 0));
    }
}
