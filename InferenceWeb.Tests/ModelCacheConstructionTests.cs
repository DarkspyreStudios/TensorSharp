using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class ModelCacheConstructionTests
{
    [Fact]
    public void FailedCopyRetainsItsUnreturnedDestinationAndSourceView()
    {
        var allocator = new UnregisteredAllocator();
        using var source = new Tensor(allocator, DType.Float32, 2, 2);
        source.SetElementsAsFloat([1, 2, 3, 4]);
        Tensor pending = source.Transpose();
        Tensor view = pending;
        var retained = new List<Tensor>();
        try
        {
            var error = Assert.Throws<ApplicationException>(() =>
                ModelDisposalOwnership.NewConstructionContiguous(view, ref pending, retained));
            Assert.Equal("None of the registered handlers match the arguments for copy", error.Message);
            Assert.Same(view, Assert.Single(retained));
            Assert.NotNull(pending);
            Assert.NotSame(view, pending);
            Assert.Equal(view.Sizes, pending.Sizes);

            var storage = Assert.IsType<UnregisteredStorage>(pending.Storage);
            Assert.NotEqual(IntPtr.Zero, storage.Backing.buffer);
            pending.Dispose();
            Assert.Equal(IntPtr.Zero, storage.Backing.buffer);
            view.Dispose();
            Assert.Equal([1f, 2f, 3f, 4f], source.GetElementsAsFloat(4));
        }
        finally
        {
            pending?.Dispose();
            foreach (var tensor in retained) tensor.Dispose();
        }
    }

    private sealed class UnregisteredAllocator : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public Storage Allocate(DType elementType, long elementCount)
            => new UnregisteredStorage(this, elementType, elementCount);
        public float GetAllocatedMemoryRatio() => 0;
    }

    // The missing backend handler supplies a copy failure without initializing a native backend.
    private sealed class UnregisteredStorage(IAllocator allocator, DType type, long count)
        : Storage(allocator, type, count)
    {
        internal CpuStorage Backing { get; } = new(allocator, type, count);
        protected override void Destroy() => Backing.Release();
        public override IntPtr PtrAtElement(long index) => Backing.PtrAtElement(index);
        public override string LocationDescription() => "Copy failure fixture";
        public override int[] GetElementsAsInt(long index, int length) => Backing.GetElementsAsInt(index, length);
        public override void SetElementsAsInt(long index, int[] value) => Backing.SetElementsAsInt(index, value);
        public override float GetElementAsFloat(long index) => Backing.GetElementAsFloat(index);
        public override float[] GetElementsAsFloat(long index, int length) => Backing.GetElementsAsFloat(index, length);
        public override void SetElementAsFloat(long index, float value) => Backing.SetElementAsFloat(index, value);
        public override void SetElementsAsFloat(long index, float[] value) => Backing.SetElementsAsFloat(index, value);
        public override void SetElementsAsHalf(long index, TensorSharp.half[] value) => Backing.SetElementsAsHalf(index, value);
        public override void CopyToStorage(long index, IntPtr source, long bytes) => Backing.CopyToStorage(index, source, bytes);
        public override void CopyFromStorage(IntPtr target, long index, long bytes) => Backing.CopyFromStorage(target, index, bytes);
    }
}
