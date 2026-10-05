using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class ModelWeightRetirementTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublishedBindingIsRemovedOnlyAfterCheckedRelease(bool failRelease)
    {
        var weights = new Dictionary<string, Tensor>();
        var failure = new InvalidOperationException("The weight release did not complete.");
        Tensor? weight = null;
        var allocator = new ReleaseAllocator(() => Assert.Same(weight, weights["weight"]),
            failRelease ? failure : null);
        using var actual = new Tensor(allocator, DType.Float32, 1);
        weight = actual;
        weights.Add("weight", actual);

        if (failRelease)
        {
            Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
                ModelDisposalOwnership.ReleasePublishedWeight(weights, "weight")));
            Assert.Same(actual, Assert.Single(weights).Value);
        }
        else
        {
            ModelDisposalOwnership.ReleasePublishedWeight(weights, "weight");
            Assert.Empty(weights);
        }
        Assert.Equal(IntPtr.Zero, Assert.IsType<ReleaseStorage>(actual.Storage).buffer);
    }

    private sealed class ReleaseAllocator(Action beforeRelease, Exception? failure) : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public Storage Allocate(DType elementType, long elementCount)
            => new ReleaseStorage(this, elementType, elementCount, beforeRelease, failure);
        public float GetAllocatedMemoryRatio() => 0;
    }

    private sealed class ReleaseStorage(IAllocator allocator, DType type, long count,
        Action beforeRelease, Exception? failure) : CpuStorage(allocator, type, count)
    {
        protected override void Destroy()
        {
            try { beforeRelease(); }
            finally { base.Destroy(); }
            if (failure is not null) throw failure;
        }
    }
}
