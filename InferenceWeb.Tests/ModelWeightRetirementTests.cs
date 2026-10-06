using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class ModelWeightRetirementTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MlxStackedCacheReleasePrecedesBackingOwnerReleaseAndPreservesFailure(bool failRelease)
    {
        using var backing = new QuantizedWeight(new byte[] { 7, 8 }, ggmlType: 0, ne0: 1, ne1: 2);
        var stacked = new StackedExpertWeights(backing.Data, 0, 1, 1, 2, 2,
            isExternalView: false, ownerToken: backing, ownedBuffer: IntPtr.Zero);
        using QuantizedWeight expert = QuantizedWeight.CreateExpertView(stacked, 0);
        IntPtr expertKey = expert.EnsureDeviceCacheKey();
        Assert.NotEqual(stacked.Data, expertKey);
        var calls = new List<IntPtr>();
        var original = new IOException("Stacked native cache release failed.");
        Exception? failure = Record.Exception(() => ModelDisposalOwnership.ReleaseMlxWeightCaches(
            [expert], [stacked], key =>
            {
                Assert.Equal(7, System.Runtime.InteropServices.Marshal.ReadByte(backing.Data));
                Assert.True(expert.HasHostData);
                calls.Add(key);
                if (key == stacked.Data && failRelease) throw original;
            }));

        Assert.Equal(new[] { expertKey, stacked.Data }, calls);
        Assert.Equal(7, System.Runtime.InteropServices.Marshal.ReadByte(backing.Data));
        Assert.True(expert.HasHostData);
        if (failRelease) Assert.Same(original, failure);
        else Assert.Null(failure);
    }

    [Fact]
    public void StorageBoundBusyRefusalKeepsWholeModelCleanupRecoverableAndExecutionFenced()
    {
        string directory = Path.Combine(Path.GetTempPath(), "model-cleanup-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "model.gguf");
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0ul);
                writer.Write(0ul);
                writer.Write(0ul);
            }
            using var model = new RecoverableModel(path);
            var storage = Assert.IsType<CpuStorage>(model.Weight.Storage);
            // The native cleanup seam refuses before releasing the actual live storage.
            var busy = (Exception)Activator.CreateInstance(
                typeof(Tensor).Assembly.GetType("TensorSharp.NativeMlxCallbackBusyException", throwOnError: true)!,
                BindingFlags.Instance | BindingFlags.NonPublic, binder: null, args: [storage], culture: null)!;
            model.Refusal = busy;

            Assert.Same(busy, Record.Exception(model.Dispose));
            Assert.False(model.IsOwnershipCleanupUnsafe);
            Assert.False(model.OwnershipResourcesReleased);
            Assert.True(storage.IsOwnerExclusive());
            Assert.Equal([7f], model.Weight.GetElementsAsFloat(1));
            Assert.Throws<ObjectDisposedException>(model.ResetKVCache);
            Assert.Throws<IOException>(() => File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None).Dispose());

            model.Refusal = null;
            model.Dispose();
            Assert.True(model.OwnershipResourcesReleased);
            Assert.Equal(IntPtr.Zero, storage.buffer);
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

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

    private sealed class RecoverableModel : ModelBase
    {
        internal Exception? Refusal;
        internal Tensor Weight { get; }

        internal RecoverableModel(string path) : base(path, BackendType.Cpu)
        {
            Weight = new Tensor(_allocator, DType.Float32, 1);
            Weight.SetElementsAsFloat([7]);
            _weights.Add("weight", Weight);
        }

        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
        public override void Dispose() => DisposeBaseResources(() =>
        {
            if (Refusal is not null) throw Refusal;
        });
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
