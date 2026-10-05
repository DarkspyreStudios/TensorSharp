using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MediaConstructionCleanupTests
{
    [Fact]
    public void FailedVisionWeightPopulationReleasesTheWeightAndFileButNotTheBorrowedAllocator()
    {
        string directory = Path.Combine(Path.GetTempPath(), "vision-construction-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "projector.gguf");
            WriteProjector(path);
            var failure = new InvalidOperationException("Weight population failed.");
            var allocator = new PopulationFailureAllocator(failure);
            using var borrowed = new Tensor(allocator, DType.Float32, 1);
            borrowed.SetElementsAsFloat([7]);

            Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
                new Qwen35VisionEncoder(path, allocator)));

            Assert.Equal(1, allocator.FailedWeight!.Releases);
            Assert.Equal(IntPtr.Zero, allocator.FailedWeight.buffer);
            Assert.False(allocator.Disposed);
            Assert.Equal([7f], borrowed.GetElementsAsFloat(1));
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void WriteProjector(string path)
    {
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(1ul);
        writer.Write(0ul);
        byte[] name = Encoding.UTF8.GetBytes("v.weight");
        writer.Write((ulong)name.Length);
        writer.Write(name);
        writer.Write(1u);
        writer.Write(1ul);
        writer.Write((uint)GgmlTensorType.F32);
        writer.Write(0ul);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(1f);
    }

    private sealed class PopulationFailureAllocator(Exception failure) : IAllocator, IDisposable
    {
        private bool _borrowedAllocated;
        public FailureStorage? FailedWeight { get; private set; }
        public bool Disposed { get; private set; }
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count)
        {
            if (_borrowedAllocated) return FailedWeight = new FailureStorage(this, type, count, failure);
            _borrowedAllocated = true;
            return new CpuStorage(this, type, count);
        }
        public void Dispose() => Disposed = true;
    }

    private sealed class FailureStorage(IAllocator allocator, DType type, long count, Exception failure)
        : CpuStorage(allocator, type, count)
    {
        public int Releases { get; private set; }
        public override void SetElementsAsFloat(long index, float[] value) => throw failure;
        protected override void Destroy()
        {
            base.Destroy();
            Releases++;
        }
    }
}
