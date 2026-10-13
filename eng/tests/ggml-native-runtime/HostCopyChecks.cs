using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

internal static class HostCopyChecks
{
    internal static void Run(GgmlContext context)
    {
        var allocator = new PendingAllocator(context);
        foreach (bool strided in new[] { false, true })
        {
            using var source = new Tensor(allocator, DType.Float32, new long[] { 2, 2 }, strided ? [1, 2] : [2, 1]);
            using var destination = new Tensor(allocator, DType.Float32, 2, 2);
            source.SetElementsAsFloat([0, 0, 0, 0]);
            ((PendingStorage)source.Storage).Pending = [1, 2, 3, 4];
            ((PendingStorage)destination.Storage).Pending = [-1, -1, -1, -1];
            GgmlBasicOps.Copy(destination, source);
            float[] expected = strided ? [1, 3, 2, 4] : [1, 2, 3, 4];
            if (!destination.GetElementsAsFloat(4).SequenceEqual(expected))
                throw new InvalidOperationException("Host copy must drain source and destination writes before copying.");
        }
    }

    private sealed class PendingAllocator(GgmlContext context) : IAllocator
    {
        private readonly GgmlAllocator inner = new(context, 0);
        /// <summary>Gets the selected context backend.</summary>
        public BlasEnum BlasEnum => inner.BlasEnum;
        /// <summary>Gets the context device.</summary>
        public int DeviceId => inner.DeviceId;
        /// <summary>Reports the backing allocator memory ratio.</summary>
        public float GetAllocatedMemoryRatio() => inner.GetAllocatedMemoryRatio();
        /// <summary>Allocates real GGML storage with a deferred-write test seam.</summary>
        public Storage Allocate(DType elementType, long elementCount) => new PendingStorage(inner, elementType, elementCount);
    }

    // The real pool owns the bytes; this stand-in completes a queued write at the host-access seam.
    private sealed class PendingStorage(GgmlAllocator allocator, DType type, long count)
        : GgmlStorage(allocator, allocator.Context, type, count)
    {
        internal float[]? Pending;
        /// <summary>Drains real device work and completes the pending test write.</summary>
        public override void EnsureHostReadable()
        {
            base.EnsureHostReadable();
            if (Pending is not { } values) return;
            Pending = null;
            Marshal.Copy(values, 0, PtrAtElement(0), values.Length);
        }
    }
}
