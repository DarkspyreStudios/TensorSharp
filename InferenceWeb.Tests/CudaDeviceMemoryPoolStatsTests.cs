using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public class CudaDeviceMemoryPoolStatsTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Return_CountsBackingFreesWithoutCountingCachedBlocks(bool enabled)
    {
        int frees = 0;
        var pool = new CudaDeviceMemoryPool(
            maxCachedBytes: 256,
            maxCachedBlockBytes: 256,
            enabled: enabled,
            backingAllocate: _ => new IntPtr(1),
            backingFree: _ => frees++,
            shardCount: 1);

        pool.Return(new IntPtr(1), 256);
        pool.Return(new IntPtr(2), 256);
        pool.Return(IntPtr.Zero, 256);

        Assert.Equal(enabled ? 1 : 2, frees);
        Assert.Equal(frees, pool.GetStats().CuMemFreeCount);
        Assert.Equal(enabled ? 256 : 0, pool.GetStats().CachedBytes);
    }
}
