using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDisposable
    {
        private int disposed;
        private readonly CudaNativeCalls nativeCalls;
        internal CudaStorageCensus Census { get; }
        internal CudaNativeCalls NativeCalls => nativeCalls;

        // Striped, size-classed device-memory cache. Replaces the original single
        // global `poolSync` monitor + Dictionary<long, Stack<IntPtr>>, which became a
        // contention hot spot when high-concurrency serving churned many small
        // transient tensors. See CudaDeviceMemoryPool for the design.
        private readonly CudaDeviceMemoryPool pool;

        public CudaAllocator(int deviceId = 0) : this(deviceId, CudaNativeApi.Instance) { }

        internal CudaAllocator(int deviceId, ICudaNativeApi api)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
            ArgumentNullException.ThrowIfNull(api);
            DeviceId = deviceId;
            Census = new CudaStorageCensus(this);
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.Allocator, api, deviceId);
            if (ReferenceEquals(api, CudaNativeApi.Instance)) CudaLibraryResolver.Register();
            CudaBackend.Register();
            bool poolEnabled = !string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_POOL"), "0", StringComparison.Ordinal);
            long maxCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_MB", 512L) * 1024L * 1024L;
            long maxCachedBlockBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_BLOCK_MB", 256L) * 1024L * 1024L;
            // Budget for the global large-block cache (prefill-sized activations;
            // see CudaDeviceMemoryPool). Big enough that a 2048-token prefill
            // chunk's transients all pool, which also keeps CUDA-graph captures
            // of the prefill loop allocation-free.
            long largeCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_LARGE_MB", 1024L) * 1024L * 1024L;
            // Keep at least this much dedicated VRAM free: once the pool would push
            // free VRAM below it, returned blocks go back to the driver instead of
            // into the cache. Prevents the pool from being the thing that tips a
            // nearly-full device (a big model on a small card) into WDDM shared
            // memory. 0 disables the valve (unbounded caching, the old behaviour).
            long minFreeReserveBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MIN_FREE_MB", 256L) * 1024L * 1024L;

            try
            {
                Context = CudaContext.Create(deviceId, api);
                Stream = CudaStream.Create(Context);
                Blas = CudaCublasHandle.Create(Context);
                Blas.SetStream(Stream.Handle);
                Kernels = CudaKernels.TryCreate(Context);
                pool = new CudaDeviceMemoryPool(
                    maxCachedBytes, maxCachedBlockBytes, poolEnabled,
                    backingAllocate: RequireStorageAllocation,
                    backingFree: FreeDeviceMemory,
                    largeCachedBytesCap: largeCachedBytes,
                    queryFreeBytes: QueryFreeDeviceBytes,
                    minFreeReserveBytes: minFreeReserveBytes);
            }
            catch (Exception original)
            {
                try { ReleaseConstruction(); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        public BlasEnum BlasEnum => BlasEnum.CUDA;

        public int DeviceId { get; }

        internal CudaContext Context { get; }

        internal CudaStream Stream { get; }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        public Storage Allocate(DType elementType, long elementCount)
        {
            ThrowIfDisposed();
            nativeCalls.ThrowIfQuarantined();
            var reservation = Census.Reserve();
            try { return new CudaStorage(this, elementType, elementCount, reservation); }
            catch
            {
                Census.CancelPending(reservation);
                throw;
            }
        }

        internal void RentDeviceMemory(CudaStorage storage, long requestedBytes)
        {
            if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(CudaAllocator));
            IntPtr ptr = pool.Rent(requestedBytes, out long allocationBytes, storage.AllocateDeviceMemory);
            storage.AcceptDeviceMemory(ptr, allocationBytes);
            CudaGraphCapture.OnRent(this, ptr, allocationBytes);
        }

        internal void ReturnDeviceMemory(IntPtr ptr, long allocationBytes)
        {
            if (CudaGraphCapture.IsCapturing(this))
            {
                // While a graph capture is active the captured kernels reference
                // this block: pooling keeps it reachable for the capture owner
                // (it is tracked and stolen at capture end); a cuMemFree would
                // leave the graph pointing at freed memory, so those returns are
                // quarantined by the capture context instead.
                bool pooled = pool.TryReturnToPool(ptr, allocationBytes);
                if (!CudaGraphCapture.InterceptReturn(this, ptr, allocationBytes, wouldPool: pooled) && !pooled)
                    pool.Return(ptr, allocationBytes);
                return;
            }
            pool.Return(ptr, allocationBytes);
        }

        /// <summary>Remove a specific free block from the pool so a cached CUDA
        /// graph can own it (see <see cref="CudaPrefillGraphCache"/>).</summary>
        internal bool TryStealPooledBlock(IntPtr ptr, long allocationBytes)
        {
            return pool.TrySteal(ptr, allocationBytes);
        }

        private static IntPtr RequireStorageAllocation(long allocationBytes)
            => throw new InvalidOperationException("CUDA pool acquisition requires the actual constructing storage owner.");

        internal void ReclaimOutsideEffects()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            using var lease = nativeCalls.EnterEffect();
            Synchronize();
            pool.DrainAndFree();
        }

        private void FreeDeviceMemory(IntPtr ptr)
        {
            using var lease = nativeCalls.EnterEffect();
            Context.BindCurrent(nativeCalls);
            nativeCalls.cuCtxSynchronize();
            nativeCalls.cuMemFree(ptr);
        }

        /// <summary>Free dedicated device memory in bytes, for the pool's
        /// VRAM-pressure valve. Sampled at most every ~10 ms, so the MakeCurrent +
        /// cuMemGetInfo cost stays off the hot return path.</summary>
        private long QueryFreeDeviceBytes()
        {
            using var lease = nativeCalls.EnterEffect();
            Context.BindCurrent(nativeCalls);
            if (nativeCalls.cuMemGetInfo(out UIntPtr free, out UIntPtr _) != 0)
                return long.MaxValue; // on query failure, don't throttle caching
            return (long)free.ToUInt64();
        }

        public float GetAllocatedMemoryRatio()
        {
            if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(CudaAllocator));
            using var lease = nativeCalls.EnterEffect();
            Context.BindCurrent(nativeCalls);
            nativeCalls.ThrowOnError(nativeCalls.cuMemGetInfo(out UIntPtr free, out UIntPtr total));
            ulong totalBytes = total.ToUInt64();
            if (totalBytes == 0)
                return 0.0f;

            ulong freeBytes = free.ToUInt64();
            return (float)(1.0 - (double)freeBytes / totalBytes);
        }

        /// <summary>Free / total device memory in bytes (driver view, i.e. dedicated
        /// VRAM only — WDDM shared-memory spillover shows up as this staying pinned
        /// near zero while performance collapses).</summary>
        public (long freeBytes, long totalBytes) GetMemoryInfo()
        {
            if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(CudaAllocator));
            using var lease = nativeCalls.EnterEffect();
            Context.BindCurrent(nativeCalls);
            nativeCalls.ThrowOnError(nativeCalls.cuMemGetInfo(out UIntPtr free, out UIntPtr total));
            return ((long)free.ToUInt64(), (long)total.ToUInt64());
        }

        internal static readonly bool VramLogEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_LOG_VRAM"), "1", StringComparison.Ordinal);

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): prints free/used dedicated VRAM
        /// plus the pool's currently-cached idle bytes at <paramref name="label"/>.</summary>
        public void LogVram(string label)
        {
            if (!VramLogEnabled)
                return;
            (long free, long total) = GetMemoryInfo();
            long cached = pool.GetStats().CachedBytes;
            Console.WriteLine(
                $"[vram] {label}: free={free / (1024 * 1024)} MB used={(total - free) / (1024 * 1024)} MB " +
                $"pool_cached={cached / (1024 * 1024)} MB");
        }

        /// <summary>
        /// Snapshot of allocator pool counters for diagnostics / load testing.
        /// </summary>
        public CudaAllocatorStats GetStats() => pool.GetStats();

        public void Synchronize()
        {
            if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(CudaAllocator));
            using var lease = nativeCalls.EnterEffect();
            Context.BindCurrent(nativeCalls);
            Stream.Synchronize();
        }

        /// <summary>Free every pooled (returned) block back to the driver. For
        /// pipeline stage boundaries where the next stage's working set has a
        /// different shape profile (e.g. a diffusion denoise handing over to a
        /// video VAE): live allocations are untouched, but the caller must ensure
        /// no concurrent Rent/Return on this allocator.</summary>
        public void TrimPool()
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            Synchronize();
            pool.DrainAndFree();
        }

        public void Dispose()
        {
            if (Volatile.Read(ref disposed) != 0) return;
            var plan = CudaRetirementPlan.PrepareBareAllocator(this);
            plan.AllowStorageRelease();
            var restoration = CudaContextRestoration.Capture(nativeCalls);
            DisposeOwned(plan, restoration);
            plan.Complete();
            restoration.Restore();
        }

        internal void DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (!plan.Owns(this)) throw new InvalidOperationException("CUDA allocator cleanup requires its actual retirement plan.");
            if (Volatile.Read(ref disposed) != 0) return;
            if (Census.HasChildren) throw new InvalidOperationException("CUDA allocator storage retirement is incomplete.");
            ReleaseResources(restoration, true);
        }

        internal void DrainForRetirement(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (!plan.Completes(this)) throw new InvalidOperationException("CUDA completion requires its actual retirement plan dependency.");
            if (Volatile.Read(ref disposed) != 0) return;
            using var lease = nativeCalls.EnterEffect();
            if (Volatile.Read(ref disposed) != 0) return;
            try
            {
                Context.BindCurrent(nativeCalls);
                nativeCalls.cuCtxSynchronize();
            }
            catch (Exception failure)
            {
                restoration?.MarkCleanupFailed();
                nativeCalls.PublishFailure(lease, failure, NativeRuntimeFailureStage.Synchronization);
                throw;
            }
        }

        private void ReleaseConstruction()
        {
            var restoration = Context == null ? null : CudaContextRestoration.Capture(nativeCalls);
            ReleaseResources(restoration, false);
            restoration?.Restore();
        }

        private void ReleaseResources(CudaContextRestoration restoration, bool drain)
        {
            if (Volatile.Read(ref disposed) != 0) return;
            var scratch = CudaQuantizedOps.ScratchRetirement.Prepare(this);
            using (var lease = nativeCalls.EnterEffect())
            {
                nativeCalls.ValidateSafeRelease(lease);
                try
                {
                    if (Context != null)
                    {
                        Context.BindCurrent(nativeCalls);
                        if (drain) nativeCalls.cuCtxSynchronize();
                        scratch.Release();
                        CudaQuantizedOps.ReleaseArena(this);
                    }
                    pool?.DrainAndFree();
                    Kernels?.Dispose();
                    Blas?.Dispose();
                    Stream?.Dispose();
                    if (Context != null) Context.DisposeOwned(restoration);
                    nativeCalls.CompleteSafeRelease(lease);
                    Volatile.Write(ref disposed, 1);
                }
                catch (Exception cleanup)
                {
                    restoration?.MarkCleanupFailed();
                    nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.AllocatorRelease);
                    throw;
                }
            }
            scratch.Complete();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref disposed) != 0)
                throw new ObjectDisposedException(nameof(CudaAllocator));
            lock (Census.Gate) Census.ValidateAddition();
        }

        private static long ReadPoolLimit(string name, long defaultMb)
        {
            string value = Environment.GetEnvironmentVariable(name);
            if (long.TryParse(value, out long parsed) && parsed >= 0)
                return parsed;
            return defaultMb;
        }
    }

    /// <summary>
    /// Immutable snapshot of <see cref="CudaAllocator"/> pool counters.
    /// </summary>
    public readonly struct CudaAllocatorStats
    {
        internal CudaAllocatorStats(
            bool poolEnabled,
            long poolHitCount,
            long poolMissCount,
            long cuMemAllocCount,
            long cuMemFreeCount,
            long returnedToPoolCount,
            long cachedBytes,
            long peakCachedBytes,
            long maxCachedBytes,
            long maxCachedBlockBytes,
            int shardCount)
        {
            PoolEnabled = poolEnabled;
            PoolHitCount = poolHitCount;
            PoolMissCount = poolMissCount;
            CuMemAllocCount = cuMemAllocCount;
            CuMemFreeCount = cuMemFreeCount;
            ReturnedToPoolCount = returnedToPoolCount;
            CachedBytes = cachedBytes;
            PeakCachedBytes = peakCachedBytes;
            MaxCachedBytes = maxCachedBytes;
            MaxCachedBlockBytes = maxCachedBlockBytes;
            ShardCount = shardCount;
        }

        public bool PoolEnabled { get; }

        /// <summary>Rent calls served from the pool (no backing allocation).</summary>
        public long PoolHitCount { get; }

        /// <summary>Rent calls that fell through to a backing allocation.</summary>
        public long PoolMissCount { get; }

        /// <summary>Total cuMemAlloc driver calls issued (== <see cref="PoolMissCount"/>).</summary>
        public long CuMemAllocCount { get; }

        /// <summary>Total cuMemFree driver calls issued.</summary>
        public long CuMemFreeCount { get; }

        /// <summary>Return calls that parked a block back into the pool.</summary>
        public long ReturnedToPoolCount { get; }

        /// <summary>Bytes currently held in the pool (summed across shards).</summary>
        public long CachedBytes { get; }

        /// <summary>Sum of per-shard high-water marks (conservative upper bound, ≤ MaxCachedBytes).</summary>
        public long PeakCachedBytes { get; }

        public long MaxCachedBytes { get; }

        public long MaxCachedBlockBytes { get; }

        /// <summary>Number of independent pool shards.</summary>
        public int ShardCount { get; }

        public long TotalRentCount => PoolHitCount + PoolMissCount;

        public double PoolHitRatio => TotalRentCount == 0 ? 0.0 : (double)PoolHitCount / TotalRentCount;

        public override string ToString()
        {
            return $"CudaAllocatorStats(shards={ShardCount}, hits={PoolHitCount}, misses={PoolMissCount}, hitRatio={PoolHitRatio:P1}, " +
                   $"cuMemAlloc={CuMemAllocCount}, cuMemFree={CuMemFreeCount}, returnedToPool={ReturnedToPoolCount}, " +
                   $"cachedBytes={CachedBytes}, peakCachedBytes={PeakCachedBytes})";
        }
    }
}
