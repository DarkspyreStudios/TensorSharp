using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TensorSharp.Cuda;

// Lifetime admission only. It never holds native effect gates across the operation.
internal sealed class CudaOperationAdmission : IDisposable
{
    private readonly object _owner;
    private readonly CudaAllocator[] _allocators;
    private int _released;
    private readonly bool _cacheRetirement;

    private CudaOperationAdmission(object owner, CudaAllocator[] allocators, bool cacheRetirement = false)
    {
        _owner = owner;
        _allocators = allocators;
        _cacheRetirement = cacheRetirement;
    }

    internal static CudaOperationAdmission Enter(object owner, CudaAllocator[] actualAllocators)
    {
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentNullException.ThrowIfNull(actualAllocators);
        var allocators = (CudaAllocator[])actualAllocators.Clone();
        if (allocators.Length == 0 || allocators.Any(a => a == null)
            || allocators.Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance).Count() != allocators.Length)
            throw new InvalidOperationException("CUDA operation admission requires distinct actual allocators.");
        Array.Sort(allocators, (a, b) =>
        {
            int ordinal = a.DeviceId.CompareTo(b.DeviceId);
            return ordinal != 0 ? ordinal : a.Census.OrderingId.CompareTo(b.Census.OrderingId);
        });
        for (int i = 1; i < allocators.Length; i++)
            if (allocators[i - 1].DeviceId == allocators[i].DeviceId
                && allocators[i - 1].Census.OrderingId == allocators[i].Census.OrderingId)
                throw new InvalidOperationException("CUDA operation allocators have conflicting ordering identities.");
        var admission = new CudaOperationAdmission(owner, allocators);
        int entered = 0;
        try
        {
            foreach (CudaAllocator allocator in allocators)
            {
                Monitor.Enter(allocator.Census.Gate);
                entered++;
            }
            foreach (CudaAllocator allocator in allocators) allocator.Census.ValidateOperationAdmission();
            foreach (CudaAllocator allocator in allocators) allocator.Census.AdmitOperation();
            return admission;
        }
        finally
        {
            for (int i = entered - 1; i >= 0; i--) Monitor.Exit(allocators[i].Census.Gate);
        }
    }

    internal void ValidateAllocator(CudaAllocator allocator)
    {
        if (Volatile.Read(ref _released) != 0 || Array.IndexOf(_allocators, allocator) < 0)
            throw new InvalidOperationException("CUDA operation admission does not cover the actual allocator.");
    }

    internal static CudaOperationAdmission EnterCacheRetirement(CudaAllocator allocator)
    {
        var admission = new CudaOperationAdmission(allocator, new[] { allocator }, true);
        lock (allocator.Census.Gate)
        {
            allocator.Census.AdmitCacheRetirement();
        }
        return admission;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        int entered = 0;
        try
        {
            foreach (CudaAllocator allocator in _allocators)
            {
                Monitor.Enter(allocator.Census.Gate);
                entered++;
            }
            foreach (CudaAllocator allocator in _allocators)
                if (_cacheRetirement) allocator.Census.ReleaseCacheRetirement();
                else allocator.Census.ReleaseOperation();
        }
        finally
        {
            for (int i = entered - 1; i >= 0; i--) Monitor.Exit(_allocators[i].Census.Gate);
            GC.KeepAlive(_owner);
        }
    }
}
