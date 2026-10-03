using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TensorSharp.Cuda;

internal sealed class CudaAllocatorBusyException : InvalidOperationException
{
    internal readonly object Owner;
    internal CudaAllocatorBusyException(object owner)
        : base("CUDA retirement requires outstanding storage references and acquisitions to drain.") => Owner = owner;
}

internal sealed class CudaRetirementPlan
{
    private readonly object _owner;
    private readonly CudaAllocator[] _allocators;
    private readonly Tensor[] _tensors;
    private readonly Dictionary<CudaStorage, int> _intents;
    private readonly bool _bareAllocatorAttempt;

    private CudaRetirementPlan(object owner, IEnumerable<Tensor> tensors, IEnumerable<IAllocator> allocators,
        bool bareAllocatorAttempt = false)
    {
        _owner = owner;
        _bareAllocatorAttempt = bareAllocatorAttempt;
        _tensors = tensors.Where(t => t != null).Distinct<Tensor>(ReferenceEqualityComparer.Instance).ToArray();
        _allocators = allocators.OfType<CudaAllocator>().Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance)
            .OrderBy(a => a.DeviceId).ThenBy(a => a.Census.OrderingId).ToArray();
        _intents = new Dictionary<CudaStorage, int>(ReferenceEqualityComparer.Instance);
        foreach (Tensor tensor in _tensors)
            if (tensor.Storage is CudaStorage storage && Array.IndexOf(_allocators, storage.AllocatorImpl) >= 0)
                _intents.TryAdd(storage, 0);
        for (int i = 1; i < _allocators.Length; i++)
            if (_allocators[i - 1].DeviceId == _allocators[i].DeviceId
                && _allocators[i - 1].Census.OrderingId == _allocators[i].Census.OrderingId)
                throw new InvalidOperationException("Distinct CUDA allocators have conflicting ordering identities.");
    }

    internal static CudaRetirementPlan PrepareBareAllocator(CudaAllocator allocator)
    {
        var plan = new CudaRetirementPlan(allocator, Array.Empty<Tensor>(), new IAllocator[] { allocator }, true);
        if (!plan.Validate(false)) throw new CudaAllocatorBusyException(allocator);
        return plan;
    }

    internal static CudaRetirementPlan Prepare(object owner, IEnumerable<Tensor> tensors,
        IEnumerable<IAllocator> allocators, bool graphsPending)
    {
        var plan = new CudaRetirementPlan(owner, tensors, allocators);
        if (!plan.Validate(graphsPending)) throw new CudaAllocatorBusyException(owner);
        return plan;
    }

    private bool Validate(bool graphsPending)
    {
        int entered = 0;
        try
        {
            foreach (CudaAllocator allocator in _allocators)
            {
                Monitor.Enter(allocator.Census.Gate);
                entered++;
            }
            foreach (CudaAllocator allocator in _allocators) allocator.Census.FenceParent(_owner, graphsPending);
            foreach (Tensor tensor in _tensors)
                if (tensor.GetLiveOwnedStorageForDisposal() is CudaStorage storage && _intents.ContainsKey(storage))
                    _intents[storage]++;
            foreach (CudaAllocator allocator in _allocators)
            {
                for (var record = allocator.Census.First; record != null; record = record.Next)
                {
                    if (record.Releasing || record.State == CudaStorageReservationState.Pending
                        || record.Storage == null || !record.Storage.TryGetTarget(out var storage)
                        || !_intents.TryGetValue(storage, out int intents) || intents <= 0
                        || intents != storage.ReadReferenceCount())
                        return Decline();
                }
            }
            foreach (var (storage, count) in _intents)
                if (count != 0 && (storage.Reservation.State == CudaStorageReservationState.Released
                    || count != storage.ReadReferenceCount())) return Decline();
            return true;
        }
        finally
        {
            for (int i = entered - 1; i >= 0; i--) Monitor.Exit(_allocators[i].Census.Gate);
        }
    }

    private bool Decline()
    {
        if (_bareAllocatorAttempt) _allocators[0].Census.MarkDeclinedBareAttempt();
        return false;
    }

    internal void AllowStorageRelease()
    {
        foreach (CudaAllocator allocator in _allocators) allocator.Census.AllowStorageRelease(_owner);
    }

    internal bool Owns(CudaAllocator allocator) => Array.IndexOf(_allocators, allocator) >= 0;

    internal void Complete()
    {
        foreach (CudaAllocator allocator in _allocators) allocator.Census.CompleteParent(_owner);
    }
}
