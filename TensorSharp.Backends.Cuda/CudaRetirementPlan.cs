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

internal sealed class CudaModelCleanupEffect : IDisposable
{
    private readonly CudaNativeCalls _calls;
    private readonly NativeEffectLease _lease;

    internal CudaModelCleanupEffect(CudaNativeCalls calls)
    {
        _calls = calls;
        _lease = calls.EnterEffect();
    }

    internal void PublishFailure(Exception failure)
        => _calls.PublishFailure(_lease, failure, NativeRuntimeFailureStage.GraphRelease);

    public void Dispose() => _lease.Dispose();
}

internal sealed class CudaRetirementPlan
{
    private readonly object _owner;
    private readonly CudaAllocator[] _allocators;
    private readonly CudaAllocator[] _completionAllocators;
    private readonly Tensor[] _tensors;
    private readonly Dictionary<CudaStorage, int> _intents;
    private readonly bool _bareAllocatorAttempt;
    private CudaNativeCalls _modelCleanupCalls;

    private CudaRetirementPlan(object owner, IEnumerable<Tensor> tensors, IEnumerable<IAllocator> allocators,
        bool bareAllocatorAttempt = false)
    {
        _owner = owner;
        _bareAllocatorAttempt = bareAllocatorAttempt;
        _tensors = tensors.Where(t => t != null).Distinct<Tensor>(ReferenceEqualityComparer.Instance).ToArray();
        _allocators = allocators.OfType<CudaAllocator>().Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance)
            .OrderBy(a => a.DeviceId).ThenBy(a => a.Census.OrderingId).ToArray();
        _intents = new Dictionary<CudaStorage, int>(ReferenceEqualityComparer.Instance);
        var completion = new List<CudaAllocator>(_allocators);
        foreach (Tensor tensor in _tensors)
            if (tensor.GetLiveOwnedStorageForDisposal() is CudaStorage storage)
            {
                completion.Add(storage.AllocatorImpl);
                if (Array.IndexOf(_allocators, storage.AllocatorImpl) >= 0) _intents.TryAdd(storage, 0);
            }
        _completionAllocators = completion.Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance)
            .OrderBy(a => a.DeviceId).ThenBy(a => a.Census.OrderingId).ToArray();
        foreach (CudaAllocator allocator in _completionAllocators)
            if (!ReferenceEquals(allocator.NativeCalls.Api, _completionAllocators[0].NativeCalls.Api))
                throw new InvalidOperationException("CUDA retirement completion requires one actual native API identity.");
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

    internal static CudaRetirementPlan PrepareModel(object owner, IEnumerable<Tensor> tensors,
        IEnumerable<IAllocator> allocators)
    {
        var plan = Prepare(owner, tensors, allocators, true);
        if (plan._completionAllocators.Length != 0)
            plan._modelCleanupCalls = new CudaNativeCalls(owner, NativeOwnerRole.Model,
                plan._completionAllocators[0].NativeCalls.Api,
                plan._completionAllocators.Select(a => a.DeviceId).Distinct().ToArray());
        return plan;
    }

    internal CudaModelCleanupEffect EnterModelCleanupEffect()
        => _modelCleanupCalls == null ? null : new CudaModelCleanupEffect(_modelCleanupCalls);

    internal void PublishModelCleanupFailure(CudaModelCleanupEffect lease, Exception failure)
    {
        lease?.PublishFailure(failure);
    }

    internal static void FenceOwnership(object owner, IEnumerable<IAllocator> allocators)
    {
        var plan = new CudaRetirementPlan(owner, Array.Empty<Tensor>(), allocators);
        int entered = 0;
        try
        {
            foreach (CudaAllocator allocator in plan._allocators)
            {
                Monitor.Enter(allocator.Census.Gate);
                entered++;
            }
            foreach (CudaAllocator allocator in plan._allocators) allocator.Census.FenceParent(owner, true);
        }
        finally
        {
            for (int i = entered - 1; i >= 0; i--) Monitor.Exit(plan._allocators[i].Census.Gate);
        }
    }

    internal static CudaRetirementPlan PrepareGroup(object owner, ICudaTensorParallelRetirement group)
    {
        var allocators = new List<IAllocator>();
        group.CollectOwnedAllocators(allocators);
        return Prepare(owner, Array.Empty<Tensor>(), allocators, false);
    }

    internal CudaContextRestoration CaptureRestoration() => _completionAllocators.Length == 0
        ? null : CudaContextRestoration.Capture(_completionAllocators[0].NativeCalls);

    internal void Drain(CudaContextRestoration restoration)
    {
        foreach (CudaAllocator allocator in _completionAllocators) allocator.DrainForRetirement(this, restoration);
    }

    internal void ValidateGroupRelease(ICudaTensorParallelRetirement group)
    {
        var allocators = new List<IAllocator>();
        group.CollectOwnedAllocators(allocators);
        if (!group.OwnsCudaAllocators || allocators.Count == 0)
            throw new InvalidOperationException("CUDA group cleanup requires actual owned CUDA allocators.");
        foreach (IAllocator candidate in allocators)
            if (candidate is not CudaAllocator allocator || !Owns(allocator) || allocator.Census.HasChildren)
                throw new InvalidOperationException("CUDA group cleanup requires its fully drained owning plan.");
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

    internal bool Completes(CudaAllocator allocator) => Array.IndexOf(_completionAllocators, allocator) >= 0;

    internal void Complete()
    {
        foreach (CudaAllocator allocator in _allocators) allocator.Census.CompleteParent(_owner);
        if (_modelCleanupCalls != null)
        {
            using var lease = _modelCleanupCalls.EnterEffect();
            _modelCleanupCalls.CompleteSafeRelease(lease);
        }
    }
}
