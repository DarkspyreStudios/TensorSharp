using System.Collections.Generic;
using TensorSharp.Cuda;

namespace TensorSharp.Models;

public abstract partial class ModelBase
{
    private bool _ownershipRetirementStarted;
    private bool _ownershipResourcesReleased;
    private System.Action<CudaRetirementPlan, CudaContextRestoration> _releaseOwnedChildren;

    internal void SetOwnedChildRelease(System.Action<CudaRetirementPlan, CudaContextRestoration> release)
        => _releaseOwnedChildren = release;

    /// <summary>
    /// Collects the actual tensors and allocators this model releases. Overrides call
    /// the base method and add their initialized owned resources, without cleanup effects.
    /// Borrowed allocators do not grant disposal authority.
    /// </summary>
    protected virtual void CollectDisposalOwnership(ICollection<Tensor> ownedTensors,
        ICollection<IAllocator> ownedAllocators)
        => CollectBaseDisposalOwnership(ownedTensors, ownedAllocators, OwnsTensorParallelGroup);

    private void CollectBaseDisposalOwnership(ICollection<Tensor> ownedTensors,
        ICollection<IAllocator> ownedAllocators, bool ownsTensorParallelGroup)
    {
        ModelDisposalOwnership.AddRange(ownedTensors, _weights.Values);
        ModelDisposalOwnership.AddRows(ownedTensors, _tpWeights.Values);
        ModelDisposalOwnership.AddRange(ownedTensors, _tpWeightReplicaCache.Values);
        if (MultimodalInjector is ModelMultimodalInjector injector)
            injector.CollectDisposalOwnership(ownedTensors);
        CollectBaseOwnedAllocators(ownedAllocators, ownsTensorParallelGroup);
    }

    private void CollectBaseOwnedAllocators(ICollection<IAllocator> ownedAllocators, bool ownsTensorParallelGroup)
    {
        if (ownsTensorParallelGroup && _tpGroup is ICudaTensorParallelRetirement group && group.OwnsCudaAllocators)
            group.CollectOwnedAllocators(ownedAllocators);
        if (!_allocatorFromTensorParallelGroup && _allocator != null) ownedAllocators.Add(_allocator);
    }

    internal void CollectOwnedModelDisposalOwnership(ICollection<Tensor> tensors, ICollection<IAllocator> allocators)
        => CollectDisposalOwnership(tensors, allocators);

    internal void ThrowIfUnsafeOwnershipCleanup()
    {
        if (_ownershipCleanupFailed)
            throw new System.InvalidOperationException("Model ownership cleanup previously failed; resource release is unsafe.",
                _ownershipCleanupFailure);
    }

    internal void CollectDFlashDisposalOwnership(ICollection<Tensor> tensors)
    {
        ModelDisposalOwnership.Add(tensors, _dflashRingK);
        ModelDisposalOwnership.Add(tensors, _dflashRingV);
    }
}
