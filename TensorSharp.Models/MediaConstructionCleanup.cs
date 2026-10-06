using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using TensorSharp.Cuda;
using TensorSharp.GGML;

namespace TensorSharp.Models;

// A media child borrows its allocator. Its release plan owns only the child's actual tensors.
internal sealed class MediaConstructionCleanup
{
    private readonly object _owner;
    private readonly bool _ggml;
    private readonly Action<ICollection<Tensor>> _collect;
    private readonly Action _releaseResources;
    private readonly Action _releaseFiles;
    private readonly Action<Exception> _retainUnsafeParent;
    private bool _resourcesReleased;
    private bool _released;
    private Exception _unsafeFailure;
    private Tensor[] _ownedTensors = Array.Empty<Tensor>();

    internal MediaConstructionCleanup(object owner, IAllocator allocator,
        Action<ICollection<Tensor>> collect, Action releaseResources, Action releaseFiles,
        Action<Exception> retainUnsafeParent)
    {
        _owner = owner;
        _ggml = allocator is GgmlAllocator;
        _collect = collect;
        _releaseResources = releaseResources;
        _releaseFiles = releaseFiles;
        _retainUnsafeParent = retainUnsafeParent;
        Cleanup = new NativeConstructionCleanupHandle(owner, Release, () => _released,
            () => _unsafeFailure != null);
    }

    internal NativeConstructionCleanupHandle Cleanup { get; }

    internal static void RequireReleasedConstruction(ref NativeConstructionCleanupHandle pending)
    {
        if (pending == null) return;
        if (!pending.IsReleased)
            throw new InvalidOperationException("A failed media construction still owns resources; release its cleanup handle before loading another encoder.");
        pending = null;
    }

    internal void CollectDisposalOwnership(ICollection<Tensor> tensors)
    {
        foreach (Tensor tensor in SnapshotOwnership()) tensors.Add(tensor);
    }

    private Tensor[] SnapshotOwnership()
    {
        var tensors = new List<Tensor>();
        _collect(tensors);
        return _ownedTensors = tensors.Distinct<Tensor>(ReferenceEqualityComparer.Instance).ToArray();
    }

    private bool IsHealthyBusy(Exception failure)
        => ModelDisposalOwnership.IsHealthyStorageRefusal(failure, _ownedTensors);

    internal void RollBackConstruction(Exception original, bool fileCloseFailed = false)
    {
        try
        {
            if (fileCloseFailed) Release(original);
            else Cleanup.Dispose();
        }
        catch (Exception cleanup)
        {
            throw new NativeConstructionCleanupException(original, cleanup, Cleanup);
        }
    }

    private void Release() => Release(null);

    private void Release(Exception priorFileFailure)
    {
        if (_released) return;
        ThrowIfUnsafe();
        CudaContextRestoration restoration = null;
        if (!_resourcesReleased)
        {
            // Collect and deduplicate through the existing plan before entering any native effect.
            Tensor[] tensors = SnapshotOwnership();
            var plan = CudaRetirementPlan.PrepareModel(_owner, tensors, Array.Empty<IAllocator>());
            restoration = plan.CaptureRestoration();
            bool drainCompleted = false;
            bool resourceBodyEntered = false;
            try
            {
                plan.Drain(restoration);
                drainCompleted = true;
                bool detachGgml = _ggml && tensors.Length != 0 && GgmlNativeLoader.NativeOwnershipMayExist;
                using (var ggmlCleanup = !detachGgml ? null : GgmlNativeLoader.ReserveResourceCleanup(_owner))
                {
                    using (var effect = plan.EnterModelCleanupEffect())
                    {
                        resourceBodyEntered = true;
                        try
                        {
                            if (detachGgml) GgmlBasicOps.HostReadBarrier();
                            _releaseResources();
                        }
                        catch (Exception failure)
                        {
                            if (!IsHealthyBusy(failure)) plan.PublishModelCleanupFailure(effect, failure);
                            throw;
                        }
                    }
                    ggmlCleanup?.Complete();
                }
                plan.Complete();
                _resourcesReleased = true;
            }
            catch (Exception failure)
            {
                if (!IsHealthyBusy(failure) && (!drainCompleted || resourceBodyEntered
                    || failure is NativeRuntimeQuarantinedException
                    || (_ggml && GgmlNativeLoader.IsUnsafeCleanupRefusal(failure))))
                {
                    restoration?.MarkCleanupFailed();
                    _unsafeFailure = failure;
                    // Parent retention can allocate and take host locks only after native effects exit.
                    try { _retainUnsafeParent?.Invoke(failure); }
                    catch (Exception retentionError) { throw new AggregateException(failure, retentionError); }
                }
                else
                {
                    try { restoration?.Restore(); }
                    catch (Exception restoreError) { throw new AggregateException(failure, restoreError); }
                }
                throw;
            }
        }

        // A file-only refusal retains its recipe without repeating proven native resource release.
        ExceptionDispatchInfo fileFailure = priorFileFailure == null ? null : ExceptionDispatchInfo.Capture(priorFileFailure);
        if (fileFailure == null)
        {
            try { ReleaseFiles(); }
            catch (Exception failure) { fileFailure = ExceptionDispatchInfo.Capture(failure); }
        }
        try { restoration?.Restore(); }
        catch (Exception failure)
        {
            if (fileFailure != null) throw new AggregateException(fileFailure.SourceException, failure);
            throw;
        }
        fileFailure?.Throw();
    }

    internal void ReleaseOwned()
    {
        if (_released) return;
        ThrowIfUnsafe();
        // The actual parent's plan already owns the complete census and native completion barrier.
        if (!_resourcesReleased)
        {
            try { _releaseResources(); }
            catch (Exception failure)
            {
                if (!IsHealthyBusy(failure)) _unsafeFailure = failure;
                throw;
            }
            _resourcesReleased = true;
        }
        ReleaseFiles();
    }

    private void ReleaseFiles()
    {
        _releaseFiles();
        _ownedTensors = Array.Empty<Tensor>();
        _released = true;
        Cleanup.CompleteRelease(_owner);
    }

    private void ThrowIfUnsafe()
    {
        if (_unsafeFailure != null) ExceptionDispatchInfo.Capture(_unsafeFailure).Throw();
    }
}
