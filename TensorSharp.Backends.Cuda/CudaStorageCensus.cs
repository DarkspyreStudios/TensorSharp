#nullable enable
using System;

namespace TensorSharp.Cuda;

internal enum CudaStorageReservationState { Pending, AttachedNoNative, NativeStarted, Released }

internal sealed class CudaStorageReservation
{
    internal readonly CudaStorageCensus Census;
    internal CudaStorageReservation? Next;
    internal WeakReference<CudaStorage>? Storage;
    internal CudaStorageReservationState State;
    internal bool Releasing;

    internal CudaStorageReservation(CudaStorageCensus census) => Census = census;
}

internal sealed class CudaStorageCensus
{
    internal readonly object Gate = new();
    internal readonly Guid OrderingId = Guid.NewGuid();
    internal readonly CudaAllocator Allocator;
    private CudaStorageReservation? _first;
    private bool _retirementRequested;
    private object? _pendingParent;
    private bool _parentGraphsPending;
    private bool _bareBusyDeclined;
    private int _activeOperations;
    private bool _cacheRetirementActive;

    internal CudaStorageCensus(CudaAllocator allocator) => Allocator = allocator;

    internal CudaStorageReservation Reserve()
    {
        var reservation = new CudaStorageReservation(this);
        lock (Gate)
        {
            ValidateAddition();
            reservation.Next = _first;
            _first = reservation;
        }
        return reservation;
    }

    internal void Attach(CudaStorageReservation reservation, CudaStorage storage)
    {
        var target = new WeakReference<CudaStorage>(storage, true);
        lock (Gate)
        {
            Validate(reservation);
            if (reservation.State != CudaStorageReservationState.Pending)
                throw new InvalidOperationException("The CUDA storage reservation is already attached.");
            reservation.Storage = target;
            reservation.State = CudaStorageReservationState.AttachedNoNative;
        }
    }

    internal void StartNative(CudaStorageReservation reservation)
    {
        lock (Gate)
        {
            Validate(reservation);
            ValidateAddition();
            if (reservation.State != CudaStorageReservationState.AttachedNoNative)
                throw new InvalidOperationException("The CUDA storage reservation cannot begin acquisition.");
            reservation.State = CudaStorageReservationState.NativeStarted;
        }
    }

    internal void ValidateAddition()
    {
        if (!System.Threading.Monitor.IsEntered(Gate))
            throw new InvalidOperationException("CUDA reference admission requires its census gate.");
        if (_retirementRequested)
            throw new ObjectDisposedException(nameof(CudaAllocator), "CUDA allocator retirement refuses new storage references.");
    }

    internal void ValidateOperationAdmission()
    {
        ValidateAddition();
        Allocator.NativeCalls.ThrowIfQuarantined();
        if (_cacheRetirementActive) throw new CudaAllocatorBusyException(Allocator);
        if (_activeOperations == int.MaxValue)
            throw new InvalidOperationException("CUDA operation admission count is exhausted.");
    }

    internal void AdmitOperation()
    {
        if (!System.Threading.Monitor.IsEntered(Gate))
            throw new InvalidOperationException("CUDA operation admission requires its census gate.");
        _activeOperations++;
    }

    internal void AdmitCacheRetirement()
    {
        ValidateOperationAdmission();
        if (_activeOperations != 0) throw new CudaAllocatorBusyException(Allocator);
        _cacheRetirementActive = true;
        _activeOperations++;
    }

    internal void ReleaseCacheRetirement()
    {
        if (!System.Threading.Monitor.IsEntered(Gate) || !_cacheRetirementActive || _activeOperations != 1)
            throw new InvalidOperationException("CUDA cache retirement requires its exclusive counted reservation.");
        _cacheRetirementActive = false;
        _activeOperations--;
    }

    internal void ReleaseOperation()
    {
        if (!System.Threading.Monitor.IsEntered(Gate) || _activeOperations <= 0)
            throw new InvalidOperationException("CUDA operation release requires an active census reservation.");
        _activeOperations--;
    }

    internal bool HasActiveOperations
    {
        get
        {
            if (!System.Threading.Monitor.IsEntered(Gate))
                throw new InvalidOperationException("CUDA operation inspection requires its census gate.");
            return _activeOperations != 0;
        }
    }

    internal bool BeginRelease(CudaStorageReservation reservation)
    {
        lock (Gate)
        {
            if (reservation.State == CudaStorageReservationState.Released) return false;
            Validate(reservation);
            reservation.Releasing = true;
            return _parentGraphsPending;
        }
    }

    internal void Complete(CudaStorageReservation reservation)
    {
        lock (Gate)
        {
            if (reservation.State == CudaStorageReservationState.Released) return;
            Validate(reservation);
            CudaStorageReservation? prior = null;
            for (var current = _first; current != null; current = current.Next)
            {
                if (ReferenceEquals(current, reservation))
                {
                    if (prior == null) _first = current.Next;
                    else prior.Next = current.Next;
                    current.Next = null;
                    current.State = CudaStorageReservationState.Released;
                    return;
                }
                prior = current;
            }
            throw new InvalidOperationException("The CUDA storage reservation is not active.");
        }
    }

    internal void CancelPending(CudaStorageReservation reservation)
    {
        lock (Gate)
        {
            if (reservation.State == CudaStorageReservationState.Pending) Complete(reservation);
        }
    }

    internal void FenceParent(object parent, bool graphsPending)
    {
        lock (Gate)
        {
            Allocator.NativeCalls.ThrowIfQuarantined();
            _retirementRequested = true;
            if (_pendingParent != null && !ReferenceEquals(_pendingParent, parent))
            {
                if (!_bareBusyDeclined || !ReferenceEquals(_pendingParent, Allocator) || _parentGraphsPending)
                    throw new InvalidOperationException("CUDA allocator retirement already belongs to a different owner.");
            }
            _bareBusyDeclined = false;
            _pendingParent = parent;
            _parentGraphsPending |= graphsPending;
        }
    }

    internal void MarkDeclinedBareAttempt()
    {
        if (!System.Threading.Monitor.IsEntered(Gate) || !ReferenceEquals(_pendingParent, Allocator)
            || _parentGraphsPending)
            throw new InvalidOperationException("Only the actual declined bare allocator preflight is transferable.");
        Allocator.NativeCalls.ThrowIfQuarantined();
        _bareBusyDeclined = true;
    }

    internal void AllowStorageRelease(object parent)
    {
        lock (Gate)
        {
            if (!ReferenceEquals(_pendingParent, parent))
                throw new InvalidOperationException("CUDA graph retirement belongs to a different owner.");
            _parentGraphsPending = false;
        }
    }

    internal void CompleteParent(object parent)
    {
        lock (Gate)
        {
            if (!ReferenceEquals(_pendingParent, parent) || _first != null || _parentGraphsPending)
                throw new InvalidOperationException("CUDA parent retirement is incomplete.");
            _pendingParent = null;
        }
    }

    internal CudaStorageReservation? First
    {
        get
        {
            if (!System.Threading.Monitor.IsEntered(Gate))
                throw new InvalidOperationException("CUDA census inspection requires its gate.");
            return _first;
        }
    }

    internal bool HasChildren
    {
        get { lock (Gate) return _first != null; }
    }

    private void Validate(CudaStorageReservation reservation)
    {
        if (!ReferenceEquals(reservation.Census, this) || reservation.State == CudaStorageReservationState.Released)
            throw new InvalidOperationException("The CUDA storage reservation does not belong to this census.");
    }
}
