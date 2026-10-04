#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading;

namespace TensorSharp.GGML;

public static partial class GgmlNativeLoader
{
    internal enum NativeAdmissionMode { CurrentBackend, IdentityOnly, CompileOnly, BackendEffect, Cleanup }
    private enum CallPhase { Reserved, Native, BorrowedCallback, Unwinding, Completed }
    private enum AcquisitionState { Reserved, ReturnedZero, Owned, ReleaseAttempted, Released }

    private sealed class GgmlProcessOwner
    {
        internal readonly Dictionary<long, ResourceRecord> Resources = s_resources;
        internal readonly Dictionary<(string Kind, IntPtr Handle), NativeResource> NativeResources = s_nativeResources;
        internal readonly Dictionary<long, NativeCallLease> Calls = s_calls;
        internal NativeOwnerRegistration? CudaDependency;
        internal bool MayUseCudaPrimaryContexts;
        internal bool CudaAttachmentInProgress;
        internal ExceptionDispatchInfo? CudaAdmissionFailure;
        internal IntPtr Library;
        internal GgmlNativeBuildIdentity? Identity;
        internal ExceptionDispatchInfo? ShutdownFailure;
    }

    private sealed class ResourceRecord(GgmlRuntimeResourceKind kind, object? owner = null,
        NativeOwnerRegistration? registration = null)
    {
        internal readonly GgmlRuntimeResourceKind Kind = kind;
        internal readonly WeakReference<object>? Owner = owner == null ? null : new(owner, true);
        internal readonly WeakReference<NativeOwnerRegistration>? Registration = registration == null ? null : new(registration, true);
        internal bool CudaAttached;
        internal object? RefusedOwner;
        internal Exception? CleanupFailure;
    }

    private sealed class NativeResource
    {
        internal readonly string Kind;
        internal readonly NativeOwnerRegistration Registration;
        internal IntPtr Handle;
        internal int Calls;
        internal bool Releasing;
        internal bool ReleaseReturned;
        internal bool CudaAttached;
        internal AcquisitionState State;
        internal Exception? CleanupFailure;
        internal NativeResource? ConflictingAcquisition;
        internal readonly GgmlProcessOwner Process = s_processOwnerToken;

        internal NativeResource(string kind)
        {
            Kind = kind;
            Registration = NativeQuarantineAuthority.Register(this, NativeOwnerRole.NativeHandle);
        }

        internal void CompleteRelease()
        {
            using var effect = Registration.EnterEffect();
            effect.CompleteSafeRelease(this);
            State = AcquisitionState.Released;
        }
    }

    private static void CheckProcessDependency()
    {
        s_processOwnerToken.CudaAdmissionFailure?.Throw();
        s_processOwnerToken.CudaDependency?.ThrowIfQuarantined();
    }

    internal static bool NativeOwnershipMayExist
    {
        get { lock (s_gate) return s_nativeUseStarted || s_processOwnerToken.MayUseCudaPrimaryContexts; }
    }

    internal static bool IsUnsafeCleanupRefusal(Exception error)
    {
        lock (s_gate) return error is NativeRuntimeQuarantinedException || s_runtimeState == GgmlRuntimeState.Poisoned;
    }

    private static void EstablishCudaDependency()
    {
        List<(ResourceRecord Record, NativeOwnerRegistration Registration)> owners;
        NativeResource[] native;
        lock (s_gate)
        {
            CheckProcessDependency();
            if (s_processOwnerToken.MayUseCudaPrimaryContexts) return;
            if (s_processOwnerToken.CudaAttachmentInProgress || s_calls.Count != 0 || s_cleanup != null)
                throw new InvalidOperationException("CUDA dependency attachment cannot overlap native calls or cleanup.");
            ClaimProcess();
            s_processOwnerToken.CudaDependency = NativeQuarantineAuthority.Register(s_processOwnerToken, NativeOwnerRole.NativeHandle);
            s_processOwnerToken.CudaAttachmentInProgress = true;
            s_processOwnerToken.MayUseCudaPrimaryContexts = true;
            owners = s_resources.Values.Where(record => record.Registration != null)
                .Select(record => (Record: record, Registration: record.Registration!.TryGetTarget(out var registration) ? registration : null))
                .Where(pair => pair.Registration != null).Select(pair => (pair.Record, pair.Registration!)).ToList();
            native = s_nativeResources.Values.ToArray();
        }
        try
        {
            s_processOwnerToken.CudaDependency.AttachCudaDependentGgml();
            foreach (var (record, registration) in owners)
            {
                registration.AttachCudaDependentGgml();
                record.CudaAttached = true;
            }
            foreach (var resource in native)
            {
                resource.Registration.AttachCudaDependentGgml();
                resource.CudaAttached = true;
            }
        }
        catch (Exception error)
        {
            lock (s_gate)
            {
                s_processOwnerToken.CudaAdmissionFailure = ExceptionDispatchInfo.Capture(error);
                foreach (var (record, registration) in owners)
                {
                    record.RefusedOwner = registration.Owner;
                    record.CleanupFailure ??= error;
                }
            }
            throw;
        }
        finally { lock (s_gate) s_processOwnerToken.CudaAttachmentInProgress = false; }
    }

    internal static void EnsureEffectiveCudaDependency(NativeOwnerRegistration registration, GgmlBackendType? requestedBackend)
    {
        if (requestedBackend == GgmlBackendType.Cuda) EstablishCudaDependency();
        ResourceRecord? record;
        bool attach;
        lock (s_gate)
        {
            CheckProcessDependency();
            if (s_processOwnerToken.CudaAttachmentInProgress)
                throw new InvalidOperationException("CUDA dependency attachment is in progress.");
            record = s_resources.Values.FirstOrDefault(value => value.Registration != null &&
                value.Registration.TryGetTarget(out var actual) && ReferenceEquals(actual, registration));
            attach = s_processOwnerToken.MayUseCudaPrimaryContexts && record?.CudaAttached != true;
        }
        if (attach)
        {
            registration.AttachCudaDependentGgml();
            if (record != null) { lock (s_gate) record.CudaAttached = true; }
        }
        registration.ThrowIfQuarantined();
    }

    internal static OwnedResourceLease AcquireOwnedLease(GgmlRuntimeResourceKind kind, object actualOwner,
        NativeOwnerRegistration registration, GgmlBackendType? requestedBackend = null)
    {
        ArgumentNullException.ThrowIfNull(actualOwner);
        if (!ReferenceEquals(registration.Owner, actualOwner))
            throw new InvalidOperationException("The GGML resource registration must belong to its actual owner.");
        long id;
        ResourceRecord record;
        lock (s_gate)
        {
            EnsureOperational();
            CheckCleanupAdmission();
            ClaimProcess();
            id = ++s_nextLease;
            record = new(kind, actualOwner, registration);
            s_resources.Add(id, record);
        }
        var lease = new OwnedResourceLease(id);
        try { EnsureEffectiveCudaDependency(registration, requestedBackend); }
        catch
        {
            lease.Dispose();
            throw;
        }
        return lease;
    }

    internal sealed class OwnedResourceLease : IDisposable
    {
        private readonly long _id;
        private readonly ResourceRecord _record;
        private bool _released;
        internal OwnedResourceLease(long id) { _id = id; _record = s_resources[id]; }

        internal void RetainCleanupFailure(object actualOwner, Exception error)
        {
            lock (s_gate)
            {
                if (_record.Owner == null || !_record.Owner.TryGetTarget(out var owner) || !ReferenceEquals(owner, actualOwner))
                    throw new InvalidOperationException("Cleanup failure must retain the actual GGML owner.");
                _record.RefusedOwner = actualOwner;
                _record.CleanupFailure ??= error;
            }
        }

        internal bool HasCleanupFailure { get { lock (s_gate) return _record.CleanupFailure != null; } }

        public void Dispose()
        {
            lock (s_gate)
            {
                if (_released || _record.CleanupFailure != null) return;
                _released = true;
                s_resources.Remove(_id);
                ReleaseUnusedProcessClaim();
            }
        }
    }

    internal static NativeCallLease EnterNativeCall(string? kind = null, IntPtr handle = default,
        NativeAdmissionMode mode = NativeAdmissionMode.CurrentBackend, GgmlBackendType? knownBackend = null)
    {
        if (mode == NativeAdmissionMode.BackendEffect && knownBackend == GgmlBackendType.Cuda)
            EstablishCudaDependency();
        NativeCallLease call = ReserveCall(kind, handle);
        try { call.ResumeNative(); }
        catch { call.Dispose(); throw; }
        return call;
    }

    private static NativeCallLease ReserveCall(string? kind = null, IntPtr handle = default)
    {
        lock (s_gate)
        {
            if (!(s_runtimeState == GgmlRuntimeState.Stopping && s_teardownThread == Environment.CurrentManagedThreadId))
                EnsureOperational();
            CheckCleanupAdmission(nativeCall: true);
            ClaimProcess();
            var call = new NativeCallLease(++s_nextLease, kind, handle);
            s_calls.Add(call.Id, call);
            return call;
        }
    }

    internal sealed class NativeCallLease : IDisposable
    {
        private readonly long _id;
        private readonly NativeResource? _resource;
        private readonly int _thread = Environment.CurrentManagedThreadId;
        private CallPhase _phase;
        private NativeEffectLease? _effect;
        private ExceptionDispatchInfo? _callbackFailure;
        private NativeResource? _acquisition;
        internal long Id => _id;
        internal NativeCallLease(long id, string? kind, IntPtr handle)
        {
            _id = id;
            if (kind != null && handle != IntPtr.Zero)
            {
                if (!s_nativeResources.TryGetValue((kind, handle), out _resource) || _resource.Releasing)
                    throw new InvalidOperationException("The native handle is not owned or is being released.");
                if (_resource.CleanupFailure != null) ExceptionDispatchInfo.Capture(_resource.CleanupFailure).Throw();
                _resource.Calls++;
            }
        }

        internal void RetainAcquisition(object acquisition) => _acquisition = (NativeResource)acquisition;

        internal void ResumeNative()
        {
            if (_thread != Environment.CurrentManagedThreadId)
                throw new InvalidOperationException("The GGML native phase must resume on its originating thread.");
            CheckProcessDependency();
            var registration = _resource?.Registration ?? s_processOwnerToken.CudaDependency;
            _effect = registration?.EnterEffect();
            _phase = CallPhase.Native;
        }

        internal bool InvokeBorrowed(Func<bool> callback)
        {
            try
            {
                if (_phase != CallPhase.Native || _thread != Environment.CurrentManagedThreadId)
                    throw new InvalidOperationException("The GGML callback must belong to its active native invocation.");
                _effect?.Dispose();
                _effect = null;
                _phase = CallPhase.BorrowedCallback;
                if (!callback()) { _phase = CallPhase.Unwinding; return false; }
                ResumeNative();
                return true;
            }
            catch (Exception error)
            {
                _callbackFailure ??= ExceptionDispatchInfo.Capture(error);
                _phase = CallPhase.Unwinding;
                return false;
            }
        }

        internal void ThrowCallbackFailure() => _callbackFailure?.Throw();

        public void Dispose()
        {
            if (_phase == CallPhase.Completed) return;
            _effect?.Dispose();
            _effect = null;
            lock (s_gate)
            {
                if (s_calls.Remove(_id) && _resource != null) _resource.Calls--;
                _phase = CallPhase.Completed;
                ReleaseUnusedProcessClaim();
            }
            GC.KeepAlive(_acquisition);
            _acquisition = null;
        }
    }

    internal sealed class NativeResourceReservation : IDisposable
    {
        private readonly NativeResource _resource;
        private readonly NativeCallLease _call;
        internal NativeResourceReservation(string kind)
        {
            bool attach;
            lock (s_gate)
            {
                _resource = new(kind);
                _call = ReserveCall();
                _call.RetainAcquisition(_resource);
                attach = s_processOwnerToken.MayUseCudaPrimaryContexts;
            }
            try
            {
                if (attach)
                {
                    _resource.Registration.AttachCudaDependentGgml();
                    _resource.CudaAttached = true;
                }
                _call.ResumeNative();
            }
            catch { _call.Dispose(); throw; }
        }

        internal IntPtr Track(IntPtr handle)
        {
            _resource.Handle = handle;
            _resource.State = handle == IntPtr.Zero ? AcquisitionState.ReturnedZero : AcquisitionState.Owned;
            if (handle == IntPtr.Zero)
            {
                CheckProcessDependency();
                return handle;
            }
            lock (s_gate)
            {
                if (!s_nativeResources.TryAdd((_resource.Kind, handle), _resource))
                {
                    s_nativeResources[(_resource.Kind, handle)].ConflictingAcquisition = _resource;
                    s_runtimeState = GgmlRuntimeState.Poisoned;
                    throw new InvalidOperationException("Native allocation returned an already-owned handle; a fresh process is required.");
                }
            }
            CheckProcessDependency();
            return handle;
        }
        public void Dispose()
        {
            try
            {
                if (_resource.State is AcquisitionState.Reserved or AcquisitionState.ReturnedZero)
                {
                    // No native payload exists. A terminal scope already retains the empty record.
                    try { _resource.CompleteRelease(); }
                    catch (NativeRuntimeQuarantinedException error) { _resource.CleanupFailure ??= error; }
                }
            }
            finally { _call.Dispose(); }
        }
    }

    internal static NativeResourceReservation ReserveNativeHandle(string kind) => new(kind);

    internal static IntPtr TrackNativeHandle(NativeResourceReservation reservation, IntPtr handle)
        => reservation.Track(handle);

    private sealed class CleanupState(object owner)
    {
        internal readonly object Owner = owner;
        internal readonly int Thread = Environment.CurrentManagedThreadId;
        internal int Depth;
        internal bool Incomplete;
    }
    private static CleanupState? s_cleanup;

    private static void CheckCleanupAdmission(bool nativeCall = false)
    {
        CheckProcessDependency();
        if (s_processOwnerToken.CudaAttachmentInProgress ||
            (s_cleanup != null && (!nativeCall || s_cleanup.Thread != Environment.CurrentManagedThreadId)))
            throw new InvalidOperationException("The GGML runtime is performing exclusive resource cleanup.");
    }

    internal static ResourceCleanupReservation ReserveResourceCleanup(object actualOwner)
    {
        lock (s_gate)
        {
            EnsureOperational();
            CheckProcessDependency();
            if (s_processOwnerToken.CudaAttachmentInProgress || s_calls.Count != 0 ||
                (s_cleanup != null && s_cleanup.Thread != Environment.CurrentManagedThreadId))
                throw new InvalidOperationException("Resource memory cannot be released while native calls or other cleanup are active.");
            s_cleanup ??= new(actualOwner);
            s_cleanup.Depth++;
            return new ResourceCleanupReservation();
        }
    }

    internal sealed class ResourceCleanupReservation : IDisposable
    {
        private readonly CleanupState _state;
        private bool _disposed;
        private bool _completed;
        internal ResourceCleanupReservation() => _state = s_cleanup!;
        internal void Complete() => _completed = true;
        public void Dispose()
        {
            lock (s_gate)
            {
                if (_disposed) return;
                if (_state.Thread != Environment.CurrentManagedThreadId || !ReferenceEquals(s_cleanup, _state))
                    throw new InvalidOperationException("The GGML cleanup reservation must exit on its originating thread.");
                _disposed = true;
                if (!_completed) _state.Incomplete = true;
                if (--_state.Depth == 0) s_cleanup = null;
            }
        }
    }
}
