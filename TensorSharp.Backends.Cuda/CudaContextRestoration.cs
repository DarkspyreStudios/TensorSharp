using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

// Release-only propagation. A prior opaque handle grants no device or lifetime authority.
internal sealed class CudaContextRestoration
{
    private readonly ICudaNativeApi _api;
    private readonly int _threadId;
    private readonly IntPtr _previous;
    private readonly List<(IntPtr Handle, bool Released)> _contexts = new();
    private bool _restored;
    private bool _cleanupFailed;

    private CudaContextRestoration(ICudaNativeApi api, IntPtr previous)
    {
        _api = api;
        _previous = previous;
        _threadId = Environment.CurrentManagedThreadId;
    }

    internal static CudaContextRestoration Capture(CudaNativeCalls ownerCalls)
    {
        ownerCalls.ThrowOnError(ownerCalls.cuCtxGetCurrent(out IntPtr previous));
        return new CudaContextRestoration(ownerCalls.Api, previous);
    }

    internal void Validate(ICudaNativeApi api)
    {
        if (_threadId != Environment.CurrentManagedThreadId || !ReferenceEquals(api, _api) || _restored || _cleanupFailed)
            throw new InvalidOperationException("Context restoration requires its original thread, API and active cleanup.");
    }

    internal int ReserveRelease(IntPtr actualOwnedHandle, ICudaNativeApi api)
    {
        Validate(api);
        int index = _contexts.Count;
        // Reserve bookkeeping before native release so completion does not allocate.
        _contexts.Add((actualOwnedHandle, false));
        return index;
    }

    internal void RecordReleased(int index)
    {
        Validate(_api);
        _contexts[index] = (_contexts[index].Handle, true);
    }

    internal void MarkCleanupFailed() => _cleanupFailed = true;

    internal void Restore()
    {
        Validate(_api);
        _restored = true;
        if (_previous == IntPtr.Zero || _contexts.Exists(c => c.Released && c.Handle == _previous)) return;

        // All outer effects must exit before this unresolved borrowed-context restoration.
        var owner = new object();
        var calls = new CudaNativeCalls(owner, NativeOwnerRole.NativeHandle, _api);
        using var lease = calls.EnterEffect();
        try { calls.ThrowOnError(calls.cuCtxSetCurrent(_previous)); }
        finally { calls.CompleteSafeRelease(lease); }
    }
}
