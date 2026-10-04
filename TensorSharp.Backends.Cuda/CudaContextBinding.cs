using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

// Borrowed context association. It does not retain or release a primary context.
internal readonly struct CudaContextBinding
{
    private readonly CudaContext _owner;
    private readonly IntPtr _borrowedHandle;
    internal int DeviceId { get; }
    internal ICudaNativeApi Api { get; }

    private CudaContextBinding(CudaContext owner, IntPtr borrowedHandle, int deviceId, ICudaNativeApi api)
    {
        _owner = owner;
        _borrowedHandle = borrowedHandle;
        DeviceId = deviceId;
        Api = api;
    }

    internal static CudaContextBinding FromOwner(CudaContext owner)
    {
        ArgumentNullException.ThrowIfNull(owner);
        return new(owner, IntPtr.Zero, owner.DeviceId, owner.NativeCalls.Api);
    }

    internal bool MatchesOwner(CudaContext owner) => ReferenceEquals(_owner, owner)
        || (_owner == null && DeviceId == owner.DeviceId && ReferenceEquals(Api, owner.NativeCalls.Api)
            && _borrowedHandle != IntPtr.Zero && _borrowedHandle == owner.Handle);

    internal static CudaContextBinding Discover(ICudaNativeApi api)
    {
        var discovery = new object();
        var calls = new CudaNativeCalls(discovery, NativeOwnerRole.NativeHandle, api);
        using var lease = calls.EnterEffect();
        try
        {
            calls.ThrowOnError(calls.cuCtxGetCurrent(out IntPtr context));
            calls.ThrowOnError(calls.cuCtxGetDevice(out int device));
            calls.ThrowOnError(calls.cuDeviceGetCount(out int count));
            for (int ordinal = 0; ordinal < count; ordinal++)
            {
                calls.ThrowOnError(calls.cuDeviceGet(out int candidate, ordinal));
                if (candidate == device) return new(null, context, ordinal, api);
            }
            throw new InvalidOperationException("The current CUDA context device has no driver ordinal.");
        }
        finally { calls.CompleteSafeRelease(lease); }
    }

    internal void BindCurrent(CudaNativeCalls calls)
    {
        if (!calls.CoversDevice(DeviceId))
            throw new InvalidOperationException("The CUDA effect does not cover the context device.");
        if (_owner != null) _owner.BindCurrent(calls);
        else calls.ThrowOnError(calls.cuCtxSetCurrent(_borrowedHandle));
    }
}
