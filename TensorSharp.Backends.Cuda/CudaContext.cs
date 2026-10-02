using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaContext : IDisposable
    {
        private IntPtr context;
        private int device;
        private readonly CudaNativeCalls nativeCalls;

        private CudaContext(int deviceId, ICudaNativeApi api)
        {
            DeviceId = deviceId;
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle, api, deviceId);
        }

        public int DeviceId { get; }

        public IntPtr Handle => context;

        public static CudaContext Create(int deviceId)
        {
            CudaLibraryResolver.Register();
            return Create(deviceId, CudaNativeApi.Instance);
        }

        internal static CudaContext Create(int deviceId, ICudaNativeApi api)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
            ArgumentNullException.ThrowIfNull(api);
            InitializeDriver(api);

            var owner = new CudaContext(deviceId, api);
            using var lease = owner.nativeCalls.EnterEffect();
            try
            {
                owner.nativeCalls.ThrowOnError(owner.nativeCalls.cuDeviceGet(out owner.device, deviceId));
                // Reserve the owner before the driver can return a primary-context reference.
                int result = owner.nativeCalls.cuDevicePrimaryCtxRetain(out owner.context, owner.device);
                owner.nativeCalls.ThrowOnError(result);
                owner.nativeCalls.ThrowOnError(owner.nativeCalls.cuCtxSetCurrent(owner.context));
                return owner;
            }
            catch (Exception original)
            {
                try { owner.Release(lease, drain: false); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        private static void InitializeDriver(ICudaNativeApi api)
        {
            var discovery = new object();
            var calls = new CudaNativeCalls(discovery, NativeOwnerRole.NativeHandle, api);
            using var lease = calls.EnterEffect();
            try { calls.ThrowOnError(calls.cuInit(0)); }
            finally { calls.CompleteSafeRelease(lease); }
        }

        /// <summary>
        /// True once this owner's primary-context reference has been released.
        /// Other owners can still retain the same device primary context.
        /// </summary>
        public bool IsDisposed => Volatile.Read(ref context) == IntPtr.Zero;

        internal CudaNativeCalls NativeCalls => nativeCalls;

        public void MakeCurrent()
        {
            if (IsDisposed) throw new ObjectDisposedException(nameof(CudaContext));
            using var lease = nativeCalls.EnterEffect();
            BindCurrent(nativeCalls);
        }

        internal void BindCurrent(CudaNativeCalls calls)
        {
            if (!calls.CoversDevice(DeviceId))
                throw new InvalidOperationException("The CUDA effect does not cover the context device.");
            nativeCalls.ThrowIfQuarantined();
            if (context == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaContext));
            calls.ThrowOnError(calls.cuCtxSetCurrent(context));
        }

        public void Dispose()
        {
            if (IsDisposed) return;
            IntPtr previous;
            using (var lease = nativeCalls.EnterEffect())
            {
                if (IsDisposed) return;
                previous = Release(lease, drain: true);
            }
            if (previous != IntPtr.Zero) RestoreBorrowedContext(previous, nativeCalls.Api);
        }

        private IntPtr Release(NativeEffectLease lease, bool drain)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                IntPtr previous = IntPtr.Zero;
                if (context != IntPtr.Zero)
                {
                    nativeCalls.ThrowOnError(nativeCalls.cuCtxGetCurrent(out IntPtr current));
                    if (drain)
                    {
                        previous = current == context ? IntPtr.Zero : current;
                        BindCurrent(nativeCalls);
                        nativeCalls.cuCtxSynchronize();
                        current = context;
                    }
                    if (current == context)
                        nativeCalls.ThrowOnError(nativeCalls.cuCtxSetCurrent(IntPtr.Zero));

                    nativeCalls.cuDevicePrimaryCtxRelease(device);
                    Volatile.Write(ref context, IntPtr.Zero);
                }
                nativeCalls.CompleteSafeRelease(lease);
                return previous;
            }
            catch (Exception cleanup)
            {
                nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.ContextRelease);
                throw;
            }
        }

        private static void RestoreBorrowedContext(IntPtr previous, ICudaNativeApi api)
        {
            // A borrowed handle alone is not ordinal proof. No known-device gate is held here.
            var restoration = new object();
            var calls = new CudaNativeCalls(restoration, NativeOwnerRole.NativeHandle, api);
            using var lease = calls.EnterEffect();
            try { calls.ThrowOnError(calls.cuCtxSetCurrent(previous)); }
            finally { calls.CompleteSafeRelease(lease); }
        }
    }
}
