using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaStream : IDisposable
    {
        private IntPtr stream;
        private readonly CudaContextBinding context;
        private readonly CudaNativeCalls nativeCalls;

        private CudaStream(CudaContextBinding context)
        {
            this.context = context;
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle, context.Api, context.DeviceId);
        }

        public IntPtr Handle => stream;
        internal bool IsFromOwner(CudaContext owner) => context.IsFromOwner(owner);

        public static CudaStream Create()
        {
            return Create(CudaNativeApi.Instance);
        }

        internal static CudaStream Create(ICudaNativeApi api) => Create(CudaContextBinding.Discover(api));
        internal static CudaStream Create(CudaContext context) => Create(CudaContextBinding.FromOwner(context));

        private static CudaStream Create(CudaContextBinding context)
        {
            var owner = new CudaStream(context);
            using var lease = owner.nativeCalls.EnterEffect();
            try
            {
                context.BindCurrent(owner.nativeCalls);
                owner.nativeCalls.ThrowOnError(owner.nativeCalls.cuStreamCreate(out owner.stream, 0));
                return owner;
            }
            catch (Exception original)
            {
                try { owner.Release(lease); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        public void Synchronize()
        {
            if (stream == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStream));

            nativeCalls.ThrowIfQuarantined();

            // A synchronize on a stream that is capturing a CUDA graph means some
            // op needs device data on the host mid-capture; that cannot be part of
            // a graph. Abort the capture (the site catches this, re-runs plainly).
            CudaGraphCapture.OnStreamSynchronize(stream);

            using var lease = nativeCalls.EnterEffect();
            context.BindCurrent(nativeCalls);
            nativeCalls.cuStreamSynchronize(stream);
        }

        public void Dispose()
        {
            if (Volatile.Read(ref stream) == IntPtr.Zero) return;
            using var lease = nativeCalls.EnterEffect();
            if (stream == IntPtr.Zero) return;
            Release(lease);
        }

        private void Release(NativeEffectLease lease)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                if (stream != IntPtr.Zero)
                {
                    context.BindCurrent(nativeCalls);
                    nativeCalls.cuStreamSynchronize(stream);
                    nativeCalls.cuStreamDestroy(stream);
                    Volatile.Write(ref stream, IntPtr.Zero);
                }
                nativeCalls.CompleteSafeRelease(lease);
            }
            catch (Exception cleanup)
            {
                nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.WorkerRetirement);
                throw;
            }
        }
    }
}
