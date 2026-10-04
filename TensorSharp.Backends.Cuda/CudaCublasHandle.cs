using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaCublasHandle : IDisposable
    {
        private IntPtr handle;
        private readonly CudaContext context;
        private readonly CudaNativeCalls nativeCalls;

        private CudaCublasHandle()
        {
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle,
                CudaNativeApi.Instance);
        }

        private CudaCublasHandle(CudaContext context)
        {
            this.context = context;
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle,
                context.NativeCalls.Api, context.DeviceId);
        }

        public IntPtr Handle => handle;

        public static CudaCublasHandle Create() => CreateOwned(new CudaCublasHandle());

        internal static CudaCublasHandle Create(CudaContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            return CreateOwned(new CudaCublasHandle(context));
        }

        private static CudaCublasHandle CreateOwned(CudaCublasHandle owner)
        {
            using var lease = owner.nativeCalls.EnterEffect();
            try
            {
                owner.context?.BindCurrent(owner.nativeCalls);
                owner.nativeCalls.cublasCreate(out owner.handle).ThrowOnCublasError();
                owner.nativeCalls.cublasSetMathMode(owner.handle, CublasApi.CUBLAS_TENSOR_OP_MATH).ThrowOnCublasError();
                return owner;
            }
            catch (Exception original)
            {
                try { owner.Release(lease, drain: false); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        internal CudaNativeCalls NativeCalls => nativeCalls;

        public void SetStream(IntPtr stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            using var lease = nativeCalls.EnterEffect();
            context?.BindCurrent(nativeCalls);
            nativeCalls.cublasSetStream(handle, stream).ThrowOnCublasError();
        }

        public void Dispose()
        {
            if (handle == IntPtr.Zero) return;
            using var lease = nativeCalls.EnterEffect();
            if (handle == IntPtr.Zero) return;
            Release(lease, drain: true);
        }

        private void Release(NativeEffectLease lease, bool drain)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                if (handle != IntPtr.Zero)
                {
                    context?.BindCurrent(nativeCalls);
                    if (drain && context != null) nativeCalls.cuCtxSynchronize();
                    nativeCalls.cublasDestroy(handle);
                    Volatile.Write(ref handle, IntPtr.Zero);
                }
                nativeCalls.CompleteSafeRelease(lease);
            }
            catch (Exception cleanup)
            {
                nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.ContextRelease);
                throw;
            }
        }
    }
}
