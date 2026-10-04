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

        private CudaCublasHandle(IntPtr handle)
        {
            this.handle = handle;
        }

        private CudaCublasHandle(CudaContext context)
        {
            this.context = context;
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle,
                context.NativeCalls.Api, context.DeviceId);
        }

        public IntPtr Handle => handle;

        public static CudaCublasHandle Create()
        {
            CublasApi.cublasCreate(out IntPtr handle).ThrowOnCublasError();
            try
            {
                CublasApi.cublasSetMathMode(handle, CublasApi.CUBLAS_TENSOR_OP_MATH).ThrowOnCublasError();
                return new CudaCublasHandle(handle);
            }
            catch
            {
                CublasApi.cublasDestroy(handle);
                throw;
            }
        }

        internal static CudaCublasHandle Create(CudaContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            var owner = new CudaCublasHandle(context);
            using var lease = owner.nativeCalls.EnterEffect();
            try
            {
                context.BindCurrent(owner.nativeCalls);
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

        internal CudaNativeCalls NativeCalls => nativeCalls
            ?? throw new InvalidOperationException("Standalone cuBLAS scope is not qualified for owner-scoped operations.");

        public void SetStream(IntPtr stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            if (nativeCalls == null)
            {
                CublasApi.cublasSetStream(handle, stream).ThrowOnCublasError();
                return;
            }

            using var lease = nativeCalls.EnterEffect();
            context.BindCurrent(nativeCalls);
            nativeCalls.cublasSetStream(handle, stream).ThrowOnCublasError();
        }

        public void Dispose()
        {
            if (nativeCalls != null)
            {
                if (handle == IntPtr.Zero) return;
                using var lease = nativeCalls.EnterEffect();
                if (handle == IntPtr.Zero) return;
                Release(lease, drain: true);
                return;
            }

            IntPtr nativeHandle = Interlocked.Exchange(ref handle, IntPtr.Zero);
            if (nativeHandle != IntPtr.Zero)
                CublasApi.cublasDestroy(nativeHandle);
        }

        private void Release(NativeEffectLease lease, bool drain)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                if (handle != IntPtr.Zero)
                {
                    context.BindCurrent(nativeCalls);
                    if (drain) nativeCalls.cuCtxSynchronize();
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
