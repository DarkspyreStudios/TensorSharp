using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaModule : IDisposable
    {
        private readonly Dictionary<string, IntPtr> functions = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        private IntPtr module;
        private readonly CudaContextBinding context;
        private readonly CudaNativeCalls nativeCalls;

        internal CudaContextBinding Context => context;

        private CudaModule(CudaContextBinding context)
        {
            this.context = context;
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle, context.Api, context.DeviceId);
        }

        public static CudaModule LoadFromFile(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes);
        }

        public static CudaModule LoadFromBytes(byte[] ptxBytes)
        {
            ArgumentNullException.ThrowIfNull(ptxBytes);
            return LoadFromBytes(ptxBytes, CudaContextBinding.Discover(CudaNativeApi.Instance));
        }

        internal static CudaModule LoadFromBytes(byte[] ptxBytes, CudaContext context)
            => LoadFromBytes(ptxBytes, CudaContextBinding.FromOwner(context));

        private static unsafe CudaModule LoadFromBytes(byte[] ptxBytes, CudaContextBinding context)
        {
            if (ptxBytes == null)
                throw new ArgumentNullException(nameof(ptxBytes));

            byte[] terminated = ptxBytes;
            if (terminated.Length == 0 || terminated[terminated.Length - 1] != 0)
            {
                terminated = new byte[ptxBytes.Length + 1];
                Buffer.BlockCopy(ptxBytes, 0, terminated, 0, ptxBytes.Length);
            }

            var owner = new CudaModule(context);
            using var lease = owner.nativeCalls.EnterEffect();
            try
            {
                context.BindCurrent(owner.nativeCalls);
                fixed (byte* ptx = terminated)
                    owner.nativeCalls.ThrowOnError(owner.nativeCalls.cuModuleLoadData(out owner.module, (IntPtr)ptx));
                return owner;
            }
            catch (Exception original)
            {
                try { owner.Release(lease, drain: false); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        public IntPtr GetFunction(string name)
        {
            ArgumentNullException.ThrowIfNull(name);
            using var lease = nativeCalls.EnterEffect();
            context.BindCurrent(nativeCalls);
            if (!functions.TryGetValue(name, out IntPtr function))
            {
                nativeCalls.ThrowOnError(nativeCalls.cuModuleGetFunction(out function, module, name));
                functions.Add(name, function);
            }

            return function;
        }

        public void Dispose()
        {
            if (module == IntPtr.Zero) return;
            using var lease = nativeCalls.EnterEffect();
            if (module == IntPtr.Zero) return;
            Release(lease, drain: true);
        }

        private void Release(NativeEffectLease lease, bool drain)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                if (module != IntPtr.Zero)
                {
                    context.BindCurrent(nativeCalls);
                    if (drain) nativeCalls.cuCtxSynchronize();
                    nativeCalls.cuModuleUnload(module);
                    module = IntPtr.Zero;
                }
                functions.Clear();
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
