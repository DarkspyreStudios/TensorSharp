// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Exact BF16 boundaries and Gemma rotary arithmetic without activation readbacks.
/// Older optional LTX modules return false before dispatch and keep the host path.</summary>
public static class CudaLtx25PromptOps
{
    private sealed class Kernels : IDisposable
    {
        private CudaModule _module;
        internal readonly IntPtr Round, Rope;
        internal Kernels(CudaAllocator allocator)
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_ltx25.ptx");
            if (path == null) return;
            allocator.Context.MakeCurrent();
            try
            {
                _module = CudaModule.LoadFromFile(path);
                Round = _module.GetFunction("ts_ltx25_round_bf16");
                Rope = _module.GetFunction("ts_ltx25_gemma_rope_bf16");
            }
            catch { _module?.Dispose(); _module = null; Round = Rope = IntPtr.Zero; }
        }
        public void Dispose() { _module?.Dispose(); _module = null; }
    }

    private static readonly ConditionalWeakTable<CudaAllocator, Kernels> Modules = new();
    private static readonly object ModuleSync = new();
    private static Kernels GetKernels(CudaAllocator allocator)
    {
        // ConditionalWeakTable factories may race; serialize native module
        // creation so every loaded module has one allocator-owned lifetime.
        lock (ModuleSync) return Modules.GetValue(allocator, static value => new Kernels(value));
    }

    internal static void ReleaseForAllocator(CudaAllocator allocator)
    {
        lock (ModuleSync)
            if (Modules.TryGetValue(allocator, out var module))
            { Modules.Remove(allocator); module.Dispose(); }
    }

    public static unsafe bool TryRoundBf16(Tensor tensor)
    {
        if (!CudaKernelOps.TryGetContiguousFloat(tensor, out var storage, out var pointer, out int count) || count < 1)
            return false;
        var allocator = storage.AllocatorImpl;
        allocator.Context.MakeCurrent();
        var kernels = GetKernels(allocator);
        if (kernels.Round == IntPtr.Zero) return false;
        storage.EnsureDeviceCurrent();
        void** args = stackalloc void*[2] { &pointer, &count };
        CudaDriverApi.cuLaunchKernel(kernels.Round, checked((uint)(((long)count + 255) / 256)), 1, 1,
            256, 1, 1, 0, allocator.Stream.Handle, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        storage.MarkDeviceModified();
        return true;
    }

    /// <summary>Mutate [tokens,heads*dim] using host-rounded [tokens,rotated]
    /// trigonometric tables. Unrotated channels and storage outside views remain unchanged.</summary>
    public static unsafe bool TryGemmaRope(Tensor input, Tensor cosine, Tensor sine, int heads, int dim, int rotated)
    {
        if (input == null || input.DimensionCount != 2 || heads <= 0 || dim <= 0 || dim % 2 != 0 ||
            rotated <= 0 || rotated > dim / 2 || input.Sizes[0] is < 1 or > int.MaxValue ||
            input.Sizes[1] != (long)heads * dim ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var x, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(cosine, out var cs, out var c, out int cn) ||
            !CudaKernelOps.TryGetContiguousFloat(sine, out var ss, out var s, out int sn) ||
            cn != input.Sizes[0] * rotated || sn != cn ||
            !ReferenceEquals(xs.AllocatorImpl, cs.AllocatorImpl) || !ReferenceEquals(xs.AllocatorImpl, ss.AllocatorImpl) ||
            ReferenceEquals(xs, cs) || ReferenceEquals(xs, ss)) return false;
        var allocator = xs.AllocatorImpl;
        allocator.Context.MakeCurrent();
        var kernels = GetKernels(allocator);
        if (kernels.Rope == IntPtr.Zero) return false;
        xs.EnsureDeviceCurrent(); cs.EnsureDeviceCurrent(); ss.EnsureDeviceCurrent();
        int tokens = (int)input.Sizes[0];
        long pairs = (long)tokens * heads * rotated;
        void** args = stackalloc void*[7] { &x, &c, &s, &tokens, &heads, &dim, &rotated };
        CudaDriverApi.cuLaunchKernel(kernels.Rope, checked((uint)((pairs + 255) / 256)), 1, 1,
            256, 1, 1, 0, allocator.Stream.Handle, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        xs.MarkDeviceModified();
        return true;
    }
}
