using System.Runtime.CompilerServices;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Runtime.ExceptionServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.CudaQuarantineFixture;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length != 1) throw new ArgumentException("One controlled fixture mode is required.");
        switch (args[0])
        {
            case "context-clean": ContextClean(); break;
            case "context-composed-clean": ContextComposedClean(false); break;
            case "context-composed-external": ContextComposedClean(true); break;
            case "context-composed-drain-refusal": ContextComposedRefusal(); break;
            case "context-composed-restore-refusal": ReleaseContextPair(true, true); break;
            case "context-composed-thread-refusal": ContextComposedThreadRefusal(); break;
            case "context-independent-postfault-release": ContextIndependentPostfaultRelease(); break;
            case "context-release-refusal": ContextReleaseRefusal(); break;
            case "context-drain-refusal": ContextDrainRefusal(); break;
            case "context-construction-rollback": ConstructionRollback(false); break;
            case "context-construction-rollback-refusal": ConstructionRollback(true); break;
            case "stream-clean": StreamClean(false); break;
            case "stream-ambient-clean": StreamClean(true); break;
            case "stream-sync-refusal": StreamRefusal(false); break;
            case "stream-destroy-refusal": StreamRefusal(true); break;
            case "stream-construction-rollback": StreamConstructionRollback(false); break;
            case "stream-construction-rollback-refusal": StreamConstructionRollback(true); break;
            case "module-clean": ModuleClean(); break;
            case "module-drain-refusal": ModuleRefusal(false); break;
            case "module-unload-refusal": ModuleRefusal(true); break;
            case "module-construction-rollback": ModuleConstructionRollback(false); break;
            case "module-construction-rollback-refusal": ModuleConstructionRollback(true); break;
            case "blas-clean": BlasClean(); break;
            case "blas-drain-refusal": BlasRefusal(false); break;
            case "blas-destroy-refusal": BlasRefusal(true); break;
            case "blas-construction-rollback": BlasConstructionRollback(false); break;
            case "blas-construction-rollback-refusal": BlasConstructionRollback(true); break;
            case "kernels-construction-rollback": KernelsConstructionRollback(false); break;
            case "kernels-construction-rollback-refusal": KernelsConstructionRollback(true); break;
            case "kernels-clean": KernelsClean(); break;
            case "kernels-drain-refusal": KernelsRefusal(false); break;
            case "kernels-free-refusal": KernelsRefusal(true); break;
            case "kernels-resize-refusal": KernelsResizeRefusal(); break;
            case "kernels-allocation-rollback": KernelsAllocationRollback(false); break;
            case "kernels-allocation-rollback-refusal": KernelsAllocationRollback(true); break;
            case "kernels-ordinary-faults": KernelsOrdinaryFaults(); break;
            case "pool-small-failed-free-retention": PoolFailedFreeRetention(false); break;
            case "pool-large-failed-free-retention": PoolFailedFreeRetention(true); break;
            case "foreign-context-clean": ForeignContext(false); break;
            case "foreign-context-release-refusal": ForeignContext(true); break;
            case "foreign-composed-clean": ForeignComposed(false); break;
            case "foreign-composed-drain-refusal": ForeignComposed(true); break;
            case "foreign-stream-clean": ForeignStream(false); break;
            case "foreign-stream-sync-refusal": ForeignStream(true); break;
            case "foreign-module-clean": ForeignModule(false); break;
            case "foreign-module-drain-refusal": ForeignModule(true); break;
            case "foreign-blas-clean": ForeignBlas(false); break;
            case "foreign-blas-drain-refusal": ForeignBlas(true); break;
            case "foreign-kernels-clean": ForeignKernels(false); break;
            case "foreign-kernels-drain-refusal": ForeignKernels(true); break;
            default: throw new ArgumentException("Unknown controlled fixture mode.");
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            mode = args[0],
            passed = true,
            nativeExecution = false,
            fixtureMvid = typeof(Program).Assembly.ManifestModule.ModuleVersionId,
            cudaMvid = typeof(CudaContext).Assembly.ManifestModule.ModuleVersionId,
            coreMvid = typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId,
            observation = "actual-managed-class/instance-injected-native-calls-only"
        }));
        return 0;
    }

    private static void Assert(bool value, string message)
    {
        if (!value) throw new InvalidOperationException(message);
    }

    private static void KernelsConstructionRollback(bool refuse)
    {
        var original = new InvalidOperationException("Controlled required kernel lookup failure.");
        var cleanup = new InvalidOperationException("Controlled kernel module cleanup refusal.");
        var api = new RecordingCudaApi { FunctionFailure = original };
        var context = CudaContext.Create(0, api);
        var module = CudaModule.LoadFromBytes(new byte[] { 1 }, context);
        if (refuse) api.ModuleUnloadFailure = cleanup;
        try { CudaKernels.CreateOwned(module); throw new InvalidOperationException("Construction unexpectedly succeeded."); }
        catch (AggregateException error) when (refuse)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "Kernel construction loses original or cleanup failure.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Kernel construction cleanup failure is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuse && ReferenceEquals(error, original)) { }
        Assert(api.ModuleUnloadCount == 1, "Failed kernel construction loses its transferred loaded module.");
        if (!refuse) context.Dispose();
    }

    private static void PoolFailedFreeRetention(bool large)
    {
        long size = large ? 2L << 20 : 256;
        int allocations = 0;
        var attempts = new List<IntPtr>();
        var original = new InvalidOperationException("Controlled pool backing cleanup refusal.");
        bool refuse = true;
        var pool = new CudaDeviceMemoryPool(size * 4, size * 4, true,
            _ => new IntPtr(6000 + ++allocations), ptr =>
            {
                attempts.Add(ptr);
                if (refuse) throw original;
            }, shardCount: 1);
        IntPtr first = pool.Rent(size, out long firstBytes);
        IntPtr second = pool.Rent(size, out long secondBytes);
        pool.Return(first, firstBytes);
        pool.Return(second, secondBytes);
        try { pool.DrainAndFree(); throw new Exception("Refused pool drain succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(pool.GetStats().CachedBytes == firstBytes + secondBytes, "Refused first free changes cached ownership accounting.");
        refuse = false;
        pool.DrainAndFree();
        Assert(attempts.Count == 3 && attempts[0] == second && attempts[1] == second && attempts[2] == first,
            "Pool loses the actual refused block before successful callback completion.");
        Assert(pool.GetStats().CachedBytes == 0, "Successful pool drain leaves cached byte ownership.");
    }

    private static object? InvokeKernelBoundary(CudaKernels kernels, string method, params object[] arguments)
    {
        try { return typeof(CudaKernels).GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(kernels, arguments); }
        catch (TargetInvocationException error) when (error.InnerException != null)
        { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
    }

    private static IntPtr Scratch(CudaKernels kernels, long bytes)
        => (IntPtr)InvokeKernelBoundary(kernels, "EnsureGdnSplitScratch", bytes)!;

    private static void KernelsClean()
    {
        var roots = CreateAndReleaseKernels();
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy actual kernels/module/context/API roots remain retained.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateAndReleaseKernels()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var module = CudaModule.LoadFromBytes(new byte[] { 1 }, context);
        var kernels = CudaKernels.CreateOwned(module);
        api.Current = IntPtr.Zero;
        kernels.LaunchFillF32(IntPtr.Zero, 1, 0, IntPtr.Zero);
        Assert(api.KernelLaunchCount == 1 && api.Current == context.Handle, "Actual kernel launch does not rebind its verified context.");
        IntPtr first = Scratch(kernels, 16);
        Assert(Scratch(kernels, 8) == first && api.MemoryAllocateCount == 1, "Scratch cache reallocates an adequate owned block.");
        IntPtr second = Scratch(kernels, 32);
        Assert(second != first && api.MemoryFreeCount == 1 && api.ContextSyncCount == 1,
            "Scratch resize does not drain before freeing its owned block.");
        InvokeKernelBoundary(kernels, "EnsureGdnPackedSharedCapacity", 65536u);
        InvokeKernelBoundary(kernels, "EnsureFlash2SharedCapacity", 256, 65536u);
        InvokeKernelBoundary(kernels, "EnsureFlash2SharedCapacity", 512, 65536u);
        Assert(api.AttributeCount == 5, "Actual shared-memory configuration does not use the injected owner path.");
        kernels.Dispose();
        kernels.Dispose();
        Assert(api.MemoryFreeCount == 2 && api.ModuleUnloadCount == 1 && api.ContextSyncCount == 3,
            "Kernel cleanup does not drain and release scratch before one checked module unload.");
        int effects = api.TotalCalls;
        try { kernels.LaunchFillF32(IntPtr.Zero, 1, 0, IntPtr.Zero); throw new Exception("Disposed kernel owner entered."); }
        catch (InvalidOperationException) { }
        Assert(api.TotalCalls == effects, "Retired kernel owner executes a native call.");
        context.Dispose();
        return [new(kernels), new(module), new(context), new(api)];
    }

    private static void KernelsRefusal(bool free)
    {
        var roots = FailKernelsRelease(free);
        Collect();
        Assert(roots.All(r => r.IsAlive), "Unsafe actual kernel scratch/module/context/API graph is lost.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailKernelsRelease(bool free)
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var module = CudaModule.LoadFromBytes(new byte[] { 1 }, context);
        var kernels = CudaKernels.CreateOwned(module);
        Scratch(kernels, 16);
        var original = new InvalidOperationException("Controlled kernel cleanup refusal.");
        if (free) api.MemoryFreeFailure = original;
        else api.DrainFailure = original;
        try { kernels.Dispose(); throw new Exception("Unsafe kernel cleanup succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(api.MemoryFreeCount == (free ? 1 : 0) && api.ModuleUnloadCount == 0 && api.ReleaseCount == 0,
            "Refused kernel cleanup releases dependent owned resources.");
        Assert(NativeRuntimeQuarantine.TryGetFailure(original, out _), "Kernel cleanup loses its recorded actual cause.");
        int effects = api.TotalCalls;
        try { kernels.Dispose(); throw new Exception("Quarantined kernel teardown repeated."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { Scratch(kernels, 8); throw new Exception("Quarantined scratch cache published."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { kernels.LaunchFillF32(IntPtr.Zero, 1, 0, IntPtr.Zero); throw new Exception("Quarantined kernel launched."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == effects, "Quarantined kernels execute later effects.");
        return [new(kernels), new(module), new(context), new(api)];
    }

    private static void KernelsResizeRefusal()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var module = CudaModule.LoadFromBytes(new byte[] { 1 }, context);
        var kernels = CudaKernels.CreateOwned(module);
        IntPtr old = Scratch(kernels, 16);
        var original = new InvalidOperationException("Controlled scratch resize drain refusal.");
        api.DrainFailure = original;
        try { Scratch(kernels, 32); throw new Exception("Refused scratch resize succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        var actual = (IntPtr)typeof(CudaKernels).GetField("gdnSplitScratch", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(kernels)!;
        Assert(actual == old && api.MemoryAllocateCount == 1 && api.MemoryFreeCount == 0,
            "Scratch resize refusal loses old backing ownership or reaches allocate/free.");
    }

    private static void KernelsAllocationRollback(bool refuse)
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var module = CudaModule.LoadFromBytes(new byte[] { 1 }, context);
        var kernels = CudaKernels.CreateOwned(module);
        var original = new InvalidOperationException("Controlled scratch allocation failure after OUT ownership.");
        var cleanup = new InvalidOperationException("Controlled scratch rollback refusal.");
        api.MemoryAllocateFailure = original;
        if (refuse) api.MemoryFreeFailure = cleanup;
        try { Scratch(kernels, 16); throw new Exception("Failed scratch allocation succeeded."); }
        catch (AggregateException error) when (refuse)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "Scratch rollback loses original or cleanup failure.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Unsafe scratch rollback is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuse && ReferenceEquals(error, original)) { }
        Assert(api.MemoryFreeCount == 1 && api.ModuleUnloadCount == 0, "Scratch allocation rollback does not release exactly its local allocation.");
        if (!refuse)
        {
            api.MemoryAllocateFailure = null;
            Assert(NativeRuntimeQuarantine.Observe().Failures.Count == 0, "Ordinary cleaned scratch allocation failure quarantines runtime.");
            Scratch(kernels, 16);
            kernels.Dispose();
            context.Dispose();
        }
    }

    private static void KernelsOrdinaryFaults()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var kernels = CudaKernels.CreateOwned(CudaModule.LoadFromBytes(new byte[] { 1 }, context));
        var original = new InvalidOperationException("Controlled ordinary launch/attribute error.");
        api.KernelLaunchFailure = original;
        try { kernels.LaunchFillF32(IntPtr.Zero, 1, 0, IntPtr.Zero); throw new Exception("Launch failure succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        api.KernelLaunchFailure = null;
        api.AttributeFailure = original;
        try { InvokeKernelBoundary(kernels, "EnsureGdnPackedSharedCapacity", 65536u); throw new Exception("Attribute failure succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        api.AttributeFailure = null;
        InvokeKernelBoundary(kernels, "EnsureGdnPackedSharedCapacity", 65536u);
        Assert(api.AttributeCount == 2 && NativeRuntimeQuarantine.Observe().Failures.Count == 0,
            "Ordinary effect failure publishes unsafe cleanup or advances attribute cache before success.");
        kernels.Dispose();
        context.Dispose();
    }

    private static void ContextClean()
    {
        WeakReference[] roots = CreateAndReleaseContext();
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy context/API roots survive explicit disposal.");
        Assert(NativeRuntimeQuarantine.Observe().Failures.Count == 0, "Healthy cleanup records quarantine.");
    }

    private static void ContextComposedClean(bool external)
    {
        WeakReference[] roots = ReleaseContextPair(external, false);
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy actual composed context/token/parent/API roots remain retained.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] ReleaseContextPair(bool external, bool restoreFailure)
    {
        var api = new RecordingCudaApi();
        var first = CudaContext.Create(0, api);
        var second = CudaContext.Create(1, api);
        first.MakeCurrent();
        if (external) api.Current = new IntPtr(7777);
        object[] actualOwner = { first, second };
        var calls = new CudaNativeCalls(actualOwner, NativeOwnerRole.Allocator, api, 0, 1);
        var restoration = CudaContextRestoration.Capture(calls);
        var original = new InvalidOperationException("Controlled post-release borrowed restoration refusal.");
        if (restoreFailure) { api.RefusedBind = new IntPtr(7777); api.RestoreFailure = original; }
        using (var lease = calls.EnterEffect())
        {
            calls.ValidateSafeRelease(lease);
            second.DisposeOwned(restoration);
            first.DisposeOwned(restoration);
            calls.CompleteSafeRelease(lease);
        }
        try { restoration.Restore(); Assert(!restoreFailure, "Borrowed restoration unexpectedly succeeded."); }
        catch (InvalidOperationException error) when (restoreFailure && ReferenceEquals(error, original)) { }
        Assert(first.IsDisposed && second.IsDisposed && api.ReleaseCount == 2,
            "Composed cleanup does not release both actual contexts.");
        Assert(api.Current == (external && !restoreFailure ? new IntPtr(7777) : IntPtr.Zero),
            "Composed cleanup restores an already released owned context or loses a borrowed context.");
        Assert(api.BindThreads.All(id => id == Environment.CurrentManagedThreadId),
            "Context restoration crosses executing threads.");
        Assert(api.BindContexts.Count(h => h == new IntPtr(7777)) == (external ? 1 : 0),
            "Composed cleanup does not attempt borrowed restoration exactly once.");
        Assert(NativeRuntimeQuarantine.Observe().Failures.Count == 0,
            "Post-release restoration failure publishes through a retired owner.");
        return [new(first), new(second), new(actualOwner), new(restoration), new(api)];
    }

    private static void ContextComposedRefusal()
    {
        WeakReference[] roots = FailContextPair();
        Collect();
        Assert(roots.All(r => r.IsAlive), "Failed composed parent loses actual context/API owners.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailContextPair()
    {
        var original = new InvalidOperationException("Controlled composed child drain refusal.");
        var api = new RecordingCudaApi();
        var first = CudaContext.Create(0, api);
        var second = CudaContext.Create(1, api);
        first.MakeCurrent();
        object[] actualOwner = { first, second };
        var calls = new CudaNativeCalls(actualOwner, NativeOwnerRole.Allocator, api, 0, 1);
        var restoration = CudaContextRestoration.Capture(calls);
        api.DrainFailure = original;
        try
        {
            using var lease = calls.EnterEffect();
            calls.ValidateSafeRelease(lease);
            second.DisposeOwned(restoration);
            first.DisposeOwned(restoration);
            calls.CompleteSafeRelease(lease);
            restoration.Restore();
            throw new InvalidOperationException("Unsafe composed cleanup succeeded.");
        }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(api.ReleaseCount == 0 && !first.IsDisposed && !second.IsDisposed && api.Current == second.Handle,
            "Child drain refusal continues dependent cleanup or restores the prior context.");
        int effects = api.TotalCalls;
        try { restoration.Restore(); throw new Exception("Unsafe cleanup token permits restoration."); }
        catch (InvalidOperationException) { }
        try { calls.ThrowIfQuarantined(); throw new InvalidOperationException("Actual parent is not fenced."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == effects && NativeRuntimeQuarantine.TryGetFailure(original, out _),
            "Composed refusal loses its cause or enters native effects.");
        return [new(first), new(second), new(actualOwner), new(api)];
    }

    private static void ContextComposedThreadRefusal()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var restoration = CudaContextRestoration.Capture(context.NativeCalls);
        int effects = api.TotalCalls;
        Task attempt = Task.Run(() =>
        {
            try { context.DisposeOwned(restoration); throw new Exception("Cross-thread token was accepted."); }
            catch (InvalidOperationException) { }
        });
        Assert(attempt.Wait(TimeSpan.FromSeconds(3)), "Cross-thread token refusal did not settle.");
        Assert(api.TotalCalls == effects && !context.IsDisposed,
            "Cross-thread token enters cleanup before refusing.");
        context.DisposeOwned(restoration);
        restoration.Restore();
        try { restoration.Restore(); throw new Exception("Restoration token was reused."); }
        catch (InvalidOperationException) { }
    }

    private static void ContextIndependentPostfaultRelease()
    {
        WeakReference independent = ReleaseIndependentAfterFailure();
        Collect();
        Assert(!independent.IsAlive, "Released independent context remains retained by another device failure.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference ReleaseIndependentAfterFailure()
    {
        var api = new RecordingCudaApi();
        var failed = CudaContext.Create(0, api);
        var independent = CudaContext.Create(1, api);
        var original = new InvalidOperationException("Controlled other-device drain refusal.");
        api.DrainFailure = original;
        try { failed.Dispose(); throw new Exception("Drain refusal succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        api.DrainFailure = null;
        Assert(api.Current == failed.Handle, "The prior context is not the failed external context.");
        try { independent.Dispose(); throw new Exception("Quarantined wildcard restoration was admitted."); }
        catch (NativeRuntimeQuarantinedException error)
        {
            Assert(error.Failure.DeviceOrdinal == 0 && ReferenceEquals(error.InnerException, original),
                "Post-success restoration refuses with an unrelated cause.");
        }
        Assert(independent.IsDisposed && !failed.IsDisposed && api.ReleaseCount == 1,
            "Other-device failure prevents known independent release or changes failed ownership.");
        Assert(NativeRuntimeQuarantine.Observe().Failures.All(f => f.DeviceOrdinal == 0),
            "Post-success restoration poisons the safely released independent device.");
        int effects = api.TotalCalls;
        independent.Dispose();
        Assert(api.TotalCalls == effects, "Already released independent context retries teardown.");
        return new WeakReference(independent);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateAndReleaseContext()
    {
        var api = new RecordingCudaApi();
        CudaContext context = CudaContext.Create(2, api);
        Assert(context.DeviceId == 2 && context.Handle != IntPtr.Zero, "Actual context was not acquired.");
        api.Current = IntPtr.Zero;
        context.MakeCurrent();
        Assert(api.Current == context.Handle, "MakeCurrent did not rebind the actual context.");
        api.Current = new IntPtr(7777);
        context.Dispose();
        context.Dispose();
        Assert(context.IsDisposed && context.Handle == IntPtr.Zero && api.ReleaseCount == 1,
            "Healthy context release is not idempotent.");
        Assert(api.Current == new IntPtr(7777), "Context cleanup does not restore the unrelated borrowed context.");
        try { context.MakeCurrent(); throw new InvalidOperationException("Disposed context entered."); }
        catch (ObjectDisposedException) { }
        return [new WeakReference(context), new WeakReference(api)];
    }

    private static void ContextReleaseRefusal()
    {
        WeakReference[] roots = FailContextRelease();
        Collect();
        Assert(roots.All(r => r.IsAlive), "Unsafe cleanup loses the actual context/API owner graph.");
    }

    private static void ContextDrainRefusal()
    {
        var original = new InvalidOperationException("Controlled current-context drain refusal.");
        var api = new RecordingCudaApi { DrainFailure = original };
        CudaContext context = CudaContext.Create(5, api);
        IntPtr handle = context.Handle;
        try { context.Dispose(); throw new InvalidOperationException("Context release omitted its checked drain."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(api.ReleaseCount == 0 && context.Handle == handle && !context.IsDisposed,
            "Drain refusal releases the primary context or clears actual ownership.");
        Assert(NativeRuntimeQuarantine.TryGetFailure(original, out var failure)
            && failure?.Stage == NativeRuntimeFailureStage.Synchronization,
            "Context drain refusal does not retain its exact synchronization cause.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailContextRelease()
    {
        var original = new InvalidOperationException("Controlled primary-context release refusal.");
        var api = new RecordingCudaApi { ReleaseFailure = original };
        CudaContext context = CudaContext.Create(3, api);
        IntPtr handle = context.Handle;
        try { context.Dispose(); throw new InvalidOperationException("Cleanup refusal was accepted."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(!context.IsDisposed && context.Handle == handle, "Failed release clears actual ownership.");
        Assert(NativeRuntimeQuarantine.TryGetFailure(original, out var failure)
            && failure?.DeviceOrdinal == 3, "Original cleanup cause is not recorded for its device.");
        int calls = api.TotalCalls;
        try { context.MakeCurrent(); throw new InvalidOperationException("Quarantined context entered."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { context.Dispose(); throw new InvalidOperationException("Quarantined context retried release."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == calls && api.ReleaseCount == 1, "Quarantine permits native entry/repeated teardown.");
        return [new WeakReference(context), new WeakReference(api)];
    }

    private static void ConstructionRollback(bool refuseCleanup)
    {
        var original = new InvalidOperationException("Controlled current-context refusal.");
        var cleanup = new InvalidOperationException("Controlled construction cleanup refusal.");
        var api = new RecordingCudaApi
        {
            BindFailure = original,
            ReleaseFailure = refuseCleanup ? cleanup : null
        };
        try { CudaContext.Create(4, api); throw new InvalidOperationException("Failed construction succeeded."); }
        catch (AggregateException error) when (refuseCleanup)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "Rollback loses original or cleanup identity.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Unsafe constructor rollback is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuseCleanup && ReferenceEquals(error, original)) { }
        Assert(api.ReleaseCount == 1, "Constructor rollback does not release its acquired primary reference.");
        Assert(NativeRuntimeQuarantine.Observe().Failures.Count == (refuseCleanup ? 1 : 0),
            "Ordinary construction failure is misclassified as unsafe cleanup.");
    }

    private static void StreamClean(bool ambient)
    {
        WeakReference[] roots = CreateAndReleaseStream(ambient);
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy stream/context/API roots survive explicit cleanup.");
    }

    private static void BlasClean()
    {
        WeakReference[] roots = CreateAndReleaseBlas();
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy actual known cuBLAS/context/API graph remains retained.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateAndReleaseBlas()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var blas = CudaCublasHandle.Create(context);
        api.Current = IntPtr.Zero;
        blas.SetStream(new IntPtr(2001));
        Assert(api.Current == context.Handle && api.BlasSetStreamCount == 1,
            "Known cuBLAS handle does not rebind its actual context.");
        blas.Dispose();
        blas.Dispose();
        Assert(blas.Handle == IntPtr.Zero && api.ContextSyncCount == 1 && api.BlasDestroyCount == 1,
            "Known cuBLAS cleanup does not drain before one checked destroy.");
        try { blas.SetStream(IntPtr.Zero); throw new Exception("Disposed cuBLAS handle entered."); }
        catch (ObjectDisposedException) { }
        context.Dispose();
        return [new(blas), new(context), new(api)];
    }

    private static void BlasRefusal(bool destroy)
    {
        WeakReference[] roots = FailBlasRelease(destroy);
        Collect();
        Assert(roots.All(r => r.IsAlive), "Unsafe known cuBLAS cleanup loses its actual owner graph.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailBlasRelease(bool destroy)
    {
        var original = new InvalidOperationException("Controlled known cuBLAS cleanup refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        var blas = CudaCublasHandle.Create(context);
        IntPtr handle = blas.Handle;
        if (destroy) api.BlasDestroyFailure = original;
        else api.DrainFailure = original;
        try { blas.Dispose(); throw new Exception("Known cuBLAS cleanup refusal succeeded."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(blas.Handle == handle && api.BlasDestroyCount == (destroy ? 1 : 0) && api.ReleaseCount == 0,
            "cuBLAS refusal clears the handle or releases dependent context ownership.");
        Assert(NativeRuntimeQuarantine.TryGetFailure(original, out var failure) && failure?.DeviceOrdinal == 0,
            "Known cuBLAS refusal loses its exact recorded cause.");
        int effects = api.TotalCalls;
        try { blas.SetStream(IntPtr.Zero); throw new Exception("Quarantined cuBLAS handle entered."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { blas.Dispose(); throw new Exception("Quarantined cuBLAS teardown repeated."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == effects, "Quarantined cuBLAS owner enters native effects.");
        return [new(blas), new(context), new(api)];
    }

    private static void BlasConstructionRollback(bool refuse)
    {
        var original = new InvalidOperationException("Controlled cuBLAS math-mode failure.");
        var cleanup = new InvalidOperationException("Controlled constructor cuBLAS destroy refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(0, api);
        api.BlasMathFailure = original;
        if (refuse) api.BlasDestroyFailure = cleanup;
        try { CudaCublasHandle.Create(context); throw new Exception("Failed cuBLAS construction succeeded."); }
        catch (AggregateException error) when (refuse)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "cuBLAS rollback loses original or cleanup cause.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Constructor cuBLAS refusal is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuse && ReferenceEquals(error, original)) { }
        Assert(api.BlasDestroyCount == 1 && api.ContextSyncCount == 0 && api.ReleaseCount == 0,
            "Constructor rollback destroys no handle, drains nonexistent work or disposes borrowed context.");
        if (!refuse)
        {
            Assert(NativeRuntimeQuarantine.Observe().Failures.Count == 0,
                "Ordinary cuBLAS construction error poisons its context.");
            context.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateAndReleaseStream(bool ambient)
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        var stream = ambient ? CudaStream.Create(api) : CudaStream.Create(context);
        api.Current = IntPtr.Zero;
        stream.Synchronize();
        Assert(api.Current == context.Handle && api.StreamSyncCount == 1,
            "Actual stream synchronization does not rebind its associated context.");
        stream.Dispose();
        stream.Dispose();
        Assert(stream.Handle == IntPtr.Zero && api.StreamSyncCount == 2 && api.StreamDestroyCount == 1,
            "Stream cleanup does not drain before exactly one destroy.");
        try { stream.Synchronize(); throw new InvalidOperationException("Disposed stream entered."); }
        catch (ObjectDisposedException) { }
        context.Dispose();
        return [new(stream), new(context), new(api)];
    }

    private static void StreamRefusal(bool destroy)
    {
        WeakReference[] roots = FailStreamRelease(destroy);
        Collect();
        Assert(roots.All(r => r.IsAlive), "Unsafe stream cleanup loses its actual dependent owner graph.");
    }

    private static void StreamConstructionRollback(bool refuseCleanup)
    {
        var original = new InvalidOperationException("Controlled stream acquisition refusal.");
        var cleanup = new InvalidOperationException("Controlled stream construction cleanup refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        api.StreamCreateFailure = original;
        api.StreamDestroyFailure = refuseCleanup ? cleanup : null;
        try { CudaStream.Create(context); throw new InvalidOperationException("Failed stream construction succeeded."); }
        catch (AggregateException error) when (refuseCleanup)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "Stream rollback loses either original cause.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Unsafe stream rollback is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuseCleanup && ReferenceEquals(error, original)) { }
        Assert(api.StreamDestroyCount == 1, "Stream rollback loses its returned native handle.");
        if (!refuseCleanup) context.Dispose();
    }

    private static void ModuleClean()
    {
        WeakReference[] roots = CreateAndReleaseModule();
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy module/context/API roots survive explicit cleanup.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateAndReleaseModule()
    {
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        var module = CudaModule.LoadFromBytes([0], context);
        IntPtr function = module.GetFunction("controlled-function");
        Assert(function != IntPtr.Zero && module.GetFunction("controlled-function") == function
            && api.FunctionCount == 1, "Actual module does not preserve function caching.");
        module.Dispose();
        module.Dispose();
        Assert(api.ModuleUnloadCount == 1, "Module cleanup is not exactly once.");
        context.Dispose();
        return [new(module), new(context), new(api)];
    }

    private static void ModuleRefusal(bool unload)
    {
        WeakReference[] roots = FailModuleRelease(unload);
        Collect();
        Assert(roots.All(r => r.IsAlive), "Unsafe module cleanup loses its actual dependent owner graph.");
    }

    private static void ModuleConstructionRollback(bool refuseCleanup)
    {
        var original = new InvalidOperationException("Controlled module acquisition refusal.");
        var cleanup = new InvalidOperationException("Controlled module construction cleanup refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        api.ModuleLoadFailure = original;
        api.ModuleUnloadFailure = refuseCleanup ? cleanup : null;
        try { CudaModule.LoadFromBytes([0], context); throw new InvalidOperationException("Failed module construction succeeded."); }
        catch (AggregateException error) when (refuseCleanup)
        {
            Assert(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original)
                && ReferenceEquals(error.InnerExceptions[1], cleanup), "Module rollback loses either original cause.");
            Assert(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Unsafe module rollback is not recorded.");
        }
        catch (InvalidOperationException error) when (!refuseCleanup && ReferenceEquals(error, original)) { }
        Assert(api.ModuleUnloadCount == 1, "Module rollback loses its returned native handle.");
        if (!refuseCleanup) context.Dispose();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailModuleRelease(bool unload)
    {
        var original = new InvalidOperationException("Controlled module cleanup refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        var module = CudaModule.LoadFromBytes([0], context);
        module.GetFunction("controlled-function");
        if (unload) api.ModuleUnloadFailure = original;
        else api.DrainFailure = original;
        try { module.Dispose(); throw new InvalidOperationException("Module cleanup refusal was accepted."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(api.ModuleUnloadCount == (unload ? 1 : 0), "Module unload occurs after failed drain.");
        int calls = api.TotalCalls;
        try { module.GetFunction("controlled-function"); throw new InvalidOperationException("Failed module publishes a cached function."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { module.Dispose(); throw new InvalidOperationException("Failed module retries cleanup."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { context.Dispose(); throw new InvalidOperationException("Dependent context releases after module failure."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == calls && api.ReleaseCount == 0, "Module quarantine permits dependent native entry.");
        return [new(module), new(context), new(api)];
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] FailStreamRelease(bool destroy)
    {
        var original = new InvalidOperationException("Controlled stream cleanup refusal.");
        var api = new RecordingCudaApi();
        var context = CudaContext.Create(1, api);
        var independent = CudaContext.Create(2, api);
        var stream = CudaStream.Create(context);
        IntPtr streamHandle = stream.Handle;
        if (destroy) api.StreamDestroyFailure = original;
        else api.StreamSyncFailure = original;
        try { stream.Dispose(); throw new InvalidOperationException("Stream cleanup refusal was accepted."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        Assert(stream.Handle == streamHandle && api.StreamDestroyCount == (destroy ? 1 : 0),
            "Unsafe stream cleanup clears the handle or destroys after failed synchronization.");
        int calls = api.TotalCalls;
        try { stream.Synchronize(); throw new InvalidOperationException("Failed stream entered."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { stream.Dispose(); throw new InvalidOperationException("Failed stream retried cleanup."); }
        catch (NativeRuntimeQuarantinedException) { }
        try { context.Dispose(); throw new InvalidOperationException("Dependent context released after stream refusal."); }
        catch (NativeRuntimeQuarantinedException) { }
        Assert(api.TotalCalls == calls && api.ReleaseCount == 0,
            "Unsafe dependent cleanup reaches native calls.");
        // This independently registered known device has no dependency on the failed stream.
        independent.MakeCurrent();
        independent.Dispose();
        Assert(api.ReleaseCount == 1, "Known independent context is incorrectly quarantined.");
        return [new(stream), new(context), new(api)];
    }

    private static void Collect()
    {
        for (int i = 0; i < 4; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
    }

    private static void ForeignContext(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: false);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign actual context/Core/CUDA/fixture roots do not match explicit cleanup outcome.");
    }

    private static void ForeignStream(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: true);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign actual stream/dependent owners do not match explicit cleanup outcome.");
    }

    private static void ForeignModule(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: false, module: true);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign actual module/dependent owners do not match explicit cleanup outcome.");
    }

    private static void ForeignComposed(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: false, composed: true);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign composed context/parent generation roots do not match cleanup outcome.");
    }

    private static void ForeignBlas(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: false, blas: true);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign actual known cuBLAS roots do not match cleanup outcome.");
    }

    private static void ForeignKernels(bool unsafeRelease)
    {
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease, stream: false, kernels: true);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease), "Foreign actual kernel roots do not match checked cleanup outcome.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] ExecuteForeignContext(bool unsafeRelease, bool stream, bool module = false, bool composed = false, bool blas = false, bool kernels = false)
    {
        var context = new FixtureLoadContext();
        Assembly fixture = context.LoadFromAssemblyPath(typeof(Program).Assembly.Location);
        Type entry = fixture.GetType(typeof(Program).FullName!, true)!;
        string operationName = kernels ? nameof(RunForeignKernels) : blas ? nameof(RunForeignBlas) : composed ? nameof(RunForeignComposed) : module ? nameof(RunForeignModule)
            : stream ? nameof(RunForeignStream) : nameof(RunForeignContext);
        var operation = entry.GetMethod(operationName, BindingFlags.Static | BindingFlags.Public)!;
        var ownedRoots = (WeakReference[])operation.Invoke(null, [unsafeRelease])!;
        Assembly cuda = context.Assemblies.Single(a => a.GetName().Name == "TensorSharp.Backends.Cuda");
        Assembly core = context.Assemblies.Single(a => a.GetName().Name == "TensorSharp.Core");
        Assert(AssemblyLoadContext.GetLoadContext(cuda) == context && AssemblyLoadContext.GetLoadContext(core) == context
            && cuda.ManifestModule.ModuleVersionId == typeof(CudaContext).Assembly.ManifestModule.ModuleVersionId,
            "Actual foreign managed dependency provenance is not the compiled CUDA graph.");
        WeakReference[] roots = [.. ownedRoots, new(context), new(fixture), new(cuda), new(core)];
        context.Unload();
        return roots;
    }

    public static WeakReference[] RunForeignContext(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaContext).Assembly)?.IsCollectible == true,
            "Actual CUDA context does not execute in the foreign collectible generation.");
        return unsafeRelease ? FailContextRelease() : CreateAndReleaseContext();
    }

    public static WeakReference[] RunForeignStream(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaStream).Assembly)?.IsCollectible == true,
            "Actual CUDA stream does not execute in the foreign collectible generation.");
        return unsafeRelease ? FailStreamRelease(false) : CreateAndReleaseStream(false);
    }

    public static WeakReference[] RunForeignModule(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaModule).Assembly)?.IsCollectible == true,
            "Actual CUDA module does not execute in the foreign collectible generation.");
        return unsafeRelease ? FailModuleRelease(false) : CreateAndReleaseModule();
    }

    public static WeakReference[] RunForeignComposed(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaContext).Assembly)?.IsCollectible == true,
            "Actual context composition does not execute in the foreign generation.");
        return unsafeRelease ? FailContextPair() : ReleaseContextPair(true, false);
    }

    public static WeakReference[] RunForeignBlas(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaCublasHandle).Assembly)?.IsCollectible == true,
            "Actual known cuBLAS owner does not execute in the foreign generation.");
        return unsafeRelease ? FailBlasRelease(false) : CreateAndReleaseBlas();
    }

    public static WeakReference[] RunForeignKernels(bool unsafeRelease)
    {
        Assert(AssemblyLoadContext.GetLoadContext(typeof(CudaKernels).Assembly)?.IsCollectible == true,
            "Actual kernel owner does not execute in the foreign generation.");
        return unsafeRelease ? FailKernelsRelease(false) : CreateAndReleaseKernels();
    }
}

internal sealed class FixtureLoadContext : AssemblyLoadContext
{
    private readonly AssemblyDependencyResolver _resolver = new(typeof(Program).Assembly.Location);
    internal FixtureLoadContext() : base(isCollectible: true) { }
    protected override Assembly? Load(AssemblyName name)
    {
        string? path = _resolver.ResolveAssemblyToPath(name);
        return path == null ? null : LoadFromAssemblyPath(path);
    }
}

internal sealed class RecordingCudaApi : RefusingCudaApi
{
    internal IntPtr Current;
    internal Exception? ReleaseFailure;
    internal Exception? BindFailure;
    internal Exception? DrainFailure;
    internal Exception? StreamSyncFailure;
    internal Exception? StreamDestroyFailure;
    internal Exception? StreamCreateFailure;
    internal Exception? ModuleUnloadFailure;
    internal Exception? ModuleLoadFailure;
    internal Exception? FunctionFailure;
    internal int StreamSyncCount;
    internal int StreamDestroyCount;
    internal int ModuleUnloadCount;
    internal int FunctionCount;
    internal int ReleaseCount;
    internal int TotalCalls;
    internal IntPtr RefusedBind;
    internal Exception? RestoreFailure;
    internal readonly List<int> BindThreads = new();
    internal readonly List<IntPtr> BindContexts = new();
    internal int ContextSyncCount;
    internal int BlasDestroyCount;
    internal int BlasSetStreamCount;
    internal Exception? BlasDestroyFailure;
    internal Exception? BlasMathFailure;
    internal int MemoryAllocateCount;
    internal int MemoryFreeCount;
    internal int KernelLaunchCount;
    internal int AttributeCount;
    internal Exception? MemoryAllocateFailure;
    internal Exception? MemoryFreeFailure;
    internal Exception? KernelLaunchFailure;
    internal Exception? AttributeFailure;

    public override int cuInit(uint flags) { TotalCalls++; return 0; }
    public override int cuDeviceGet(out int device, int ordinal) { TotalCalls++; device = ordinal + 100; return 0; }
    public override int cuDeviceGetCount(out int count) { TotalCalls++; count = 8; return 0; }
    public override int cuCtxGetDevice(out int device)
    { TotalCalls++; device = checked((int)Current.ToInt64() - 1000); return 0; }
    public override int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device)
    { TotalCalls++; ctx = new IntPtr(1000 + device); return 0; }
    public override int cuCtxSetCurrent(IntPtr ctx)
    {
        TotalCalls++;
        BindThreads.Add(Environment.CurrentManagedThreadId);
        BindContexts.Add(ctx);
        if (ctx == RefusedBind && RestoreFailure != null) throw RestoreFailure;
        if (ctx != IntPtr.Zero && BindFailure != null) throw BindFailure;
        Current = ctx;
        return 0;
    }
    public override int cuCtxGetCurrent(out IntPtr ctx) { TotalCalls++; ctx = Current; return 0; }
    public override int cuCtxSynchronize()
    {
        TotalCalls++;
        ContextSyncCount++;
        if (DrainFailure != null) throw DrainFailure;
        return 0;
    }
    public override int cuDevicePrimaryCtxRelease(int device)
    {
        TotalCalls++;
        ReleaseCount++;
        if (ReleaseFailure != null) throw ReleaseFailure;
        return 0;
    }
    public override int cuStreamCreate(out IntPtr stream, uint flags)
    {
        TotalCalls++;
        stream = new IntPtr(2001);
        if (StreamCreateFailure != null) throw StreamCreateFailure;
        return 0;
    }
    public override int cuStreamSynchronize(IntPtr stream)
    {
        TotalCalls++;
        StreamSyncCount++;
        if (StreamSyncFailure != null) throw StreamSyncFailure;
        return 0;
    }
    public override int cuStreamDestroy(IntPtr stream)
    {
        TotalCalls++;
        StreamDestroyCount++;
        if (StreamDestroyFailure != null) throw StreamDestroyFailure;
        return 0;
    }
    public override int cuModuleLoadData(out IntPtr module, IntPtr image)
    {
        TotalCalls++;
        module = new IntPtr(3001);
        if (ModuleLoadFailure != null) throw ModuleLoadFailure;
        return 0;
    }
    public override int cuModuleGetFunction(out IntPtr function, IntPtr module, string name)
    {
        TotalCalls++; FunctionCount++; function = new IntPtr(3002);
        if (FunctionFailure != null) throw FunctionFailure;
        return 0;
    }
    public override int cuModuleUnload(IntPtr module)
    {
        TotalCalls++;
        ModuleUnloadCount++;
        if (ModuleUnloadFailure != null) throw ModuleUnloadFailure;
        return 0;
    }

    public override int cublasCreate(out IntPtr handle)
    { TotalCalls++; handle = new IntPtr(4001); return 0; }
    public override int cublasSetMathMode(IntPtr handle, int mode)
    {
        TotalCalls++;
        if (BlasMathFailure != null) throw BlasMathFailure;
        return 0;
    }
    public override int cublasSetStream(IntPtr handle, IntPtr stream)
    { TotalCalls++; BlasSetStreamCount++; return 0; }
    public override int cublasDestroy(IntPtr handle)
    {
        TotalCalls++;
        BlasDestroyCount++;
        if (BlasDestroyFailure != null) throw BlasDestroyFailure;
        return 0;
    }

    public override int cuMemAlloc(out IntPtr ptr, UIntPtr bytes)
    {
        TotalCalls++; MemoryAllocateCount++; ptr = new IntPtr(5000 + MemoryAllocateCount);
        if (MemoryAllocateFailure != null) throw MemoryAllocateFailure;
        return 0;
    }
    public override int cuMemFree(IntPtr ptr)
    {
        TotalCalls++; MemoryFreeCount++;
        if (MemoryFreeFailure != null) throw MemoryFreeFailure;
        return 0;
    }
    public override int cuFuncSetAttribute(IntPtr function, int attribute, int value)
    {
        TotalCalls++; AttributeCount++;
        if (AttributeFailure != null) throw AttributeFailure;
        return 0;
    }
    public override int cuLaunchKernel(IntPtr function, uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, IntPtr stream,
        IntPtr kernelParams, IntPtr extra)
    {
        TotalCalls++; KernelLaunchCount++;
        if (KernelLaunchFailure != null) throw KernelLaunchFailure;
        return 0;
    }
}
