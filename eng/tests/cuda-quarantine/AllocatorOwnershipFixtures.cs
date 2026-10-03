using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.CudaQuarantineFixture;

internal static class AllocatorOwnershipFixtures
{
    internal static void Construction(bool partial)
    {
        var original = new InvalidOperationException("Controlled allocator construction refusal.");
        var api = new RecordingCudaApi();
        if (partial) api.BlasMathFailure = original;
        else api.DeviceFailure = original;
        try { _ = new CudaAllocator(4, api); throw new Exception("Construction unexpectedly succeeds."); }
        catch (InvalidOperationException failure) when (ReferenceEquals(failure, original)) { }
        Require(api.ReleaseCount == (partial ? 1 : 0) && api.StreamDestroyCount == (partial ? 1 : 0)
            && api.BlasDestroyCount == (partial ? 1 : 0) && api.MemoryFreeCount == 0,
            "Constructor cleanup loses or invents actual context/stream/cuBLAS ownership.");
        Require(!NativeRuntimeQuarantine.TryGetFailure(original, out _),
            "Clean constructor rollback classifies an ordinary construction error as quarantine.");
    }

    internal static void BeforeNativeRefusal()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var reservation = allocator.Census.Reserve();
        var parent = new object();
        allocator.Census.FenceParent(parent, true);
        int calls = api.TotalCalls;
        Expect<ObjectDisposedException>(() => new CudaStorage(allocator, DType.Float32, 4, reservation));
        Require(api.TotalCalls == calls && api.MemoryAllocateCount == 0 && !allocator.Census.HasChildren,
            "Before-native refusal performs effects or leaves an unattached storage reservation.");
        Require(NativeRuntimeQuarantine.Observe().State == NativeRuntimeQuarantineState.NoRecordedFailure,
            "No-resource construction refusal falsely poisons pending graph ownership.");
        var plan = CudaRetirementPlan.Prepare(parent, Array.Empty<Tensor>(), new[] { allocator }, true);
        plan.AllowStorageRelease();
        var restore = CudaContextRestoration.Capture(allocator.NativeCalls);
        allocator.DisposeOwned(plan, restore);
        plan.Complete();
        restore.Restore();
    }

    internal static void ParentTransfer()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var tensor = new Tensor(allocator, DType.Float32, 4);
        int calls = api.TotalCalls;
        ExpectBusy(allocator.Dispose);
        var parent = new object();
        var plan = CudaRetirementPlan.Prepare(parent, new[] { tensor }, new[] { allocator }, true);
        Require(api.TotalCalls == calls, "Healthy Busy parent transfer enters native effects.");
        Expect<ObjectDisposedException>(() => allocator.Allocate(DType.Float32, 1));
        plan.AllowStorageRelease();
        tensor.Dispose();
        var restore = CudaContextRestoration.Capture(allocator.NativeCalls);
        allocator.DisposeOwned(plan, restore);
        plan.Complete();
        restore.Restore();
        Require(allocator.Context.IsDisposed && api.MemoryFreeCount == 1,
            "The explicit owning plan cannot finish a declined bare allocator attempt.");
    }

    internal static void InflightParentRefusal()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        Exception? error = null;
        api.BeforeContextSynchronize = () =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException("Controlled cleanup gate does not open.");
        };
        var worker = new Thread(() =>
        {
            try { allocator.Dispose(); }
            catch (Exception failure) { error = failure; }
        })
        { IsBackground = true };
        try
        {
            worker.Start();
            Require(entered.Wait(TimeSpan.FromSeconds(5)), "Successful bare cleanup does not enter its API.");
            int calls = api.TotalCalls;
            Expect<InvalidOperationException>(() => CudaRetirementPlan.Prepare(new object(),
                Array.Empty<Tensor>(), new[] { allocator }, true));
            Require(api.TotalCalls == calls, "A different parent enters effects during admitted bare cleanup.");
        }
        finally
        {
            release.Set();
            Require(worker.Join(TimeSpan.FromSeconds(5)), "Admitted cleanup does not drain.");
            api.BeforeContextSynchronize = null;
        }
        Require(error == null && allocator.Context.IsDisposed, "Parent conflict changes the successful bare cleanup.");
    }

    internal static void GraphParentRefusal()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var tensor = new Tensor(allocator, DType.Float32, 4);
        var owner = new object();
        var plan = CudaRetirementPlan.Prepare(owner, new[] { tensor }, new[] { allocator }, true);
        int calls = api.TotalCalls;
        Expect<InvalidOperationException>(() => CudaRetirementPlan.Prepare(new object(),
            new[] { tensor }, new[] { allocator }, true));
        Require(api.TotalCalls == calls, "A different parent enters effects or steals graph-pending ownership.");
        plan.AllowStorageRelease();
        tensor.Dispose();
        var restore = CudaContextRestoration.Capture(allocator.NativeCalls);
        allocator.DisposeOwned(plan, restore);
        plan.Complete();
        restore.Restore();
    }

    internal static void Busy()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var tensor = new Tensor(allocator, DType.Float32, 4);
        var view = tensor.CopyRef();
        int calls = api.TotalCalls;
        ExpectBusy(allocator.Dispose);
        Require(api.TotalCalls == calls && !allocator.Context.IsDisposed, "Busy performs native cleanup.");
        Expect<ObjectDisposedException>(() => allocator.Allocate(DType.Float32, 1));
        Expect<ObjectDisposedException>(() => tensor.CopyRef());
        tensor.Dispose();
        calls = api.TotalCalls;
        ExpectBusy(allocator.Dispose);
        Require(api.TotalCalls == calls, "A live escaped view does not keep cleanup Busy.");
        view.Dispose();
        allocator.Dispose();
        calls = api.TotalCalls;
        allocator.Dispose();
        Require(allocator.Context.IsDisposed && api.TotalCalls == calls
            && api.ReleaseCount == 1 && api.MemoryFreeCount == 1, "Explicit retry does not release exactly once.");
        Expect<ObjectDisposedException>(() => allocator.Synchronize());
        Expect<ObjectDisposedException>(() => allocator.GetMemoryInfo());
        Expect<ObjectDisposedException>(() => tensor.Storage.GetElementAsFloat(0));
        Require(api.TotalCalls == calls, "Disposed metadata/access refusal invokes native effects.");
    }

    internal static void OwnedPlan()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var tensor = new Tensor(allocator, DType.Float32, 4);
        var view = tensor.CopyRef();
        var parent = new object();
        int calls = api.TotalCalls;
        ExpectBusy(() => CudaRetirementPlan.Prepare(parent, new[] { tensor, tensor }, new[] { allocator }, true));
        Require(api.TotalCalls == calls, "Incomplete tensor intents perform native cleanup.");
        var plan = CudaRetirementPlan.Prepare(parent, new[] { tensor, view, tensor }, new[] { allocator }, true);
        Require(api.TotalCalls == calls, "Owned preflight enters native effects.");
        plan.AllowStorageRelease();
        tensor.Dispose();
        view.Dispose();
        var restore = CudaContextRestoration.Capture(allocator.NativeCalls);
        allocator.DisposeOwned(plan, restore);
        plan.Complete();
        restore.Restore();
        Require(allocator.Context.IsDisposed && api.MemoryFreeCount == 1, "Healthy model-owned references block disposal.");
    }

    internal static void PendingAcquisition()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        CudaStorage? storage = null;
        Exception? error = null;
        api.BeforeMemoryAllocate = () =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException("Controlled allocation gate does not open.");
        };
        var worker = new Thread(() =>
        {
            try { storage = (CudaStorage)allocator.Allocate(DType.Float32, 4); }
            catch (Exception failure) { error = failure; }
        })
        { IsBackground = true };
        try
        {
            worker.Start();
            Require(entered.Wait(TimeSpan.FromSeconds(5)), "Allocation does not enter its actual API.");
            int calls = api.TotalCalls;
            ExpectBusy(allocator.Dispose);
            Require(api.TotalCalls == calls, "Pending native acquisition permits destructive cleanup.");
        }
        finally
        {
            release.Set();
            Require(worker.Join(TimeSpan.FromSeconds(5)), "Allocation worker does not drain.");
            api.BeforeMemoryAllocate = null;
        }
        Require(error == null && storage != null, "Already admitted acquisition does not return its true owner.");
        storage!.Release();
        allocator.Dispose();
    }

    internal static void Copy()
    {
        var api = new RecordingCudaApi { TrackStorageData = true };
        var allocator = new CudaAllocator(4, api);
        var source = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var destination = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        source.SetElementsAsFloat(0, new[] { 1f, 2f, 3f, 4f });
        destination.CopyDeviceFrom(source);
        Require(destination.GetElementsAsFloat(0, 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }),
            "Actual Storage mirror/copy methods change data in the controlled API.");
        source.Release();
        destination.Release();
        allocator.Dispose();
        Require(api.DeviceBytes.Count == 0, "Explicit cleanup retains a controlled device allocation.");
    }

    internal static void LargePeerCopy()
    {
        var api = new RecordingCudaApi { PeerAccessible = true };
        var sourceAllocator = new CudaAllocator(4, api);
        var destinationAllocator = new CudaAllocator(5, api);
        long length = (long)int.MaxValue + 1;
        var source = (CudaStorage)sourceAllocator.Allocate(DType.UInt8, length);
        var destination = (CudaStorage)destinationAllocator.Allocate(DType.UInt8, length);
        try
        {
            destination.CopyDeviceFrom(source);
            Require(api.PeerCopyCount == 1 && api.PeerCopyBytes == (ulong)length
                && api.HostStageReadCount == 0 && api.HostStageWriteCount == 0,
                "Opaque large peer transfer requires a host staging allocation.");
        }
        finally
        {
            source.Release();
            destination.Release();
            sourceAllocator.Dispose();
            destinationAllocator.Dispose();
        }
    }

    internal static void FallbackCopy()
    {
        var api = new RecordingCudaApi { TrackStorageData = true };
        var sourceAllocator = new CudaAllocator(4, api);
        var destinationAllocator = new CudaAllocator(5, api);
        var source = (CudaStorage)sourceAllocator.Allocate(DType.Float32, 4);
        var destination = (CudaStorage)destinationAllocator.Allocate(DType.Float32, 4);
        try
        {
            source.SetElementsAsFloat(0, new[] { 1f, 2f, 3f, 4f });
            destination.CopyDeviceFrom(source);
            Require(api.PeerCopyCount == 0 && api.HostStageReadCount == 1 && api.HostStageWriteCount == 1
                && destination.GetElementsAsFloat(0, 4).SequenceEqual(new[] { 1f, 2f, 3f, 4f }),
                "Bounded fallback does not preserve the actual entered bytes.");
        }
        finally
        {
            source.Release();
            destination.Release();
            sourceAllocator.Dispose();
            destinationAllocator.Dispose();
        }
    }

    internal static void Rollback(bool refuse)
    {
        var original = new InvalidOperationException("Controlled allocation OUT then throw.");
        var cleanup = new InvalidOperationException("Controlled rollback drain refusal.");
        var api = new RecordingCudaApi { MemoryAllocateFailure = original };
        var allocator = new CudaAllocator(4, api);
        if (refuse) api.DrainFailure = cleanup;
        try { allocator.Allocate(DType.Float32, 4); throw new Exception("Allocation unexpectedly succeeds."); }
        catch (AggregateException aggregate) when (refuse)
        {
            Require(ReferenceEquals(aggregate.InnerExceptions[0], original)
                && ReferenceEquals(aggregate.InnerExceptions[1], cleanup), "Rollback changes original/cleanup identities.");
        }
        catch (InvalidOperationException failure) when (!refuse && ReferenceEquals(failure, original)) { }
        Require(api.MemoryFreeCount == 0, "Rollback frees a buffer after refusal or bypasses its live pool owner.");
        if (refuse)
        {
            Require(NativeRuntimeQuarantine.TryGetFailure(cleanup, out _), "Unsafe rollback has no authenticated failure.");
            Require(allocator.Census.HasChildren, "Unsafe rollback drops the actual storage reservation.");
        }
        else
        {
            Require(!allocator.Census.HasChildren, "Clean rollback leaves a storage reservation.");
            Require(allocator.GetStats().CachedBytes >= 256, "Clean rollback loses the transferred pool allocation.");
            allocator.Dispose();
            Require(api.MemoryFreeCount == 1, "The actual pool owner does not release its transferred allocation.");
        }
    }

    internal static void CopyDisposal(bool peer = false)
    {
        Environment.SetEnvironmentVariable("TENSORSHARP_CUDA_POOL", "0");
        var api = new RecordingCudaApi { TrackStorageData = true, PeerAccessible = peer };
        var allocator = new CudaAllocator(4, api);
        var destinationAllocator = peer ? new CudaAllocator(5, api) : allocator;
        var source = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var destination = (CudaStorage)destinationAllocator.Allocate(DType.Float32, 4);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var retiring = new ManualResetEventSlim();
        Exception? copyError = null;
        Exception? releaseError = null;
        api.BeforeDeviceCopy = () =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException("Controlled transfer gate does not open.");
        };
        var copy = new Thread(() =>
        {
            try { destination.CopyDeviceFrom(source); }
            catch (Exception error) { copyError = error; }
        })
        { IsBackground = true };
        var dispose = new Thread(() =>
        {
            retiring.Set();
            try { source.Release(); }
            catch (Exception error) { releaseError = error; }
        })
        { IsBackground = true };
        bool disposalStarted = false;
        try
        {
            copy.Start();
            Require(entered.Wait(TimeSpan.FromSeconds(5)), "Actual transfer does not enter its controlled API.");
            dispose.Start();
            disposalStarted = true;
            Require(retiring.Wait(TimeSpan.FromSeconds(5)), "Source retirement does not start.");
            Require(!dispose.Join(TimeSpan.FromMilliseconds(100)) && api.MemoryFreeCount == 0,
                "Source retirement passes the admitted transfer before enqueue completes.");
        }
        finally
        {
            release.Set();
            Require(copy.Join(TimeSpan.FromSeconds(5)), "Transfer does not settle after its gate opens.");
            if (disposalStarted) Require(dispose.Join(TimeSpan.FromSeconds(5)), "Source retirement does not drain.");
            api.BeforeDeviceCopy = null;
        }
        Require(copyError == null && releaseError == null && api.MemoryFreeCount == 1,
            "Transfer/retirement changes failures or loses the actual source cleanup.");
        destination.Release();
        allocator.Dispose();
        if (peer) destinationAllocator.Dispose();
        Require(api.MemoryFreeCount == 2, "Transfer cleanup does not retire the two actual buffers exactly once.");
    }

    internal static void DrainRefusal()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var storage = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var error = new InvalidOperationException("Controlled storage drain refusal.");
        ExpectBusy(allocator.Dispose);
        api.DrainFailure = error;
        try { storage.Release(); throw new Exception("Release unexpectedly succeeds."); }
        catch (InvalidOperationException failure) when (ReferenceEquals(failure, error)) { }
        Require(api.MemoryFreeCount == 0 && storage.DeviceBuffer != IntPtr.Zero
            && allocator.Census.HasChildren, "Refused drain releases or forgets actual storage.");
        Require(NativeRuntimeQuarantine.TryGetFailure(error, out _), "Storage failure is not recorded.");
        int calls = api.TotalCalls;
        Expect<NativeRuntimeQuarantinedException>(allocator.Dispose);
        Expect<NativeRuntimeQuarantinedException>(() => CudaRetirementPlan.Prepare(new object(),
            Array.Empty<Tensor>(), new[] { allocator }, true));
        Require(api.TotalCalls == calls, "Quarantine permits declined-owner transfer or more native effects.");
        Require(storage.IsRetainedFinalizerFailure(error)
            && !storage.IsRetainedFinalizerFailure(new AggregateException(error)),
            "Finalizer recognition accepts an arbitrary exception graph or rejects its authenticated cause.");
        try { storage.EnsureDeviceCurrent(); throw new Exception("Quarantined storage permits effects."); }
        catch (NativeRuntimeQuarantinedException refusal)
        { Require(storage.IsRetainedFinalizerFailure(refusal), "The exact retained refusal is not authenticated."); }
    }

    internal static void ParentFinalizer()
    {
        var roots = CreatePendingParent();
        for (int i = 0; i < 8; i++)
        { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }
        Require(roots.All(root => root.IsAlive), "Unsafe finalizer does not retain actual parent/allocator/storage.");
        var storage = (CudaStorage)roots[2].Target!;
        var api = (RecordingCudaApi)roots[3].Target!;
        Require(api.MemoryFreeCount == 0 && storage.DeviceBuffer != IntPtr.Zero,
            "Pending parent graphs allow destructive storage finalization.");
        Require(NativeRuntimeQuarantine.Observe().State
            == NativeRuntimeQuarantineState.FailedRequiresProcessExit, "Unsafe parent cleanup has no terminal observation.");
    }

    internal static void SiblingQuarantine()
    {
        var roots = CreateQuarantinedSiblings();
        var api = (RecordingCudaApi)roots[3].Target!;
        int calls = api.TotalCalls;
        for (int i = 0; i < 8; i++)
        { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }
        Require(roots.All(root => root.IsAlive), "An affected sibling storage loses the actual retained owner graph.");
        Require(api.TotalCalls == calls && api.MemoryFreeCount == 0,
            "Affected storage finalizers perform native cleanup after another owner quarantines the device.");
        var sibling = (CudaStorage)roots[2].Target!;
        try { sibling.EnsureDeviceCurrent(); throw new Exception("Quarantined sibling permits effects."); }
        catch (NativeRuntimeQuarantinedException refusal)
        { Require(sibling.IsRetainedFinalizerFailure(refusal), "The existing authority did not authenticate the affected sibling."); }
        Require(api.TotalCalls == calls, "Sibling refusal reaches a native call.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreateQuarantinedSiblings()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var first = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var second = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var cause = new InvalidOperationException("Controlled other stream owner synchronization refusal.");
        api.StreamSyncFailure = cause;
        try { allocator.Stream.Synchronize(); throw new Exception("Stream synchronization unexpectedly succeeds."); }
        catch (InvalidOperationException failure) when (ReferenceEquals(failure, cause)) { }
        Require(NativeRuntimeQuarantine.TryGetFailure(cause, out _), "The actual stream owner did not publish its failure.");
        return new[] { new WeakReference(allocator, true), new WeakReference(first, true),
            new WeakReference(second, true), new WeakReference(api, true) };
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] CreatePendingParent()
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var storage = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var parent = new object();
        ExpectBusy(() => CudaRetirementPlan.Prepare(parent, Array.Empty<Tensor>(), new[] { allocator }, true));
        return new[] { new WeakReference(parent, true), new WeakReference(allocator, true),
            new WeakReference(storage, true), new WeakReference(api, true) };
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static WeakReference[] ReleaseForeign(bool refuse)
    {
        var api = new RecordingCudaApi();
        var allocator = new CudaAllocator(4, api);
        var storage = (CudaStorage)allocator.Allocate(DType.Float32, 4);
        var roots = new[] { new WeakReference(allocator, true), new WeakReference(storage, true),
            new WeakReference(api, true), new WeakReference(allocator.Context, true) };
        if (refuse)
        {
            var error = new InvalidOperationException("Foreign controlled storage cleanup refusal.");
            api.DrainFailure = error;
            try { storage.Release(); throw new Exception("Cleanup unexpectedly succeeds."); }
            catch (InvalidOperationException failure) when (ReferenceEquals(failure, error)) { }
            Require(api.MemoryFreeCount == 0 && storage.DeviceBuffer != IntPtr.Zero,
                "Foreign refused cleanup loses the real storage.");
        }
        else
        {
            storage.Release();
            allocator.Dispose();
            Require(api.MemoryFreeCount == 1 && allocator.Context.IsDisposed,
                "Foreign healthy cleanup does not explicitly release ownership.");
        }
        return roots;
    }

    private static void ExpectBusy(Action action) => Expect<CudaAllocatorBusyException>(action);

    private static void Expect<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T) { return; }
        throw new InvalidOperationException("Expected refusal does not occur: " + typeof(T).Name);
    }

    private static void Require(bool condition, string error)
    { if (!condition) throw new InvalidOperationException(error); }
}
