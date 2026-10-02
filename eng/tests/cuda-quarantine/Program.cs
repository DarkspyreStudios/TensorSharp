using System.Runtime.CompilerServices;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
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
            case "context-release-refusal": ContextReleaseRefusal(); break;
            case "context-drain-refusal": ContextDrainRefusal(); break;
            case "context-construction-rollback": ConstructionRollback(false); break;
            case "context-construction-rollback-refusal": ConstructionRollback(true); break;
            case "foreign-context-clean": ForeignContext(false); break;
            case "foreign-context-release-refusal": ForeignContext(true); break;
            default: throw new ArgumentException("Unknown controlled fixture mode.");
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            mode = args[0], passed = true, nativeExecution = false,
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

    private static void ContextClean()
    {
        WeakReference[] roots = CreateAndReleaseContext();
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy context/API roots survive explicit disposal.");
        Assert(NativeRuntimeQuarantine.Observe().Failures.Count == 0, "Healthy cleanup records quarantine.");
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
        WeakReference[] roots = ExecuteForeignContext(unsafeRelease);
        Collect();
        Assert(roots.All(r => r.IsAlive == unsafeRelease),
            "Foreign actual context/Core/CUDA/fixture roots do not match explicit cleanup outcome.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] ExecuteForeignContext(bool unsafeRelease)
    {
        var context = new FixtureLoadContext();
        Assembly fixture = context.LoadFromAssemblyPath(typeof(Program).Assembly.Location);
        Type entry = fixture.GetType(typeof(Program).FullName!, true)!;
        var operation = entry.GetMethod(nameof(RunForeignContext), BindingFlags.Static | BindingFlags.Public)!;
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
    internal int ReleaseCount;
    internal int TotalCalls;

    public override int cuInit(uint flags) { TotalCalls++; return 0; }
    public override int cuDeviceGet(out int device, int ordinal) { TotalCalls++; device = ordinal + 100; return 0; }
    public override int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device)
    { TotalCalls++; ctx = new IntPtr(1000 + device); return 0; }
    public override int cuCtxSetCurrent(IntPtr ctx)
    {
        TotalCalls++;
        if (ctx != IntPtr.Zero && BindFailure != null) throw BindFailure;
        Current = ctx;
        return 0;
    }
    public override int cuCtxGetCurrent(out IntPtr ctx) { TotalCalls++; ctx = Current; return 0; }
    public override int cuCtxSynchronize()
    {
        TotalCalls++;
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
}
