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
            case "foreign-context-clean": ForeignContext(false); break;
            case "foreign-context-release-refusal": ForeignContext(true); break;
            case "foreign-stream-clean": ForeignStream(false); break;
            case "foreign-stream-sync-refusal": ForeignStream(true); break;
            case "foreign-module-clean": ForeignModule(false); break;
            case "foreign-module-drain-refusal": ForeignModule(true); break;
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

    private static void StreamClean(bool ambient)
    {
        WeakReference[] roots = CreateAndReleaseStream(ambient);
        Collect();
        Assert(roots.All(r => !r.IsAlive), "Healthy stream/context/API roots survive explicit cleanup.");
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

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] ExecuteForeignContext(bool unsafeRelease, bool stream, bool module = false)
    {
        var context = new FixtureLoadContext();
        Assembly fixture = context.LoadFromAssemblyPath(typeof(Program).Assembly.Location);
        Type entry = fixture.GetType(typeof(Program).FullName!, true)!;
        string operationName = module ? nameof(RunForeignModule)
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
    internal int StreamSyncCount;
    internal int StreamDestroyCount;
    internal int ModuleUnloadCount;
    internal int FunctionCount;
    internal int ReleaseCount;
    internal int TotalCalls;

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
    { TotalCalls++; FunctionCount++; function = new IntPtr(3002); return 0; }
    public override int cuModuleUnload(IntPtr module)
    {
        TotalCalls++;
        ModuleUnloadCount++;
        if (ModuleUnloadFailure != null) throw ModuleUnloadFailure;
        return 0;
    }
}
