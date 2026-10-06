using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;

if (args.Length != 3 || args[0] is not ("selected" or "selected-metal" or "default" or "ambiguous-default" or "reject-variant" or "reject-legacy" or
    "retire-selected-cpu" or "retire-default-cpu" or "retire-selected-metal" or "retire-default-metal"))
{
    Console.Error.WriteLine("Usage: ggml-native-runtime <selected|selected-metal|default|ambiguous-default|reject-variant|reject-legacy|retire-selected-cpu|retire-default-cpu|retire-selected-metal|retire-default-metal> <absolute-bridge-directory> <variant>");
    return 2;
}

try
{
    string mode = args[0], directory = Path.GetFullPath(args[1]), variant = args[2];
    if (mode.StartsWith("retire-", StringComparison.Ordinal))
        return LoadedGenerationRetirement.Run(mode, directory, variant);
    GgmlBackendType backend = mode == "selected-metal" ? GgmlBackendType.Metal : GgmlBackendType.Cpu;
    string entry = Path.Combine(directory, GgmlNativeLoader.EntryLibraryName);
    byte[] bytes = File.ReadAllBytes(entry);
    var candidate = new GgmlNativeCandidate(directory, GgmlNativeLoader.TensorSharpBuild,
        GgmlNativeLoader.RuntimeIdentifier, variant, backend,
        [new(GgmlNativeLoader.EntryLibraryName, bytes.LongLength, Convert.ToHexStringLower(SHA256.HashData(bytes)))]);
    Require(GgmlNativeLoader.Current == null, "The process starts without a selected bridge.");
    Require(GgmlNativeLoader.Check(candidate) == null, "The candidate passes filesystem validation before loading.");

    if (mode is "reject-variant" or "reject-legacy")
    {
        GgmlNativeCandidate rejected = mode == "reject-variant" ? candidate with { Variant = "wrong-variant" } : candidate;
        GgmlNativeSelection refusal = GgmlNativeLoader.Select([rejected, candidate]);
        Require(refusal.State == GgmlNativeSelectionState.PartiallyInitialized, "A loaded identity refusal requires a fresh process.");
        Require(refusal.Untried.Count == 1 && refusal.Untried[0].Directory == candidate.Directory &&
            refusal.Untried[0].Files.SequenceEqual(candidate.Files), "The next immutable candidate is not loaded after a loaded refusal.");
        Require(refusal.Refusals.Count == 1 && refusal.Refusals[0].Code == GgmlNativeRefusalCodes.LoadFailed, "The identity refusal has a structured diagnostic.");
        if (mode == "reject-variant")
            Require(refusal.Identity?.NativeAbi == GgmlNativeLoader.NativeAbi, "The refusal exercises a real matching-ABI bridge with a different declared variant.");
        else
            Require(refusal.Identity == null, "A bridge without an exact ABI identity is not accepted as a legacy match.");
        try
        {
            GgmlNativeLoader.Select([candidate]);
            throw new InvalidOperationException("A second selection cannot recover a loaded refusal in this process.");
        }
        catch (InvalidOperationException error) when (error.Message.Contains("earlier selection", StringComparison.Ordinal))
        {
        }
        Console.WriteLine(JsonSerializer.Serialize(new { mode, state = refusal.State.ToString(), refused = refusal.Refusals[0].Message, untried = refusal.Untried.Count }));
        return 0;
    }

    GgmlNativeSelection selected;
    GgmlRuntimePlan plan = new(GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.NativeAbi, GgmlNativeLoader.RuntimeIdentifier, [], RequestedBackend: backend);
    if (mode is "default" or "ambiguous-default")
    {
        GgmlRuntimePlan defaultPlan = plan with { Candidates = null, DefaultNuGetProbing = true };
        string searchDirectories = (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") as string) ?? string.Empty;
        if (mode == "ambiguous-default")
        {
            string duplicate = Path.Combine(Path.GetTempPath(), "ggml-duplicate-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(duplicate);
            File.Copy(entry, Path.Combine(duplicate, GgmlNativeLoader.EntryLibraryName));
            AppContext.SetData("NATIVE_DLL_SEARCH_DIRECTORIES", directory + Path.PathSeparator + duplicate + Path.PathSeparator + searchDirectories);
            GgmlNativeLoader.Configure(defaultPlan);
            Require((await GgmlNativeLoader.InitializeAsync()).State == GgmlInitializationState.Unavailable, "Competing default bridges are refused before native loading.");
            Require(GgmlNativeLoader.Current == null, "Ambiguity does not select a native library.");
            Directory.Delete(duplicate, recursive: true);
        }
        AppContext.SetData("NATIVE_DLL_SEARCH_DIRECTORIES", directory + Path.PathSeparator + searchDirectories);
        GgmlNativeLoader.Configure(defaultPlan);
        selected = (await GgmlNativeLoader.InitializeAsync()).Selection;
        Require(selected.Candidate == null && selected.LibraryPath == entry, "Default probing reports its actual absolute native bridge path.");
    }
    else
    {
        GgmlNativeLoader.Configure(plan with { Rid = "wrong-rid" });
        Require((await GgmlNativeLoader.InitializeAsync()).State == GgmlInitializationState.Unsupported, "A wrong-RID plan never loads native code.");
        GgmlNativeLoader.Configure(plan);
        Refused(() => GgmlNativeLoader.AcquireLease(GgmlRuntimeResourceKind.Model), "Resources cannot bind a configured, uninitialized plan.");
        Require((await GgmlNativeLoader.InitializeAsync()).State == GgmlInitializationState.Unavailable, "An explicit empty plan never falls through to default probing.");
        Require(GgmlNativeLoader.State == GgmlRuntimeState.Unavailable, "No-load failures remain reconfigurable.");
        GgmlNativeSelection unavailable = GgmlNativeLoader.Select([candidate with { Files = [candidate.Files[0] with { Sha256 = new string('0', 64) }] }]);
        Require(unavailable.State == GgmlNativeSelectionState.Unavailable, "A hash refusal loads no bridge and permits retry.");
        Require(unavailable.Refusals.Single().Code == GgmlNativeRefusalCodes.HashMismatch, "The hash refusal retains its reason.");
        var mutableCandidates = new List<GgmlNativeCandidate> { candidate with { NativeAbi = new string('0', 64) }, candidate };
        GgmlNativeLoader.Configure(plan with { Candidates = mutableCandidates });
        mutableCandidates.Clear();
        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        try { await GgmlNativeLoader.InitializeAsync(canceled.Token); throw new InvalidOperationException("Canceled initialization wait was accepted."); }
        catch (OperationCanceledException) { }
        object ownerGate = typeof(GgmlNativeLoader).GetField("s_gate", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Task<GgmlInitializationResult> first;
        Task<GgmlInitializationResult> second;
        Monitor.Enter(ownerGate);
        try
        {
            first = GgmlNativeLoader.InitializeAsync();
            second = GgmlNativeLoader.InitializeAsync();
            Require(!first.IsCompleted && !second.IsCompleted, "Concurrent callers wait for the pending initialization.");
            using var waitingCaller = new CancellationTokenSource();
            Task<GgmlInitializationResult> waiting = GgmlNativeLoader.InitializeAsync(waitingCaller.Token);
            waitingCaller.Cancel();
            try { waiting.GetAwaiter().GetResult(); throw new InvalidOperationException("Cancellation did not stop the pending caller wait."); }
            catch (OperationCanceledException) { }
            Require(!first.IsCompleted, "Canceled wait leaves the shared native initialization running.");
            Require(!GgmlNativeLoader.Shutdown().Released, "Shutdown cannot race initialization.");
        }
        finally { Monitor.Exit(ownerGate); }
        GgmlInitializationResult shared = await first;
        Require(ReferenceEquals(shared, await second), "Concurrent callers receive the same initialization result.");
        selected = shared.Selection;
        Require(selected.Refusals.Single().Code == GgmlNativeRefusalCodes.NotSelected, "A different ABI plan is refused before loading.");
    }
    Require(selected.State == GgmlNativeSelectionState.Loaded, "The requested real backend initializes after pre-load refusals.");
    Require(selected.Identity?.NativeAbi == GgmlNativeLoader.NativeAbi && selected.Identity.GgmlCommit == GgmlNativeLoader.GgmlCommit,
        "The actual bridge reports the managed ABI and pinned upstream revision.");
    GgmlInitializationResult ready = await GgmlNativeLoader.InitializeAsync();
    Require(ready.ActualBackend == backend, "The initialized backend matches the requested backend without fallback.");
    int gpuDevices = backend == GgmlBackendType.Cpu ? 0 : GgmlBasicOps.GetGpuDeviceCount(backend);
    string? gpuDescription = gpuDevices > 0 ? GgmlBasicOps.GetGpuDeviceDescription(backend, 0) : null;
    if (backend != GgmlBackendType.Cpu)
        Require(gpuDevices > 0 && !string.IsNullOrWhiteSpace(gpuDescription), "A GPU run requires a detected device and its actual description.");
    GgmlNativeLoader.Configure(ready.Plan);
    Require(ReferenceEquals(await GgmlNativeLoader.InitializeAsync(), ready), "Repeated identical configuration shares the existing owner result.");
    Require(GgmlDeepSeek4Native.NPast(IntPtr.Zero) == 0, "Other interop classes bind to the same selected bridge.");
    Refused(() => GgmlNativeLoader.Configure(plan), "A used runtime cannot reconfigure.");
    Refused(() => GgmlEmbeddingNative.TSGgml_EmbeddingLoad("not-loaded.gguf", "cuda", 0, 1), "A standalone whole-model loader cannot switch the owner's backend.");
    WeakReference foreign = RefuseOtherRuntimeOwner();
    for (int i = 0; i < 10 && foreign.IsAlive; i++) { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }
    Require(!foreign.IsAlive, "The BCL process-owner token does not pin a foreign collectible assembly.");

    var context = new GgmlContext([0], backend);
    MethodInfo enter = typeof(GgmlNativeLoader).GetMethod("EnterNativeCall", BindingFlags.NonPublic | BindingFlags.Static)!;
    using ((IDisposable)enter.Invoke(null, [null, IntPtr.Zero, Type.Missing, Type.Missing])!)
    {
        Refused(context.Dispose, "Context memory cannot be freed during a native call.");
        Refused(() => context.ReleasePooledMemory(), "Pooled memory cannot be trimmed during a native call.");
    }
    var allocator = new GgmlAllocator(context, 0);
    float[] values;
    using (var input = new Tensor(allocator, DType.Float32, 4))
    using (var output = new Tensor(allocator, DType.Float32, 4))
    {
        Require(!GgmlNativeLoader.Shutdown().Released, "Live tensors and contexts refuse teardown.");
        Refused(context.Dispose, "Context disposal refuses live tensor storage.");
        Refused(GgmlBasicOps.Shutdown, "The public basic-ops shutdown cannot bypass leases.");
        Refused(() => GgmlBasicOps.RecreateBackend(), "Backend recreation cannot invalidate live contexts/tensors.");
        input.SetElementsAsFloat([1, 2, 3, 4]);
        GgmlBasicOps.Add(output, input, 5);
        values = output.GetElementsAsFloat(4);
        Require(values.SequenceEqual(new float[] { 6, 7, 8, 9 }), "A real GGML operation returns the expected tensor values on the requested backend.");
    }
    Require(!GgmlNativeLoader.Shutdown().Released, "An active context refuses shutdown even after its tensors are disposed.");
    context.Dispose();
    using (var parallelContext = new GgmlContext([0], backend))
    using (var group = new GgmlTensorParallelGroup(parallelContext))
    using (var started = new ManualResetEventSlim())
    using (var finish = new ManualResetEventSlim())
    {
        Task work = Task.Run(() => group.RunPerRank(_ => { started.Set(); finish.Wait(); }));
        Require(started.Wait(TimeSpan.FromSeconds(10)), "Rank work starts in the test worker.");
        try
        {
            Refused(group.Dispose, "A group cannot dispose while rank work remains active.");
            Require(!GgmlNativeLoader.Shutdown().Released, "A group and its context retain the runtime until worker teardown.");
        }
        finally { finish.Set(); }
        await work;
    }
    Refused(() => new Tensor(allocator, DType.Float32, 1), "A disposed context cannot allocate storage.");
    using (GgmlNativeLoader.AcquireLease(GgmlRuntimeResourceKind.Model))
        Require(!GgmlNativeLoader.Shutdown().Released, "Model leases independently refuse teardown.");
    IntPtr allocation = GgmlBasicOps.AlignedAlloc(128);
    Require(allocation != IntPtr.Zero && !GgmlNativeLoader.Shutdown().Released, "Owned native allocations independently refuse teardown.");
    GgmlBasicOps.AlignedFree(allocation);
    Refused(() => GgmlBasicOps.AlignedFree(allocation), "A freed native handle is not freed twice.");
    using ((IDisposable)enter.Invoke(null, [null, IntPtr.Zero, Type.Missing, Type.Missing])!)
        Require(!GgmlNativeLoader.Shutdown().Released, "An active native call independently refuses teardown.");
    Type native = typeof(GgmlNativeLoader).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!;
    Require((bool)native.GetField("s_earlyTunablesApplied", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!, "Import binding applies tunables after identity selection.");
    GgmlNativeShutdownResult shutdown = GgmlNativeLoader.Shutdown();
    Require(shutdown.Released && GgmlNativeLoader.Shutdown().Released, "Teardown after disposing the test tensors is idempotent: " + shutdown.Diagnostic);
    Require(GgmlNativeLoader.State == GgmlRuntimeState.Stopped, "Successful shutdown is terminal.");
    Refused(() => GgmlBasicOps.AlignedAlloc(128), "A cached native import cannot execute after shutdown.");
    Refused(() => GgmlNativeLoader.InitializeAsync(), "Initialization cannot follow terminal shutdown.");
    Console.WriteLine(JsonSerializer.Serialize(new { mode, state = selected.State.ToString(), backend = ready.ActualBackend.ToString(), gpuDevices, gpuDescription, selected.Identity, values, shutdown.Released }));
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(error.ToString());
    return 1;
}

static void Require(bool condition, string message)
{
    if (!condition)
        throw new InvalidOperationException(message);
}

static void Refused(Action action, string message)
{
    try { action(); }
    catch (InvalidOperationException) { return; }
    throw new InvalidOperationException(message);
}

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference RefuseOtherRuntimeOwner()
{
    var context = new AssemblyLoadContext("foreign-ggml-owner", isCollectible: true);
    Assembly assembly = context.LoadFromAssemblyPath(typeof(GgmlNativeLoader).Assembly.Location);
    Type loader = assembly.GetType(typeof(GgmlNativeLoader).FullName!)!;
    object plan = Activator.CreateInstance(assembly.GetType(typeof(GgmlRuntimePlan).FullName!)!,
        GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.NativeAbi, GgmlNativeLoader.RuntimeIdentifier,
        null, true, Enum.Parse(assembly.GetType(typeof(GgmlBackendType).FullName!)!, "Cpu"))!;
    loader.GetMethod("Configure")!.Invoke(null, [plan]);
    Task task = (Task)loader.GetMethod("InitializeAsync")!.Invoke(null, [CancellationToken.None])!;
    task.GetAwaiter().GetResult();
    object result = task.GetType().GetProperty("Result")!.GetValue(task)!;
    Require(result.GetType().GetProperty("State")!.GetValue(result)!.ToString() == "RequiresProcessRestart", "A second managed runtime cannot load a native bridge into the owned process.");
    var weak = new WeakReference(context);
    context.Unload();
    return weak;
}
