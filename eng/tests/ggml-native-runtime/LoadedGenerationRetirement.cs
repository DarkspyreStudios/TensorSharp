#nullable enable

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;

internal static class LoadedGenerationRetirement
{
    internal static int Run(string mode, string directory, string variant)
    {
        RetirementEvidence evidence = LoadUseAndUnload(mode, directory, variant);
        for (int attempt = 0; attempt < 30 && evidence.Roots.Any(root => root.IsAlive); attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Thread.Sleep(10);
        }
        if (evidence.Roots.Any(root => root.IsAlive))
            throw new InvalidOperationException("A loaded and stopped foreign GGML generation remains rooted after its invocation frame returns.");
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            mode,
            evidence.NativeEvidence,
            collectedRoots = evidence.Roots.Length,
            actualModelQualification = "not-run",
            multiDeviceWorkers = "not-run"
        }));
        return 0;
    }

    private sealed record RetirementEvidence(WeakReference[] Roots, string NativeEvidence);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static RetirementEvidence LoadUseAndUnload(string mode, string directory, string variant)
    {
        var context = new ForeignGeneration(Path.GetDirectoryName(typeof(ForeignNativeRetirement).Assembly.Location)!);
        Assembly fixture = context.LoadFromAssemblyPath(typeof(ForeignNativeRetirement).Assembly.Location);
        Type runner = fixture.GetType(nameof(ForeignNativeRetirement), throwOnError: true)!;
        MethodInfo run = runner.GetMethod(nameof(ForeignNativeRetirement.Run))!;
        string nativeEvidence;
        try
        {
            nativeEvidence = (string)run.Invoke(null, [mode, directory, variant])!;
        }
        catch (TargetInvocationException error)
        {
            // Keep foreign exceptions out of the permanent verification frame.
            throw new InvalidOperationException("Foreign GGML retirement fixture failed: " + error.InnerException);
        }
        Assembly backend = context.Assemblies.Single(assembly => assembly.GetName().Name == "TensorSharp.Backends.GGML");
        Assembly core = context.Assemblies.Single(assembly => assembly.GetName().Name == "TensorSharp.Core");
        WeakReference[] roots = [new(context), new(fixture), new(backend), new(core)];
        context.Unload();
        return new(roots, nativeEvidence);
    }

    private sealed class ForeignGeneration(string directory) : AssemblyLoadContext("loaded-ggml-retirement-" + Guid.NewGuid(), isCollectible: true)
    {
        protected override Assembly? Load(AssemblyName name)
        {
            if (name.Name is not ("ggml-native-runtime" or "TensorSharp.Core" or "TensorSharp.Backends.GGML"))
                return null;
            string path = Path.Combine(directory, name.Name + ".dll");
            if (!File.Exists(path)) throw new FileNotFoundException("The foreign fixture requires its real private dependency.", path);
            return LoadFromAssemblyPath(path);
        }
    }
}

public static class ForeignNativeRetirement
{
    public static string Run(string mode, string directory, string variant)
    {
        AssemblyLoadContext foreign = AssemblyLoadContext.GetLoadContext(typeof(ForeignNativeRetirement).Assembly)!;
        Require(foreign.IsCollectible && foreign != AssemblyLoadContext.Default, "The fixture generation is genuinely foreign and collectible.");
        foreach (Type type in new[] { typeof(ForeignNativeRetirement), typeof(GgmlNativeLoader), typeof(GgmlContext), typeof(GgmlAllocator),
            typeof(GgmlTensorParallelGroup), typeof(Tensor), typeof(OpRegistry) })
            Require(AssemblyLoadContext.GetLoadContext(type.Assembly) == foreign, "All real fixture/backend/Core types belong to the foreign generation: " + type.FullName);

        GgmlBackendType backend = mode.EndsWith("metal", StringComparison.Ordinal) ? GgmlBackendType.Metal : GgmlBackendType.Cpu;
        bool defaultProbing = mode.Contains("default", StringComparison.Ordinal);
        string entry = Path.Combine(directory, GgmlNativeLoader.EntryLibraryName);
        byte[] bytes = File.ReadAllBytes(entry);
        string bridgeHash = Convert.ToHexStringLower(SHA256.HashData(bytes));
        var candidate = new GgmlNativeCandidate(directory, GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.RuntimeIdentifier,
            variant, backend, [new(GgmlNativeLoader.EntryLibraryName, bytes.LongLength, bridgeHash)]);
        Require(GgmlNativeLoader.Current == null, "The foreign owner starts without native selection.");
        Require(GgmlNativeLoader.Check(candidate) == null, "The real candidate passes exact staged-file validation.");
        object? previousSearch = AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES");
        try
        {
            if (defaultProbing) AppContext.SetData("NATIVE_DLL_SEARCH_DIRECTORIES", directory);
            GgmlNativeLoader.Configure(new(GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.NativeAbi, GgmlNativeLoader.RuntimeIdentifier,
                defaultProbing ? null : [candidate], DefaultNuGetProbing: defaultProbing, RequestedBackend: backend));
            GgmlInitializationResult initialized = GgmlNativeLoader.InitializeAsync().GetAwaiter().GetResult();
            Require(initialized.State == GgmlInitializationState.Ready && initialized.ActualBackend == backend,
                "The real foreign owner initializes the requested backend without fallback: " + initialized.Diagnostic);
            Require(initialized.Selection.LibraryPath == entry && initialized.ActualNativeAbi == GgmlNativeLoader.NativeAbi &&
                initialized.Selection.Identity?.GgmlCommit == GgmlNativeLoader.GgmlCommit && initialized.ActualVariant == variant,
                "The real selection reports its absolute bridge and exact source/ABI identity.");
            Require((initialized.Selection.Candidate == null) == defaultProbing, "The selection preserves explicit/default plan provenance.");
            Require(ProcessExitHooked(), "A live selected/default owner registers its process-exit cleanup.");
            GgmlNativeLoader.Configure(initialized.Plan);
            Require(ReferenceEquals(initialized, GgmlNativeLoader.InitializeAsync().GetAwaiter().GetResult()), "Identical initialization is idempotent.");
            int gpuDevices = backend == GgmlBackendType.Cpu ? 0 : GgmlBasicOps.GetGpuDeviceCount(backend);
            string gpuDescription = gpuDevices == 0 ? "none" : GgmlBasicOps.GetGpuDeviceDescription(backend, 0);
            Require(backend == GgmlBackendType.Cpu || gpuDevices > 0 && !string.IsNullOrWhiteSpace(gpuDescription),
                "A Metal case requires actual detected hardware.");

            var context = new GgmlContext([0], backend);
            var allocator = new GgmlAllocator(context, 0);
            float[] values;
            using (var input = new Tensor(allocator, DType.Float32, 4))
            using (var output = new Tensor(allocator, DType.Float32, 4))
            {
                Busy("Live foreign context/tensors retain native ownership.");
                Refused(context.Dispose, "Live tensor storage prevents context disposal.");
                input.SetElementsAsFloat([1, 2, 3, 4]);
                GgmlBasicOps.Add(output, input, 5);
                values = output.GetElementsAsFloat(4);
                Require(values.SequenceEqual(new float[] { 6, 7, 8, 9 }), "Real foreign GGML arithmetic produces the expected values.");
                GgmlBasicOps.HostReadBarrier();
            }
            Busy("Context ownership outlives disposed tensors.");
            context.Dispose();
            using (var rankContext = new GgmlContext([0], backend))
            using (var group = new GgmlTensorParallelGroup(rankContext))
            using (var started = new ManualResetEventSlim())
            using (var finish = new ManualResetEventSlim())
            {
                Require(group.Degree == 1, "The fixture proves single-rank work, not multi-device workers.");
                Task work = Task.Run(() => group.RunPerRank(_ => { started.Set(); finish.Wait(); }));
                try
                {
                    Require(started.Wait(TimeSpan.FromSeconds(10)), "The foreign rank-one callback starts.");
                    Refused(group.Dispose, "Active rank-one work prevents group disposal.");
                    Busy("Pending rank-one work retains the foreign group and context.");
                }
                finally { finish.Set(); }
                Require(work.Wait(TimeSpan.FromSeconds(10)), "The pending rank-one task drains before teardown.");
                work.GetAwaiter().GetResult();
            }
            using (GgmlNativeLoader.AcquireLease(GgmlRuntimeResourceKind.Model))
                Busy("A Model-kind lease prevents teardown; this is not an actual loaded-model qualification.");
            IntPtr allocation = GgmlBasicOps.AlignedAlloc(128);
            Require(allocation != IntPtr.Zero, "The selected bridge creates a real aligned allocation.");
            Busy("A real native allocation retains foreign ownership.");
            GgmlBasicOps.AlignedFree(allocation);
            MethodInfo enter = typeof(GgmlNativeLoader).GetMethod("EnterNativeCall", BindingFlags.NonPublic | BindingFlags.Static)!;
            using ((IDisposable)enter.Invoke(null, [null, IntPtr.Zero, Type.Missing, Type.Missing])!)
                Busy("A manually held native-call lease prevents teardown; no blocked P/Invoke is claimed.");
            GgmlNativeShutdownResult shutdown = GgmlNativeLoader.Shutdown();
            Require(shutdown.Released && GgmlNativeLoader.State == GgmlRuntimeState.Stopped, "Guarded teardown succeeds after all owned work drains: " + shutdown.Diagnostic);
            Require(!ProcessExitHooked(), "Successful native teardown removes the external process-exit root.");
            Require(ReferenceEquals(shutdown, GgmlNativeLoader.Shutdown()), "Guarded shutdown is idempotent.");
            Refused(() => GgmlBasicOps.AlignedAlloc(128), "Cached imports cannot execute after terminal shutdown.");
            Refused(() => GgmlNativeLoader.InitializeAsync(), "The stopped owner cannot initialize again.");
            return $"backend={backend};variant={variant};rid={GgmlNativeLoader.RuntimeIdentifier};abi={GgmlNativeLoader.NativeAbi};" +
                $"source={initialized.Selection.Identity!.SourceCommit};ggml={GgmlNativeLoader.GgmlCommit};bridgeSha256={bridgeHash};" +
                $"gpuDevices={gpuDevices};gpuDescription={gpuDescription};values={string.Join(',', values)};state=Stopped;released=true;rankOneWork=drained";
        }
        finally { AppContext.SetData("NATIVE_DLL_SEARCH_DIRECTORIES", previousSearch); }
    }

    private static bool ProcessExitHooked() => (bool)typeof(GgmlNativeLoader)
        .GetField("s_processExitHooked", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;

    private static void Busy(string message)
    {
        Require(!GgmlNativeLoader.Shutdown().Released, message);
        Require(GgmlNativeLoader.State == GgmlRuntimeState.Ready && ProcessExitHooked(), "Busy teardown retains a Ready owner and its cleanup hook.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static void Refused(Action action, string message)
    {
        try { action(); }
        catch (InvalidOperationException) { return; }
        throw new InvalidOperationException(message);
    }
}
