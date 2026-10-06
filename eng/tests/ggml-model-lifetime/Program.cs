#nullable enable

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length != 3 || args[0] is not ("normal" or "refusal" or "phase-order" or "borrowed-tp" or "constructor-matrix" or "partial-weight" or
    "derived-cleanup-failure" or "base-cleanup-failure" or "local-cleanup-failure" or "dispose-cleanup-failure" or
    "raw-quantized-read-refusal" or "stacked-quantized-read-refusal" or "stacked-owner-insertion-refusal" or "stacked-partial-view-refusal" or "bonsai-unregister-refusal" or
    "local-quantized-transfer" or "quantized-fusion-ownership" or "local-bonsai-transfer-refusal" or "quantized-fusion-source-refusal" or "bonsai-registration-refusal" or
    "observe-tp-column-owner" or "observe-tp-generic-column" or "observe-tp-generic-row" or "observe-tp-generic-copy" or "observe-tp-concatenated" or "observe-tp-separate" or
    "observe-tp-concatenated-bias" or "observe-tp-separate-bias" or "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order" or
    "observe-tp-sync-retirement" or "tp-sync-preflight-refusal" or
    "tp-quantized-copy-row" or "tp-quantized-copy-concatenated" or "tp-quantized-requantize-separate" or
    "tp-broadcast-ownership" or "tp-broadcast-source-disposal" or "tp-broadcast-partial-copy" or "tp-broadcast-second-copy" or
    "tp-broadcast-temporary-retirement" or "tp-broadcast-source-cleanup-refusal" or "tp-broadcast-rollback-refusal" or "tp-broadcast-temporary-cleanup-refusal" or
    "vision-normal" or "vision-mismatch" or "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal" or
    "execution-cleanup-failure" or "execution-worker-cleanup-failure" or "execution-dispatch-cleanup-failure" or "vision-execution-cleanup-failure") || args[1] is not ("cpu" or "metal"))
    throw new ArgumentException("Expected a model-lifetime mode, cpu|metal and an absolute bridge directory.");

Retirement.Run(args[0], args[1], Path.GetFullPath(args[2]));

internal static class Retirement
{
    private sealed record Evidence(WeakReference[] Roots, string[] Names, string Report);

    internal static void Run(string mode, string backend, string directory)
    {
        Evidence evidence = LoadAndUnload(mode, backend, directory);
        for (int attempt = 0; attempt < 30 && evidence.Roots.Any(root => root.IsAlive); attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Thread.Sleep(10);
        }
        string[] retained = evidence.Names.Where((_, index) => evidence.Roots[index].IsAlive).ToArray();
        bool unsafeCleanup = (mode.EndsWith("cleanup-failure", StringComparison.Ordinal) && mode != "base-cleanup-failure") || mode is "bonsai-unregister-refusal" or "local-bonsai-transfer-refusal" or "quantized-fusion-source-refusal" or "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order" or "observe-tp-sync-retirement" or "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal" or "tp-broadcast-source-cleanup-refusal" or "tp-broadcast-rollback-refusal" or "tp-broadcast-temporary-cleanup-refusal";
        if (!unsafeCleanup && retained.Length != 0)
            throw new InvalidOperationException("Disposed real model generation remains rooted: " + string.Join(",", retained));
        if (unsafeCleanup && retained.Length != evidence.Roots.Length)
            throw new InvalidOperationException("A terminal cleanup failure must retain the complete unsafe foreign generation.");
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            mode,
            backend,
            evidence.Report,
            retained,
            actualModel = mode.StartsWith("observe-tp-", StringComparison.Ordinal) || mode.StartsWith("tp-quantized-", StringComparison.Ordinal) || mode.StartsWith("tp-broadcast-", StringComparison.Ordinal) || mode is "tp-view-unregister-refusal" or "tp-sync-preflight-refusal"
                ? "generated ownership-only sources, logical ranks on one real GGML context, no forward" :
                mode is "vision-normal" or "vision-mismatch" or "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal" or "vision-execution-cleanup-failure"
                ? "generated one-layer F32 DeepSeek41 text and real vision companion, lifetime only" :
                mode is "raw-quantized-read-refusal" or "stacked-quantized-read-refusal" or
                "stacked-owner-insertion-refusal" or "stacked-partial-view-refusal" or "quantized-fusion-ownership" or "quantized-fusion-source-refusal" ? "generated Q4_0 ownership-only GGUF, no forward" :
                mode is "bonsai-unregister-refusal" or "bonsai-registration-refusal" or "local-quantized-transfer" or "local-bonsai-transfer-refusal" ?
                    "generated F32 metadata context and Q2_0 owner, no forward" : "generated tiny F32 HunyuanDense",
            pretrainedModelQualification = "not-run",
            wholeNativeExecutor = mode switch
            {
                "vision-normal" => "real-text-load/vision-attach/dispose-lifetime-only;forward-not-run",
                "vision-mismatch" => "real-text-load/reset/dispose-lifetime-only;forward-not-run",
                "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal"
                    => "real-text-load/managed-cleanup-refusal/retention-lifetime-only;forward-not-run",
                "vision-execution-cleanup-failure" => "real-text-load/vision-attach/managed-storage-cleanup-refusal/admission-fences;forward-not-run",
                _ => "not-run"
            },
            multiDeviceWorkers = "not-run"
        }));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Evidence LoadAndUnload(string mode, string backend, string directory)
    {
        string assemblyPath = typeof(ForeignModelLifetime).Assembly.Location;
        var context = new Generation(Path.GetDirectoryName(assemblyPath)!);
        Assembly fixture = context.LoadFromAssemblyPath(assemblyPath);
        string report;
        try
        {
            report = (string)fixture.GetType(nameof(ForeignModelLifetime), true)!
                .GetMethod(nameof(ForeignModelLifetime.Run))!.Invoke(null, [mode, backend, directory])!;
        }
        catch (TargetInvocationException error)
        {
            throw new InvalidOperationException("Foreign real-model fixture failed: " + error.InnerException);
        }
        Assembly[] assemblies = context.Assemblies.ToArray();
        var roots = new List<WeakReference> { new(context) };
        var names = new List<string> { "collectible ALC" };
        foreach (Assembly assembly in assemblies)
        {
            roots.Add(new(assembly));
            names.Add(assembly.GetName().Name!);
        }
        context.Unload();
        return new(roots.ToArray(), names.ToArray(), report);
    }

    private sealed class Generation(string directory) : AssemblyLoadContext("real-model-lifetime-" + Guid.NewGuid(), true)
    {
        protected override Assembly? Load(AssemblyName name)
        {
            if (name.Name != "ggml-model-lifetime" && !name.Name!.StartsWith("TensorSharp.", StringComparison.Ordinal)) return null;
            string path = Path.Combine(directory, name.Name + ".dll");
            if (!File.Exists(path)) throw new FileNotFoundException("The foreign model requires its actual private dependency.", path);
            return LoadFromAssemblyPath(path);
        }
    }
}

public static partial class ForeignModelLifetime
{
    public static string Run(string mode, string backendName, string directory)
    {
        AssemblyLoadContext generation = AssemblyLoadContext.GetLoadContext(typeof(ForeignModelLifetime).Assembly)!;
        foreach (Type type in new[] { typeof(ForeignModelLifetime), typeof(ModelBase), typeof(HunyuanDenseModel),
            typeof(GgufFile), typeof(TensorSharp.Tensor), typeof(GgmlNativeLoader) })
            Require(generation.IsCollectible && AssemblyLoadContext.GetLoadContext(type.Assembly) == generation,
                "Fixture, Models, Runtime, Core and GGML all belong to the foreign generation.");
        GgmlBackendType nativeBackend = backendName == "cpu" ? GgmlBackendType.Cpu : GgmlBackendType.Metal;
        BackendType backend = backendName == "cpu" ? BackendType.GgmlCpu : BackendType.GgmlMetal;
        string entry = Path.Combine(directory, GgmlNativeLoader.EntryLibraryName);
        byte[] bytes = File.ReadAllBytes(entry);
        string hash = Convert.ToHexStringLower(SHA256.HashData(bytes));
        var candidate = new GgmlNativeCandidate(directory, GgmlNativeLoader.TensorSharpBuild,
            GgmlNativeLoader.RuntimeIdentifier, "metal", nativeBackend,
            [new(GgmlNativeLoader.EntryLibraryName, bytes.LongLength, hash)]);
        GgmlNativeLoader.Configure(new(GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.NativeAbi,
            GgmlNativeLoader.RuntimeIdentifier, [candidate], RequestedBackend: nativeBackend));
        GgmlInitializationResult initialized = GgmlNativeLoader.InitializeAsync().GetAwaiter().GetResult();
        Require(initialized.State == GgmlInitializationState.Ready && initialized.ActualBackend == nativeBackend &&
            initialized.ActualNativeAbi == GgmlNativeLoader.NativeAbi && initialized.Selection.LibraryPath == entry,
            "The real pinned bridge initializes the exact requested backend/ABI.");

        string scratch = Path.Combine(Environment.GetEnvironmentVariable("TMPDIR")
            ?? throw new InvalidOperationException("TMPDIR must name repository scratch."), "model-lifetime-" + Guid.NewGuid());
        Directory.CreateDirectory(scratch);
        string modelPath = Path.Combine(scratch, "tiny-hunyuan-dense.gguf");
        try
        {
            if (mode is "execution-cleanup-failure" or "execution-worker-cleanup-failure" or "execution-dispatch-cleanup-failure" or "vision-execution-cleanup-failure")
            {
                string executionWork = ExerciseExecutionFences(mode, scratch, backend);
                GgmlNativeShutdownResult executionShutdown = GgmlNativeLoader.Shutdown();
                Require(!executionShutdown.Released && ModelLeaseCount() == 1 && GgmlNativeLoader.State == GgmlRuntimeState.Ready,
                    "Controlled model ownership refusal retains its actual owners and refuses shutdown without poisoning the runtime.");
                return $"bridgeSha256={hash};nativeAbi={GgmlNativeLoader.NativeAbi};work={executionWork};shutdownReleased={executionShutdown.Released};diagnostic={executionShutdown.Diagnostic}";
            }
            if (mode is "vision-normal" or "vision-mismatch" or "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal")
            {
                string visionWork = ExerciseVisionLifetime(mode, scratch, backend);
                GgmlNativeShutdownResult visionShutdown = GgmlNativeLoader.Shutdown();
                bool refused = mode is "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal";
                Require(refused ? !visionShutdown.Released && ModelLeaseCount() == 1 : visionShutdown.Released && ModelLeaseCount() == 0,
                    "Actual vision lifetime shutdown matches the explicit-clean or controlled-terminal observation.");
                return $"bridgeSha256={hash};nativeAbi={GgmlNativeLoader.NativeAbi};work={visionWork};shutdownReleased={visionShutdown.Released};diagnostic={visionShutdown.Diagnostic}";
            }
            new DenseDecoderSyntheticModelBuilder
            {
                Architecture = "hunyuan-dense",
                PreTokenizer = "hunyuan-dense",
                IncludeMistralControlTokens = false,
                IncludeQkNorms = true,
                RopeBase = 10000f,
                ValueHeadDim = mode is "refusal" or "borrowed-tp" ? 8 : DenseDecoderSyntheticModelBuilder.HeadDim,
                IncludeTokenizer = mode != "constructor-matrix"
            }.Write(modelPath);
            if (mode is "raw-quantized-read-refusal" or "stacked-quantized-read-refusal" or
                "stacked-owner-insertion-refusal" or "stacked-partial-view-refusal" or "quantized-fusion-ownership" or "quantized-fusion-source-refusal" or
                "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order")
                WriteQuantizedOwnershipFixture(modelPath, mode is not ("raw-quantized-read-refusal" or "quantized-fusion-ownership" or "quantized-fusion-source-refusal" or "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order"));
            string modelHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath)));
            string work = mode switch
            {
                "normal" => UseModel(modelPath, backend),
                "borrowed-tp" => RefuseBorrowedGroup(modelPath, backend),
                "phase-order" => RefuseWithPendingCompute(modelPath, backend),
                "constructor-matrix" => RefuseConstructorMatrix(modelPath, backend),
                "partial-weight" => RefuseWeightRead(modelPath, backend, failCleanup: false),
                "local-cleanup-failure" => RefuseWeightRead(modelPath, backend, failCleanup: true),
                "derived-cleanup-failure" => RefuseDerivedCleanup(modelPath, backend),
                "base-cleanup-failure" => RefuseBaseCleanup(modelPath, backend),
                "dispose-cleanup-failure" => RefuseNormalDispose(modelPath, backend),
                "raw-quantized-read-refusal" or "stacked-quantized-read-refusal" or
                    "stacked-owner-insertion-refusal" or "stacked-partial-view-refusal" => ExerciseQuantizedTransfer(mode, modelPath, backend),
                "bonsai-unregister-refusal" => RefuseBonsaiUnregister(modelPath, backend),
                "local-quantized-transfer" => ExerciseLocalQuantizedTransfer(modelPath, backend),
                "quantized-fusion-ownership" => ExerciseQuantizedFusion(modelPath, backend),
                "local-bonsai-transfer-refusal" => ExerciseLocalQuantizedTransfer(modelPath, backend, refuseCleanup: true),
                "quantized-fusion-source-refusal" => ExerciseQuantizedFusion(modelPath, backend, refuseSource: true),
                "bonsai-registration-refusal" => ExerciseBonsaiRegistration(modelPath, backend),
                "observe-tp-column-owner" => ObserveTpColumnOwner(modelPath, backend),
                "observe-tp-generic-column" or "observe-tp-generic-row" or "observe-tp-generic-copy" or "observe-tp-concatenated" or "observe-tp-separate" or
                    "observe-tp-concatenated-bias" or "observe-tp-separate-bias" => ObserveTpPartialShard(mode, modelPath, backend),
                "tp-view-unregister-refusal" => ExerciseTpViewRefusal(modelPath, backend, poisonLate: false),
                "observe-tp-view-cleanup-order" => ExerciseTpViewRefusal(modelPath, backend, poisonLate: true),
                "observe-tp-sync-retirement" => ExerciseTpSynchronization(modelPath, backend, preflight: false),
                "tp-sync-preflight-refusal" => ExerciseTpSynchronization(modelPath, backend, preflight: true),
                "tp-quantized-copy-row" or "tp-quantized-copy-concatenated" or "tp-quantized-requantize-separate" => ExerciseTpQuantizedCopies(mode, modelPath, backend),
                "tp-broadcast-ownership" or "tp-broadcast-source-disposal" or "tp-broadcast-partial-copy" or "tp-broadcast-second-copy" or
                    "tp-broadcast-temporary-retirement" => ExerciseGgmlBroadcast(mode, modelPath, backend),
                "tp-broadcast-source-cleanup-refusal" or "tp-broadcast-rollback-refusal" or "tp-broadcast-temporary-cleanup-refusal" => ExerciseBroadcastCleanupRefusal(mode, modelPath, backend),
                _ => RefuseModel(modelPath, backend)
            };
            // Finalization is diagnostic here. A failed construction must not require GC to retire its model lease.
            bool unsafeCleanup = (mode.EndsWith("cleanup-failure", StringComparison.Ordinal) && mode != "base-cleanup-failure") || mode is "bonsai-unregister-refusal" or "local-bonsai-transfer-refusal" or "quantized-fusion-source-refusal" or "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order" or "observe-tp-sync-retirement" or "tp-broadcast-source-cleanup-refusal" or "tp-broadcast-rollback-refusal" or "tp-broadcast-temporary-cleanup-refusal";
            for (int attempt = 0; unsafeCleanup && attempt < 10; attempt++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            int modelLeases = ModelLeaseCount();
            if ((mode.EndsWith("cleanup-failure", StringComparison.Ordinal) && mode != "base-cleanup-failure")) VerifyRetainedOwner(mode);
            if (mode is "bonsai-unregister-refusal" or "local-bonsai-transfer-refusal" or "quantized-fusion-source-refusal") VerifyQuantizedRetention(mode);
            if (mode is "tp-view-unregister-refusal" or "observe-tp-view-cleanup-order" or "observe-tp-sync-retirement") VerifyTpRetention(mode);
            if (mode is "tp-broadcast-source-cleanup-refusal" or "tp-broadcast-rollback-refusal" or "tp-broadcast-temporary-cleanup-refusal") VerifyBroadcastRetention();
            GgmlNativeShutdownResult shutdown = GgmlNativeLoader.Shutdown();
            if (!unsafeCleanup) Require(shutdown.Released && modelLeases == 0,
                (mode.StartsWith("observe-tp-", StringComparison.Ordinal)
                    ? "Diagnostic TP observation leaves no native leases at shutdown; GC is not explicit ownership cleanup: "
                    : "Explicit real model cleanup drains guarded shutdown without GC: ") + shutdown.Diagnostic);
            else Require(!shutdown.Released && modelLeases == 1, "Terminal unsafe ownership refuses shutdown and retains its model lease: " + shutdown.Diagnostic);
            return $"bridgeSha256={hash};nativeAbi={GgmlNativeLoader.NativeAbi};modelSha256={modelHash};" +
                $"work={work};modelLeases={modelLeases};shutdownReleased={shutdown.Released};diagnostic={shutdown.Diagnostic}";
        }
        finally
        {
            File.Delete(modelPath);
            File.Delete(modelPath + ".empty");
            Directory.Delete(scratch);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string UseModel(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        Require(model is HunyuanDenseModel && model.Config.NumLayers == 2, "The actual complete two-layer architecture loads.");
        Require(!GgmlNativeLoader.Shutdown().Released, "Guarded shutdown refuses a real live model.");
        float[] prefill = model.ForwardRefill([65, 66, 67]);
        Require(prefill.Length == 256 && prefill.All(float.IsFinite), "The complete real native-backed prefill produces finite logits.");
        float[] decode = model.Forward([68]);
        Require(decode.Length == 256 && decode.All(float.IsFinite), "The complete real native-backed decode produces finite logits.");
        return "full-two-layer-prefill-and-decode";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseModel(string path, BackendType backend)
    {
        try { _ = new HunyuanDenseModel(path, backend); }
        catch (NotSupportedException error) when (error.Message.Contains("equal key/value head dims", StringComparison.Ordinal))
        {
            return error.GetType().Name + ":" + error.Message;
        }
        throw new InvalidOperationException("The controlled production constructor must refuse unequal head dimensions.");
    }

    private static int ModelLeaseCount()
    {
        var resources = (IDictionary)typeof(GgmlNativeLoader).GetField("s_resources", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        return resources.Values.Cast<object>().Count(resource =>
            (GgmlRuntimeResourceKind)resource.GetType().GetField("Kind", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(resource)! == GgmlRuntimeResourceKind.Model);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseConstructorMatrix(string path, BackendType backend)
    {
        var refused = new List<string>();
        foreach (Type type in typeof(ModelBase).Assembly.GetTypes().Where(type => !type.IsAbstract && type.IsSubclassOf(typeof(ModelBase))))
        {
            ConstructorInfo constructor = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Single(ctor => ctor.GetParameters() is [{ ParameterType: var first }, { ParameterType: var second }, ..]
                    && first == typeof(string) && second == typeof(BackendType));
            ParameterInfo[] parameters = constructor.GetParameters();
            object?[] arguments = parameters.Select((parameter, index) => index switch
            {
                0 => (object)path,
                1 => backend,
                _ => parameter.DefaultValue
            }).ToArray();
            try
            {
                var model = (ModelBase)constructor.Invoke(arguments);
                model.Dispose();
                throw new InvalidOperationException("A malformed model unexpectedly completed construction: " + type.Name);
            }
            catch (TargetInvocationException error)
            {
                Require(error.InnerException is not AggregateException, "The malformed model rollback must itself succeed: " + type.Name + ": " + error.InnerException);
                Require(ModelLeaseCount() == 0, "Each failed constructor releases its model lease without GC: " + type.Name);
                Require(RuntimeResourceCount() == 0, "Each failed constructor explicitly releases all context/tensor leases: " + type.Name);
                refused.Add(type.Name + ":" + error.InnerException!.GetType().Name);
            }
        }
        Require(refused.Count == 18, "Every concrete direct/indirect ModelBase implementation is included.");
        return string.Join(",", refused);
    }

    private static int RuntimeResourceCount()
    {
        var resources = (IDictionary)typeof(GgmlNativeLoader).GetField("s_resources", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        return resources.Count;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseWeightRead(string path, BackendType backend, bool failCleanup)
    {
        ReadFailingQwen3.SourcePath = path;
        ReadFailingQwen3.FailCleanup = failCleanup;
        ReadFailingQwen3.Last = null;
        try { _ = new ReadFailingQwen3(path, backend); }
        catch (Exception error) when (error is EndOfStreamException or AggregateException)
        {
            Require(ReadFailingQwen3.Last!.TryGetTarget(out ReadFailingQwen3? model), "The construction frame still exposes the inspected model.");
            if (failCleanup)
            {
                Require(error is TensorSharp.NativeConstructionCleanupException aggregate && !aggregate.Cleanup.IsReleased &&
                    aggregate.InnerExceptions[0] is AggregateException work && work.InnerExceptions[0] is EndOfStreamException &&
                    work.InnerExceptions[1] is InvalidOperationException &&
                    ReferenceEquals(work.InnerExceptions[1], aggregate.InnerExceptions[1]),
                    "The local transfer records both the actual read failure and injected storage cleanup failure: " + error);
                Require(model!.LoadedWeights == 1 && RuntimeResourceCount() == 4,
                    "The first owned weight, unregistered second storage, context and Model lease remain retained.");
            }
            else
            {
                Require(error is EndOfStreamException && error.StackTrace!.Contains("LoadWeights", StringComparison.Ordinal),
                    "Clean rollback preserves the actual file read exception and its original loading stack.");
                Require(model!.LoadedWeights == 0 && RuntimeResourceCount() == 0,
                    "Partial dictionary weights and local transfer allocation release explicitly without GC.");
            }
            Require(ReadFailingQwen3.VirtualDisposeCalls == 0, "Construction rollback never invokes subclass virtual Dispose.");
            return failCleanup ? "actual-F32-read-failure;injected-managed-storage-cleanup-refusal" : "actual-F32-read-failure-after-first-owned-weight";
        }
        throw new InvalidOperationException("The controlled actual file read must fail.");
    }

    private sealed class ReadFailingQwen3 : Qwen3Model
    {
        internal static string SourcePath = string.Empty;
        internal static bool FailCleanup;
        internal static WeakReference<ReadFailingQwen3>? Last;
        internal static int VirtualDisposeCalls;
        internal int LoadedWeights => _weights.Count;
        public ReadFailingQwen3(string path, BackendType backend) : base(path, backend) { }

        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            if (_weights.Count == 1)
            {
                Last = new(this);
                File.WriteAllBytes(SourcePath + ".empty", []);
                FieldInfo stream = typeof(GgufFile).GetField("_stream", BindingFlags.Instance | BindingFlags.NonPublic)!;
                var old = (FileStream)stream.GetValue(_gguf)!;
                stream.SetValue(_gguf, File.OpenRead(SourcePath + ".empty"));
                old.Dispose();
                if (FailCleanup)
                {
                    var original = (GgmlAllocator)_allocator;
                    typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .SetValue(this, new CleanupFailingAllocator(original.Context, original.DeviceId));
                }
            }
            return base.IsQuantizedLinearWeight(info);
        }

        public override void Dispose()
        {
            VirtualDisposeCalls++;
            throw new InvalidOperationException("A subclass Dispose must not run during its base construction.");
        }
    }

    private sealed class CleanupFailingAllocator(GgmlContext context, int device) : GgmlAllocator(context, device), TensorSharp.IAllocator
    {
        public new TensorSharp.Storage Allocate(TensorSharp.DType type, long count) => new CleanupFailingStorage(this, Context, type, count);
    }

    private sealed class CleanupFailingStorage : GgmlStorage
    {
        internal CleanupFailingStorage(GgmlAllocator allocator, GgmlContext context, TensorSharp.DType type, long count)
            : base(allocator, context, type, count)
        {
            SetElementsAsFloat(0, [9, 10, 11, 12]);
        }
        protected override void Destroy() => throw new InvalidOperationException("controlled managed storage cleanup failure before native memory free");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseWithPendingCompute(string path, BackendType backend)
    {
        try { _ = new PhaseOrderModel(path, backend); }
        catch (InvalidDataException error) when (error.Message == "controlled phase-order construction failure")
        {
            Require(RuntimeResourceCount() == 0, "Pending compute drains before graph and buffer phases, then owners explicitly release without GC.");
            return "actual-native-async-flash-attention;observed-initial-drain=" + PhaseOrderModel.InitialDrain + ";graph-phase-pending-drain=0;cache-empty-before-buffer-free;no-captured-graph-qualification";
        }
        finally { GgmlBasicOps.SetAsyncCompute(false); }
        throw new InvalidOperationException("The controlled phase-order constructor must fail.");
    }

    private sealed class PhaseOrderModel : ModelBase
    {
        internal static int InitialDrain;
        private readonly TensorSharp.Tensor _input;
        private static int ProbeGuardedBarrier() => (int)typeof(GgmlNativeLoader).Assembly.GetType("TensorSharp.GGML.GgmlNative", true)!
            .GetMethod("TSGgml_HostReadBarrier", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, null)!;

        internal PhaseOrderModel(string path, BackendType backend) : base(path, backend)
        {
            try
            {
                var allocator = (GgmlAllocator)_allocator;
                _input = new(new PhaseOrderAllocator(allocator.Context, allocator.DeviceId), TensorSharp.DType.Float32, 32);
                _input.SetElementsAsFloat(Enumerable.Repeat(4f, 32).ToArray());
                _weights.Add("result", new(_allocator, TensorSharp.DType.Float32, 32));
                _weights.Add("kCache", new(_allocator, TensorSharp.DType.Float32, 1, 32, 32));
                _weights.Add("vCache", new(_allocator, TensorSharp.DType.Float32, 1, 32, 32));
                GgmlBasicOps.SetAsyncCompute(true);
                QueueAttention();
                InitialDrain = ProbeGuardedBarrier();
                if (backend == BackendType.GgmlMetal)
                    Require(InitialDrain == 1, "The real Metal operation demonstrably defers compute before the ordering test.");
                QueueAttention();
                throw new InvalidDataException("controlled phase-order construction failure");
            }
            catch (Exception original)
            {
                RollBackFailedConstruction(original, () => _input?.Dispose(), releaseDerivedGraphs: () =>
                {
                    Require(ProbeGuardedBarrier() == 0, "The pipeline drains deferred work BEFORE entering the first family graph teardown phase.");
                    Require(_weights["result"].GetElementsAsFloat(32).All(value => Math.Abs(value - 4) < 0.00001), "Drained actual native output remains valid before any graph/buffer teardown.");
                    if (backend == BackendType.GgmlMetal)
                        Require(GgmlBasicOps.DeviceCopyCacheResidentBytes() > 0, "The real native attention leaves live cache bindings until the later generic cache phase.");
                });
                throw;
            }
        }
        private void QueueAttention() => GgmlBasicOps.FlashAttnDecode(_input, _input, _input,
            _weights["kCache"], _weights["vCache"], _weights["result"], 1, 1, 32, 32, 0, 1f / MathF.Sqrt(32));
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private sealed class PhaseOrderAllocator(GgmlContext context, int device) : GgmlAllocator(context, device), TensorSharp.IAllocator
    {
        public new TensorSharp.Storage Allocate(TensorSharp.DType type, long count) => new PhaseOrderStorage(this, Context, type, count);
    }

    private sealed class PhaseOrderStorage(GgmlAllocator allocator, GgmlContext context, TensorSharp.DType type, long count)
        : GgmlStorage(allocator, context, type, count)
    {
        protected override void Destroy()
        {
            Require(GgmlBasicOps.DeviceCopyCacheResidentBytes() == 0, "The generic native cache phase precedes the first derived buffer free.");
            base.Destroy();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseDerivedCleanup(string path, BackendType backend)
    {
        try { _ = new CleanupFailingModel(path, backend); }
        catch (AggregateException error)
        {
            Require(error.InnerExceptions[0] is InvalidDataException && error.InnerExceptions[1] is InvalidOperationException,
                "The original construction failure and dependency cleanup failure remain separately observable.");
            Require(RuntimeResourceCount() == 4, "Derived cleanup failure releases neither the real derived tensor nor base weight/context/Model ownership.");
            return "injected-derived-dependency-cleanup-failure;actual-native-tensor-owned";
        }
        throw new InvalidOperationException("Derived cleanup must report both failures.");
    }

    private sealed class CleanupFailingModel : ModelBase
    {
        private readonly TensorSharp.Tensor _held;
        public CleanupFailingModel(string path, BackendType backend) : base(path, backend)
        {
            _held = new(_allocator, TensorSharp.DType.Float32, 4);
            _held.SetElementsAsFloat([1, 2, 3, 4]);
            _weights.Add("controlled-base-weight", new(_allocator, TensorSharp.DType.Float32, 4));
            _weights["controlled-base-weight"].SetElementsAsFloat([5, 6, 7, 8]);
            try { throw new InvalidDataException("controlled model construction failure"); }
            catch (Exception original)
            {
                RollBackFailedConstruction(original, () => _held.Dispose(),
                    releaseDerivedGraphs: static () => throw new InvalidOperationException("controlled derived graph cleanup failure before any model buffer release"));
                throw;
            }
        }
        internal bool HeldValuesIntact => _held.GetElementsAsFloat(4).SequenceEqual(new float[] { 1, 2, 3, 4 }) &&
            _weights.Count == 1 && _weights["controlled-base-weight"].GetElementsAsFloat(4).SequenceEqual(new float[] { 5, 6, 7, 8 });
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private sealed class DisposeFailingModel : ModelBase
    {
        internal const string Refusal = "controlled normal Dispose graph cleanup refusal before any model buffer release";
        private readonly TensorSharp.Tensor _held;
        internal int GraphCleanupCalls { get; private set; }

        public DisposeFailingModel(string path, BackendType backend) : base(path, backend)
        {
            _held = new(_allocator, TensorSharp.DType.Float32, 4);
            _held.SetElementsAsFloat([1, 2, 3, 4]);
            _weights.Add("controlled-base-weight", new(_allocator, TensorSharp.DType.Float32, 4));
            _weights["controlled-base-weight"].SetElementsAsFloat([5, 6, 7, 8]);
        }

        internal bool ValuesIntact => _held.GetElementsAsFloat(4).SequenceEqual(new float[] { 1, 2, 3, 4 }) &&
            _weights["controlled-base-weight"].GetElementsAsFloat(4).SequenceEqual(new float[] { 5, 6, 7, 8 });
        public override void Dispose() => DisposeBaseResources(() => _held.Dispose(), releaseDerivedGraphs: RefuseGraphCleanup);
        private void RefuseGraphCleanup()
        {
            GraphCleanupCalls++;
            throw new InvalidOperationException(Refusal);
        }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseNormalDispose(string path, BackendType backend)
    {
        var model = new DisposeFailingModel(path, backend);
        Require(RuntimeResourceCount() == 4 && model.ValuesIntact, "The model completes construction with real owned native buffers.");
        try { model.Dispose(); }
        catch (InvalidOperationException original) when (original.Message == DisposeFailingModel.Refusal)
        {
            Require(original.StackTrace!.Contains("RefuseGraphCleanup", StringComparison.Ordinal),
                "Ordinary Dispose preserves its original graph-refusal exception type and stack.");
            for (int attempt = 0; attempt < 2; attempt++)
            {
                try { model.Dispose(); }
                catch (InvalidOperationException refusal) when (ReferenceEquals(refusal.InnerException, original))
                {
                    Require(model.GraphCleanupCalls == 1 && RuntimeResourceCount() == 4 && model.ValuesIntact,
                        "Repeated teardown refuses before any phase, preserving the first failure and actual owners.");
                    continue;
                }
                throw new InvalidOperationException("Repeated Dispose must refuse without rerunning its graph phase.");
            }
            return "controlled-normal-Dispose-graph-phase-refusal;firstExceptionPreserved=true;graphCleanupCalls=1;" +
                "repeatedTeardownRefused=true;actual-model-and-native-storage-owned;not-actual-GPU-failure";
        }
        throw new InvalidOperationException("Normal Dispose must report its controlled graph-phase refusal.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseBaseCleanup(string path, BackendType backend)
    {
        IDisposable call = (IDisposable)typeof(GgmlNativeLoader).GetMethod("EnterNativeCall", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, [null, IntPtr.Zero, Type.Missing, Type.Missing])!;
        TextWriter previous = Console.Out;
        var original = new IOException("controlled base-constructor output failure");
        TensorSharp.NativeConstructionCleanupException? failure = null;
        try
        {
            Console.SetOut(new FailingOutput(original));
            try { _ = new BaseConstructionProbe(path, backend); }
            catch (TensorSharp.NativeConstructionCleanupException error)
            {
                Require(ReferenceEquals(error.InnerExceptions[0], original) && error.InnerExceptions[1] is InvalidOperationException,
                    "Base rollback retains the original failure and actual resource-cleanup refusal while a controlled call lease is held.");
                Require(ModelLeaseCount() == 1 && RuntimeResourceCount() == 2,
                    "Base rollback refusal retains its context and Model lease.");
                Require(!error.Cleanup.IsReleased &&
                    typeof(TensorSharp.NativeConstructionCleanupHandle).GetField("_owner", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .GetValue(error.Cleanup) is BaseConstructionProbe,
                    "The actual failed construction remains owned by its release-only recovery handle.");
                failure = error;
            }
        }
        finally { Console.SetOut(previous); call.Dispose(); }
        Require(failure != null, "Base rollback must report its cleanup refusal.");
        var retained = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(retained.Count == 0, "A pre-effect Busy refusal does not poison the failed construction.");
        failure!.Cleanup.Dispose();
        failure.Cleanup.Dispose();
        Require(failure.Cleanup.IsReleased && ModelLeaseCount() == 0 && RuntimeResourceCount() == 0,
            "Explicit recovery after the call drains releases the actual context and model lease without replaying construction.");
        return "injected-base-output-failure;real-pre-effect-Busy;actual-construction-owned;explicit-release-only-recovery";
    }

    private sealed class FailingOutput(Exception error) : TextWriter
    {
        public override Encoding Encoding => Encoding.UTF8;
        public override void WriteLine(string? value) => throw error;
    }

    private sealed class BaseConstructionProbe(string path, BackendType backend, TensorSharp.ITensorParallelGroup? group = null)
        : ModelBase(path, backend, tpGroup: group!)
    {
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseBorrowedGroup(string path, BackendType backend)
    {
        using (var group = new BorrowedGroup(new BorrowedAllocator()))
        {
            try { _ = new BaseConstructionProbe(path + ".missing", BackendType.Cuda, group); }
            catch (FileNotFoundException) { }
            Require(group.AllocatorRequests == 1 && group.DisposeCalls == 0 && group.Allocator.DisposeCalls == 0,
                "Base constructor rollback never disposes a caller-owned disposable allocator or group.");
            try { _ = new HunyuanDenseModel(path, BackendType.Cuda, tpGroup: group); }
            catch (NotSupportedException error) when (error.Message.Contains("equal key/value head dims", StringComparison.Ordinal)) { }
            Require(group.AllocatorRequests == 2 && group.DisposeCalls == 0 && group.Allocator.DisposeCalls == 0,
                "Derived constructor rollback never disposes a caller-owned disposable allocator or group.");
        }

        var context = new GgmlContext([0], backend == BackendType.GgmlCpu ? GgmlBackendType.Cpu : GgmlBackendType.Metal);
        using (var local = new GgmlTensorParallelGroup(context, ownsContext: true))
        {
            var nested = new BorrowedNestedGroup(local);
            int callerResources = RuntimeResourceCount();
            // The constructor reuses the supplied real context; this does not exercise a CUDA device.
            try { _ = new BaseConstructionProbe(path + ".missing", BackendType.GgmlCuda, nested); }
            catch (FileNotFoundException) { }
            Require(nested.DisposeCalls == 0 && RuntimeResourceCount() == callerResources,
                "Base rollback leaves the nested caller-owned native context/group live.");
            try { _ = new HunyuanDenseModel(path, BackendType.GgmlCuda, tpGroup: nested); }
            catch (NotSupportedException error) when (error.Message.Contains("equal key/value head dims", StringComparison.Ordinal)) { }
            Require(nested.DisposeCalls == 0 && RuntimeResourceCount() == callerResources,
                "Derived rollback leaves the nested caller-owned native context/group live.");
            using var tensor = new TensorSharp.Tensor(local.GetAllocator(0), TensorSharp.DType.Float32, 4);
            tensor.SetElementsAsFloat([1, 2, 3, 4]);
            Require(tensor.GetElementsAsFloat(4).SequenceEqual(new float[] { 1, 2, 3, 4 }),
                "The real borrowed context remains usable after both failed constructors.");
        }
        Require(RuntimeResourceCount() == 0, "The caller explicitly drains its own resources without GC.");
        return "borrowed-disposable-allocator-contract-and-real-nested-GGML-context;no-CUDA-device-qualification";
    }

    private sealed class BorrowedAllocator : TensorSharp.IAllocator, IDisposable
    {
        internal int DisposeCalls;
        public TensorSharp.BlasEnum BlasEnum => TensorSharp.BlasEnum.DotNet;
        public int DeviceId => 0;
        public TensorSharp.Storage Allocate(TensorSharp.DType type, long count) => throw new NotSupportedException();
        public float GetAllocatedMemoryRatio() => 0;
        public void Dispose() => DisposeCalls++;
    }

    private sealed class BorrowedGroup(BorrowedAllocator allocator) : TensorSharp.ITensorParallelGroup
    {
        internal BorrowedAllocator Allocator => allocator;
        internal int AllocatorRequests;
        internal int DisposeCalls;
        public int Degree => 1;
        public bool IsActive => false;
        public int GlobalDegree => 1;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public TensorSharp.IAllocator GetAllocator(int rank) { AllocatorRequests++; return allocator; }
        public void AllReduce(TensorSharp.Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int operation, int[] payload) => throw new NotSupportedException();
        public (int, int[]) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { DisposeCalls++; allocator.Dispose(); }
    }

    private sealed class BorrowedNestedGroup(GgmlTensorParallelGroup local) : TensorSharp.ITensorParallelGroup, TensorSharp.INestedTensorParallelGroup
    {
        internal int DisposeCalls;
        public TensorSharp.ITensorParallelGroup LocalGroup => local;
        public int Degree => local.Degree;
        public bool IsActive => local.IsActive;
        public int GlobalDegree => local.GlobalDegree;
        public int GlobalRankOffset => local.GlobalRankOffset;
        public int NodeCount => 1;
        public TensorSharp.IAllocator GetAllocator(int rank) => local.GetAllocator(rank);
        public void AllReduce(TensorSharp.Tensor[] tensors) => local.AllReduce(tensors);
        public void Synchronize() => local.Synchronize();
        public void Barrier() => local.Barrier();
        public void CrossNodeAllReduce(float[] buffer, int count) => throw new NotSupportedException();
        public void BroadcastControl(int operation, int[] payload) => throw new NotSupportedException();
        public (int, int[]) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { DisposeCalls++; local.Dispose(); }
    }

    private static void VerifyRetainedOwner(string mode)
    {
        var owners = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(owners.Count == 1, "Only the terminal failed model enters the generation-local retention collection.");
        object model = owners[0]!;
        if (mode == "derived-cleanup-failure")
            Require(model is CleanupFailingModel retained && retained.HeldValuesIntact, "Actual native storage values survive finalizer drainage.");
        if (mode == "local-cleanup-failure")
        {
            var resources = (IList)typeof(ModelBase).GetField("_failedOwnershipResources", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            Require(resources.Count == 1 && resources[0] is TensorSharp.Tensor, "The local unregistered allocation is strongly retained with its model.");
            var tensor = (TensorSharp.Tensor)resources[0]!;
            Require(tensor.Storage.GetElementsAsFloat(0, 4).SequenceEqual(new float[] { 9, 10, 11, 12 }),
                "The real unregistered native storage remains strongly alive with intact values after finalizer drainage.");
            Require(RuntimeResourceCount() == 4, "Finalizers release none of the unsafe owned allocations.");
        }
        if (mode == "dispose-cleanup-failure")
        {
            Require(model is DisposeFailingModel retained && retained.ValuesIntact && retained.GraphCleanupCalls == 1,
                "The normally constructed model and both actual native storage values survive finalizer drainage without graph retries.");
            Require(ModelLeaseCount() == 1 && RuntimeResourceCount() == 4,
                "All context/tensor/Model ownership remains strongly retained rather than an abandoned numeric lease alone.");
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
