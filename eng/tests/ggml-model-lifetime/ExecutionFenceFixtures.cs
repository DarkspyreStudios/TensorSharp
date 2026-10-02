#nullable enable

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    private static string ExerciseExecutionFences(string mode, string scratch, BackendType backend)
    {
        string path = Path.Combine(scratch, "execution.gguf");
        string vision = Path.Combine(scratch, "execution.vision.gguf");
        string[] settings = ["MAX_CONTEXT", "TS_DSV4_UBATCH", "TS_DSV4_THREADS"];
        string?[] previous = settings.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            Environment.SetEnvironmentVariable(settings[0], "32");
            Environment.SetEnvironmentVariable(settings[1], "1");
            Environment.SetEnvironmentVariable(settings[2], "2");
            if (mode == "vision-execution-cleanup-failure")
            {
                ulong fingerprint = DeepSeek41LifetimeFixture.WriteText(path);
                DeepSeek41LifetimeFixture.WriteVision(vision, fingerprint, 32);
            }
            else new DenseDecoderSyntheticModelBuilder
            {
                Architecture = "hunyuan-dense",
                PreTokenizer = "hunyuan-dense",
                IncludeMistralControlTokens = false,
                IncludeQkNorms = true,
                RopeBase = 10000f
            }.Write(path);
            ExecutionEvidence evidence = UseFailedOwner(mode, path, vision, backend);
            for (int attempt = 0; attempt < 10; attempt++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            var retained = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
            Require(evidence.Model.IsAlive && evidence.Tensor.IsAlive && evidence.Storage.IsAlive && retained.Count == 1 &&
                ReferenceEquals(retained[0], evidence.Model.Target) && ModelLeaseCount() == 1 && RuntimeResourceCount() == evidence.ResourceCount &&
                GgmlNativeLoader.State == GgmlRuntimeState.Ready,
                "Actual model, failed tensor/storage and all native ownership survive finalizer drainage on an operational runtime.");
            return evidence.Report + ";actual-model-tensor-storage-retained;owner-ready;controlled-managed-refusal-not-GPU-fault;no-physical-multidevice";
        }
        finally
        {
            for (int index = 0; index < settings.Length; index++) Environment.SetEnvironmentVariable(settings[index], previous[index]);
            File.Delete(path);
            File.Delete(vision);
        }
    }

    private sealed record ExecutionEvidence(WeakReference Model, WeakReference Tensor, WeakReference Storage, int ResourceCount, string Report);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static ExecutionEvidence UseFailedOwner(string mode, string path, string vision, BackendType backend)
    {
        ModelBase model = mode == "vision-execution-cleanup-failure" ? new DeepSeek41Model(path, backend) : new HunyuanDenseModel(path, backend);
        if (model is DeepSeek41Model loadedVision) loadedVision.LoadVisionEncoder(vision);
        else
        {
            Require(model.ForwardRefill([65, 66, 67]).All(float.IsFinite) && model.Forward([68]).All(float.IsFinite),
                "The normally constructed complete two-layer model performs real prefill/decode before controlled ownership failure.");
        }
        var allocator = (GgmlAllocator)typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var held = new Tensor(new CleanupFailingAllocator(allocator.Context, allocator.DeviceId), DType.Float32, 4);
        Exception original = new InvalidOperationException("execution fixture has not triggered its real cleanup refusal");
        var group = new ExecutionControlGroup(allocator, mode == "execution-cleanup-failure" ? 0 : 1);
        if (model is HunyuanDenseModel)
        {
            typeof(ModelBase).GetField("_tpGroup", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, group);
            if (mode == "execution-cleanup-failure") model.BeginDistributedDriver();
        }
        if (mode == "execution-dispatch-cleanup-failure") group.OnReceive = () => original = RefuseRealOwnedTensor(model, held);
        else original = RefuseRealOwnedTensor(model, held);

        long before = NativeLeaseSequence();
        int resources = RuntimeResourceCount();
        string report;
        if (mode is "execution-worker-cleanup-failure" or "execution-dispatch-cleanup-failure")
        {
            Exception? refusal = null;
            try { model.RunDistributedWorkerLoop(); }
            catch (Exception error) { refusal = error; }
            Require(refusal is InvalidOperationException && ReferenceEquals(refusal.InnerException, original) &&
                group.Receives == (mode == "execution-dispatch-cleanup-failure" ? 1 : 0) && NativeLeaseSequence() == before,
                "Worker terminal refusal propagates before receive or, when failure occurs during logical receive, before direct Core dispatch.");
            report = $"worker-admission;logicalReceives={group.Receives};nativeLeaseSequenceUnchanged=true";
        }
        else if (model is HunyuanDenseModel hunyuan)
        {
            byte[] cache = new byte[checked((int)model.ComputeKVBlockByteSize(1))];
            Array.Fill(cache, (byte)0xa5);
            string fingerprint = model.KVStateFingerprint;
            object? sequenceLength = typeof(ModelBase).GetField("_cacheSeqLen", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);
            object? cacheK = typeof(HunyuanDenseModel).GetField("_kvCacheK", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);
            object? cacheV = typeof(HunyuanDenseModel).GetField("_kvCacheV", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);
            ExpectTerminal(() => model.Forward([69]), original);
            ExpectTerminal(() => model.ForwardRefill([69]), original);
            ExpectTerminal(model.ResetKVCache, original);
            ExpectTerminal(() => model.TruncateKVCache(1), original);
            ExpectTerminal(() => model.TryTruncateKVCache(1), original);
            ExpectTerminal(() => model.TryExtractKVBlock(0, 1, cache), original);
            ExpectTerminal(() => model.TryInjectKVBlock(0, 1, cache), original);
            ExpectTerminal(() => model.PrepareForPrefill(1024), original);
            ExpectTerminal(() => hunyuan.ForwardBatch(null!), original);
            ExpectTerminal(model.ReleaseGgmlDeviceResidency, original);
            ExpectTerminal(model.TrimIdleMemory, original);
            ExpectTerminal(model.WarmUpKernels, original);
            ExpectTerminal(model.BeginDistributedDriver, original);
            Require(group.Broadcasts == 0 && NativeLeaseSequence() == before && model.KVStateFingerprint == fingerprint && cache.All(value => value == 0xa5) &&
                Equals(sequenceLength, typeof(ModelBase).GetField("_cacheSeqLen", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)) &&
                ReferenceEquals(cacheK, typeof(HunyuanDenseModel).GetField("_kvCacheK", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)) &&
                ReferenceEquals(cacheV, typeof(HunyuanDenseModel).GetField("_kvCacheV", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)),
                "Failed real-model APIs accept no native call/control broadcast and preserve KV metadata and caller snapshot storage.");
            model.ResetForwardTiming();
            Require(model.Config.HiddenSize > 0 && model.SampleGreedy([0, 1]) == 1,
                "Pure metadata, diagnostics and independent managed sampling remain available.");
            report = "real-prefill-decode;13-shared-family-admission-checks;no-broadcast-native-call-KV-mutation;batch-null-is-boundary-only";
        }
        else
        {
            var multimodal = (DeepSeek41Model)model;
            var injector = model.MultimodalInjector;
            var handles = NativeHandleKeys();
            using var incoming = new Tensor(allocator, DType.Float32, 1, 32);
            incoming.SetElementsAsFloat(Enumerable.Repeat(2f, 32).ToArray());
            long mediaBefore = NativeLeaseSequence();
            ExpectTerminal(() => multimodal.LoadVisionEncoder(vision), original);
            ExpectTerminal(() => multimodal.SetVisionEmbeddings(incoming, 0), original);
            ExpectTerminal(() => injector.ProcessPromptTokens([], [], "failed-owner"), original);
            ExpectTerminal(() => injector.QueuePromptEmbeddings(0, "failed-owner"), original);
            ExpectTerminal(() => injector.TrimPreparedPrompt(0, "failed-owner"), original);
            ExpectTerminal(() => injector.ClearPreparedPromptState("failed-owner"), original);
            ExpectTerminal(((IDisposable)injector).Dispose, original);
            MethodInfo expand = typeof(DeepSeek41Model).GetMethods(BindingFlags.Instance | BindingFlags.NonPublic)
                .Single(method => method.Name.EndsWith(".ExpandMultimodalPrompt", StringComparison.Ordinal));
            ExpectTerminal(() => InvokeExecutionBoundary(expand, model, [injector, new List<ChatMessage>(), new List<int>()]), original);
            string[] directNames = ["BindSequenceCache", "AdoptPrimaryCacheToFused", "RestorePrimaryCache", "OnSequenceReleased", "RetainSequenceCache",
                "RetainSequenceCacheAs", "TryRebindRetainedCache", "DiscardRetainedCache", "SpecEnsureCapacity", "SpecSnapshotRecurrentState", "SpecRestoreRecurrentState",
                "SpecRewindCache", "DraftStep", "DraftBlock", "DraftCatchUp", "SpecForward"];
            foreach (string name in directNames)
            {
                MethodInfo method = model.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.Public)!;
                object?[] parameters = method.GetParameters().Select(parameter =>
                    parameter.ParameterType == typeof(string) ? (object)"failed-owner" : parameter.ParameterType == typeof(int[]) ? Array.Empty<int>() :
                    parameter.ParameterType == typeof(float[]) ? Array.Empty<float>() : parameter.ParameterType.IsValueType ? Activator.CreateInstance(parameter.ParameterType) : null).ToArray();
                ExpectTerminal(() => InvokeExecutionBoundary(method, model, parameters), original);
            }
            Require(NativeLeaseSequence() == mediaBefore && handles.ToHashSet().SetEquals(NativeHandleKeys()) &&
                !injector.HasPendingEmbeddings("failed-owner") && injector.GetPreparedMediaSpans("failed-owner").Count == 0,
                "Real loaded vision refuses media operations before native handles or prepared prompt state change.");
            Require(incoming.GetElementsAsFloat(32).All(value => value == 2f),
                "Rejected embeddings remain caller-owned and readable; healthy model ownership transfer is unchanged.");
            report = "real-DeepSeek41-load-attach;8-media-admission-checks;16-sequence-draft-method-boundary-only-checks;caller-embedding-owned;no-draft-forward-or-encode";
        }
        Require(ReferenceEquals(ExecutionCleanupFailure(model), original), "Admission refusal preserves the exact first ownership cleanup diagnostic.");
        ExpectTerminal(model.Dispose, original);
        Require(GgmlNativeLoader.State == GgmlRuntimeState.Ready, "Model terminal refusal does not poison the runtime.");
        return new(new(model), new(held), new(held.Storage), resources, report);
    }

    private static Exception RefuseRealOwnedTensor(ModelBase model, Tensor tensor)
    {
        try { typeof(ModelBase).GetMethod("RetireOwnedResource", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, [tensor]); }
        catch (TargetInvocationException error) when (error.InnerException is InvalidOperationException failure &&
            failure.Message.Contains("before native memory free", StringComparison.Ordinal))
        {
            Require(ReferenceEquals(ExecutionCleanupFailure(model), failure) && GgmlNativeLoader.State == GgmlRuntimeState.Ready,
                "Actual owned-resource cleanup failure sets the existing terminal model flag without seeding or poisoning the owner.");
            return failure;
        }
        throw new InvalidOperationException("Real tensor cleanup must refuse before native freeing.");
    }

    private static void ExpectTerminal(Action operation, Exception original)
    {
        Exception? refusal = null;
        try { operation(); }
        catch (Exception error) { refusal = error; }
        Require(refusal is InvalidOperationException && ReferenceEquals(refusal.InnerException, original),
            "A failed actual model must refuse admission with its exact first cleanup failure before other work or fallback.");
    }

    private static long NativeLeaseSequence() => (long)typeof(GgmlNativeLoader).GetField("s_nextLease", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;

    private static Exception? ExecutionCleanupFailure(ModelBase model)
        => (Exception?)typeof(ModelBase).GetField("_ownershipCleanupFailure", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);

    private static void InvokeExecutionBoundary(MethodInfo method, object target, object?[] parameters)
    {
        try { method.Invoke(target, parameters); }
        catch (TargetInvocationException error) when (error.InnerException != null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error.InnerException).Throw();
        }
    }

    private sealed class ExecutionControlGroup(GgmlAllocator allocator, int offset) : ITensorParallelGroup
    {
        internal int Broadcasts;
        internal int Receives;
        internal Action? OnReceive;
        public int Degree => 1;
        public bool IsActive => false;
        public int GlobalDegree => 1;
        public int GlobalRankOffset => offset;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => allocator;
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Logical control fixture has no physical collective.");
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int operation, int[] payload) => Broadcasts++;
        public (int, int[]) ReceiveControl()
        {
            Receives++;
            if (Receives > 1) throw new InvalidOperationException("Unexpected repeated logical receive.");
            OnReceive?.Invoke();
            return (1, [69]);
        }
        public void Dispose() { }
    }
}
