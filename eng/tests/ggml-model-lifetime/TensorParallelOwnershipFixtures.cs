#nullable enable

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ObserveTpColumnOwner(string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        WeakReference source = model.ShardOwnedColumnSource();
        Require(source.IsAlive && model.QuantShardCount == 2 && model.SourceQuantCount == 0,
            "Two actual external views keep the removed owned source wrapper alive before disposal.");
        model.Dispose();
        Require(HasUndisposedHostOwner(source),
            "Current model disposal retires the views but does not explicitly dispose their removed backing owner.");
        DrainFinalizers();
        Require(!source.IsAlive && RuntimeResourceCount() == 0,
            "The removed QuantizedWeight wrapper collects without explicit backing-owner disposal.");
        return "red-column-backing-owner-undisposed-before-collection;actual-owner-weak-reference-collected;" +
            "raw-heap-liveness-not-measured;no-post-disposal-pointer-read;logical-ranks-on-one-real-context-not-multidevice";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool HasUndisposedHostOwner(WeakReference owner)
        => owner.Target is QuantizedWeight weight && weight.HasHostData && weight.Data != IntPtr.Zero;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ObserveTpPartialShard(string mode, string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        PartialShardEvidence evidence = model.RefuseSecondShard(mode);
        Require(evidence.UnregisteredStorages.All(storage => storage.IsAlive) && model.F32ShardCount == 0 &&
            RuntimeResourceCount() == 2 + model.SourceTensorCount + evidence.UnregisteredStorages.Length,
            "The real first shard allocates before rank two refuses, but no model-owned array receives it.");
        DrainFinalizers();
        Require(evidence.UnregisteredStorages.All(storage => !storage.IsAlive) && RuntimeResourceCount() == 2 + model.SourceTensorCount,
            "The unregistered first shard is lost and its actual native storage finalizes while the model stays live.");
        bool leakedSourceView = mode is "observe-tp-generic-column" or "observe-tp-generic-row" or "observe-tp-generic-copy";
        try { model.Dispose(); }
        catch (InvalidOperationException error) when (leakedSourceView && error.Message.Contains("tensor storages", StringComparison.Ordinal))
        {
            model.ContextCleanupRefused = true;
        }
        Require(leakedSourceView
                ? model.ContextCleanupRefused && model.SourceStorageOwned && ModelLeaseCount() == 1 && RuntimeResourceCount() == 3
                : RuntimeResourceCount() == 0,
            "The generic route also loses its second Narrow reference: context cleanup refuses before terminal model release.");
        return $"red-{mode};unregisteredNativeStorages={evidence.UnregisteredStorages.Length};real-first-shard-storage-lost-to-finalizer;reserved-array-absent;" +
            (leakedSourceView ? "unregistered-source-Narrow-reference-leaked;context-cleanup-terminal-refusal;" : "") +
            "finalization-is-defect-evidence-not-explicit-shard-cleanup;logical-ranks-on-one-real-context-not-multidevice";
    }

    private sealed record PartialShardEvidence(WeakReference[] UnregisteredStorages);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseTpViewRefusal(string path, BackendType backend, bool poisonLate)
    {
        var model = new LogicalRankModel(path, backend);
        model.ShardMappedColumnSource();
        QuantizedWeight view = model.FirstQuantShard;
        RegisterTransform(view);
        Require(model.MappingOpen && RegisteredBonsaiKeys(view) == 1,
            "The real mapped shard has a live actual backing mapping and a native Bonsai registration.");
        if (!poisonLate)
        {
            PoisonOwner();
            try { view.Dispose(); }
            catch (InvalidOperationException) { }
            Require(view.HasHostData && RegisteredBonsaiKeys(view) == 1 && model.MappingOpen,
                "An actual guarded view unregister refusal preserves the view identity and backing mapping before model cleanup.");
        }
        else model.PoisonAfterGenericCaches = true;
        bool refused = false;
        try { model.Dispose(); }
        catch (InvalidOperationException) { refused = true; }
        Require(refused && view.HasHostData && RegisteredBonsaiKeys(view) == 1,
            "Shared disposal preserves the failed actual view owner and registration after its managed refusal.");
        Require(model.MappingOpen == !poisonLate,
            "Early refusal blocks backing-map release; late refusal exposes the current map-before-TP-view order.");
        return poisonLate
            ? "red-mapping-disposed-before-failed-TP-view-unregister;managed-owner-retained-but-mapping-already-closed;no-pointer-read-after-poison;not-GPU-fault"
            : "positive-real-TP-view-unregister-refusal-blocks-backing-map-release;actual-model-and-view-retained;not-GPU-fault";
    }

    private static int RegisteredBonsaiKeys(QuantizedWeight weight)
        => ((ICollection<IntPtr>)typeof(QuantizedWeight).GetField("_bonsaiRegisteredKeys", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(weight)!).Count;

    private static void VerifyTpRetention(string mode)
    {
        var owners = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(owners.Count == 1 && owners[0] is LogicalRankModel,
            "The generation retains the actual failed logical-rank model after finalizer drainage.");
        var model = (LogicalRankModel)owners[0]!;
        if (mode is "observe-tp-generic-column" or "observe-tp-generic-row" or "observe-tp-generic-copy")
        {
            Require(model.ContextCleanupRefused && RuntimeResourceCount() == 2 && !model.SourceStorageOwned,
                "After terminal context refusal, finalizer drainage releases the abandoned source storage but not the retained context/Model.");
            return;
        }
        Require(RuntimeResourceCount() == 2 && model.QuantShardCount == 2 && model.FirstQuantShard.HasHostData &&
            RegisteredBonsaiKeys(model.FirstQuantShard) == 1,
            "The actual failed view, context/Model ownership and registration remain, not only numeric lease IDs.");
        Require(model.MappingOpen == (mode == "tp-view-unregister-refusal"),
            "The red late-refusal mapping is already closed; the positive early-refusal mapping stays owned.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseTpSynchronization(string path, BackendType backend, bool preflight)
    {
        var model = new LogicalRankModel(path, backend);
        model.AddSourceTensor("probe.weight", 4, 4);
        model.Group.RefuseSynchronization = true;
        if (preflight)
        {
            bool refused = false;
            try { model.Group.Synchronize(); }
            catch (IOException) { refused = true; }
            Require(refused && model.GraphCleanupCalls == 0 && model.SourceTensorCount == 1 &&
                RuntimeResourceCount() == 3 && model.SourceStorageOwned,
                "Controlled caller preflight synchronization refuses before graph/cache/weight retirement.");
            model.Group.RefuseSynchronization = false;
        }
        model.Dispose();
        Require(model.GraphCleanupCalls == 1 && !model.SourceStorageOwned && RuntimeResourceCount() == 0,
            "The model executes the controlled graph-phase marker and real storage cleanup, then releases its native resources.");
        Require(model.Group.SynchronizeCalls == (preflight ? 1 : 0),
            "Current shared disposal does not consult the supplied group's configured synchronization refusal.");
        return preflight
            ? "positive-caller-preflight-refusal-before-any-retirement;not-production-drain-proof;controlled-logical-group-not-CUDA-stream"
            : "red-model-dispose-retires-graph-callback-and-real-storage-without-group-synchronization;configured-refusal-not-consulted;not-GPU-fault";
    }

    private static void DrainFinalizers()
    {
        for (int attempt = 0; attempt < 10; attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
    }

    private static bool StorageDestroyed(GgmlStorage storage)
        => (IntPtr)typeof(GgmlStorage).GetField("buffer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(storage)! == IntPtr.Zero;

    private sealed class LogicalRankModel : ModelBase
    {
        internal LogicalRankGroup Group { get; }
        private readonly RefusingRankAllocator _rankAllocator;
        private WeakReference? _sourceStorage;
        internal bool PoisonAfterGenericCaches;
        internal bool ContextCleanupRefused;
        internal int GraphCleanupCalls;
        protected override bool OwnsTensorParallelGroup => false;
        internal int QuantShardCount => _tpQuantWeights.Values.Sum(shards => shards.Count(weight => weight != null));
        internal int F32ShardCount => _tpWeights.Values.Sum(shards => shards.Count(weight => weight != null));
        internal int SourceTensorCount => _weights.Count;
        internal int SourceQuantCount => _quantWeights.Count;
        internal QuantizedWeight FirstQuantShard => _tpQuantWeights.Single().Value[0];
        internal bool MappingOpen => typeof(GgufFile).GetField("_mappedView", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(_gguf) != null;
        internal bool SourceStorageOwned => _sourceStorage?.Target is GgmlStorage storage && !StorageDestroyed(storage);

        internal LogicalRankModel(string path, BackendType backend) : this(path, backend, new LogicalRankGroup()) { }

        private LogicalRankModel(string path, BackendType backend, LogicalRankGroup group) : base(path, backend, tpGroup: group)
        {
            Group = group;
            var allocator = (GgmlAllocator)_allocator;
            _rankAllocator = new RefusingRankAllocator(allocator.Context, allocator.DeviceId);
            group.Allocator = _rankAllocator;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal WeakReference ShardOwnedColumnSource()
        {
            var source = new QuantizedWeight(new byte[144], (int)GgmlTensorType.Q4_0, 128, 2);
            var weak = new WeakReference(source);
            _quantWeights.Add("probe.weight", source);
            ShardWeightsForTensorParallelism(["probe"], []);
            Require(_tpQuantWeights["probe.weight"].All(view => ReferenceEquals(
                typeof(QuantizedWeight).GetField("_ownerToken", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(view), source)),
                "Both actual column views borrow the exact removed source owner.");
            return weak;
        }

        internal void ShardMappedColumnSource()
        {
            var info = _gguf.Tensors.Single().Value;
            Require(_gguf.TryGetTensorDataPointer(info, out IntPtr data), "The Q4 GGUF supplies a real mapped source.");
            _quantWeights.Add("probe.weight", QuantizedWeight.CreateExternalView(data, 144, (int)GgmlTensorType.Q4_0, 128, 2, _gguf));
            ShardWeightsForTensorParallelism(["probe"], []);
        }

        internal void AddSourceTensor(string name, params long[] shape)
        {
            var source = new Tensor(_rankAllocator, DType.Float32, shape);
            source.SetElementsAsFloat(Enumerable.Repeat(1f, (int)source.ElementCount()).ToArray());
            _weights.Add(name, source);
            _sourceStorage = new WeakReference(source.Storage);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal PartialShardEvidence RefuseSecondShard(string mode)
        {
            bool biases = mode is "observe-tp-concatenated-bias" or "observe-tp-separate-bias";
            bool separate = mode is "observe-tp-separate" or "observe-tp-separate-bias";
            AddSourceTensor("source0.weight", biases ? [4] : separate ? [4, 3] : [4, 4]);
            if (separate) AddSourceTensor("source1.weight", biases ? [4] : [4, 3]);
            bool refuseCopy = mode == "observe-tp-generic-copy";
            _rankAllocator.BeginSecondAllocationRefusal(allowSecond: refuseCopy);
            Action<object?[]>? originalHook = OpRegistry.PreInvokeHook;
            int copies = 0;
            if (refuseCopy) OpRegistry.PreInvokeHook = arguments =>
            {
                originalHook?.Invoke(arguments);
                if (arguments is [Tensor, Tensor] && ++copies == 2)
                    throw new IOException("controlled second shard copy dispatch refusal");
            };
            Exception? failure = null;
            try
            {
                switch (mode)
                {
                    case "observe-tp-generic-column":
                    case "observe-tp-generic-copy": ShardWeightsForTensorParallelism(["source"], []); break;
                    case "observe-tp-generic-row": ShardWeightsForTensorParallelism([], ["source"]); break;
                    case "observe-tp-concatenated": ShardConcatenatedColumnParallel("source0.weight", 2, 2); break;
                    case "observe-tp-separate": ShardSeparateColumnParallel("fused.weight", ["source0.weight", "source1.weight"], [4, 4]); break;
                    case "observe-tp-concatenated-bias": ShardConcatenatedBiasColumnParallel("source0.weight", 2, 2); break;
                    case "observe-tp-separate-bias": ShardSeparateBiasColumnParallel("fused.bias", ["source0.weight", "source1.weight"], [4, 4]); break;
                    default: throw new ArgumentException("Unknown TP allocation route.", nameof(mode));
                }
            }
            catch (IOException error) { failure = error; }
            finally { OpRegistry.PreInvokeHook = originalHook; }
            Require(failure != null && _rankAllocator.ShardAllocations.Count == (refuseCopy ? 2 : 1) && _rankAllocator.AllocationCalls == 2,
                "The second allocation or copy refuses after the first real shard allocates; the dispatch hook restores before cleanup.");
            return new(_rankAllocator.ShardAllocations.ToArray());
        }

        public override void Dispose() => DisposeBaseResources(
            () => { if (PoisonAfterGenericCaches) PoisonOwner(); },
            releaseDerivedGraphs: () => GraphCleanupCalls++);

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private sealed class RefusingRankAllocator(GgmlContext context, int device) : GgmlAllocator(context, device), IAllocator
    {
        internal readonly List<WeakReference> ShardAllocations = [];
        internal int AllocationCalls;
        private bool _refuseSecond;
        private bool _recordShards;
        internal void BeginSecondAllocationRefusal(bool allowSecond) { AllocationCalls = 0; ShardAllocations.Clear(); _refuseSecond = !allowSecond; _recordShards = true; }
        public new Storage Allocate(DType type, long count)
        {
            if (_recordShards && ++AllocationCalls == 2 && _refuseSecond) throw new IOException("controlled second logical-rank storage allocation refusal");
            Storage storage = base.Allocate(type, count);
            if (_recordShards) ShardAllocations.Add(new WeakReference(storage));
            return storage;
        }
    }

    private sealed class LogicalRankGroup : ITensorParallelGroup
    {
        internal IAllocator Allocator = null!;
        internal bool RefuseSynchronization;
        internal int SynchronizeCalls;
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => rank is 0 or 1 ? Allocator : throw new ArgumentOutOfRangeException(nameof(rank));
        public void RunPerRank(Action<int> body) { body(0); body(1); }
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Logical ownership ranks do not qualify collectives.");
        public void Synchronize() { SynchronizeCalls++; if (RefuseSynchronization) throw new IOException("controlled logical-group synchronization refusal"); }
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int operation, int[] payload) => throw new NotSupportedException();
        public (int, int[]) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
