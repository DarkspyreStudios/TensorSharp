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
        Require(HasDisposedHostOwner(source),
            "Model disposal explicitly retires the removed backing owner after its views, before finalizer drainage.");
        DrainFinalizers();
        Require(!source.IsAlive && RuntimeResourceCount() == 0,
            "The explicitly disposed backing owner collects after successful cleanup.");
        return "explicit-column-backing-owner-disposed-before-finalizers;actual-owner-weak-reference-collected;" +
            "raw-heap-liveness-not-measured;no-post-disposal-pointer-read;logical-ranks-on-one-real-context-not-multidevice";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool HasDisposedHostOwner(WeakReference reference)
        => reference.Target is QuantizedWeight owner && !owner.HasHostData && owner.Data == IntPtr.Zero;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ObserveTpPartialShard(string mode, string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        PartialShardEvidence evidence = model.RefuseSecondShard(mode);
        Require(evidence.ShardStorages.All(storage => storage.IsAlive) && model.F32ShardCount == evidence.ShardStorages.Length &&
            RuntimeResourceCount() == 2 + model.SourceTensorCount + evidence.ShardStorages.Length,
            "Every allocated real shard transfers to the reserved model-owned array before a later allocation/copy refusal.");
        DrainFinalizers();
        Require(evidence.ShardStorages.All(storage => storage.IsAlive) &&
            RuntimeResourceCount() == 2 + model.SourceTensorCount + evidence.ShardStorages.Length,
            "Diagnostic finalizer drainage does not release the still-owned partial shards.");
        model.Dispose();
        Require(evidence.ShardStorages.All(storage => storage.Target is GgmlStorage actual && StorageDestroyed(actual)) &&
            !model.SourceStorageOwned && RuntimeResourceCount() == 0,
            "Explicit model disposal destroys every actual partial shard and source storage, with no leaked Narrow or finalizer dependence.");
        return $"explicit-{mode};ownedNativeStorages={evidence.ShardStorages.Length};reserved-array-retains-all-partial-shards;" +
            "actual-native-storage-destroyed-by-explicit-dispose-before-finalizers;logical-ranks-on-one-real-context-not-multidevice";
    }

    private sealed record PartialShardEvidence(WeakReference[] ShardStorages);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseTpQuantizedCopies(string mode, string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        WeakReference[] sources = model.ShardCopiedQuantSources(mode);
        Require(model.QuantShardCount == 2 && model.SourceQuantCount == 0 && sources.All(
            reference => reference.Target is QuantizedWeight source && !source.HasHostData),
            "Actual independent quantized destinations transfer before their original sources explicitly retire.");
        QuantizedWeight[] shards = model.QuantShards;
        Require(shards.All(shard => shard.HasHostData && !shard.HasExternalHostView && shard.RawBytes > 0),
            "Every completed raw-copy or Q8 requantization shard has an actual owned host buffer.");
        model.Dispose();
        Require(shards.All(shard => !shard.HasHostData && shard.Data == IntPtr.Zero) && RuntimeResourceCount() == 0,
            "Explicit model disposal clears all actual quantized host owners before finalizer drainage.");
        return "actual-quantized-copy-or-requantize;explicit-source-and-shard-disposal;no-raw-post-disposal-read;logical-ranks-not-multidevice";
    }

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
        Require(model.MappingOpen,
            "Early and late guarded view unregister refusals both block backing-map release.");
        return poisonLate
            ? "late-failed-TP-view-unregister-preserves-backing-map;actual-managed-owner-retained;no-pointer-read-after-poison;not-GPU-fault"
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
        if (mode == "observe-tp-sync-retirement")
        {
            Require(model.Group.SynchronizeCalls == 1 && model.GraphCleanupCalls == 0 &&
                RuntimeResourceCount() == 3 && model.SourceStorageOwned && model.SourceTensorCount == 1,
                "A failed production synchronization retains the actual source storage and model before graph/cache/weight retirement.");
            return;
        }
        Require(RuntimeResourceCount() == 2 && model.QuantShardCount == 2 && model.FirstQuantShard.HasHostData &&
            RegisteredBonsaiKeys(model.FirstQuantShard) == 1,
            "The actual failed view, context/Model ownership and registration remain, not only numeric lease IDs.");
        Require(model.MappingOpen, "Every refused TP view keeps its actual backing mapping owned.");
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
        if (!preflight)
        {
            Exception? failure = null;
            try { model.Dispose(); }
            catch (IOException error) { failure = error; }
            Require(failure != null && model.Group.SynchronizeCalls == 1 && model.GraphCleanupCalls == 0 &&
                model.SourceTensorCount == 1 && model.SourceStorageOwned && RuntimeResourceCount() == 3,
                "Production disposal consults synchronization before any graph/cache/storage retirement and retains the actual owners on refusal.");
            try { model.Dispose(); }
            catch (InvalidOperationException repeated)
            {
                Require(ReferenceEquals(repeated.InnerException, failure) && model.Group.SynchronizeCalls == 1,
                    "Terminal repeated teardown preserves the original synchronization fault without retrying work.");
            }
            return "production-logical-group-sync-refusal-before-graphs-caches-storage;actual-model-source-retained;not-CUDA-stream-or-GPU-fault";
        }
        model.Dispose();
        Require(model.GraphCleanupCalls == 1 && !model.SourceStorageOwned && RuntimeResourceCount() == 0,
            "The model executes the controlled graph-phase marker and real storage cleanup, then releases its native resources.");
        Require(model.Group.SynchronizeCalls == 2,
            "Successful production cleanup also synchronizes after the separate caller preflight observation.");
        return "caller-preflight-refusal-before-retirement;explicit-production-sync-cleanup;controlled-logical-group-not-CUDA-stream";
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
        internal int GraphCleanupCalls;
        protected override bool OwnsTensorParallelGroup => false;
        internal int QuantShardCount => _tpQuantWeights.Values.Sum(shards => shards.Count(weight => weight != null));
        internal int F32ShardCount => _tpWeights.Values.Sum(shards => shards.Count(weight => weight != null));
        internal int SourceTensorCount => _weights.Count;
        internal int SourceQuantCount => _quantWeights.Count;
        internal QuantizedWeight FirstQuantShard => _tpQuantWeights.Single().Value[0];
        internal QuantizedWeight[] QuantShards => _tpQuantWeights.Single().Value;
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

        internal WeakReference[] ShardCopiedQuantSources(string mode)
        {
            var source = new QuantizedWeight(new byte[288], (int)GgmlTensorType.Q4_0, 128, 4);
            _quantWeights.Add("source0.weight", source);
            var sources = new List<WeakReference> { new(source) };
            switch (mode)
            {
                case "tp-quantized-copy-row": ShardWeightsForTensorParallelism([], ["source"]); break;
                case "tp-quantized-copy-concatenated": ShardConcatenatedColumnParallel("source0.weight", 2, 2); break;
                case "tp-quantized-requantize-separate":
                    var second = new QuantizedWeight(new byte[288], (int)GgmlTensorType.Q4_0, 128, 4);
                    _quantWeights.Add("source1.weight", second);
                    sources.Add(new(second));
                    ShardSeparateColumnParallel("fused.weight", ["source0.weight", "source1.weight"], [4, 4]);
                    break;
                default: throw new ArgumentException("Unknown quantized TP copy route.", nameof(mode));
            }
            return sources.ToArray();
        }

        internal void AddSourceTensor(string name, params long[] shape)
        {
            var source = new Tensor(_rankAllocator, DType.Float32, shape);
            source.SetElementsAsFloat(Enumerable.Repeat(1f, (int)source.ElementCount()).ToArray());
            _weights.Add(name, source);
            _sourceStorage = new WeakReference(source.Storage);
        }

        internal Tensor CreateBroadcastSource()
        {
            var source = new Tensor(_rankAllocator, DType.Float32, 2, 2);
            source.SetElementsAsFloat([1f, 2f, 3f, 4f]);
            return source;
        }

        internal Tensor[] Broadcast(Tensor source) => BroadcastTensorToAllRanks(source);
        internal void OwnBroadcastSource(Tensor source) => _weights.Add("broadcast.source", source);

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
            ownsTensorParallelGroup: OwnsTensorParallelGroup,
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
