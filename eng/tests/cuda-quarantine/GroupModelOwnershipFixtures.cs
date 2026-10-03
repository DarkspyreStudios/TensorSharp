using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.CudaQuarantineFixture;

internal static class GroupModelOwnershipFixtures
{
    internal static void GroupBusy()
    {
        var firstApi = new RecordingCudaApi();
        var secondApi = new RecordingCudaApi();
        var first = new CudaAllocator(0, firstApi);
        var second = new CudaAllocator(1, secondApi);
        var group = new TensorParallelGroup(new[] { first, second });
        var tensor = new Tensor(second, DType.Float32, 4);
        int calls = firstApi.TotalCalls + secondApi.TotalCalls;
        Exception? failure = Capture(group.Dispose);
        Console.WriteLine($"Group observation: failure={failure?.GetType().Name}, callsBefore={calls}, callsAfter={firstApi.TotalCalls + secondApi.TotalCalls}, firstReleased={first.Context.IsDisposed}, secondReleased={second.Context.IsDisposed}.");
        Require(failure is CudaAllocatorBusyException && firstApi.TotalCalls + secondApi.TotalCalls == calls
            && !first.Context.IsDisposed && !second.Context.IsDisposed,
            "Group Busy destroys an earlier rank before preflighting every owned allocator.");
        tensor.Dispose();
        group.Dispose();
        Require(first.Context.IsDisposed && second.Context.IsDisposed,
            "Group Busy caches completion and prevents the explicit drained retry.");
    }

    internal static void ModelBusy()
    {
        WithMetadata(path =>
        {
            var api = new RecordingCudaApi();
            var allocator = new CudaAllocator(0, api);
            var group = new TensorParallelGroup(new[] { allocator });
            var model = new MinimalModel(path, group);
            var escaped = model.Weight.CopyRef();
            int calls = api.TotalCalls;
            Exception? failure = Capture(model.Dispose);
            Console.WriteLine($"Model observation: failure={failure?.GetType().Name}, cleanupCount={model.CleanupCount}, callsBefore={calls}, callsAfter={api.TotalCalls}, allocatorReleased={allocator.Context.IsDisposed}.");
            Require(failure is CudaAllocatorBusyException && model.CleanupCount == 0
                && api.TotalCalls == calls && !allocator.Context.IsDisposed,
                "Model Busy runs derived cleanup or destroys weights before the complete ownership preflight.");
            Require(!NativeRuntimeQuarantine.TryGetFailure(failure!, out _), "Healthy model Busy records quarantine.");
            escaped.Dispose();
            model.Dispose();
            Require(model.CleanupCount == 1 && allocator.Context.IsDisposed,
                "Model healthy Busy cannot retry its actual owned weight and group cleanup.");
        });
    }

    internal static void ConstructorBusy()
    {
        WithMetadata(path =>
        {
            var api = new RecordingCudaApi();
            var allocator = new CudaAllocator(0, api);
            var group = new TensorParallelGroup(new[] { allocator });
            var original = new InvalidOperationException("Controlled derived construction failure.");
            MinimalModel? partial = null;
            Tensor? escaped = null;
            Exception? failure = Capture(() => _ = new MinimalModel(path, group, original, (model, view) =>
            {
                partial = model;
                escaped = view;
            }, new RecordingCudaApi()));
            Console.WriteLine($"Constructor observation: failure={failure?.GetType().Name}, originalPreserved={failure is AggregateException a && ReferenceEquals(a.InnerExceptions[0], original)}, cleanupCount={partial?.CleanupCount}.");
            Require(failure is AggregateException aggregate && aggregate.InnerExceptions.Count == 2
                && ReferenceEquals(aggregate.InnerExceptions[0], original)
                && aggregate.InnerExceptions[1] is CudaAllocatorBusyException
                && partial != null && escaped != null && partial.CleanupCount == 0,
                "Failed-construction rollback destroys derived ownership before refusing an escaped reference.");
            escaped!.Dispose();
            partial!.Dispose();
            Require(partial.CleanupCount == 1 && allocator.Context.IsDisposed,
                "Failed-construction healthy Busy does not permit explicit parent retry.");
        });
    }

    private sealed class MinimalModel : ModelBase
    {
        internal readonly Tensor Weight;
        internal int CleanupCount;
        private readonly CudaAllocator? _derivedAllocator;
        private readonly Tensor? _derivedTensor;

        internal MinimalModel(string path, TensorParallelGroup group, Exception? original = null,
            Action<MinimalModel, Tensor>? capture = null, RecordingCudaApi? derivedApi = null)
            : base(path, BackendType.Cuda, tpGroup: group)
        {
            Weight = new Tensor(group.GetAllocator(0), DType.Float32, 4);
            _weights.Add("fixture.weight", Weight);
            if (original == null) return;
            try
            {
                _derivedAllocator = new CudaAllocator(1, derivedApi!);
                _derivedTensor = new Tensor(_derivedAllocator, DType.Float32, 4);
                capture!(this, _derivedTensor.CopyRef());
                throw original;
            }
            catch (Exception error)
            {
                RollBackFailedConstruction(error, ReleaseDerived);
                throw;
            }
        }

        private void ReleaseDerived()
        {
            CleanupCount++;
            _derivedTensor?.Dispose();
            _derivedAllocator?.Dispose();
        }
        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
        public override void Dispose() => DisposeBaseResources(ReleaseDerived);
    }

    private static void WithMetadata(Action<string> action)
    {
        string directory = Path.Combine("tmp", "cuda-quarantine", "generated-metadata");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0ul);
                writer.Write(0ul);
                writer.Write(0ul);
            }
            action(path);
        }
        finally { File.Delete(path); }
    }

    private static Exception? Capture(Action action)
    {
        try { action(); return null; }
        catch (Exception failure) { return failure; }
    }

    private static void Require(bool value, string message)
    {
        if (!value) throw new InvalidOperationException(message);
    }
}
