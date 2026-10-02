#nullable enable

using System.Runtime.CompilerServices;
using System.Collections;
using System.Reflection;
using System.Runtime.Loader;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    private static readonly float[] BroadcastSourceValues = [1f, 2f, 3f, 4f];
    private static readonly float[] BroadcastMutatedValues = [9f, 8f, 7f, 6f];
    private static WeakReference[]? _broadcastRetainedResources;
    private static int _broadcastRetainedCount;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseGgmlBroadcast(string mode, string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        Tensor? source = null;
        Tensor[]? outputs = null;
        var captured = new List<Tensor>();
        Action<object?[]>? originalHook = OpRegistry.PreInvokeHook;
        var original = new IOException("controlled GGML broadcast copy refusal");
        try
        {
            source = model.CreateBroadcastSource();
            var sourceStorage = (GgmlStorage)source.Storage;
            if (mode is "tp-broadcast-partial-copy" or "tp-broadcast-second-copy")
            {
                int refusedCopy = mode == "tp-broadcast-second-copy" ? 2 : 1;
                OpRegistry.PreInvokeHook = arguments =>
                {
                    originalHook?.Invoke(arguments);
                    if (arguments is [Tensor result, Tensor input] && ReferenceEquals(input, source))
                    {
                        captured.Add(result);
                        if (captured.Count == refusedCopy) throw original;
                    }
                };
                Exception? failure = null;
                try { outputs = model.Broadcast(source); }
                catch (IOException error) { failure = error; }
                finally { OpRegistry.PreInvokeHook = originalHook; }
                Require(ReferenceEquals(failure, original) && captured.Count == refusedCopy &&
                    captured.All(copy => StorageDestroyed((GgmlStorage)copy.Storage)) && !StorageDestroyed(sourceStorage) && RuntimeResourceCount() == 3,
                    "A refused copy explicitly retires every actual partial destination and preserves the original error and borrowed source.");
            }
            else
            {
                outputs = model.Broadcast(source);
                if (mode == "tp-broadcast-temporary-retirement")
                {
                    InvokeBroadcastCleanup(model, "DisposeTensorParallelBroadcast", outputs, source);
                    Require(outputs.All(output => StorageDestroyed((GgmlStorage)output.Storage)) && StorageDestroyed(sourceStorage),
                        "Owning temporary cleanup releases every independently owned rank, including rank zero, and its owned original source.");
                }
                else if (mode == "tp-broadcast-source-disposal")
                {
                    InvokeBroadcastCleanup(model, "RetireTensorParallelBroadcastSource", source, outputs);
                    Require(StorageDestroyed(sourceStorage) && outputs.All(output => !StorageDestroyed((GgmlStorage)output.Storage)),
                        "Caller source disposal does not destroy any returned GGML broadcast storage.");
                }
                else
                {
                    Require(outputs.Length == 2 && outputs.All(output => !ReferenceEquals(output, source) &&
                        !ReferenceEquals(output.Storage, sourceStorage)) && !ReferenceEquals(outputs[0].Storage, outputs[1].Storage),
                        "Every returned GGML rank owns independent storage, not the borrowed original or another rank.");
                    outputs[1].SetElementsAsFloat(BroadcastMutatedValues);
                    Require(source.GetElementsAsFloat(4).SequenceEqual(BroadcastSourceValues) &&
                        outputs[0].GetElementsAsFloat(4).SequenceEqual(BroadcastSourceValues),
                        "Mutation of one owned rank leaves source and other rank unchanged.");
                    source.Dispose();
                    Require(outputs[0].GetElementsAsFloat(4).SequenceEqual(BroadcastSourceValues),
                        "A returned independent output remains readable after explicit source retirement.");
                }
            }
            return "actual-GGML-broadcast-owned-ranks;borrowed-input-preserved;explicit-partial-rollback;logical-ranks-not-multidevice";
        }
        finally
        {
            OpRegistry.PreInvokeHook = originalHook;
            // Baseline red teardown owns captured destinations that production failed to retire.
            foreach (Tensor tensor in captured) tensor.Dispose();
            if (outputs != null) foreach (Tensor output in outputs) output.Dispose();
            source?.Dispose();
            model.Dispose();
        }
    }

    private static void InvokeBroadcastCleanup(ModelBase model, string method, params object?[] arguments) =>
        InvokeExecutionBoundary(typeof(ModelBase).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!, model, arguments);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseBroadcastCleanupRefusal(string mode, string path, BackendType backend)
    {
        var model = new LogicalRankModel(path, backend);
        var allocator = (GgmlAllocator)model.Group.Allocator;
        var generation = AssemblyLoadContext.GetLoadContext(typeof(ForeignModelLifetime).Assembly);
        Require(generation != null && generation != AssemblyLoadContext.Default &&
            new[] { typeof(OpRegistry), typeof(GgmlBasicOps), typeof(CleanupFailingStorage), typeof(ModelBase) }.All(type =>
                AssemblyLoadContext.GetLoadContext(type.Assembly) == generation && type.Assembly.ManifestModule.ModuleVersionId != Guid.Empty),
            "The finite fixture registration, real copy implementation and storage subclass belong to the same actual private generation.");
        int actualCopies = 0;
        Type[] types = [typeof(GgmlStorage), typeof(CleanupFailingStorage)];
        foreach (Type resultType in types)
            foreach (Type sourceType in types)
                if (resultType == typeof(CleanupFailingStorage) || sourceType == typeof(CleanupFailingStorage))
                    OpRegistry.Register("copy", arguments =>
                    {
                        GgmlBasicOps.Copy((Tensor)arguments[0]!, (Tensor)arguments[1]!);
                        actualCopies++;
                        return null;
                    }, [new ArgCountConstraint(2), new ArgStorageTypeConstraint(0, resultType, false), new ArgStorageTypeConstraint(1, sourceType, false)]);
        bool refuseSource = mode == "tp-broadcast-source-cleanup-refusal";
        bool refuseTemporary = mode == "tp-broadcast-temporary-cleanup-refusal";
        var source = refuseSource
            ? new Tensor(new CleanupFailingAllocator(allocator.Context, allocator.DeviceId), DType.Float32, 2, 2)
            : model.CreateBroadcastSource();
        Tensor[] outputs;
        Exception failure;
        if (refuseSource || refuseTemporary)
        {
            if (refuseTemporary) model.Group.Allocator = new CleanupFailingAllocator(allocator.Context, allocator.DeviceId);
            outputs = model.Broadcast(source);
            try
            {
                if (refuseTemporary) InvokeBroadcastCleanup(model, "DisposeTensorParallelBroadcast", outputs, source);
                else InvokeBroadcastCleanup(model, "RetireTensorParallelBroadcastSource", source, outputs);
                throw new InvalidOperationException("Broadcast cleanup unexpectedly succeeded.");
            }
            catch (InvalidOperationException error) { failure = error; }
            Require(ReferenceEquals(ExecutionCleanupFailure(model), failure), "Caller cleanup preserves its exact original refusal.");
        }
        else
        {
            // This owning caller retains its borrowed input independently of the broadcast output rollback owner.
            model.OwnBroadcastSource(source);
            model.Group.Allocator = new CleanupFailingAllocator(allocator.Context, allocator.DeviceId);
            var copied = new List<Tensor>();
            var original = new IOException("controlled second broadcast copy refusal with failed rollback");
            Action<object?[]>? hook = OpRegistry.PreInvokeHook;
            try
            {
                OpRegistry.PreInvokeHook = arguments =>
                {
                    hook?.Invoke(arguments);
                    if (arguments is [Tensor result, Tensor input] && ReferenceEquals(input, source))
                    {
                        copied.Add(result);
                        if (copied.Count == 2) throw original;
                    }
                };
                try { _ = model.Broadcast(source); throw new InvalidOperationException("Copy refusal unexpectedly succeeded."); }
                catch (AggregateException error)
                {
                    Require(error.InnerExceptions.Count == 2 && ReferenceEquals(error.InnerExceptions[0], original) &&
                        ReferenceEquals(error.InnerExceptions[1], ExecutionCleanupFailure(model)),
                        "Failed rollback preserves the original copy error and exact cleanup error together.");
                    failure = error.InnerExceptions[1];
                }
            }
            finally { OpRegistry.PreInvokeHook = hook; }
            outputs = copied.ToArray();
            Require(outputs.Length == 2, "Both real destinations exist before the controlled second-copy refusal.");
        }
        Tensor[] all = [source, .. outputs];
        Require(actualCopies == (refuseSource || refuseTemporary ? 2 : 1), "The exact fixture storage handler executes actual GGML copies before the managed cleanup refusal.");
        _broadcastRetainedResources = [new(model), .. all.SelectMany(tensor => new[] { new WeakReference(tensor), new WeakReference(tensor.Storage) })];
        _broadcastRetainedCount = RuntimeResourceCount();
        ExpectTerminal(model.Dispose, failure);
        ExpectTerminal(model.Dispose, failure);
        Require(all.All(tensor => !StorageDestroyed((GgmlStorage)tensor.Storage)) && ModelLeaseCount() == 1,
            "Terminal cleanup refusal preserves actual source and all returned/partial copy storages, and refuses repeated model cleanup.");
        return "actual-GGML-broadcast-cleanup-refusal;original-errors-preserved;source-and-whole-copy-array-retained;managed-refusal-not-GPU-fault";
    }

    private static void VerifyBroadcastRetention()
    {
        var retained = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(_broadcastRetainedResources != null && _broadcastRetainedResources.All(reference => reference.IsAlive) &&
            retained.Count == 1 && ReferenceEquals(retained[0], _broadcastRetainedResources[0].Target) &&
            RuntimeResourceCount() == _broadcastRetainedCount && GgmlNativeLoader.State == GgmlRuntimeState.Ready,
            "The actual failed model, source, every copy and native storage ownership survive outer-frame finalizer drainage.");
    }
}
