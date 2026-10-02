#nullable enable

using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    private static readonly float[] BroadcastSourceValues = [1f, 2f, 3f, 4f];
    private static readonly float[] BroadcastMutatedValues = [9f, 8f, 7f, 6f];

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
            if (mode == "tp-broadcast-partial-copy")
            {
                OpRegistry.PreInvokeHook = arguments =>
                {
                    originalHook?.Invoke(arguments);
                    if (arguments is [Tensor result, Tensor input] && ReferenceEquals(input, source))
                    {
                        captured.Add(result);
                        throw original;
                    }
                };
                Exception? failure = null;
                try { outputs = model.Broadcast(source); }
                catch (IOException error) { failure = error; }
                finally { OpRegistry.PreInvokeHook = originalHook; }
                Require(ReferenceEquals(failure, original) && captured.Count == 1 &&
                    StorageDestroyed((GgmlStorage)captured[0].Storage) && !StorageDestroyed(sourceStorage) && RuntimeResourceCount() == 3,
                    "A failed first copy explicitly retires its untransferred actual destination and preserves the original error and borrowed source.");
            }
            else
            {
                outputs = model.Broadcast(source);
                if (mode == "tp-broadcast-source-disposal")
                {
                    source.Dispose();
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
}
