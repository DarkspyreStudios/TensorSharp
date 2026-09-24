using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public class Qwen35BatchedDecodeDiagnosticsTests
{
    [Fact]
    public void DeclineReason_ReachesSchedulerInterface_AndTracksLatestAttempt()
    {
        // No weights/backend are needed to exercise the early admission gates.
        var model = (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        IBatchedPagedModel fused = model;
        Assert.False(fused.TryForwardBatchedFusedDecode(
            ["a", "b", "c"], [1, 2, 3], [8, 9, 10], new float[3][]));
        Assert.Contains("backend", fused.BatchedFusedDecodeDeclineReason);

        typeof(ModelBase).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, BackendType.GgmlCuda);
        Assert.False(fused.TryForwardBatchedFusedDecodeSampled(
            ["a", "b", "c"], [1, 2, 3], [8, 9, 10], new int[3]));
        Assert.Equal("per-sequence caches have not been initialized", fused.BatchedFusedDecodeDeclineReason);
    }
}
