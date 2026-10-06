using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class TensorParallelCallbackFailureTests
{
    [Theory]
    [InlineData(typeof(Qwen35Model))]
    [InlineData(typeof(GptOssModel))]
    [InlineData(typeof(MuseGlimmerModel))]
    public void ModelCallbackLeavesTheOriginalReducerFailureForTheNativeBoundaryOwner(Type modelType)
    {
        // Exercise only the callback state, without loading weights or a native runtime.
        object model = RuntimeHelpers.GetUninitializedObject(modelType);
        var group = new Reducer();
        typeof(ModelBase).GetField("_tpGroup", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, group);
        modelType.GetField("_tpCrossNodeBuf", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, Array.Empty<float>());
        var callback = (GgmlBasicOps.CrossNodeAllReduce)modelType
            .GetProperty("TpCrossNodeCallback", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        IntPtr buffer = Marshal.AllocHGlobal(2 * sizeof(float));
        try
        {
            Marshal.Copy(new[] { 2f, 5f }, 0, buffer, 2);
            Assert.True(callback(IntPtr.Zero, buffer, 2));
            var result = new float[2];
            Marshal.Copy(buffer, result, 0, result.Length);
            Assert.Equal(new[] { 4f, 10f }, result);

            var original = new InvalidOperationException("The cross-node exchange failed.");
            group.Failure = original;
            Assert.Same(original, Assert.Throws<InvalidOperationException>(() => callback(IntPtr.Zero, buffer, 2)));
            Marshal.Copy(buffer, result, 0, result.Length);
            Assert.Equal(new[] { 4f, 10f }, result);
            Assert.Equal(2, group.Calls);
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RankRestorationLeavesTheOriginalPendingErrorUnduplicated(bool restoreThrowsSameError)
    {
        var original = new InvalidOperationException("Plan execution failed.");
        int restoredRank = -1;
        Exception observed = Assert.Throws<InvalidOperationException>((Action)(() =>
        {
            Exception? operationFailure = null;
            try { throw original; }
            catch (Exception failure) { operationFailure = failure; throw; }
            finally
            {
                TensorParallelRankRestoration.Restore(3, operationFailure, rank =>
                {
                    restoredRank = rank;
                    if (restoreThrowsSameError) throw original;
                });
            }
        }));

        Assert.Same(original, observed);
        Assert.Equal(3, restoredRank);
    }

    [Fact]
    public void RankRestorationPreservesTheOriginalAndItsIndependentFailure()
    {
        var original = new InvalidOperationException("Plan execution failed.");
        var restoreError = new ArgumentOutOfRangeException("rank", "The previous rank could not be restored.");
        AggregateException observed = Assert.Throws<AggregateException>((Action)(() =>
        {
            Exception? operationFailure = null;
            try { throw original; }
            catch (Exception failure) { operationFailure = failure; throw; }
            finally { TensorParallelRankRestoration.Restore(3, operationFailure, _ => throw restoreError); }
        }));

        Assert.Equal(2, observed.InnerExceptions.Count);
        Assert.Same(original, observed.InnerExceptions[0]);
        Assert.Same(restoreError, observed.InnerExceptions[1]);
        Assert.Same(restoreError, Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorParallelRankRestoration.Restore(3, null, _ => throw restoreError)));
    }

    private sealed class Reducer : ITensorParallelGroup, INestedTensorParallelGroup
    {
        internal Exception? Failure;
        internal int Calls;
        public int Degree => 1;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public ITensorParallelGroup LocalGroup => this;

        public void CrossNodeAllReduce(float[] buffer, int count)
        {
            Calls++;
            if (Failure != null) throw Failure;
            for (int index = 0; index < count; index++) buffer[index] *= 2;
        }

        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() => throw new NotSupportedException();
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
