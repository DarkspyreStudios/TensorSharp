using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxStorageReleaseTests
{
    [Fact]
    public void FailedStreamCompletionPreservesTheArrayReference()
    {
        var pointer = new IntPtr(1);
        var array = Unsafe.BitCast<IntPtr, MlxNative.MlxArray>(pointer);
        var failure = new InvalidOperationException("Stream completion failed.");
        var stage = NativeRuntimeFailureStage.StorageRelease;

        Exception observed = Assert.Throws<InvalidOperationException>(() =>
            MlxStorage.ReleaseDeviceArray(ref array, ref stage, () => throw failure, _ => { }));

        Assert.Same(failure, observed);
        Assert.Equal(pointer, array.Ctx);
        Assert.Equal(NativeRuntimeFailureStage.Synchronization, stage);
    }
}
