using System.Reflection;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxClosureErrorTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(7)]
    public void ClosureStatusPreservesCallbackAndNativeFailures(int nativeStatus)
    {
        var callback = new InvalidOperationException("Trace operation failed.");
        var cleanup = new InvalidOperationException("Trace cleanup failed.");
        var invocation = new MlxNative.ClosureInvocation();
        invocation.Traces.Add(new MlxNative.ClosureTrace { Error = callback, CleanupError = cleanup });
        MethodInfo check = typeof(MlxNative).GetMethod("CheckClosureStatus", BindingFlags.Static | BindingFlags.NonPublic)!;

        // Supply a native return code without invoking the expensive native execution seam.
        var wrapper = Assert.Throws<TargetInvocationException>(() =>
            check.Invoke(null, [invocation, nativeStatus, "applying compiled MLX closure"]));
        var errors = Assert.IsType<AggregateException>(wrapper.InnerException).Flatten().InnerExceptions;

        Assert.Equal(nativeStatus == 0 ? 2 : 3, errors.Count);
        Assert.Same(callback, Assert.Single(errors, error => ReferenceEquals(error, callback)));
        Assert.Same(cleanup, Assert.Single(errors, error => ReferenceEquals(error, cleanup)));
        Assert.Same(cleanup, invocation.CleanupError);
        if (nativeStatus != 0)
        {
            Exception native = Assert.Single(errors, error => !ReferenceEquals(error, callback) && !ReferenceEquals(error, cleanup));
            Assert.Contains("applying compiled MLX closure", native.Message);
            Assert.Contains($"error code {nativeStatus}", native.Message);
        }
    }
}
