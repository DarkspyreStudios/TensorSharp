using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxDeviceInitializationTests
{
    [Fact]
    public void MetalAvailabilityQueryFailurePreservesNativeError()
    {
        var failure = Assert.Throws<InvalidOperationException>(() => MlxNative.CheckMetalAvailability(7, false));

        Assert.Contains("checking MLX Metal availability", failure.Message);
        Assert.Contains("error code 7", failure.Message);
        Assert.Throws<PlatformNotSupportedException>(() => MlxNative.CheckMetalAvailability(0, false));
        MlxNative.CheckMetalAvailability(0, true);
    }
}
