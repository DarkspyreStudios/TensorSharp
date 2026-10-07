using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

public sealed class HostRuntimeRetirementTests
{
    [Fact]
    public void RuntimeShutdownFollowsActualContainerOwnedModelDisposal()
    {
        var model = new OwnedModel();
        int shutdowns = 0;
        HostRuntimeRetirement.Run(() =>
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Services.AddSingleton(_ => model);
            using IHost host = builder.Build();
            Assert.Same(model, host.Services.GetRequiredService<OwnedModel>());
            Assert.False(model.Disposed);
        }, () =>
        {
            Assert.True(model.Disposed);
            shutdowns++;
        });
        Assert.Equal(1, shutdowns);
    }

    [Fact]
    public void HostFailureStillRetiresRuntimeAndPreservesOriginalCause()
    {
        var original = new IOException("host run failed");
        int shutdowns = 0;
        Exception observed = Assert.Throws<IOException>(() => HostRuntimeRetirement.Run(
            () => throw original, () => shutdowns++));
        Assert.Same(original, observed);
        Assert.Equal(1, shutdowns);
    }

    [Fact]
    public void IndependentHostAndRuntimeFailuresRemainOrderedWithoutReplay()
    {
        var original = new IOException("host run failed");
        var cleanup = new InvalidOperationException("runtime retirement failed");
        int shutdowns = 0;
        var observed = Assert.Throws<AggregateException>(() => HostRuntimeRetirement.Run(
            () => throw original, () => { shutdowns++; throw cleanup; }));
        Assert.Collection(observed.InnerExceptions,
            first => Assert.Same(original, first), second => Assert.Same(cleanup, second));
        Assert.Equal(1, shutdowns);
    }

    [Fact]
    public void RuntimeFailureCannotBecomeSuccessfulHostExit()
    {
        var cleanup = new InvalidOperationException("runtime retirement failed");
        Exception observed = Assert.Throws<InvalidOperationException>(() => HostRuntimeRetirement.Run(
            () => { }, () => throw cleanup));
        Assert.Same(cleanup, observed);
    }

    private sealed class OwnedModel : IDisposable
    {
        internal bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }
}
