using System;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.MLX;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class MlxWorkerRetirementTests
{
    [Fact]
    public async Task RetirementDrainsAcceptedWorkAndClearsCachesBeforeClosingTheWorker()
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var cachesCleared = new ManualResetEventSlim();
        int completed = 0;
        int cleanups = 0;
        int retirements = 0;
        using var worker = new MlxWorker(current =>
        {
            Assert.True(current.IsOnWorkerThread);
            Assert.Equal(42, Volatile.Read(ref completed));
            Interlocked.Increment(ref cleanups);
            cachesCleared.Set();
        }, () =>
        {
            Assert.True(cachesCleared.IsSet);
            Interlocked.Increment(ref retirements);
        });
        Task<int> work = Task.Run(() => worker.Invoke(() =>
        {
            entered.Set();
            Assert.True(release.Wait(TimeSpan.FromSeconds(5)));
            Volatile.Write(ref completed, 42);
            return 42;
        }));
        Task retirement = null;
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
            retirement = Task.Run(worker.Dispose);
            Assert.False(cachesCleared.IsSet);
            release.Set();
            Assert.Equal(42, await work.WaitAsync(TimeSpan.FromSeconds(5)));
            await retirement.WaitAsync(TimeSpan.FromSeconds(5));
            InvalidOperationException refused = Assert.Throws<InvalidOperationException>(() => worker.Invoke(() => 0));
            Assert.Equal("Native ownership was already safely released.", refused.Message);
            worker.Dispose();
            Assert.Equal(1, cleanups);
            Assert.Equal(1, retirements);
        }
        finally
        {
            release.Set();
            await work.WaitAsync(TimeSpan.FromSeconds(5));
            if (retirement is not null) await retirement.WaitAsync(TimeSpan.FromSeconds(5));
        }
    }
}
