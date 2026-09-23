// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Server;
using TensorSharp.Server.Jev;

namespace InferenceWeb.Tests;

public sealed class DiffusionSchedulerShutdownTests
{
    private static DiffusionBatchScheduler CancelledRequestsOnlyScheduler()
    {
        // This fixture deliberately has no tokenizer, weights, allocator or backend.
        // Only already-canceled requests are submitted while its worker is alive;
        // any accidental admission into model execution fails the test.
        var model = (DiffusionGemmaModel)RuntimeHelpers.GetUninitializedObject(typeof(DiffusionGemmaModel));
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { VocabSize = 16 });
        return new DiffusionBatchScheduler(model, null, 2);
    }

    [Fact]
    public async Task DisposedSchedulerRejectsLateSubmissionsAndRepeatedDisposalIsSafe()
    {
        var scheduler = CancelledRequestsOnlyScheduler();
        scheduler.Dispose();
        scheduler.Dispose();
        var handle = scheduler.Submit([1], new DiffusionEbParams(), CancellationToken.None);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handle.Completion);
        await handle.Previews.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(0, scheduler.ActiveCount);
    }

    [Fact]
    public async Task CancelledSubmissionsRacingShutdownAllCompleteWithoutNativeExecution()
    {
        using var scheduler = CancelledRequestsOnlyScheduler();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        using var start = new ManualResetEventSlim();
        var submitters = Enumerable.Range(0, 32).Select(_ => Task.Run(async () =>
        {
            start.Wait();
            for (int n = 0; n < 8; ++n)
            {
                var handle = scheduler.Submit([1], new DiffusionEbParams(), cancellation.Token);
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handle.Completion);
                await handle.Previews.Completion.WaitAsync(TimeSpan.FromSeconds(5));
            }
        })).ToArray();
        var shutdown = Task.Run(() => { start.Wait(); scheduler.Dispose(); });
        start.Set();
        await Task.WhenAll(submitters.Append(shutdown)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(0, scheduler.ActiveCount);
    }

    [Fact]
    public async Task CancellationDoesNotReleaseModelLeaseWhilePhysicalWorkIsStillRunning()
    {
        var gate = new JevExecutionGate(2);
        using var cancellation = new CancellationTokenSource();
        using var release = new ManualResetEventSlim();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = gate.ExecuteAsync(ct =>
        {
            started.SetResult();
            release.Wait(); // Simulate a native dispatch that cannot be interrupted.
            ct.ThrowIfCancellationRequested();
            return 1;
        }, cancellation.Token);
        try
        {
            await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
            cancellation.Cancel();
            Assert.False(work.IsCompleted);
            var changed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var change = Task.Run(() => { using var lease = gate.BeginChange(); changed.SetResult(); });
            Assert.False(changed.Task.IsCompleted);
            release.Set();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => work);
            await change.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(2, await gate.ExecuteAsync(ct => 2, CancellationToken.None));
        }
        finally { release.Set(); }
    }

    [Fact]
    public async Task ShutdownDiscardsAdmittedWaitersBeforeTheirDelegatesRun()
    {
        var gate = new JevExecutionGate(3);
        using var release = new ManualResetEventSlim();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = gate.ExecuteAsync(ct => { started.SetResult(); release.Wait(); return 1; }, CancellationToken.None);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        bool queuedRan = false;
        var queued = gate.ExecuteAsync(ct => { queuedRan = true; return 2; }, CancellationToken.None);
        Exception? shutdownError = null;
        var shutdown = new Thread(() =>
        {
            try { using var lease = gate.BeginChange(shutdown: true); }
            catch (Exception error) { shutdownError = error; }
        }) { IsBackground = true };
        shutdown.Start();
        try
        {
            // BeginChange closes admission before blocking on active work. This
            // dedicated thread does no other blocking, so its wait proves closure.
            Assert.True(SpinWait.SpinUntil(() => (shutdown.ThreadState & ThreadState.WaitSleepJoin) != 0, TimeSpan.FromSeconds(5)));
            await Assert.ThrowsAsync<ObjectDisposedException>(() => gate.ExecuteAsync(ct => 3, CancellationToken.None));
            release.Set();
            Assert.Equal(1, await work);
            await Assert.ThrowsAsync<ObjectDisposedException>(() => queued);
            Assert.False(queuedRan);
        }
        finally
        {
            release.Set();
            Assert.True(shutdown.Join(TimeSpan.FromSeconds(5)));
        }
        Assert.Null(shutdownError);
    }
}
