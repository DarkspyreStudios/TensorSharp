// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Whether the host is allowed to run the model on the GPU right now.
    ///
    /// <para>
    /// It exists for one rule that iOS does not negotiate: <b>an app that is not
    /// frontmost may not submit work to the GPU.</b> A Metal command buffer committed
    /// from the background comes back <c>MTLCommandBufferStatusError</c> with
    /// <c>kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted</c>, and
    /// ggml-metal's response is not to retry — it sets a sticky <c>has_error</c> flag
    /// whose own comment reads "once set, graph_compute will return GGML_STATUS_FAILED
    /// until the backend is recreated". So one command buffer submitted a moment after
    /// the user swipes away does not spoil one token: it spoils the model for the rest
    /// of the process, and every message afterwards fails too.
    /// </para>
    /// <para>
    /// The answer is to stop asking. The gate is CLOSED while the app is not frontmost,
    /// and everything that would submit GPU work waits on it instead: the
    /// <see cref="InferenceEngine"/>'s step loop between steps (the engine runs on its
    /// own thread and would otherwise keep decoding into an unbounded channel whether
    /// or not anyone was reading), and the host's stream wrapper between pulls. A turn
    /// does not end; it carries on from the same token when the user comes back.
    /// </para>
    /// <para>
    /// It lives in the runtime, next to the engine that honours it, precisely so that
    /// nothing here knows about iOS: the app's head opens and closes it from the
    /// lifecycle notifications, a server never touches it, and a test drives it
    /// directly. Open is the only state a process that never closes it can observe,
    /// and in that state <see cref="Wait"/> and <see cref="WaitAsync"/> cost one
    /// volatile read.
    /// </para>
    /// </summary>
    public sealed class ComputeGate
    {
        private readonly object _lock = new object();
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private TaskCompletionSource _open = Opened();
        private TimeSpan _closedAt;
        private TimeSpan _totalClosed;
        private long _closures;

        private static TaskCompletionSource Opened()
        {
            var source = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            source.SetResult();
            return source;
        }

        /// <summary>Whether work may be submitted right now.</summary>
        public bool IsOpen
        {
            get { lock (_lock) return _open.Task.IsCompleted; }
        }

        /// <summary>
        /// How many times the gate has been closed since it was made.
        ///
        /// <para>
        /// A counter rather than a flag, so a caller can ask a question no flag can
        /// answer: "was the app ever away WHILE I was running?" — which is what decides
        /// whether a failed turn is worth blaming on the GPU. By the time a failure is
        /// handled the app is always in front again, so "is it open now" says nothing.
        /// </para>
        /// </summary>
        public long Closures
        {
            get { lock (_lock) return _closures; }
        }

        /// <summary>Total time spent closed, for the log. Still accruing while closed.</summary>
        public TimeSpan TotalClosed
        {
            get
            {
                lock (_lock)
                    return _open.Task.IsCompleted ? _totalClosed : _totalClosed + (_clock.Elapsed - _closedAt);
            }
        }

        /// <summary>Raised after the gate opens (true) or closes (false), on the caller's thread.</summary>
        public event Action<bool>? Changed;

        /// <summary>Work may be submitted again. Idempotent.</summary>
        public void Open()
        {
            TaskCompletionSource release;
            lock (_lock)
            {
                if (_open.Task.IsCompleted)
                    return;
                _totalClosed += _clock.Elapsed - _closedAt;
                release = _open;
            }
            // Completed OUTSIDE the lock. The continuations are generation threads
            // waiting to take their next step, and RunContinuationsAsynchronously keeps
            // them off this one -- which is the UI thread, arriving from a lifecycle
            // notification.
            release.TrySetResult();
            Changed?.Invoke(true);
        }

        /// <summary>No more work may be submitted. Idempotent.</summary>
        public void Close()
        {
            lock (_lock)
            {
                if (!_open.Task.IsCompleted)
                    return;
                _open = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                _closedAt = _clock.Elapsed;
                _closures++;
            }
            Changed?.Invoke(false);
        }

        /// <summary>
        /// Wait until work may be submitted.
        ///
        /// <para>
        /// Returns a completed task in the ordinary case — the app is in front and this
        /// is on the path of every single token — so the common case allocates nothing
        /// and does not yield.
        /// </para>
        /// </summary>
        public Task WaitAsync(CancellationToken cancellationToken = default)
        {
            Task open;
            lock (_lock)
            {
                if (_open.Task.IsCompleted)
                    return Task.CompletedTask;
                open = _open.Task;
            }
            return cancellationToken.CanBeCanceled
                ? open.WaitAsync(cancellationToken)
                : open;
        }

        /// <summary>
        /// Block the calling thread until work may be submitted.
        ///
        /// <para>
        /// For the engine's own worker thread, which is synchronous by design: it runs
        /// one scheduler step after another and has no async context to yield to. A
        /// closed gate parks it here between two steps, which is exactly the point
        /// where no command buffer is in flight.
        /// </para>
        /// </summary>
        /// <exception cref="OperationCanceledException">The wait was cancelled, typically by the engine shutting down.</exception>
        public void Wait(CancellationToken cancellationToken = default)
        {
            Task open;
            lock (_lock)
            {
                if (_open.Task.IsCompleted)
                    return;
                open = _open.Task;
            }
            // Task.Wait wraps a cancellation in an AggregateException; the callers want
            // the OperationCanceledException itself, the way every other blocking wait
            // in the engine reports it.
            try
            {
                open.Wait(cancellationToken);
            }
            catch (AggregateException ex) when (ex.InnerException is OperationCanceledException cancelled)
            {
                throw cancelled;
            }
        }
    }
}
