using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Threading;

namespace TensorSharp.MLX
{
    internal abstract class MlxNativeResources
    {
        internal bool SafelyReleased;
        internal virtual NativeRuntimeFailureStage CleanupFailureStage => NativeRuntimeFailureStage.GraphRelease;
    }

    public sealed class MlxWorker : IDisposable
    {
        private readonly BlockingCollection<IWorkItem> queue = new();
        private readonly List<MlxNativeResources> nativeResources = new();
        private readonly Thread thread;
        private readonly NativeOwnerRegistration nativeOwner;
        private int workerThreadId;
        private int disposed;

        public static MlxWorker Shared { get; } = new MlxWorker();

        private MlxWorker()
        {
            nativeOwner = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Worker);
            nativeOwner.AttachMlxSharedRuntime();
            thread = new Thread(Run)
            {
                IsBackground = true,
                Name = "TensorSharp MLX worker"
            };
            thread.Start();
        }

        // Returns true when called from inside an Invoke on the worker
        // thread itself — i.e. from a re-entrant context like an mlx_compile
        // trace callback. In that case, nesting another Invoke would deadlock
        // (worker thread is busy running us). Callers can detect this and run
        // the work inline.
        public bool IsOnWorkerThread => Thread.CurrentThread.ManagedThreadId == Volatile.Read(ref workerThreadId);

        public T Invoke<T>(Func<T> func)
        {
            if (func == null)
                throw new ArgumentNullException(nameof(func));
            ThrowIfDisposed();

            // Re-entrant: we're already on the worker. Run inline, otherwise
            // we'd block waiting for ourselves.
            if (IsOnWorkerThread)
                return func();

            NativeQuarantineAuthority.ValidateMlxWorkerDispatch(nativeOwner);
            var item = new WorkItem<T>(func);
            queue.Add(item);
            return item.GetResult();
        }

        public void Invoke(Action action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            Invoke(() =>
            {
                action();
                return 0;
            });
        }

        internal T InvokeNative<T>(Func<T> func)
        {
            if (func == null)
                throw new ArgumentNullException(nameof(func));

            return Invoke(() =>
            {
                using NativeEffectLease effect = nativeOwner.EnterEffect();
                MlxNative.InstallCurrentErrorHandler();
                return func();
            });
        }

        internal void InvokeNative(Action action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            InvokeNative(() =>
            {
                action();
                return 0;
            });
        }

        internal T InvokeNative<T>(MlxNativeResources resources, Func<NativeEffectLease, T> operation, Action cleanup)
        {
            ArgumentNullException.ThrowIfNull(resources);
            ArgumentNullException.ThrowIfNull(operation);
            ArgumentNullException.ThrowIfNull(cleanup);
            return Invoke(() =>
            {
                // Install the actual resource carrier before its first native acquisition.
                nativeResources.Add(resources);
                bool started = false;
                try
                {
                    using NativeEffectLease effect = nativeOwner.EnterEffect();
                    MlxNative.InstallCurrentErrorHandler();
                    started = true;
                    Exception error = null;
                    T result = default;
                    try { result = operation(effect); }
                    catch (Exception original) { error = original; }

                    bool unsafeCleanup = false;
                    try { nativeOwner.ThrowIfQuarantined(); }
                    catch (NativeRuntimeQuarantinedException refusal)
                    {
                        unsafeCleanup = true;
                        if (error == null) error = refusal;
                    }
                    if (!unsafeCleanup)
                    {
                        try
                        {
                            cleanup();
                            resources.SafelyReleased = true;
                        }
                        catch (Exception cleanupError)
                        {
                            error = MlxNative.JoinNativeErrors(error, cleanupError);
                            try { effect.PublishFailure(this, cleanupError, resources.CleanupFailureStage); }
                            catch (Exception publication) { error = MlxNative.JoinNativeErrors(error, publication); }
                        }
                    }
                    if (error != null) ExceptionDispatchInfo.Capture(error).Throw();
                    return result;
                }
                finally
                {
                    if (!started || resources.SafelyReleased) nativeResources.Remove(resources);
                }
            });
        }

        private void Run()
        {
            Volatile.Write(ref workerThreadId, Thread.CurrentThread.ManagedThreadId);
            foreach (IWorkItem item in queue.GetConsumingEnumerable())
                item.Execute();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref disposed) != 0)
                throw new ObjectDisposedException(nameof(MlxWorker));
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            queue.CompleteAdding();
        }

        private interface IWorkItem
        {
            void Execute();
        }

        private sealed class WorkItem<T> : IWorkItem
        {
            private readonly Func<T> func;
            private readonly ManualResetEventSlim completed = new(false);
            private T result;
            private Exception exception;

            public WorkItem(Func<T> func)
            {
                this.func = func;
            }

            public void Execute()
            {
                try
                {
                    result = func();
                }
                catch (Exception ex)
                {
                    exception = ex;
                }
                finally
                {
                    completed.Set();
                }
            }

            public T GetResult()
            {
                completed.Wait();
                if (exception != null)
                    ExceptionDispatchInfo.Capture(exception).Throw();
                return result;
            }
        }

    }
}
