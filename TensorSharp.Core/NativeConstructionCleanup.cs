using System;
using System.Runtime.ExceptionServices;
using System.Threading;

namespace TensorSharp;

/// <summary>Preserves construction and cleanup errors with release-only recovery authority.</summary>
public sealed class NativeConstructionCleanupException : AggregateException
{
    internal NativeConstructionCleanupException(Exception loadError, Exception cleanupError,
        NativeConstructionCleanupHandle cleanup)
        : base("Native-backed construction failed and its owned-resource cleanup did not complete.",
            loadError, cleanupError)
        => Cleanup = cleanup ?? throw new ArgumentNullException(nameof(cleanup));

    public NativeConstructionCleanupHandle Cleanup { get; }
}

/// <summary>
/// Retains an actual failed construction for explicit, synchronous resource release.
/// This handle exposes no usable partial object and performs no finalizer cleanup.
/// </summary>
public sealed class NativeConstructionCleanupHandle : IDisposable
{
    private object? _owner;
    private Action? _release;
    private Func<bool>? _isReleased;
    private Func<bool>? _isUnsafe;
    private ExceptionDispatchInfo? _terminalFailure;
    private int _attempt;
    private int _released;

    internal NativeConstructionCleanupHandle(object owner, Action release,
        Func<bool> isReleased, Func<bool> isUnsafe)
    {
        _owner = owner ?? throw new ArgumentNullException(nameof(owner));
        _release = release ?? throw new ArgumentNullException(nameof(release));
        _isReleased = isReleased ?? throw new ArgumentNullException(nameof(isReleased));
        _isUnsafe = isUnsafe ?? throw new ArgumentNullException(nameof(isUnsafe));
    }

    public bool IsReleased => Volatile.Read(ref _released) != 0;

    public void Dispose()
    {
        if (IsReleased) return;
        if (Interlocked.CompareExchange(ref _attempt, 1, 0) != 0)
            throw new InvalidOperationException("Construction cleanup is already in progress.");
        object? owner = _owner;
        try
        {
            if (IsReleased) return;
            _terminalFailure?.Throw();
            Action release = _release!;
            Func<bool> isReleased = _isReleased!;
            Func<bool> isUnsafe = _isUnsafe!;
            try
            {
                release();
            }
            catch (Exception error)
            {
                if (IsReleased || isReleased()) FinishRelease();
                else if (isUnsafe()) _terminalFailure ??= ExceptionDispatchInfo.Capture(error);
                throw;
            }
            if (!IsReleased && !isReleased())
            {
                var failure = new InvalidOperationException("Construction cleanup returned without proven resource release.");
                _terminalFailure = ExceptionDispatchInfo.Capture(failure);
                throw failure;
            }
            FinishRelease();
        }
        finally
        {
            GC.KeepAlive(owner);
            Volatile.Write(ref _attempt, 0);
        }
    }

    internal void CompleteRelease(object actualOwner)
    {
        if (IsReleased) return;
        if (!ReferenceEquals(_owner, actualOwner))
            throw new InvalidOperationException("Construction cleanup requires its actual reserved owner.");
        Volatile.Write(ref _released, 1);
        if (Volatile.Read(ref _attempt) == 0) FinishRelease();
    }

    private void FinishRelease()
    {
        Volatile.Write(ref _released, 1);
        _release = null;
        _isReleased = null;
        _isUnsafe = null;
        _owner = null;
    }
}
