using System;
using System.Collections.Generic;

namespace TensorSharp;

public enum NativeRuntimeScopeKind { CudaPrimaryDevice = 0, MlxSharedRuntime = 1 }
public enum NativeRuntimeQuarantineState { NoRecordedFailure = 0, FailedRequiresProcessExit = 1 }
public enum NativeRuntimeFailureStage
{
    Synchronization = 0, GraphRelease = 1, StorageRelease = 2, CacheRelease = 3,
    AllocatorRelease = 4, ContextRelease = 5, WorkerRetirement = 6
}

/// <summary>Immutable facts about unsafe cleanup. This is not a native readiness observation.</summary>
public sealed class NativeRuntimeFailure
{
    internal NativeRuntimeFailure(object[] cell)
    {
        ScopeId = (Guid)cell[0];
        ScopeKind = (NativeRuntimeScopeKind)(int)cell[1];
        DeviceOrdinal = (int)cell[2] < 0 ? null : (int)cell[2];
        FailureId = (Guid)cell[3];
        Stage = (NativeRuntimeFailureStage)(int)cell[4];
        Revision = (long)cell[5];
        RecordedAt = (DateTimeOffset)cell[6];
    }

    public Guid FailureId { get; }
    public Guid ScopeId { get; }
    public NativeRuntimeScopeKind ScopeKind { get; }
    public int? DeviceOrdinal { get; }
    public NativeRuntimeFailureStage Stage { get; }
    public long Revision { get; }
    public DateTimeOffset RecordedAt { get; }
}

public sealed class NativeRuntimeQuarantineSnapshot
{
    internal NativeRuntimeQuarantineSnapshot(long revision, NativeRuntimeFailure[] failures)
    {
        Revision = revision;
        Failures = Array.AsReadOnly(failures);
        State = failures.Length == 0 ? NativeRuntimeQuarantineState.NoRecordedFailure
            : NativeRuntimeQuarantineState.FailedRequiresProcessExit;
    }

    public long Revision { get; }
    public NativeRuntimeQuarantineState State { get; }
    public IReadOnlyList<NativeRuntimeFailure> Failures { get; }
}

public sealed class NativeRuntimeQuarantinedException : InvalidOperationException
{
    internal NativeRuntimeQuarantinedException(NativeRuntimeFailure failure, Exception cause)
        : base("Native resource cleanup failed; the affected runtime requires application-process exit.", cause)
    {
        Failure = failure;
    }

    public NativeRuntimeFailure Failure { get; }
}

/// <summary>Process-wide recorded unsafe cleanup, not a probe, reset, or release API.</summary>
public static class NativeRuntimeQuarantine
{
    public static NativeRuntimeQuarantineSnapshot Observe() => NativeQuarantineAuthority.Observe();

    /// <summary>Matches only recorded exception identities or their inner-exception graph.</summary>
    public static bool TryGetFailure(Exception error, out NativeRuntimeFailure? failure)
    {
        ArgumentNullException.ThrowIfNull(error);
        return NativeQuarantineAuthority.TryGetFailure(error, out failure);
    }
}
