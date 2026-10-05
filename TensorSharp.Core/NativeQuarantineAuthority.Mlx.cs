using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TensorSharp;

internal enum NativeMlxCallbackCallKind { CreatePayloadClosure, CompileClosure, ApplyClosure, ReleasePayloadReference }
internal enum NativeMlxCallbackKind { Trace = 1, PayloadDestructor = 2 }

internal sealed class NativeMlxCallbackBusyException : InvalidOperationException
{
    private readonly object owner;
    internal NativeMlxCallbackBusyException(object owner) : base("MLX cleanup is busy with an admitted compiled call or release.") => this.owner = owner;
    internal bool IsFor(object actualOwner) => ReferenceEquals(owner, actualOwner);
}

internal sealed class NativeMlxCallbackCallLease : IDisposable
{
    internal readonly NativeOwnerRegistration Registration;
    internal readonly object[] Frame;
    internal readonly int ThreadId = Environment.CurrentManagedThreadId;
    internal readonly NativeMlxReleaseReservation? Release;
    internal bool Held;
    internal bool Disposed;

    internal NativeMlxCallbackCallLease(NativeOwnerRegistration registration, NativeMlxCallbackCallKind kind, NativeMlxReleaseReservation? release)
    {
        Registration = registration;
        Release = release;
        Frame = new object[] { Guid.NewGuid(), registration.Cell[0], Guid.Empty, (int)kind, 0, 0, 0L };
    }

    internal void EnterManagedCallback(NativeMlxCallbackKind kind) => NativeQuarantineAuthority.EnterManagedCallback(this, kind);
    internal void ResumeNative() => NativeQuarantineAuthority.ResumeMlxNative(this);
    internal void NativeReturned() => NativeQuarantineAuthority.MlxNativeReturned(this);
    internal NativeRuntimeFailure PublishFailure(object actualOwner, Exception cleanupError, NativeRuntimeFailureStage stage)
        => NativeQuarantineAuthority.Publish(this, actualOwner, cleanupError, stage);
    internal void CompleteSafeRelease(object actualOwner, NativeMlxReleaseReservation reservation)
        => NativeQuarantineAuthority.Complete(this, actualOwner, reservation);
    public void Dispose() => NativeQuarantineAuthority.Exit(this);
}

internal sealed class NativeMlxReleaseReservation : IDisposable
{
    internal readonly NativeOwnerRegistration Registration;
    internal readonly object[] Cell;
    internal bool Disposed;

    internal NativeMlxReleaseReservation(NativeOwnerRegistration registration)
    {
        Registration = registration;
        Cell = new object[] { Guid.NewGuid(), registration.Cell[0], 0, 0, registration.Cell };
    }

    internal NativeEffectLease EnterEffect() => NativeQuarantineAuthority.EnterMlxRelease(this);
    internal NativeMlxCallbackCallLease EnterCallbackRelease() => NativeQuarantineAuthority.EnterMlxCallbackRelease(this);
    public void Dispose() => NativeQuarantineAuthority.Exit(this);
}

internal static partial class NativeQuarantineAuthority
{
    private static Dictionary<int, List<object[]>> CallbackFrames(object[] s) => (Dictionary<int, List<object[]>>)s[10];
    private static Dictionary<Guid, object[]> MlxReleases(object[] s) => (Dictionary<Guid, object[]>)s[11];

    internal static void ValidateMlxWorkerDispatch(NativeOwnerRegistration registration)
    {
        lock (registration.State[1])
        {
            ValidateMlxOwner(registration, registration.Owner, false);
            int thread = Environment.CurrentManagedThreadId;
            if (Frames(registration.State).ContainsKey(thread) || CallbackFrames(registration.State).ContainsKey(thread))
                throw new InvalidOperationException("MLX native work cannot cross workers beneath an active native effect or compiled callback.");
        }
    }

    private static void ValidateMlxCells(object[] s)
    {
        int count = 0;
        int activeThread = 0;
        foreach (var (thread, frames) in CallbackFrames(s))
        {
            if (thread <= 0 || frames == null || frames.Count == 0 || activeThread != 0)
                throw new InvalidOperationException(ProtocolError);
            activeThread = thread;
            Guid parent = Guid.Empty;
            int index = 0;
            foreach (object[] f in frames)
            {
                if (f == null || f.Length != 7 || f[0] is not Guid id || id == Guid.Empty
                    || f[1] is not Guid owner || owner == Guid.Empty || f[2] is not Guid parentId || parentId != parent
                    || f[3] is not int kind || kind < 0 || kind > 3
                    || f[4] is not int phase || phase < 0 || phase > 2
                    || f[5] is not int callback || callback < 0 || callback > 2
                    || ((phase == 1) != (callback != 0)) || f[6] is not long epoch || epoch < 0
                    || (index < frames.Count - 1 && phase != 1))
                    throw new InvalidOperationException(ProtocolError);
                parent = id;
                count = checked(count + 1);
                index++;
            }
        }
        if (Scopes(s).TryGetValue(Mlx, out var scope))
        {
            if ((int)scope[9] != count || (int)scope[10] != activeThread)
                throw new InvalidOperationException(ProtocolError);
        }
        else if (count != 0) throw new InvalidOperationException(ProtocolError);
        foreach (var (id, c) in MlxReleases(s))
        {
            if (c == null || c.Length != 5 || c[0] is not Guid cellId || id != cellId || id == Guid.Empty
                || c[1] is not Guid owner || owner == Guid.Empty || c[2] is not int phase || phase < 0 || phase > 2
                || c[3] is not int thread || thread < 0 || (phase == 0 && thread != 0) || (phase == 1 && thread == 0)
                || c[4] is not object[] original || original.Length != 8 || (Guid)original[0] != owner
                || (phase != 2 && (!Owners(s).TryGetValue(owner, out var current) || !ReferenceEquals(original, current))))
                throw new InvalidOperationException(ProtocolError);
        }
    }

    private static void ValidateMlxOwner(NativeOwnerRegistration r, object actualOwner, bool callback)
    {
        ValidateRegistration(r);
        if (!ReferenceEquals(r.Owner, actualOwner) || ((HashSet<string>)r.Cell[3]).Count != 1
            || !((HashSet<string>)r.Cell[3]).Contains(Mlx) || (callback && (int)r.Cell[1] != (int)NativeOwnerRole.NativeHandle))
            throw new InvalidOperationException("MLX admission requires the actual owner and its frozen shared-runtime scope.");
        ThrowFailure(r);
    }

    // Replacement containers are allocated without metadata/native gates. Installation
    // copies into reserved capacity only after its count snapshot is revalidated.
    private static (Dictionary<int, List<object[]>> Frames, List<object[]> Stack, int Threads, int Depth) PrepareMlxCallbackCapacity(object[] s, int thread)
    {
        int dictionaries;
        int frames;
        lock (s[1])
        {
            ValidateCells(s);
            dictionaries = CallbackFrames(s).Count;
            frames = CallbackFrames(s).TryGetValue(thread, out var stack) ? stack.Count : 0;
        }
        return (new Dictionary<int, List<object[]>>(checked(dictionaries + 1)),
            new List<object[]>(checked(frames + 1)), dictionaries, frames);
    }

    private static int PrepareMlxReleaseCapacity(object[] s)
    {
        while (true)
        {
            int count;
            lock (s[1]) { ValidateCells(s); count = MlxReleases(s).Count; }
            var replacement = new Dictionary<Guid, object[]>(checked(count + 1));
            lock (s[1])
            {
                var current = MlxReleases(s);
                if (current.Count > count) continue;
                foreach (var pair in current) replacement.Add(pair.Key, pair.Value);
                s[11] = replacement;
                return replacement.EnsureCapacity(0);
            }
        }
    }

    private static bool CheckMlxAdmission(NativeOwnerRegistration r, NativeMlxReleaseReservation? release)
    {
        if (!((HashSet<string>)r.Cell[3]).Contains(Mlx))
        {
            if (CallbackFrames(r.State).ContainsKey(Environment.CurrentManagedThreadId))
                throw new InvalidOperationException("Compiled callbacks cannot widen native scopes.");
            return true;
        }
        var scope = Scopes(r.State)[Mlx];
        int calls = (int)scope[9];
        if (release != null)
        {
            ValidateMlxReservation(release, false);
            if (calls != 0) throw new NativeMlxCallbackBusyException(r.Owner);
            return true;
        }
        if (calls == 0) return true;
        int thread = Environment.CurrentManagedThreadId;
        if ((int)scope[10] != thread) return false;
        var stack = CallbackFrames(r.State)[thread];
        if ((int)stack[^1][4] != 1 || ((HashSet<string>)r.Cell[3]).Count != 1)
            throw new InvalidOperationException("Only short MLX effects may enter an active managed trace callback.");
        return true;
    }

    private static void WaitForMlxAdmission(NativeOwnerRegistration r, NativeMlxReleaseReservation? release)
    {
        while (true)
        {
            ManualResetEventSlim drain;
            lock (r.State[1])
            {
                ValidateRegistration(r);
                ThrowFailure(r);
                if (CheckMlxAdmission(r, release)) return;
                if (Frames(r.State).ContainsKey(Environment.CurrentManagedThreadId))
                    throw new InvalidOperationException("MLX callback drain cannot wait beneath another native effect.");
                drain = (ManualResetEventSlim)Scopes(r.State)[Mlx][11];
            }
            drain.Wait();
        }
    }

    internal static NativeMlxReleaseReservation ReserveMlxRelease(NativeOwnerRegistration r, object actualOwner)
    {
        var reservation = new NativeMlxReleaseReservation(r);
        while (true)
        {
            int capacity = PrepareMlxReleaseCapacity(r.State);
            lock (r.State[1])
            {
                ValidateMlxOwner(r, actualOwner, false);
                if ((int)Scopes(r.State)[Mlx][9] != 0)
                    throw new NativeMlxCallbackBusyException(actualOwner);
                foreach (var cell in MlxReleases(r.State).Values)
                    if ((Guid)cell[1] == (Guid)r.Cell[0]) throw new NativeMlxCallbackBusyException(actualOwner);
                if (MlxReleases(r.State).Count >= capacity) continue;
                int count = checked((int)r.Cell[7] + 1);
                MlxReleases(r.State).Add((Guid)reservation.Cell[0], reservation.Cell);
                r.Cell[7] = count;
            }
            return reservation;
        }
    }

    private static void ValidateMlxReservation(NativeMlxReleaseReservation reservation, bool executing)
    {
        var r = reservation.Registration;
        if (reservation.Disposed || !ReferenceEquals(reservation.Cell[4], r.Cell)
            || !MlxReleases(r.State).TryGetValue((Guid)reservation.Cell[0], out var cell) || !ReferenceEquals(cell, reservation.Cell)
            || (int)cell[2] != (executing ? 1 : 0)
            || (executing && (int)cell[3] != Environment.CurrentManagedThreadId))
            throw new InvalidOperationException("MLX release requires its exact active reservation.");
        ValidateMlxOwner(r, r.Owner, false);
    }

    private static void BeginMlxRelease(NativeMlxReleaseReservation release)
    {
        ValidateMlxReservation(release, false);
        release.Cell[2] = 1;
        release.Cell[3] = Environment.CurrentManagedThreadId;
    }

    private static void EndMlxRelease(NativeMlxReleaseReservation release)
    {
        int phase = (int)release.Cell[2];
        if (phase == 0) return; // A failed pre-acquisition attempt never began execution.
        if (release.Disposed || (int)release.Cell[3] != Environment.CurrentManagedThreadId || phase < 1 || phase > 2)
            throw new InvalidOperationException("MLX release exit requires the executing thread.");
        release.Cell[3] = 0;
        if (phase == 1) release.Cell[2] = 0;
    }

    internal static NativeEffectLease EnterMlxRelease(NativeMlxReleaseReservation release)
    {
        lock (release.Registration.State[1]) ValidateMlxReservation(release, false);
        return Enter(release.Registration, release);
    }

    internal static void ValidateMlxRelease(NativeEffectLease lease, object actualOwner, NativeMlxReleaseReservation reservation)
    {
        lock (lease.Registration.State[1])
        {
            ValidateHeldOwnerGates(lease, actualOwner);
            if (!ReferenceEquals(lease.MlxReleaseReservation, reservation) || !ReferenceEquals(reservation.Registration, lease.Registration))
                throw new InvalidOperationException("MLX release cannot substitute another reservation.");
            ValidateMlxReservation(reservation, true);
            if ((int)Scopes(lease.Registration.State)[Mlx][9] != 0)
                throw new NativeMlxCallbackBusyException(actualOwner);
        }
    }

    private static void CompleteMlxRelease(NativeOwnerRegistration r, object[] ordinaryFrame)
    {
        foreach (var cell in MlxReleases(r.State).Values)
        {
            if (ReferenceEquals(cell[4], r.Cell) && (int)cell[2] == 1 && (int)cell[3] == Environment.CurrentManagedThreadId)
            {
                if ((Guid)ordinaryFrame[0] != (Guid)r.Cell[0] || (int)Scopes(r.State)[Mlx][9] != 0)
                    throw new InvalidOperationException("MLX release must complete outside compiled calls.");
                cell[2] = 2;
                return;
            }
        }
        throw new InvalidOperationException("MLX release requires a pre-admitted destructive reservation.");
    }

    internal static void Exit(NativeMlxReleaseReservation reservation)
    {
        var r = reservation.Registration;
        lock (r.State[1])
        {
            if (reservation.Disposed || !MlxReleases(r.State).TryGetValue((Guid)reservation.Cell[0], out var cell)
                || !ReferenceEquals(cell, reservation.Cell) || (int)cell[3] != 0 || (int)cell[2] == 1)
                throw new InvalidOperationException("MLX release reservations exit only after actual execution settles.");
            MlxReleases(r.State).Remove((Guid)cell[0]);
            r.Cell[7] = checked((int)r.Cell[7] - 1);
            reservation.Disposed = true;
        }
    }

    internal static NativeMlxCallbackCallLease EnterMlxCallbackCall(NativeOwnerRegistration r, object actualOwner, NativeMlxCallbackCallKind kind)
    {
        if (kind < NativeMlxCallbackCallKind.CreatePayloadClosure || kind >= NativeMlxCallbackCallKind.ReleasePayloadReference)
            throw new ArgumentOutOfRangeException(nameof(kind));
        return EnterMlxCallback(new NativeMlxCallbackCallLease(r, kind, null), actualOwner);
    }

    internal static NativeMlxCallbackCallLease EnterMlxCallbackRelease(NativeMlxReleaseReservation release)
        => EnterMlxCallback(new NativeMlxCallbackCallLease(release.Registration, NativeMlxCallbackCallKind.ReleasePayloadReference, release), release.Registration.Owner);

    private static NativeMlxCallbackCallLease EnterMlxCallback(NativeMlxCallbackCallLease lease, object actualOwner)
    {
        var r = lease.Registration;
        object[] s = r.State;
        while (true)
        {
            ManualResetEventSlim? drain = null;
            lock (s[1])
            {
                ValidateMlxOwner(r, actualOwner, true);
                if (Frames(s).ContainsKey(lease.ThreadId))
                    throw new InvalidOperationException("Compiled callback entry cannot suspend an ordinary native lease.");
                var scope = Scopes(s)[Mlx];
                if (lease.Release != null)
                {
                    ValidateMlxReservation(lease.Release, false);
                    if ((int)scope[9] != 0) throw new NativeMlxCallbackBusyException(actualOwner);
                }
                else if (MlxReleases(s).Values.Any(cell => (int)cell[2] != 2))
                    throw new NativeMlxCallbackBusyException(actualOwner);
                else if ((int)scope[9] != 0 && (int)scope[10] != lease.ThreadId)
                    drain = (ManualResetEventSlim)scope[11];
            }
            if (drain != null) { drain.Wait(); continue; }
            var prepared = PrepareMlxCallbackCapacity(s, lease.ThreadId);
            Monitor.Enter(s[7]);
            bool entered = false;
            try
            {
                lock (s[1])
                {
                    ValidateMlxOwner(r, actualOwner, true);
                    var scope = Scopes(s)[Mlx];
                    var current = CallbackFrames(s);
                    current.TryGetValue(lease.ThreadId, out var oldStack);
                    if ((int)scope[9] != 0 && (int)scope[10] != lease.ThreadId)
                    {
                        if (lease.Release != null) throw new NativeMlxCallbackBusyException(actualOwner);
                        continue;
                    }
                    if (current.Count > prepared.Threads || (oldStack?.Count ?? 0) > prepared.Depth) continue;
                    foreach (var pair in current) prepared.Frames.Add(pair.Key, pair.Value);
                    if (oldStack != null) foreach (var frame in oldStack) prepared.Stack.Add(frame);
                    var stack = prepared.Stack;
                    if (lease.Release != null)
                    {
                        ValidateMlxReservation(lease.Release, false);
                        if ((int)scope[9] != 0) throw new NativeMlxCallbackBusyException(actualOwner);
                    }
                    else if (MlxReleases(s).Values.Any(cell => (int)cell[2] != 2))
                        throw new NativeMlxCallbackBusyException(actualOwner);
                    if (stack.Count != 0 && (int)stack[^1][4] != 1)
                        throw new InvalidOperationException("Nested compiled calls require the parent's managed callback phase.");
                    int calls = checked((int)scope[9] + 1);
                    int owners = checked((int)r.Cell[7] + 1);
                    lease.Frame[2] = stack.Count == 0 ? Guid.Empty : stack[^1][0];
                    if (lease.Release != null) BeginMlxRelease(lease.Release);
                    stack.Add(lease.Frame);
                    prepared.Frames[lease.ThreadId] = stack;
                    s[10] = prepared.Frames;
                    scope[9] = calls;
                    scope[10] = lease.ThreadId;
                    ((ManualResetEventSlim)scope[11]).Reset();
                    r.Cell[7] = owners;
                    lease.Held = true;
                    entered = true;
                    return lease;
                }
            }
            finally
            {
                if (!entered)
                {
                    Monitor.Exit(s[7]);
                }
            }
        }
    }

    private static void ValidateCallback(NativeMlxCallbackCallLease lease, int phase, bool held)
    {
        if (lease.Disposed || lease.ThreadId != Environment.CurrentManagedThreadId || lease.Held != held
            || (held && !Monitor.IsEntered(lease.Registration.State[7]))
            || !CallbackFrames(lease.Registration.State).TryGetValue(lease.ThreadId, out var stack)
            || stack.Count == 0 || !ReferenceEquals(stack[^1], lease.Frame) || (int)lease.Frame[4] != phase
            || Frames(lease.Registration.State).ContainsKey(lease.ThreadId))
            throw new InvalidOperationException("MLX compiled calls require their exact synchronous top frame and phase.");
    }

    internal static void EnterManagedCallback(NativeMlxCallbackCallLease lease, NativeMlxCallbackKind kind)
    {
        if (kind != NativeMlxCallbackKind.Trace && kind != NativeMlxCallbackKind.PayloadDestructor)
            throw new ArgumentOutOfRangeException(nameof(kind));
        lock (lease.Registration.State[1])
        {
            ValidateCallback(lease, 0, true);
            lease.Frame[6] = checked((long)lease.Frame[6] + 1);
            lease.Frame[4] = 1;
            lease.Frame[5] = (int)kind;
            lease.Held = false;
        }
        Monitor.Exit(lease.Registration.State[7]);
    }

    internal static void ResumeMlxNative(NativeMlxCallbackCallLease lease)
    {
        lock (lease.Registration.State[1]) ValidateCallback(lease, 1, false);
        Monitor.Enter(lease.Registration.State[7]);
        lock (lease.Registration.State[1])
        {
            ValidateCallback(lease, 1, false);
            lease.Held = true;
            lease.Frame[4] = 0;
            lease.Frame[5] = 0;
            // Terminal refusal is reported only after the unwind can authenticate
            // its native phase and still-held originating monitor.
            ThrowFailure(lease.Registration);
        }
    }

    internal static void MlxNativeReturned(NativeMlxCallbackCallLease lease)
    {
        lock (lease.Registration.State[1])
        {
            ValidateCallback(lease, 0, true);
            lease.Frame[4] = 2;
        }
    }

    internal static NativeRuntimeFailure Publish(NativeMlxCallbackCallLease lease, object owner, Exception error, NativeRuntimeFailureStage stage)
    {
        lock (lease.Registration.State[1])
        {
            ValidateCallback(lease, 2, true);
            ValidateRegistration(lease.Registration);
            if (!ReferenceEquals(owner, lease.Registration.Owner))
                throw new InvalidOperationException("MLX fault publication requires the actual invocation owner.");
            return PublishHeld(lease.Registration, new[] { Mlx }, error, stage);
        }
    }

    internal static void Complete(NativeMlxCallbackCallLease lease, object owner, NativeMlxReleaseReservation release)
    {
        lock (lease.Registration.State[1])
        {
            ValidateCallback(lease, 2, true);
            ValidateMlxOwner(lease.Registration, owner, true);
            if (!ReferenceEquals(lease.Release, release) || (int)lease.Frame[3] != 3)
                throw new InvalidOperationException("Only the exact payload-release call can complete its owner.");
            ValidateMlxReservation(release, true);
            foreach (var frame in CallbackFrames(lease.Registration.State)[lease.ThreadId])
                if (!ReferenceEquals(frame, lease.Frame) && (Guid)frame[1] == (Guid)lease.Registration.Cell[0])
                    throw new InvalidOperationException("MLX release refuses recursive calls on the same owner.");
            ThrowFailure(lease.Registration);
            release.Cell[2] = 2;
            Owners(lease.Registration.State).Remove((Guid)lease.Registration.Cell[0]);
        }
    }

    internal static void Exit(NativeMlxCallbackCallLease lease)
    {
        object[] s = lease.Registration.State;
        lock (s[1])
        {
            ValidateCallback(lease, 2, true);
            var stack = CallbackFrames(s)[lease.ThreadId];
            stack.RemoveAt(stack.Count - 1);
            if (stack.Count == 0) CallbackFrames(s).Remove(lease.ThreadId);
            var scope = Scopes(s)[Mlx];
            scope[9] = checked((int)scope[9] - 1);
            lease.Registration.Cell[7] = checked((int)lease.Registration.Cell[7] - 1);
            if ((int)scope[9] == 0)
            {
                scope[10] = 0;
                ((ManualResetEventSlim)scope[11]).Set();
            }
            if (lease.Release != null) EndMlxRelease(lease.Release);
            lease.Held = false;
            lease.Disposed = true;
        }
        Monitor.Exit(s[7]);
    }
}
