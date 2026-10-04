using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;

namespace TensorSharp;

internal enum NativeOwnerRole { Model, Allocator, Storage, Graph, Worker, NativeHandle }

internal sealed class NativeOwnerRegistration
{
    internal readonly object[] State;
    internal readonly object[] Cell;
    internal readonly object Owner;

    internal NativeOwnerRegistration(object[] state, object[] cell, object owner)
    {
        State = state;
        Cell = cell;
        Owner = owner;
    }

    internal void AttachCudaPrimaryDevice(int deviceOrdinal)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceOrdinal);
        NativeQuarantineAuthority.Attach(this, "cuda-primary/" + deviceOrdinal.ToString(CultureInfo.InvariantCulture));
    }

    internal void AttachMlxSharedRuntime() => NativeQuarantineAuthority.Attach(this, "mlx-shared-runtime");
    internal void AttachCudaUnresolvedRuntime() => NativeQuarantineAuthority.Attach(this, "cuda-primary/*");
    internal void AttachCudaDependentGgml() => NativeQuarantineAuthority.Attach(this, "cuda-primary/*");
    internal void ThrowIfQuarantined() => NativeQuarantineAuthority.Check(this);
    internal NativeEffectLease EnterEffect() => NativeQuarantineAuthority.Enter(this);
    internal void CompleteSafeRelease() => NativeQuarantineAuthority.Complete(this);
}

internal sealed class NativeEffectLease : IDisposable
{
    internal readonly NativeOwnerRegistration Registration;
    internal readonly object[] Frame;
    internal readonly int ThreadId;
    internal readonly List<object> DeviceGates = new();
    internal bool CudaHeld;
    internal bool MlxHeld;
    internal bool Disposed;

    internal NativeEffectLease(NativeOwnerRegistration registration, object[] frame)
    {
        Registration = registration;
        Frame = frame;
        ThreadId = Environment.CurrentManagedThreadId;
    }

    internal NativeRuntimeFailure PublishFailure(object actualOwner, Exception cleanupError, NativeRuntimeFailureStage stage)
        => NativeQuarantineAuthority.Publish(this, actualOwner, cleanupError, stage);

    internal void ValidateSafeRelease(object actualOwner) => NativeQuarantineAuthority.ValidateSafeRelease(this, actualOwner);
    internal void CompleteSafeRelease(object actualOwner) => NativeQuarantineAuthority.Complete(this, actualOwner);

    public void Dispose() => NativeQuarantineAuthority.Exit(this);
}

// The process slot contains only BCL cells. Local registrations/leases never enter it.
internal static class NativeQuarantineAuthority
{
    private const string Slot = "Darkspyre.TensorSharp.NativeQuarantine";
    private const string Wildcard = "cuda-primary/*";
    private const string Mlx = "mlx-shared-runtime";
    private const string ProtocolError = "The process native quarantine protocol is incompatible.";

    private static object[] GetState()
    {
        object? value = AppDomain.CurrentDomain.GetData(Slot);
        if (value != null) return ValidateShape(value);
        lock (AppDomain.CurrentDomain)
        {
            value = AppDomain.CurrentDomain.GetData(Slot);
            if (value == null)
            {
                value = new object[]
                {
                    1, new object(), 0L,
                    new Dictionary<string, object[]>(StringComparer.Ordinal),
                    new Dictionary<Guid, object[]>(),
                    new Dictionary<Exception, List<Guid>>(ReferenceEqualityComparer.Instance),
                    new ReaderWriterLockSlim(LockRecursionPolicy.SupportsRecursion), new object(),
                    Guid.NewGuid(), new Dictionary<int, List<object[]>>()
                };
                AppDomain.CurrentDomain.SetData(Slot, value);
            }
            return ValidateShape(value);
        }
    }

    // Existing-slot validation never takes the initialization monitor beneath metadata.
    private static object[] ValidateShape(object? value)
    {
        if (value is not object[] s || s.Length != 10 || s[0] is not int p || p != 1
            || s[1]?.GetType() != typeof(object) || s[2] is not long revision || revision < 0
            || s[3] is not Dictionary<string, object[]> scopes || !ReferenceEquals(scopes.Comparer, StringComparer.Ordinal)
            || s[4] is not Dictionary<Guid, object[]>
            || s[5] is not Dictionary<Exception, List<Guid>> causes || !ReferenceEquals(causes.Comparer, ReferenceEqualityComparer.Instance)
            || s[6] is not ReaderWriterLockSlim rw || rw.RecursionPolicy != LockRecursionPolicy.SupportsRecursion
            || s[7]?.GetType() != typeof(object) || s[8] is not Guid id || id == Guid.Empty
            || s[9] is not Dictionary<int, List<object[]>>)
            throw new InvalidOperationException(ProtocolError);
        return s;
    }

    private static Dictionary<string, object[]> Scopes(object[] s) => (Dictionary<string, object[]>)s[3];
    private static Dictionary<Guid, object[]> Owners(object[] s) => (Dictionary<Guid, object[]>)s[4];
    private static Dictionary<Exception, List<Guid>> Causes(object[] s) => (Dictionary<Exception, List<Guid>>)s[5];
    private static Dictionary<int, List<object[]>> Frames(object[] s) => (Dictionary<int, List<object[]>>)s[9];

    private static void ValidateCells(object[] s)
    {
        foreach (var (key, c) in Scopes(s))
        {
            bool mlx = key == Mlx;
            int ordinal = -1;
            if (!mlx && key != Wildcard && (!key.StartsWith("cuda-primary/", StringComparison.Ordinal)
                || !int.TryParse(key.AsSpan(13), NumberStyles.None, CultureInfo.InvariantCulture, out ordinal)
                || ordinal < 0 || key != "cuda-primary/" + ordinal.ToString(CultureInfo.InvariantCulture)))
                throw new InvalidOperationException(ProtocolError);
            if (c == null || c.Length != 9 || c[0] is not Guid scopeId || scopeId == Guid.Empty
                || c[1] is not int kind || kind != (mlx ? 1 : 0) || c[2] is not int device || device != ordinal
                || c[3] is not Guid || c[4] is not int stage || stage < 0 || stage > 6
                || c[5] is not long revision || revision < 0 || c[6] is not DateTimeOffset
                || (c[7] != null && c[7] is not Exception) || c[8]?.GetType() != typeof(object)
                || (((Guid)c[3] == Guid.Empty) != (revision == 0))
                || (((Guid)c[3] == Guid.Empty) != (c[7] == null)))
                throw new InvalidOperationException(ProtocolError);
        }
        foreach (var (id, c) in Owners(s))
        {
            if (c == null || c.Length != 8 || c[0] is not Guid cellId || id != cellId || id == Guid.Empty
                || c[1] is not int role || role < 0 || role > 5 || c[2] is not WeakReference<object>
                || c[3] is not HashSet<string> keys || !ReferenceEquals(keys.Comparer, StringComparer.Ordinal)
                || keys.Any(key => !Scopes(s).ContainsKey(key)) || c[4] is not bool wild || wild != keys.Contains(Wildcard)
                || c[5] is not Guid || c[7] is not int active || active < 0
                || ((Guid)c[5] == Guid.Empty && c[6] != null))
                throw new InvalidOperationException(ProtocolError);
        }
        foreach (var (thread, frames) in Frames(s))
        {
            if (thread <= 0 || frames == null || frames.Count == 0)
                throw new InvalidOperationException(ProtocolError);
            foreach (var f in frames)
                if (f == null || f.Length != 3 || f[0] is not Guid || f[1] is not string[] keys
                    || keys.Any(key => !Scopes(s).ContainsKey(key)) || f[2] is not bool write || write != keys.Contains(Wildcard))
                    throw new InvalidOperationException(ProtocolError);
        }
        foreach (var pair in Causes(s))
            if (pair.Key == null || pair.Value == null || pair.Value.Count == 0
                || pair.Value.Any(id => !Scopes(s).Values.Any(c => (Guid)c[3] == id)))
                throw new InvalidOperationException(ProtocolError);
    }

    internal static NativeOwnerRegistration Register(object owner, NativeOwnerRole role)
    {
        ArgumentNullException.ThrowIfNull(owner);
        if (role < NativeOwnerRole.Model || role > NativeOwnerRole.NativeHandle)
            throw new ArgumentOutOfRangeException(nameof(role));
        object[] s = GetState();
        lock (s[1])
        {
            ValidateCells(s);
            var id = Guid.NewGuid();
            object[] c = { id, (int)role, new WeakReference<object>(owner, true), new HashSet<string>(StringComparer.Ordinal), false, Guid.Empty, null!, 0 };
            Owners(s).Add(id, c);
            return new NativeOwnerRegistration(s, c, owner);
        }
    }

    private static object[] Scope(object[] s, string key)
    {
        if (!Scopes(s).TryGetValue(key, out object[]? c))
        {
            int ordinal = key == Mlx || key == Wildcard ? -1 : int.Parse(key.AsSpan(13), CultureInfo.InvariantCulture);
            c = new object[] { Guid.NewGuid(), key == Mlx ? 1 : 0, ordinal, Guid.Empty, 0, 0L, default(DateTimeOffset), null!, new object() };
            Scopes(s).Add(key, c);
        }
        return c;
    }

    private static void ValidateRegistration(NativeOwnerRegistration r)
    {
        if (!ReferenceEquals(ValidateShape(AppDomain.CurrentDomain.GetData(Slot)), r.State))
            throw new InvalidOperationException(ProtocolError);
        ValidateCells(r.State);
        if (!Owners(r.State).TryGetValue((Guid)r.Cell[0], out object[]? c) || !ReferenceEquals(c, r.Cell))
            throw new InvalidOperationException("Native ownership was already safely released.");
    }

    internal static void Attach(NativeOwnerRegistration r, string key)
    {
        lock (r.State[1])
        {
            ValidateRegistration(r);
            if ((int)r.Cell[7] != 0) throw new InvalidOperationException("Native effect scopes are frozen during execution.");
            ThrowFailure(r);
            Scope(r.State, key);
            var keys = (HashSet<string>)r.Cell[3];
            keys.Add(key);
            r.Cell[4] = keys.Contains(Wildcard);
            try { ThrowFailure(r); }
            catch { keys.Remove(key); r.Cell[4] = keys.Contains(Wildcard); throw; }
        }
    }

    private static object[]? FailureCell(NativeOwnerRegistration r)
    {
        Guid id = (Guid)r.Cell[5];
        if (id != Guid.Empty) return Scopes(r.State).Values.Single(c => (Guid)c[3] == id);
        HashSet<string> keys = (HashSet<string>)r.Cell[3];
        return Scopes(r.State).Where(pair => (Guid)pair.Value[3] != Guid.Empty &&
            (keys.Contains(pair.Key) || (pair.Key == Wildcard && keys.Any(IsCuda))
            || ((bool)r.Cell[4] && IsCuda(pair.Key))))
            .Select(pair => pair.Value).OrderBy(c => (long)c[5]).FirstOrDefault();
    }

    private static bool IsCuda(string key) => key.StartsWith("cuda-primary/", StringComparison.Ordinal);

    private static void ThrowFailure(NativeOwnerRegistration r)
    {
        object[]? c = FailureCell(r);
        if (c != null) throw new NativeRuntimeQuarantinedException(new NativeRuntimeFailure(c), (Exception)c[7]);
    }

    internal static void Check(NativeOwnerRegistration r)
    {
        lock (r.State[1]) { ValidateRegistration(r); ThrowFailure(r); }
    }

    internal static bool HasCurrentThreadEffects(NativeOwnerRegistration r)
    {
        lock (r.State[1])
        {
            ValidateRegistration(r);
            return Frames(r.State).ContainsKey(Environment.CurrentManagedThreadId);
        }
    }

    internal static NativeEffectLease Enter(NativeOwnerRegistration r)
    {
        object[] s = r.State;
        object[] frame;
        int thread = Environment.CurrentManagedThreadId;
        lock (s[1])
        {
            ValidateRegistration(r);
            ThrowFailure(r);
            var keys = ((HashSet<string>)r.Cell[3]).OrderBy(key => key == Mlx ? int.MaxValue
                : key == Wildcard ? -1 : (int)Scopes(s)[key][2]).ToArray();
            if (keys.Length == 0) throw new InvalidOperationException("Native effects require an attached scope.");
            bool write = keys.Contains(Wildcard);
            if (!Frames(s).TryGetValue(thread, out List<object[]>? stack))
                Frames(s).Add(thread, stack = new List<object[]>());
            if (stack.Count != 0)
            {
                var parent = stack[^1];
                string[] parentKeys = (string[])parent[1];
                bool parentCudaWrite = (bool)parent[2];
                if (keys.Any(key => !parentKeys.Contains(key)
                    && !(parentCudaWrite && IsCuda(key) && key != Wildcard)) || (write && !parentCudaWrite))
                    throw new InvalidOperationException("Recursive native effects cannot widen scopes or upgrade CUDA gates.");
            }
            frame = new object[] { r.Cell[0], keys, write };
            stack.Add(frame);
            r.Cell[7] = checked((int)r.Cell[7] + 1);
        }
        var lease = new NativeEffectLease(r, frame);
        try
        {
            string[] keys = (string[])frame[1];
            if (keys.Any(IsCuda))
            {
                var rw = (ReaderWriterLockSlim)s[6];
                if ((bool)frame[2]) rw.EnterWriteLock(); else rw.EnterReadLock();
                lease.CudaHeld = true;
                foreach (string key in keys.Where(key => IsCuda(key) && key != Wildcard))
                {
                    object gate;
                    lock (s[1]) gate = Scopes(s)[key][8];
                    Monitor.Enter(gate);
                    lease.DeviceGates.Add(gate);
                }
            }
            if (keys.Contains(Mlx)) { Monitor.Enter(s[7]); lease.MlxHeld = true; }
            lock (s[1]) { ValidateRegistration(r); ThrowFailure(r); }
            return lease;
        }
        catch { Exit(lease); throw; }
    }

    private static void ValidateLease(NativeEffectLease lease)
    {
        if (lease.Disposed || lease.ThreadId != Environment.CurrentManagedThreadId)
            throw new InvalidOperationException("Native effect leases are active synchronous thread-affine operations.");
        var frames = Frames(lease.Registration.State);
        if (!frames.TryGetValue(lease.ThreadId, out List<object[]>? stack) || stack.Count == 0
            || !ReferenceEquals(stack[^1], lease.Frame) || (Guid)lease.Frame[0] != (Guid)lease.Registration.Cell[0])
            throw new InvalidOperationException("Native effect leases must complete in nesting order.");
    }

    internal static void Exit(NativeEffectLease lease)
    {
        object[] s = lease.Registration.State;
        lock (s[1])
        {
            ValidateLease(lease);
            var stack = Frames(s)[lease.ThreadId];
            stack.RemoveAt(stack.Count - 1);
            if (stack.Count == 0) Frames(s).Remove(lease.ThreadId);
            lease.Registration.Cell[7] = (int)lease.Registration.Cell[7] - 1;
            lease.Disposed = true;
        }
        if (lease.MlxHeld) Monitor.Exit(s[7]);
        for (int i = lease.DeviceGates.Count - 1; i >= 0; i--) Monitor.Exit(lease.DeviceGates[i]);
        if (lease.CudaHeld)
        {
            var rw = (ReaderWriterLockSlim)s[6];
            if ((bool)lease.Frame[2]) rw.ExitWriteLock(); else rw.ExitReadLock();
        }
    }

    internal static void Complete(NativeOwnerRegistration r)
    {
        lock (r.State[1])
        {
            ValidateRegistration(r);
            if (!Frames(r.State).TryGetValue(Environment.CurrentManagedThreadId, out List<object[]>? stack)
                || stack.Count == 0 || (Guid)stack[^1][0] != (Guid)r.Cell[0])
                throw new InvalidOperationException("Safe release requires the originating active effect lease.");
            if (stack.Count(frame => (Guid)frame[0] == (Guid)r.Cell[0]) != 1)
                throw new InvalidOperationException("Safe release refuses recursive effects on the same owner.");
            ThrowFailure(r);
            Owners(r.State).Remove((Guid)r.Cell[0]);
        }
    }

    private static string[] ValidateHeldOwnerGates(NativeEffectLease lease, object actualOwner)
    {
        ValidateLease(lease);
        var r = lease.Registration;
        ValidateRegistration(r);
        if (!ReferenceEquals(actualOwner, r.Owner))
            throw new InvalidOperationException("Native ownership requires the exact registered owner.");
        string[] keys = (string[])lease.Frame[1];
        if (!((HashSet<string>)r.Cell[3]).SetEquals(keys)
            || lease.CudaHeld != keys.Any(IsCuda) || lease.MlxHeld != keys.Contains(Mlx)
            || lease.DeviceGates.Count != keys.Count(key => IsCuda(key) && key != Wildcard))
            throw new InvalidOperationException("Native ownership requires the originating acquired scope gates.");
        if (lease.CudaHeld)
        {
            var cuda = (ReaderWriterLockSlim)r.State[6];
            if ((bool)lease.Frame[2] ? !cuda.IsWriteLockHeld : !cuda.IsReadLockHeld)
                throw new InvalidOperationException("The originating CUDA gate is not held by this thread.");
            string[] devices = keys.Where(key => IsCuda(key) && key != Wildcard).ToArray();
            for (int i = 0; i < devices.Length; i++)
                if (!ReferenceEquals(lease.DeviceGates[i], Scopes(r.State)[devices[i]][8])
                    || !Monitor.IsEntered(lease.DeviceGates[i]))
                    throw new InvalidOperationException("The originating device gates are not held by this thread.");
        }
        if (lease.MlxHeld && !Monitor.IsEntered(r.State[7]))
            throw new InvalidOperationException("The originating MLX gate is not held by this thread.");
        return keys;
    }

    internal static void ValidateSafeRelease(NativeEffectLease lease, object actualOwner)
    {
        lock (lease.Registration.State[1])
        {
            ValidateHeldOwnerGates(lease, actualOwner);
            var r = lease.Registration;
            if (Frames(r.State)[lease.ThreadId].Count(frame => (Guid)frame[0] == (Guid)r.Cell[0]) != 1)
                throw new InvalidOperationException("Safe release refuses recursive effects on the same owner.");
            ThrowFailure(r);
        }
    }

    internal static void Complete(NativeEffectLease lease, object actualOwner)
    {
        lock (lease.Registration.State[1])
        {
            ValidateSafeRelease(lease, actualOwner);
            Complete(lease.Registration);
        }
    }

    internal static NativeRuntimeFailure Publish(NativeEffectLease lease, object owner, Exception error, NativeRuntimeFailureStage stage)
    {
        ArgumentNullException.ThrowIfNull(error);
        if (stage < NativeRuntimeFailureStage.Synchronization || stage > NativeRuntimeFailureStage.WorkerRetirement)
            throw new ArgumentOutOfRangeException(nameof(stage));
        var r = lease.Registration;
        object[] s = r.State;
        lock (s[1])
        {
            string[] keys = ValidateHeldOwnerGates(lease, owner);
            object[][] affected = keys.Select(key => Scope(s, key)).ToArray();
            object[][] proposed = affected.Select(c => (object[])c.Clone()).ToArray();
            long revision = (long)s[2];
            foreach (var c in proposed)
            {
                if ((Guid)c[3] == Guid.Empty)
                {
                    c[3] = Guid.NewGuid(); c[4] = (int)stage;
                    c[5] = checked(++revision); c[6] = DateTimeOffset.UtcNow; c[7] = error;
                }
            }
            object[] first = proposed.OrderBy(c => (long)c[5]).First();
            bool allCuda = keys.Contains(Wildcard);
            foreach (var c in Owners(s).Values)
            {
                var otherKeys = (HashSet<string>)c[3];
                bool hit = ReferenceEquals(c, r.Cell) || otherKeys.Overlaps(keys)
                    || (allCuda && otherKeys.Any(IsCuda))
                    || ((bool)c[4] && keys.Any(IsCuda));
                if (hit && ((WeakReference<object>)c[2]).TryGetTarget(out object? actual))
                {
                    c[6] = actual;
                    if ((Guid)c[5] == Guid.Empty) c[5] = first[3];
                }
            }
            // Retain actual dependent owners before exposing any terminal scope facts.
            for (int i = 0; i < affected.Length; i++) Array.Copy(proposed[i], 3, affected[i], 3, 5);
            s[2] = revision;
            if (!Causes(s).TryGetValue(error, out List<Guid>? ids)) Causes(s).Add(error, ids = new List<Guid>());
            foreach (var c in affected) if (!ids.Contains((Guid)c[3])) ids.Add((Guid)c[3]);
            return new NativeRuntimeFailure(first);
        }
    }

    internal static NativeRuntimeQuarantineSnapshot Observe()
    {
        object[] s = GetState();
        lock (s[1])
        {
            ValidateCells(s);
            return new NativeRuntimeQuarantineSnapshot((long)s[2], Scopes(s).Values
                .Where(c => (Guid)c[3] != Guid.Empty).OrderBy(c => (long)c[5])
                .Select(c => new NativeRuntimeFailure(c)).ToArray());
        }
    }

    internal static bool TryGetFailure(Exception error, out NativeRuntimeFailure? failure)
    {
        object[] s = GetState();
        lock (s[1])
        {
            ValidateCells(s);
            var visited = new HashSet<Exception>(ReferenceEqualityComparer.Instance);
            var pending = new Stack<Exception>();
            var ids = new HashSet<Guid>();
            pending.Push(error);
            while (pending.TryPop(out Exception? current))
            {
                if (!visited.Add(current)) continue;
                if (Causes(s).TryGetValue(current, out List<Guid>? matches)) ids.UnionWith(matches);
                if (current.InnerException != null) pending.Push(current.InnerException);
                if (current is AggregateException aggregate)
                    foreach (Exception inner in aggregate.InnerExceptions) pending.Push(inner);
            }
            object[]? c = Scopes(s).Values.Where(cell => ids.Contains((Guid)cell[3])).OrderBy(cell => (long)cell[5]).FirstOrDefault();
            failure = c == null ? null : new NativeRuntimeFailure(c);
            return failure != null;
        }
    }

    internal static bool IsRetainedFailure(NativeOwnerRegistration registration, Exception error)
    {
        lock (registration.State[1])
        {
            ValidateRegistration(registration);
            Guid id = (Guid)registration.Cell[5];
            if (id == Guid.Empty || !ReferenceEquals(registration.Cell[6], registration.Owner)) return false;
            if (Causes(registration.State).TryGetValue(error, out var ids) && ids.Contains(id)) return true;
            if (error is not NativeRuntimeQuarantinedException refusal || refusal.Failure.FailureId != id)
                return false;
            return Scopes(registration.State).Values.Any(c => (Guid)c[3] == id
                && ReferenceEquals(c[7], refusal.InnerException));
        }
    }
}
