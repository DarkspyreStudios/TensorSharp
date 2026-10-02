using System.Runtime.CompilerServices;
using TensorSharp;

namespace TensorSharp.QuarantineFixture;

public sealed class Fixture
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static object[] FinalizerOrigin(bool fail)
    {
        object[] observation = { false, false, Guid.Empty, null!, false,
            typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId,
            typeof(Fixture).Assembly.ManifestModule.ModuleVersionId };
        var owner = new FinalizingOwner(observation);
        object[] result = { new WeakReference(owner, true), new WeakReference(owner.Resource, true),
            observation, new WeakReference(typeof(NativeRuntimeQuarantine).Assembly) };
        if (!fail) owner.ReleaseSafely();
        return result;
    }

    private sealed class FinalizingOwner
    {
        internal readonly byte[] Resource = new byte[128];
        private readonly NativeOwnerRegistration _registration;
        private readonly object[] _observation;

        internal FinalizingOwner(object[] observation)
        {
            _observation = observation;
            _registration = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Storage);
            _registration.AttachCudaPrimaryDevice(11);
        }

        internal void ReleaseSafely()
        {
            using var lease = _registration.EnterEffect();
            lease.ValidateSafeRelease(this);
            lease.CompleteSafeRelease(this);
            GC.SuppressFinalize(this);
        }

        public object[] ExecutingProvenance() => new object[]
        {
            GetType().Assembly.ManifestModule.ModuleVersionId,
            typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId,
            GetType().Assembly.IsCollectible,
            typeof(NativeRuntimeQuarantine).Assembly.IsCollectible
        };

        ~FinalizingOwner()
        {
            _observation[1] = ((WeakReference<object>)_registration.Cell[2]).TryGetTarget(out object? actual)
                && ReferenceEquals(actual, this);
            var original = new InvalidOperationException("Controlled finalizer cleanup refusal.");
            using var lease = _registration.EnterEffect();
            Guid id = lease.PublishFailure(this, original, NativeRuntimeFailureStage.StorageRelease).FailureId;
            _observation[3] = original;
            _observation[2] = id;
            _observation[4] = ReferenceEquals(_registration.Cell[6], this);
            _observation[0] = true;
        }
    }

    private sealed class Owner
    {
        internal readonly byte[] ActualResource = new byte[128];
    }

    private readonly Owner _owner = new();
    private readonly NativeOwnerRegistration _registration;
    private NativeEffectLease? _lease;

    public Fixture(int ordinal, string kind, int role)
    {
        _registration = NativeQuarantineAuthority.Register(_owner, (NativeOwnerRole)role);
        if (kind == "mlx") _registration.AttachMlxSharedRuntime();
        else if (kind == "wildcard") _registration.AttachCudaDependentGgml();
        else _registration.AttachCudaPrimaryDevice(ordinal);
    }

    public WeakReference OwnerReference => new(_owner, true);
    public WeakReference StorageReference => new(_owner.ActualResource, true);
    public WeakReference CoreReference => new(typeof(NativeRuntimeQuarantine).Assembly);
    public bool CoreIsCollectible => typeof(NativeRuntimeQuarantine).Assembly.IsCollectible;
    public Guid CoreMvid => typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId;
    public Guid FixtureMvid => GetType().Assembly.ManifestModule.ModuleVersionId;

    public string Check()
    {
        try { _registration.ThrowIfQuarantined(); return "healthy"; }
        catch (NativeRuntimeQuarantinedException ex) { return ex.Failure.FailureId.ToString(); }
    }

    public bool RefusalMatches(Exception error, Guid id)
    {
        try { _registration.ThrowIfQuarantined(); return false; }
        catch (NativeRuntimeQuarantinedException ex)
        {
            return ReferenceEquals(ex.InnerException, error) && ex.Failure.FailureId == id
                && NativeRuntimeQuarantine.TryGetFailure(ex, out var failure) && failure!.FailureId == id;
        }
    }

    public void Start() => _lease = _registration.EnterEffect();
    public void Stop() { _lease!.Dispose(); _lease = null; }
    public object[] RegistrationBinding => new[] { _registration.State, _registration.Cell, (object)_owner,
        typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId };
    public static WeakReference CurrentCoreReference() => new(typeof(NativeRuntimeQuarantine).Assembly);
    public int ActiveReservations { get { lock (_registration.State[1]) return (int)_registration.Cell[7]; } }
    public void CompleteActive()
    {
        _lease!.ValidateSafeRelease(_owner);
        _lease.CompleteSafeRelease(_owner);
    }

    public static string AttemptForeignEffect(object[] binding)
    {
        Require(typeof(NativeRuntimeQuarantine).Assembly.IsCollectible
            && typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId == (Guid)binding[3]
            && !ReferenceEquals(binding[2].GetType().Assembly, typeof(Fixture).Assembly),
            "The queued effect did not execute in a distinct matching private generation.");
        var registration = new NativeOwnerRegistration((object[])binding[0], (object[])binding[1], binding[2]);
        try { using var lease = registration.EnterEffect(); return "native-entry"; }
        catch (InvalidOperationException) { return "refused"; }
    }
    public void Complete()
    {
        using var lease = _registration.EnterEffect();
        _registration.CompleteSafeRelease();
    }

    public Guid Fail(Exception error)
    {
        using var lease = _registration.EnterEffect();
        return lease.PublishFailure(_owner, error, NativeRuntimeFailureStage.Synchronization).FailureId;
    }

    public Guid FailActive(Exception error)
        => _lease!.PublishFailure(_owner, error, NativeRuntimeFailureStage.StorageRelease).FailureId;

    public bool Match(Exception error, Guid id) => NativeRuntimeQuarantine.TryGetFailure(error, out var f) && f!.FailureId == id;
    public long Revision => NativeRuntimeQuarantine.Observe().Revision;
    public void AttachCuda(int ordinal) => _registration.AttachCudaPrimaryDevice(ordinal);

    public static void ConstructorFailure(Action<WeakReference, WeakReference> capture, Exception original, Exception cleanup)
    {
        _ = new FailingConstructor(capture, original, cleanup);
    }

    private sealed class FailingConstructor
    {
        private readonly byte[] _storage = new byte[128];

        internal FailingConstructor(Action<WeakReference, WeakReference> capture, Exception original, Exception cleanup)
        {
            var registration = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Model);
            registration.AttachMlxSharedRuntime();
            capture(new WeakReference(this, true), new WeakReference(_storage, true));
            using var lease = registration.EnterEffect();
            lease.PublishFailure(this, cleanup, NativeRuntimeFailureStage.StorageRelease);
            throw new AggregateException(original, cleanup);
        }
    }

    public static void LeaseNegatives()
    {
        var a = new Owner(); var b = new Owner();
        var ra = NativeQuarantineAuthority.Register(a, NativeOwnerRole.Model);
        var rb = NativeQuarantineAuthority.Register(b, NativeOwnerRole.Storage);
        ra.AttachCudaPrimaryDevice(0); rb.AttachCudaPrimaryDevice(0);
        var lease = ra.EnterEffect();
        try
        {
            Refuses(() => lease.PublishFailure(b, new Exception(), NativeRuntimeFailureStage.GraphRelease));
            Refuses(() => ra.AttachCudaPrimaryDevice(1));
            using (var nested = rb.EnterEffect())
                Refuses(() => lease.PublishFailure(a, new Exception(), NativeRuntimeFailureStage.GraphRelease));
            var wider = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.Model);
            wider.AttachCudaPrimaryDevice(1);
            Refuses(() => wider.EnterEffect());
            var upgrade = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.Model);
            upgrade.AttachCudaDependentGgml();
            Refuses(() => upgrade.EnterEffect());
            Task.Run(() =>
            {
                Refuses(() => lease.PublishFailure(a, new Exception(), NativeRuntimeFailureStage.GraphRelease));
                Refuses(() => lease.Dispose());
            }).GetAwaiter().GetResult();
            ra.CompleteSafeRelease();
        }
        finally { lease.Dispose(); }
        Refuses(() => lease.PublishFailure(a, new Exception(), NativeRuntimeFailureStage.GraphRelease));
        Refuses(() => lease.Dispose());
        using (var bLease = rb.EnterEffect()) rb.CompleteSafeRelease();
        Require(NativeRuntimeQuarantine.Observe().Revision == 0, "Rejected leases changed terminal state.");
    }

    public static void MismatchedLease()
    {
        var a = new Owner(); var b = new Owner();
        var ra = NativeQuarantineAuthority.Register(a, NativeOwnerRole.Model);
        var rb = NativeQuarantineAuthority.Register(b, NativeOwnerRole.Storage);
        ra.AttachCudaPrimaryDevice(0); rb.AttachCudaPrimaryDevice(0);
        using (var actual = ra.EnterEffect())
        {
            var mismatched = new NativeEffectLease(rb, actual.Frame);
            Refuses(() => mismatched.PublishFailure(b, new Exception(), NativeRuntimeFailureStage.StorageRelease));
            ra.CompleteSafeRelease();
        }
        using (var lease = rb.EnterEffect()) rb.CompleteSafeRelease();
        Require(NativeRuntimeQuarantine.Observe().Revision == 0, "Mismatched lease changed terminal state.");
    }

    public static void ResolvedCudaNesting()
    {
        var discovery = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.NativeHandle);
        discovery.AttachCudaDependentGgml();
        var known = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.Allocator);
        known.AttachCudaPrimaryDevice(1);
        var other = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.Storage);
        other.AttachCudaPrimaryDevice(2);
        var mlx = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.Worker);
        mlx.AttachMlxSharedRuntime();
        using (var unresolved = discovery.EnterEffect())
        {
            using (var resolved = known.EnterEffect())
            {
                Refuses(() => other.EnterEffect());
                Refuses(() => discovery.EnterEffect());
                Refuses(() => mlx.EnterEffect());
                Refuses(() => known.AttachCudaPrimaryDevice(2));
                known.CompleteSafeRelease();
            }
            discovery.CompleteSafeRelease();
        }
        using (var lease = other.EnterEffect()) other.CompleteSafeRelease();
        using (var lease = mlx.EnterEffect()) mlx.CompleteSafeRelease();
        Require(NativeRuntimeQuarantine.Observe().Revision == 0, "Resolved nesting changed terminal state.");
    }

    public static void QueuedSafeRelease()
    {
        var registration = NativeQuarantineAuthority.Register(new Owner(), NativeOwnerRole.NativeHandle);
        registration.AttachCudaPrimaryDevice(0);
        int nativeEntries = 0;
        bool refused = false;
        Exception? completionFailure = null;
        var release = registration.EnterEffect();
        var queued = Task.Run(() =>
        {
            try
            {
                using var effect = registration.EnterEffect();
                Interlocked.Increment(ref nativeEntries);
            }
            catch (InvalidOperationException) { refused = true; }
        });
        try
        {
            Require(SpinWait.SpinUntil(() =>
            {
                lock (registration.State[1]) return (int)registration.Cell[7] == 2;
            }, TimeSpan.FromSeconds(3)), "The controlled waiter did not reserve its effect.");
            release.ValidateSafeRelease(registration.Owner);
            release.CompleteSafeRelease(registration.Owner);
        }
        catch (Exception error) { completionFailure = error; }
        finally
        {
            release.Dispose();
            Require(queued.Wait(TimeSpan.FromSeconds(3)), "The controlled waiter did not drain.");
            if (completionFailure != null)
            {
                using var cleanup = registration.EnterEffect();
                registration.CompleteSafeRelease();
            }
        }
        if (completionFailure != null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(completionFailure).Throw();
        Require(refused && nativeEntries == 0, "Queued work entered native effects after safe release.");
        lock (registration.State[1])
        {
            Require((int)registration.Cell[7] == 0, "Queued reservations remained active.");
            Require(((Dictionary<int, List<object[]>>)registration.State[9]).Count == 0, "Thread frames did not drain.");
        }
    }

    public static void SafeReleaseNegatives()
    {
        var owner = new Owner();
        var registration = NativeQuarantineAuthority.Register(owner, NativeOwnerRole.Storage);
        registration.AttachCudaPrimaryDevice(0);
        int frees = 0;
        var release = registration.EnterEffect();
        try
        {
            Refuses(() => release.ValidateSafeRelease(new Owner()));
            using (var nested = registration.EnterEffect())
                Refuses(() =>
                {
                    nested.ValidateSafeRelease(owner);
                    frees++;
                });
            var forged = new NativeEffectLease(registration, release.Frame) { CudaHeld = true };
            forged.DeviceGates.Add(new object());
            Refuses(() => forged.ValidateSafeRelease(owner));
            Task.Run(() => Refuses(() => release.ValidateSafeRelease(owner))).GetAwaiter().GetResult();
            release.ValidateSafeRelease(owner);
            release.CompleteSafeRelease(owner);
        }
        finally { release.Dispose(); }
        Refuses(() => release.ValidateSafeRelease(owner));
        Require(frees == 0, "Nested release reached the controlled free callback.");
    }

    private static void Refuses(Action action)
    {
        try { action(); }
        catch (InvalidOperationException) { return; }
        throw new Exception("Expected managed admission refusal.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }
}
