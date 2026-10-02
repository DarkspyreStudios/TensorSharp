using TensorSharp;

namespace TensorSharp.QuarantineFixture;

public sealed class Fixture
{
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
