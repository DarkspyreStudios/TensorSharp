using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;

const string slot = "Darkspyre.TensorSharp.NativeQuarantine";
string mode = args.Single();
string configuration = new DirectoryInfo(AppContext.BaseDirectory).Parent!.Name;
string fixturePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,
    "../../../../native-quarantine-fixture/bin", configuration, "net10.0/TensorSharp.QuarantineFixture.dll"));
if (!File.Exists(fixturePath)) throw new FileNotFoundException("Build the actual fixture graph first.", fixturePath);
switch (mode)
{
    case "healthy":
        var healthy = Healthy(fixturePath);
        Collect();
        Require(!healthy.Alc.IsAlive && !healthy.Assembly.IsAlive && !healthy.Owner.IsAlive && !healthy.Storage.IsAlive,
            "Healthy actual foreign graph did not collect.");
        Require(NativeRuntimeQuarantine.Observe().State == NativeRuntimeQuarantineState.NoRecordedFailure, "Unexpected failure.");
        break;
    case "promotion":
    case "wildcard":
    case "mlx":
        var retained = Promotion(fixturePath, mode);
        Collect();
        Require(retained.Failed.All(w => w.IsAlive), "Failed actual foreign ownership was not retained.");
        Require(retained.Independent.All(w => !w.IsAlive), "Independent safe ownership did not collect.");
        break;
    case "lease-negatives":
        using (var f = new Foreign(fixturePath)) f.Static("LeaseNegatives");
        break;
    case "mismatched-lease":
        using (var f = new Foreign(fixturePath)) f.Static("MismatchedLease");
        break;
    case "resolved-cuda-nesting":
        using (var f = new Foreign(fixturePath)) f.Static("ResolvedCudaNesting");
        break;
    case "queued-safe-release":
        using (var f = new Foreign(fixturePath)) f.Static("QueuedSafeRelease");
        break;
    case "race":
        Race(fixturePath);
        break;
    case "causes":
        Causes(fixturePath);
        break;
    case "constructor":
        var construction = ConstructorFailure(fixturePath);
        Collect();
        Require(construction.All(w => w.IsAlive), "Constructor rollback did not retain actual ownership.");
        break;
    case "initialize":
        Initialize(fixturePath);
        break;
    case "concurrent-faults":
        ConcurrentFaults(fixturePath);
        break;
    case "queued":
        Queued(fixturePath);
        break;
    case "roles":
        var roles = Roles(fixturePath);
        Collect();
        Require(roles.All(w => w.IsAlive), "An actual dependent owner role was not retained.");
        break;
    case "cross-nesting":
        CrossNesting(fixturePath);
        break;
    case "lock-order":
        LockOrder(fixturePath);
        break;
    case "protocol":
        AppDomain.CurrentDomain.SetData(slot, new object[] { 999 });
        try { NativeRuntimeQuarantine.Observe(); throw new Exception("Invalid protocol was accepted."); }
        catch (InvalidOperationException) { }
        Require(((object[])AppDomain.CurrentDomain.GetData(slot)!)[0].Equals(999), "Invalid slot was overwritten.");
        break;
    case "malformed-cells":
        _ = NativeRuntimeQuarantine.Observe();
        var payload = (object[])AppDomain.CurrentDomain.GetData(slot)!;
        ((Dictionary<string, object[]>)payload[3]).Add("unknown-scope", new object[9]);
        try { NativeRuntimeQuarantine.Observe(); throw new Exception("Malformed scope was accepted."); }
        catch (InvalidOperationException) { }
        break;
    default: throw new ArgumentException("Unknown focused mode.");
}
Console.WriteLine(JsonSerializer.Serialize(new
{
    mode, passed = true, nativeEffects = 0, fixturePath,
    fixtureSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(fixturePath))),
    coreSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(NativeRuntimeQuarantine).Assembly.Location))),
    coreMvid = typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId,
    state = mode is "protocol" or "malformed-cells" ? "incompatible-refused" : NativeRuntimeQuarantine.Observe().State.ToString(),
    qualification = "managed-authority-only; backend integrations inert; no native/model/GPU qualification"
}));

[MethodImpl(MethodImplOptions.NoInlining)]
static (WeakReference Alc, WeakReference Assembly, WeakReference Owner, WeakReference Storage) Healthy(string path)
{
    using var f = new Foreign(path);
    f.Create(0, "cuda");
    var refs = (new WeakReference(f.Alc), new WeakReference(f.Assembly), (WeakReference)f.Get("OwnerReference"), (WeakReference)f.Get("StorageReference"));
    f.Call("Complete");
    return refs;
}

[MethodImpl(MethodImplOptions.NoInlining)]
static (WeakReference[] Failed, WeakReference[] Independent) Promotion(string path, string mode)
{
    using var a = new Foreign(path); using var b = new Foreign(path); using var independent = new Foreign(path);
    string kind = mode == "mlx" ? "mlx" : mode == "wildcard" ? "wildcard" : "cuda";
    a.Create(0, kind); b.Create(0, mode == "promotion" ? "wildcard" : kind);
    independent.Create(1, mode == "wildcard" ? "mlx" : "cuda");
    var failed = new[] { new WeakReference(a.Alc), new WeakReference(a.Assembly), (WeakReference)a.Get("OwnerReference"),
        (WeakReference)a.Get("StorageReference"), (WeakReference)a.Get("CoreReference"), new WeakReference(b.Alc),
        (WeakReference)b.Get("OwnerReference"), (WeakReference)b.Get("StorageReference"), (WeakReference)b.Get("CoreReference") };
    var safe = new[] { new WeakReference(independent.Alc), new WeakReference(independent.Assembly),
        (WeakReference)independent.Get("OwnerReference"), (WeakReference)independent.Get("StorageReference"),
        (WeakReference)independent.Get("CoreReference") };
    var error = new HostileError();
    Guid id = (Guid)a.Call("Fail", error)!;
    Require((string)b.Call("Check")! == id.ToString(), "Foreign dependent owner was not fenced.");
    Require((bool)b.Call("RefusalMatches", error, id)!, "Refusal lost the original failure inner or identity.");
    Require(NativeRuntimeQuarantine.TryGetFailure(new AggregateException(new Exception("original"), error), out var failure)
        && failure!.FailureId == id, "Constructor-shaped aggregate lost exact cause.");
    independent.Call("Complete");
    return (failed, safe);
}

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] Roles(string path)
{
    var fixtures = new List<Foreign>();
    var references = new List<WeakReference>();
    try
    {
        for (int role = 0; role < 6; role++)
        {
            var f = new Foreign(path);
            fixtures.Add(f);
            f.Create(0, "cuda", role);
            references.Add(new WeakReference(f.Alc));
            references.Add((WeakReference)f.Get("OwnerReference"));
            references.Add((WeakReference)f.Get("StorageReference"));
            references.Add((WeakReference)f.Get("CoreReference"));
        }
        var error = new HostileError();
        Guid id = (Guid)fixtures[0].Call("Fail", error)!;
        Require(fixtures.All(f => (bool)f.Call("RefusalMatches", error, id)!), "A dependent role admitted terminal work.");
        return references.ToArray();
    }
    finally { foreach (var f in fixtures) f.Dispose(); }
}

static void CrossNesting(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path); using var c = new Foreign(path); using var wildcard = new Foreign(path);
    a.Create(0, "cuda"); b.Create(0, "cuda"); c.Create(1, "cuda"); wildcard.Create(0, "wildcard");
    a.Call("Start");
    try
    {
        b.Call("Start");
        try
        {
            Refuses(() => c.Call("Start"));
            Refuses(() => wildcard.Call("Start"));
            Refuses(() => a.Call("Stop"));
            Refuses(() => b.Call("AttachCuda", 1));
        }
        finally { b.Call("Stop"); }
    }
    finally { a.Call("Stop"); }
    a.Call("Complete"); b.Call("Complete"); c.Call("Complete"); wildcard.Call("Complete");
    Require(ActiveThreads() == 0 && NativeRuntimeQuarantine.Observe().Revision == 0, "Rejected nesting left state or locks.");
}

static void Refuses(Action action)
{
    try { action(); }
    catch (InvalidOperationException) { return; }
    throw new Exception("Expected admission refusal.");
}

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] ConstructorFailure(string path)
{
    using var f = new Foreign(path);
    var original = new InvalidOperationException("original validation error");
    var cleanup = new HostileError();
    WeakReference? owner = null, storage = null;
    Action<WeakReference, WeakReference> capture = (o, s) => { owner = o; storage = s; };
    try { f.Static("ConstructorFailure", capture, original, cleanup); throw new Exception("Constructor returned."); }
    catch (AggregateException error)
    {
        Require(ReferenceEquals(error.InnerExceptions[0], original) && ReferenceEquals(error.InnerExceptions[1], cleanup),
            "Constructor original/cleanup identity was lost.");
        Require(NativeRuntimeQuarantine.TryGetFailure(error, out _), "Constructor exception does not map exact cleanup cause.");
    }
    return new[] { new WeakReference(f.Alc), new WeakReference(f.Assembly), owner!, storage! };
}

static void Initialize(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path);
    using var barrier = new Barrier(2);
    Task t1 = Task.Run(() => { barrier.SignalAndWait(); a.Create(0, "cuda"); });
    Task t2 = Task.Run(() => { barrier.SignalAndWait(); b.Create(0, "cuda"); });
    Require(Task.WaitAll(new[] { t1, t2 }, TimeSpan.FromSeconds(5)), "Initialization did not finish.");
    Require((long)a.Get("Revision") == 0 && (long)b.Get("Revision") == 0, "Initial authorities disagree.");
    var state = (object[])AppDomain.CurrentDomain.GetData("Darkspyre.TensorSharp.NativeQuarantine")!;
    lock (state[1]) Require(((Dictionary<string, object[]>)state[3]).Count == 1, "Generations created parallel scope authorities.");
    a.Call("Complete"); b.Call("Complete");
}

static void ConcurrentFaults(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path);
    a.Create(0, "cuda"); b.Create(1, "cuda");
    using var entered = new Barrier(2);
    var errorA = new HostileError(); var errorB = new HostileError();
    Task<Guid> t1 = Task.Run(() =>
    {
        a.Call("Start");
        try { entered.SignalAndWait(); return (Guid)a.Call("FailActive", errorA)!; }
        finally { a.Call("Stop"); }
    });
    Task<Guid> t2 = Task.Run(() =>
    {
        b.Call("Start");
        try { entered.SignalAndWait(); return (Guid)b.Call("FailActive", errorB)!; }
        finally { b.Call("Stop"); }
    });
    Require(Task.WaitAll(new Task[] { t1, t2 }, TimeSpan.FromSeconds(5)), "Concurrent faults did not drain.");
    Require(t1.Result != t2.Result && NativeRuntimeQuarantine.Observe().Revision == 2, "Failure revisions collided.");
    Require(NativeRuntimeQuarantine.TryGetFailure(errorA, out var fa) && fa!.FailureId == t1.Result, "CauseA did not match.");
    Require(NativeRuntimeQuarantine.TryGetFailure(errorB, out var fb) && fb!.FailureId == t2.Result, "CauseB did not match.");
}

static void Queued(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path);
    a.Create(0, "mlx"); b.Create(0, "mlx");
    Require((string)b.Call("Check")! == "healthy", "Queue admission was not healthy.");
    a.Call("Fail", new HostileError());
    int effects = 0;
    Task pending = Task.Run(() =>
    {
        try { b.Call("Start"); effects++; b.Call("Stop"); }
        catch (Exception ex) when (ex.GetType().FullName == typeof(NativeRuntimeQuarantinedException).FullName) { }
    });
    Require(pending.Wait(TimeSpan.FromSeconds(5)) && effects == 0, "Queued execution passed the terminal fence.");
}

static void Race(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path);
    a.Create(0, "cuda"); b.Create(0, "cuda");
    a.Call("Start");
    using var started = new ManualResetEventSlim();
    int effects = 0;
    Task pending = Task.Run(() =>
    {
        started.Set();
        try { b.Call("Start"); Interlocked.Increment(ref effects); b.Call("Stop"); }
        catch (Exception ex) when (ex.GetType().FullName == typeof(NativeRuntimeQuarantinedException).FullName) { }
    });
    try
    {
        Require(started.Wait(TimeSpan.FromSeconds(5)), "Contender did not start.");
        Require(SpinWait.SpinUntil(() => ActiveThreads() == 2, TimeSpan.FromSeconds(5)), "Contender did not reserve its effect before publication.");
        a.Call("FailActive", new Exception("controlled cleanup refusal"));
    }
    finally { a.Call("Stop"); }
    Require(pending.Wait(TimeSpan.FromSeconds(5)), "Contender did not drain.");
    Require(effects == 0, "An effect passed terminal publication.");
}

static int ActiveThreads()
{
    var state = (object[])AppDomain.CurrentDomain.GetData("Darkspyre.TensorSharp.NativeQuarantine")!;
    lock (state[1]) return ((Dictionary<int, List<object[]>>)state[9]).Count;
}

static void LockOrder(string path)
{
    using var a = new Foreign(path); using var b = new Foreign(path);
    a.Create(0, "cuda"); b.Create(1, "cuda");
    var state = (object[])AppDomain.CurrentDomain.GetData("Darkspyre.TensorSharp.NativeQuarantine")!;
    using var metadataHeld = new ManualResetEventSlim();
    using var domainHeld = new ManualResetEventSlim();
    Task metadata = Task.Run(() =>
    {
        lock (state[1])
        {
            metadataHeld.Set();
            Require(domainHeld.Wait(TimeSpan.FromSeconds(2)), "Domain holder did not start.");
            a.Call("Start");
            a.Call("Stop");
        }
    });
    Task domain = Task.Run(() =>
    {
        lock (AppDomain.CurrentDomain)
        {
            domainHeld.Set();
            Require(metadataHeld.Wait(TimeSpan.FromSeconds(2)), "Metadata holder did not start.");
            _ = b.Get("Revision");
        }
    });
    Require(Task.WaitAll(new[] { metadata, domain }, TimeSpan.FromSeconds(3)),
        "AppDomain/metadata lock inversion; background managed tasks remain blocked until this failing test process exits.");
    a.Call("Complete"); b.Call("Complete");
}

static void Causes(string path)
{
    using var a = new Foreign(path);
    a.Create(0, "cuda");
    a.Call("Start");
    var first = new HostileError(); var repeat = new HostileError();
    try
    {
        Guid id = (Guid)a.Call("FailActive", first)!;
        Require((Guid)a.Call("FailActive", repeat)! == id, "First failure identity changed.");
        Require(NativeRuntimeQuarantine.TryGetFailure(new AggregateException(first, repeat, first), out var result)
            && result!.FailureId == id, "Repeated causes did not match.");
        Require(!NativeRuntimeQuarantine.TryGetFailure(new HostileError(), out _), "Unrelated error matched global state.");
        Require(!NativeRuntimeQuarantine.TryGetFailure(new InvalidOperationException("pack release"), out _), "Pack error matched native failure.");
    }
    finally { a.Call("Stop"); }
}

static void Collect()
{
    for (int i = 0; i < 12; i++) { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); Thread.Sleep(10); }
}

static void Require(bool condition, string message)
{
    if (!condition) throw new Exception(message);
}

sealed class HostileError : Exception
{
    public override string Message => throw new Exception("Message must not be inspected.");
    public override System.Collections.IDictionary Data => throw new Exception("Data must not be inspected.");
    public override string ToString() => throw new Exception("ToString must not be inspected.");
}

sealed class Foreign : IDisposable
{
    public AssemblyLoadContext Alc { get; }
    public Assembly Assembly { get; }
    private readonly Type _type;
    private object? _instance;

    public Foreign(string path)
    {
        Alc = new PrivateContext(path);
        Assembly = Alc.LoadFromAssemblyPath(path);
        _type = Assembly.GetType("TensorSharp.QuarantineFixture.Fixture", true)!;
    }

    public void Create(int ordinal, string kind, int role = 0)
    {
        _instance = Activator.CreateInstance(_type, ordinal, kind, role)!;
        if (!(bool)Get("CoreIsCollectible")) throw new Exception("Fixture reused Default Core.");
        if ((Guid)Get("CoreMvid") != typeof(NativeRuntimeQuarantine).Assembly.ManifestModule.ModuleVersionId)
            throw new Exception("Fixture Core does not match the executing source graph.");
    }
    public object Get(string property) => _type.GetProperty(property)!.GetValue(_instance)!;
    public object? Call(string method, params object[] args)
    {
        try { return _type.GetMethod(method)!.Invoke(_instance, args); }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw(); throw; }
    }
    public void Static(string method, params object[] args)
    {
        try { _type.GetMethod(method)!.Invoke(null, args); }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw(); throw; }
    }
    public void Dispose() { _instance = null; Alc.Unload(); }

    private sealed class PrivateContext(string fixturePath) : AssemblyLoadContext(isCollectible: true)
    {
        private readonly AssemblyDependencyResolver _resolver = new(fixturePath);
        protected override Assembly? Load(AssemblyName name)
        {
            string? path = _resolver.ResolveAssemblyToPath(name);
            return path == null ? null : LoadFromAssemblyPath(path);
        }
    }
}
