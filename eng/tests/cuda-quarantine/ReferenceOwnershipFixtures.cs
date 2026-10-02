using System.Reflection;
using TensorSharp;

namespace TensorSharp.CudaQuarantineFixture;

internal static class ReferenceOwnershipFixtures
{
    internal static void GateAdmission()
    {
        var owner = new RecordingOwner(useGate: true);
        owner.AddRef();
        owner.Release();
        if (owner.ReadReferenceCount() != 1 || owner.Validations != 1)
            throw new InvalidOperationException("Healthy reference mutation changes its count or validation.");

        using var started = new ManualResetEventSlim();
        Exception? refusal = null;
        var worker = new Thread(() =>
        {
            started.Set();
            try { owner.AddRef(); }
            catch (Exception error) { refusal = error; }
        })
        { IsBackground = true };
        lock (owner.Gate)
        {
            worker.Start();
            if (!started.Wait(TimeSpan.FromSeconds(5)))
                throw new InvalidOperationException("Reference worker does not start.");
            owner.Retiring = true;
        }
        if (!worker.Join(TimeSpan.FromSeconds(5)))
            throw new InvalidOperationException("Reference addition does not exit after the census gate opens.");
        if (refusal is not InvalidOperationException || owner.ReadReferenceCount() != 1 || owner.Destructions != 0)
            throw new InvalidOperationException("Retirement admission adds a reference or destroys live storage.");
        owner.Release();
        if (owner.ReadReferenceCount() != 0 || owner.Destructions != 1 || owner.DestroyedUnderGate)
            throw new InvalidOperationException("Final release does not destroy exactly once outside the census gate.");
    }

    internal static void GateRefusal()
    {
        var original = new InvalidOperationException("Controlled reference admission refusal.");
        var owner = new RecordingOwner(useGate: true) { AdmissionFailure = original };
        try { owner.AddRef(); throw new Exception("Reference addition unexpectedly succeeds."); }
        catch (InvalidOperationException error) when (ReferenceEquals(error, original)) { }
        bool acquired = false;
        var worker = new Thread(() =>
        {
            acquired = Monitor.TryEnter(owner.Gate, TimeSpan.FromSeconds(2));
            if (acquired) Monitor.Exit(owner.Gate);
        })
        { IsBackground = true };
        worker.Start();
        if (!worker.Join(TimeSpan.FromSeconds(5)) || !acquired || owner.ReadReferenceCount() != 1)
            throw new InvalidOperationException("Refused reference mutation leaves its gate held or changes the count.");
        owner.Release();
    }

    internal static void ParallelRelease()
    {
        for (int i = 0; i < 64; i++)
        {
            var owner = new RecordingOwner();
            owner.AddRef();
            using var start = new ManualResetEventSlim();
            Exception? firstError = null;
            Exception? secondError = null;
            var first = new Thread(() =>
            {
                start.Wait();
                try { owner.Release(); }
                catch (Exception error) { firstError = error; }
            })
            { IsBackground = true };
            var second = new Thread(() =>
            {
                start.Wait();
                try { owner.Release(); }
                catch (Exception error) { secondError = error; }
            })
            { IsBackground = true };
            try
            {
                first.Start();
                second.Start();
            }
            finally { start.Set(); }
            if (!first.Join(TimeSpan.FromSeconds(5)) || !second.Join(TimeSpan.FromSeconds(5))
                || firstError != null || secondError != null
                || owner.ReadReferenceCount() != 0 || owner.Destructions != 1)
                throw new InvalidOperationException("Concurrent releases do not retire their exact references once.");
            try { owner.AddRef(); throw new Exception("Destroyed owner accepts a reference."); }
            catch (InvalidOperationException) { }
            try { owner.Release(); throw new Exception("Destroyed owner accepts another release."); }
            catch (InvalidOperationException) { }
            if (owner.Destructions != 1)
                throw new InvalidOperationException("A zero-count refusal repeats destruction.");
        }
    }

    internal static void TensorIntents()
    {
        using var root = new Tensor(new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet), DType.Float32, 4);
        using var copy = root.CopyRef();
        using var view = root.View(2, 2);
        Tensor same = root;
        var intents = new HashSet<Tensor>(ReferenceEqualityComparer.Instance) { root, copy, view, same };
        Storage storage = root.Storage;
        if (intents.Count != 3 || storage.ReadReferenceCount() != 3
            || intents.Any(t => !ReferenceEquals(t.GetLiveOwnedStorageForDisposal(), storage)))
            throw new InvalidOperationException("Tensor intent identities do not match their actual Storage references.");
        root.Dispose();
        root.Dispose();
        if (root.GetLiveOwnedStorageForDisposal() != null || storage.ReadReferenceCount() != 2
            || !ReferenceEquals(copy.GetLiveOwnedStorageForDisposal(), storage))
            throw new InvalidOperationException("Disposed Tensor still contributes an ownership intent or drops a borrowed view.");
        view.Dispose();
        copy.Dispose();
        if (storage.ReadReferenceCount() != 0 || ((TensorSharp.Cpu.CpuStorage)storage).buffer != IntPtr.Zero)
            throw new InvalidOperationException("Successful explicit host Storage cleanup does not retire the real allocation.");
    }

    internal static void Overflow()
    {
        var owner = new RecordingOwner();
        typeof(RefCounted).GetField("refCount", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(owner, int.MaxValue);
        bool refused = false;
        try { owner.AddRef(); }
        catch (OverflowException) { refused = true; }
        int count = (int)typeof(RefCounted).GetField("refCount", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(owner)!;
        GC.SuppressFinalize(owner);
        if (!refused || count != int.MaxValue || owner.Destructions != 0)
            throw new InvalidOperationException("Reference addition wraps an overflowing live reference count.");
    }

    private sealed class RecordingOwner : RefCounted
    {
        internal readonly object Gate = new();
        private readonly bool _useGate;
        internal bool Retiring;
        internal Exception? AdmissionFailure;
        internal int Destructions;
        internal int Validations;
        internal bool DestroyedUnderGate;

        internal RecordingOwner(bool useGate = false) => _useGate = useGate;
        internal override object? ReferenceMutationGate => _useGate ? Gate : null;
        internal override void ValidateReferenceAddition()
        {
            if (_useGate && !Monitor.IsEntered(Gate))
                throw new InvalidOperationException("Reference validation runs outside the census gate.");
            Validations++;
            if (AdmissionFailure != null) throw AdmissionFailure;
            if (Retiring) throw new InvalidOperationException("Controlled owner retirement refuses new references.");
        }
        protected override void Destroy()
        {
            DestroyedUnderGate = Monitor.IsEntered(Gate);
            Interlocked.Increment(ref Destructions);
        }
    }
}
