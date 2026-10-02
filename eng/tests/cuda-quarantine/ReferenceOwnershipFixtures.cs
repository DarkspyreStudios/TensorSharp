using System.Reflection;
using TensorSharp;

namespace TensorSharp.CudaQuarantineFixture;

internal static class ReferenceOwnershipFixtures
{
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
        internal int Destructions;
        protected override void Destroy() => Interlocked.Increment(ref Destructions);
    }
}
