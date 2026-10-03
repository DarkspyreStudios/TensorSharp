using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Thrown when an operation that cannot be captured (a stream synchronize,
    /// i.e. a host read of device data) runs while a CUDA graph capture is
    /// active. The capture site catches this, aborts the capture, blacklists
    /// the shape and re-runs the region plainly (no kernel executed during the
    /// failed capture, so a re-run is safe).
    /// </summary>
    public sealed class CudaGraphCaptureAbortedException : Exception
    {
        public CudaGraphCaptureAbortedException(string message) : base(message) { }
    }

    /// <summary>
    /// Process-wide hooks for the single active CUDA-graph stream capture.
    /// While a capture is active on an allocator's stream:
    ///  - device blocks that the pool would cuMemFree are quarantined instead
    ///    (the captured graph references them),
    ///  - every block rented or pooled is tracked so the capture owner can
    ///    remove ("steal") the free ones from the pool when the capture ends,
    ///  - host mirror buffers of storages destroyed mid-capture are donated to
    ///    the capture owner (a captured HtoD copy may reference them),
    ///  - a stream synchronize throws <see cref="CudaGraphCaptureAbortedException"/>.
    /// </summary>
    internal static class CudaGraphCapture
    {
        private static readonly object Sync = new object();
        private static CaptureContext active;

        internal sealed class CaptureContext
        {
            public CudaAllocator Allocator;
            public bool NativeBegun;
            public bool NativeEntered;
            public CudaPrefillGraphCache Owner;
            public StorageCapturePayload Payloads;
            public readonly List<CudaDecodeDynParams.GraphReference> DynReferences = new();
            public readonly HashSet<IntPtr> TrackedBlocks = new HashSet<IntPtr>();
            public readonly Dictionary<IntPtr, long> TrackedSizes = new Dictionary<IntPtr, long>();
            public readonly List<(IntPtr ptr, long bytes)> QuarantinedBlocks = new List<(IntPtr, long)>();
            public readonly List<IntPtr> DonatedHostBuffers = new List<IntPtr>();
        }

        internal static CaptureContext Begin(CudaAllocator allocator, CudaPrefillGraphCache owner, CaptureContext prepared)
        {
            lock (Sync)
            {
                if (active != null)
                    return null;
                prepared.Allocator = allocator;
                prepared.Owner = owner;
                active = prepared;
                return active;
            }
        }

        internal static void End(CaptureContext context)
        {
            lock (Sync)
            {
                if (ReferenceEquals(active, context))
                    active = null;
            }
        }

        internal static bool IsCapturing(CudaAllocator allocator)
        {
            CaptureContext ctx = active;
            return ctx != null && ReferenceEquals(ctx.Allocator, allocator);
        }

        internal sealed class StorageCapturePayload
        {
            internal readonly CudaStorage Source;
            internal CaptureContext Context;
            internal StorageCapturePayload Next;
            internal IntPtr Device;
            internal long Bytes;
            internal IntPtr Host;
            internal bool PartialAllocation;
            internal bool PoolOwned;
            internal bool Transferred;

            internal StorageCapturePayload(CudaStorage source) => Source = source;
        }

        internal static CaptureContext PrepareStorageTransfer(CudaStorage storage, StorageCapturePayload payload)
        {
            lock (Sync)
            {
                CaptureContext ctx = active;
                if (ctx == null || !ctx.NativeBegun || !ReferenceEquals(ctx.Allocator, storage.AllocatorImpl)) return null;
                if (payload.Context == null)
                {
                    payload.Context = ctx;
                    payload.Next = ctx.Payloads;
                    ctx.Payloads = payload;
                }
                if (!ReferenceEquals(payload.Context, ctx))
                    throw new InvalidOperationException("The storage payload belongs to another capture.");
                return ctx;
            }
        }

        internal static bool CanTransferStorage(CaptureContext ctx, StorageCapturePayload payload)
            => ctx != null && ReferenceEquals(active, ctx) && ctx.NativeBegun
                && ReferenceEquals(payload.Context, ctx)
                && ReferenceEquals(ctx.Allocator, payload.Source.AllocatorImpl);

        internal static void RetainDynamicOwner(CudaDecodeDynParams owner, CudaAllocator allocator)
        {
            CaptureContext ctx = active;
            if (ctx == null || !ctx.NativeBegun || !ReferenceEquals(ctx.Allocator, allocator)) return;
            foreach (CudaDecodeDynParams.GraphReference reference in ctx.DynReferences)
                if (ReferenceEquals(reference.Owner, owner)) return;
            ctx.DynReferences.EnsureCapacity(ctx.DynReferences.Count + 1);
            CudaDecodeDynParams.GraphReference acquired = owner.AcquireGraphReference(allocator);
            ctx.DynReferences.Add(acquired);
        }

        internal static void OnRent(CudaAllocator allocator, IntPtr ptr, long bytes)
        {
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || ptr == IntPtr.Zero)
                return;
            lock (Sync)
            {
                if (ctx.TrackedBlocks.Add(ptr))
                    ctx.TrackedSizes[ptr] = bytes;
            }
        }

        /// <summary>
        /// Called for every pool return while this allocator is capturing. Returns
        /// true when the block was quarantined (the caller must NOT free or pool
        /// it); false when normal pooling should proceed (the block was tracked so
        /// it can be stolen from the pool at capture end).
        /// </summary>
        internal static bool InterceptReturn(CudaAllocator allocator, IntPtr ptr, long bytes, bool wouldPool)
        {
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || ptr == IntPtr.Zero)
                return false;
            lock (Sync)
            {
                if (wouldPool)
                {
                    if (ctx.TrackedBlocks.Add(ptr))
                        ctx.TrackedSizes[ptr] = bytes;
                    return false;
                }

                // The pool would cuMemFree this block, but the captured graph
                // still references it: quarantine instead.
                ctx.TrackedBlocks.Remove(ptr);
                ctx.TrackedSizes.Remove(ptr);
                ctx.QuarantinedBlocks.Add((ptr, bytes));
                return true;
            }
        }

        internal static void OnHostBufferOrphaned(CudaAllocator allocator, IntPtr hostBuffer, out bool donated)
        {
            donated = false;
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || hostBuffer == IntPtr.Zero)
                return;
            lock (Sync)
            {
                ctx.DonatedHostBuffers.Add(hostBuffer);
                donated = true;
            }
        }

        internal static readonly bool DiagnosticLog =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_LOG"), "1", StringComparison.Ordinal);

        /// <summary>Diagnostic: a pageable host->device upload is about to run on a
        /// capturing stream (it will fail with CUDA error 900). Logs the culprit's
        /// call site so the host-dirtying code path can be fixed.</summary>
        internal static void OnCapturedHostUpload(CudaAllocator allocator, long bytes)
        {
            if (!DiagnosticLog || !IsCapturing(allocator))
                return;
            Console.WriteLine($"[cuda-graph] HtoD upload of {bytes} B inside capture at:");
            Console.WriteLine(Environment.StackTrace);
        }

        internal static void OnStreamSynchronize(IntPtr streamHandle)
        {
            CaptureContext ctx = active;
            if (ctx != null && ctx.Allocator.Stream.Handle == streamHandle)
                throw new CudaGraphCaptureAbortedException(
                    "Stream synchronization (host read of device data) is not permitted while a CUDA graph capture is active."
                    + (DiagnosticLog ? "\n" + Environment.StackTrace : string.Empty));
        }
    }

    /// <summary>
    /// Per-token dynamic parameters for CUDA-graph DECODE replay. A captured
    /// graph bakes every scalar kernel argument, but the decode step's
    /// position-dependent values (attention length, KV write position, GDN conv
    /// ring index, RoPE position) change every token. Kernels on the decode path
    /// therefore re-read those values from a small device block; the graph's
    /// first node is a captured memcpy from this object's PINNED host buffer to
    /// that device block, so a replay only needs the host ints refreshed before
    /// cuGraphLaunch (the memcpy node re-reads pinned memory on every launch).
    /// </summary>
    public sealed class CudaDecodeDynParams : IDisposable
    {
        // Slot layout must match the TS_DYN_* defines in tensorsharp_kernels.cu:
        // [0]=attend_len, [1]=kv write pos, [2]=conv ring write idx, [3]=rope pos.
        private const int SlotCount = 4;
        private const int ByteCount = SlotCount * sizeof(int);

        private readonly CudaAllocator allocator;
        private readonly CudaContext context;
        private readonly CudaStream stream;
        private readonly CudaNativeCalls nativeCalls;
        private bool retired;
        private bool retirementRequested;
        private int graphReferences;
        private IntPtr devicePtr;
        private IntPtr hostPtr;

        // Capture launchers are synchronous on one host thread, but unrelated
        // allocators may launch ordinary work concurrently. Keep the ambient
        // source thread-local and retain the owning instance so a launcher can
        // reject a dynamic pointer that belongs to another allocator/stream.
        [ThreadStatic]
        private static CudaDecodeDynParams activeInstance;

        [ThreadStatic]
        private static int captureMaxAttendLen;

        /// <summary>Largest attention length the graph being captured stays
        /// valid for (0 = unlimited). Set by launchers that bake a
        /// length-dependent launch configuration (shared memory, grid).</summary>
        public static int CaptureMaxAttendLen => captureMaxAttendLen;

        /// <summary>Returns the ambient dynamic-parameter pointer only when it
        /// belongs to the allocator issuing this launch. Otherwise the caller
        /// must use its ordinary scalar kernel arguments.</summary>
        internal static CudaDecodeDynParams GetActiveOwner(CudaAllocator launchAllocator)
        {
            CudaDecodeDynParams active = activeInstance;
            if (active == null || !ReferenceEquals(active.allocator, launchAllocator))
                return null;
            return active;
        }

        public CudaDecodeDynParams(IAllocator allocator)
            : this(allocator as CudaAllocator, (allocator as CudaAllocator)?.Context, (allocator as CudaAllocator)?.Stream)
        {
        }

        internal CudaDecodeDynParams(CudaContext context, CudaStream stream)
            : this(null, context, stream)
        {
        }

        private CudaDecodeDynParams(CudaAllocator allocator, CudaContext context, CudaStream stream)
        {
            this.allocator = allocator;
            this.context = context;
            this.stream = stream;
            if (context == null)
                return;
            ArgumentNullException.ThrowIfNull(stream);
            if (!stream.MatchesContext(context))
                throw new ArgumentException("The dynamic parameter stream belongs to another context.", nameof(stream));
            if (stream.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(stream));
            nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.NativeHandle, context.NativeCalls.Api, context.DeviceId);
            using var lease = nativeCalls.EnterEffect();
            bool allocationResultFailure = false;
            try
            {
                context.BindCurrent(nativeCalls);
                int result = nativeCalls.cuMemAlloc(out devicePtr, (UIntPtr)ByteCount);
                allocationResultFailure = result != 0;
                nativeCalls.ThrowOnError(result);
                result = nativeCalls.cuMemHostAlloc(out hostPtr, (UIntPtr)ByteCount, 0);
                allocationResultFailure = result != 0;
                nativeCalls.ThrowOnError(result);
            }
            catch (Exception original)
            {
                try { Release(lease, drain: false); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                if (allocationResultFailure && original is CudaException)
                    return;
                throw;
            }
        }

        public bool IsValid
        {
            get
            {
                if (retired || retirementRequested) return false;
                nativeCalls?.ThrowIfQuarantined();
                return !retired && devicePtr != IntPtr.Zero && hostPtr != IntPtr.Zero;
            }
        }

        internal bool MatchesConsumer(CudaAllocator launchAllocator, CudaKernels kernels)
            => ReferenceEquals(allocator, launchAllocator) && launchAllocator != null
                && MatchesConsumer(launchAllocator.Context, launchAllocator.Stream, kernels);

        internal bool MatchesConsumer(CudaContext launchContext, CudaStream launchStream, CudaKernels kernels)
            => ReferenceEquals(context, launchContext) && ReferenceEquals(stream, launchStream)
                && context != null && stream.MatchesContext(context) && kernels != null && kernels.MatchesContext(context);

        internal DeviceBorrow BorrowFor(CudaAllocator launchAllocator, CudaKernels kernels)
            => MatchesConsumer(launchAllocator, kernels)
                ? BorrowFor(launchAllocator.Context, launchAllocator.Stream, kernels) : default;

        internal DeviceBorrow BorrowFor(CudaContext launchContext, CudaStream launchStream, CudaKernels kernels)
        {
            if (!MatchesConsumer(launchContext, launchStream, kernels)) return default;
            var lease = EnterValidEffect();
            return new DeviceBorrow(this, lease, devicePtr);
        }

        internal readonly struct DeviceBorrow : IDisposable
        {
            private readonly CudaDecodeDynParams owner;
            private readonly NativeEffectLease lease;
            internal IntPtr Pointer { get; }
            internal DeviceBorrow(CudaDecodeDynParams owner, NativeEffectLease lease, IntPtr pointer)
            {
                this.owner = owner;
                this.lease = lease;
                Pointer = pointer;
            }
            public void Dispose()
            {
                lease?.Dispose();
                GC.KeepAlive(owner);
            }
        }

        private NativeEffectLease EnterValidEffect()
        {
            if (nativeCalls == null || retired || retirementRequested)
                throw new ObjectDisposedException(nameof(CudaDecodeDynParams));
            var lease = nativeCalls.EnterEffect();
            if (retired || retirementRequested || devicePtr == IntPtr.Zero || hostPtr == IntPtr.Zero || stream.Handle == IntPtr.Zero || context.IsDisposed)
            {
                lease.Dispose();
                throw new ObjectDisposedException(nameof(CudaDecodeDynParams));
            }
            return lease;
        }

        /// <summary>Writes this token's values into the pinned host block. Must
        /// run before Replay (and before EndCaptureAndLaunch when capturing).</summary>
        public unsafe void Write(int attendLen, int kvWritePos, int convWriteIdx, int ropePos)
        {
            using var lease = EnterValidEffect();
            int* p = (int*)hostPtr;
            p[0] = attendLen;
            p[1] = kvWritePos;
            p[2] = convWriteIdx;
            p[3] = ropePos;
        }

        /// <summary>Enqueues the pinned-host -> device upload on the allocator
        /// stream. Called once INSIDE the capture region so the copy becomes the
        /// graph's leading node.</summary>
        public void EnqueueUpload()
        {
            CudaGraphCapture.RetainDynamicOwner(this, allocator);
            using var lease = EnterValidEffect();
            context.BindCurrent(nativeCalls);
            nativeCalls.ThrowOnError(nativeCalls.cuMemcpyHtoDAsync(devicePtr, hostPtr, (UIntPtr)ByteCount, stream.Handle));
        }

        /// <summary>Makes this block the ambient dyn-parameter source for the
        /// decode-path kernel launchers (capture scope only; clear with
        /// <see cref="Deactivate"/> in a finally).</summary>
        public void Activate()
        {
            using var lease = EnterValidEffect();
            activeInstance = this;
            captureMaxAttendLen = 0;
        }

        public static void Deactivate()
        {
            activeInstance = null;
        }

        internal static void LimitCaptureAttendLen(int limit)
        {
            if (activeInstance != null &&
                limit > 0 &&
                (captureMaxAttendLen == 0 || limit < captureMaxAttendLen))
            {
                using var lease = activeInstance.EnterValidEffect();
                captureMaxAttendLen = limit;
            }
        }

        public void Dispose()
        {
            if (nativeCalls == null || retired) return;
            using var lease = nativeCalls.EnterEffect();
            if (retired) return;
            retirementRequested = true;
            if (graphReferences != 0) throw new CudaAllocatorBusyException(this);
            Release(lease, drain: true);
        }

        internal GraphReference AcquireGraphReference(CudaAllocator actualAllocator)
        {
            var reference = new GraphReference(this);
            using var lease = EnterValidEffect();
            if (!ReferenceEquals(allocator, actualAllocator) || actualAllocator == null
                || !ReferenceEquals(context, actualAllocator.Context) || !ReferenceEquals(stream, actualAllocator.Stream))
                throw new InvalidOperationException("The dynamic buffer belongs to another capture allocator.");
            graphReferences = checked(graphReferences + 1);
            return reference;
        }

        internal sealed class GraphReference : IDisposable
        {
            internal CudaDecodeDynParams Owner { get; private set; }
            internal GraphReference(CudaDecodeDynParams owner) => Owner = owner;
            internal void Validate()
            {
                if (Owner == null) throw new ObjectDisposedException(nameof(GraphReference));
                using var lease = Owner.EnterValidEffect();
            }
            public void Dispose()
            {
                CudaDecodeDynParams owner = Owner;
                if (owner == null) return;
                using var lease = owner.nativeCalls.EnterEffect();
                owner.graphReferences--;
                Owner = null;
            }
        }

        private void Release(NativeEffectLease lease, bool drain)
        {
            nativeCalls.ValidateSafeRelease(lease);
            try
            {
                if (devicePtr != IntPtr.Zero || hostPtr != IntPtr.Zero)
                {
                    context.BindCurrent(nativeCalls);
                    if (drain) nativeCalls.cuCtxSynchronize();
                    if (devicePtr != IntPtr.Zero)
                    {
                        nativeCalls.cuMemFree(devicePtr);
                        devicePtr = IntPtr.Zero;
                    }
                    if (hostPtr != IntPtr.Zero)
                    {
                        nativeCalls.cuMemFreeHost(hostPtr);
                        hostPtr = IntPtr.Zero;
                    }
                }
                if (ReferenceEquals(activeInstance, this)) activeInstance = null;
                nativeCalls.CompleteSafeRelease(lease);
                retired = true;
            }
            catch (Exception cleanup)
            {
                nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.StorageRelease);
                throw;
            }
        }
    }

    /// <summary>
    /// Cache of instantiated CUDA graphs for the direct-CUDA per-op prefill
    /// layer loop. A graph replays the exact kernel sequence with the exact
    /// pointers/parameters that were captured, so entries are keyed on every
    /// value that gets baked into kernel launches (sequence length, start
    /// position, KV-cache buffer identity, recurrent conv ring phase) and the
    /// graph owns the pool blocks its kernels reference. Capture happens on the
    /// SECOND run of a key (the first plain run grows every scratch buffer and
    /// lazily-created cache, so nothing allocates or synchronizes mid-capture);
    /// later runs replay in a single cuGraphLaunch, collapsing the ~900 per-op
    /// dispatches of a prefill chunk.
    /// </summary>
    public sealed class CudaPrefillGraphCache : IDisposable
    {
        public static readonly bool Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH"), "0", StringComparison.Ordinal);
        public static readonly bool DecodeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_DECODE_GRAPH"), "0", StringComparison.Ordinal);
        private static readonly bool Log =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_LOG"), "1", StringComparison.Ordinal);
        private static readonly int Capacity = ReadCapacity();
        private static int ReadCapacity()
        {
            string value = Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_MAX");
            return int.TryParse(value, out int capacity) && capacity > 0 ? capacity : 4;
        }

        internal sealed class TensorReference
        {
            internal Tensor Value;
            internal TensorReference Next;
            private Exception refusal;
            internal TensorReference(Tensor value = null) => Value = value;
            internal void Release()
            {
                if (refusal != null) throw refusal;
                if (Value == null) return;
                try { Value.Dispose(); Value = null; }
                catch (Exception error) { refusal = error; throw; }
            }
            internal Tensor Transfer()
            {
                if (refusal != null) throw refusal;
                Tensor value = Value;
                Value = null;
                return value;
            }
        }

        internal sealed class DeviceBlock
        {
            internal IntPtr Pointer;
            internal long Bytes;
        }

        internal sealed class Entry
        {
            internal IntPtr Graph;
            internal IntPtr Exec;
            internal readonly TensorReference Input = new();
            internal readonly List<TensorReference> References = new();
            internal readonly List<DeviceBlock> Blocks = new();
            internal CudaGraphCapture.CaptureContext Context;
            internal long LastUse;
            internal int MaxAttendLen;
            internal bool LaunchAttempted;
        }

        internal sealed class CaptureAttempt
        {
            internal readonly CudaPrefillGraphCache cache;
            internal readonly Entry entry = new();
            internal readonly TensorReference original = new();
            internal readonly TensorReference fallback = new();
            internal readonly TensorReference incoming = new();
            internal TensorReference result = new();
            internal CudaOperationAdmission admission;
            private readonly int thread = Environment.CurrentManagedThreadId;
            private bool settled;
            internal bool published;
            internal bool safeFallback;
            internal bool beginThrew;
            internal readonly string Key;
            internal Tensor Input => original.Value;
            internal void AcceptIncoming(Tensor value) { Validate(); cache.ValidateInput(value); incoming.Value = value; }
            internal void ReleaseIncoming() { Validate(); incoming.Release(); }
            internal CaptureAttempt(CudaPrefillGraphCache cache, string key)
            { this.cache = cache; Key = key; }

            internal void RecordLoopResult(Tensor value)
            {
                Validate();
                if (ReferenceEquals(value, original.Value)) result = original;
                else result.Value = value;
            }
            internal void Validate()
            {
                if (thread != Environment.CurrentManagedThreadId || settled || !ReferenceEquals(cache.capture, this))
                    throw new InvalidOperationException("The capture attempt is not active on its owning thread.");
                admission.ValidateAllocator(cache.allocator);
            }
            internal Tensor TakeFallbackInput()
            {
                Validate();
                if (!safeFallback || entry.LaunchAttempted)
                    throw new InvalidOperationException("The capture did not establish a safe plain fallback.");
                try
                {
                    original.Release();
                    if (result != null && !ReferenceEquals(result, original)) result.Release();
                    incoming.Release();
                    Tensor value = fallback.Transfer();
                    Finish();
                    return value;
                }
                catch { RetainFailedAttempt(); throw; }
            }
            internal Tensor TakeSuccessfulResult()
            {
                Validate();
                if (!published || !entry.LaunchAttempted)
                    throw new InvalidOperationException("The capture has no successful result.");
                try
                {
                    cache.nativeCalls.ThrowIfQuarantined();
                    fallback.Release();
                    incoming.Release();
                    if (!ReferenceEquals(result, original)) original.Release();
                    Tensor value = result.Transfer();
                    Finish();
                    return value;
                }
                catch { RetainFailedAttempt(); throw; }
            }
            private void RetainFailedAttempt()
            {
                cache.releaseBlocked = true;
                admission?.Dispose();
                admission = null;
            }
            internal void Finish()
            {
                settled = true;
                admission?.Dispose();
                admission = null;
                cache.capture = null;
            }
        }

        internal sealed class ReplayReservation
        {
            internal readonly CudaPrefillGraphCache cache;
            internal readonly Entry entry;
            internal readonly TensorReference input = new();
            internal readonly TensorReference incoming;
            internal CudaOperationAdmission admission;
            internal bool launched;
            private readonly int thread = Environment.CurrentManagedThreadId;
            internal Tensor Input => input.Value;
            internal ReplayReservation(CudaPrefillGraphCache cache, Entry entry, Tensor hidden)
            { this.cache = cache; this.entry = entry; incoming = new TensorReference(hidden); }
            internal void Validate()
            {
                if (thread != Environment.CurrentManagedThreadId || admission == null)
                    throw new InvalidOperationException("The replay reservation is not active on its owning thread.");
                admission.ValidateAllocator(cache.allocator);
                cache.nativeCalls.ThrowIfQuarantined();
            }
            internal void ReleaseIncomingHidden() { Validate(); incoming.Release(); }
            internal Tensor TransferResult()
            {
                Validate();
                if (!launched) throw new InvalidOperationException("The replay did not complete.");
                Tensor value = input.Transfer();
                Finish();
                return value;
            }
            internal Exception CleanupAfterFailure(Exception original)
            {
                try
                {
                    cache.nativeCalls.ThrowIfQuarantined();
                    incoming.Release();
                    input.Release();
                    return original;
                }
                catch (Exception cleanup)
                {
                    cache.releaseBlocked = true;
                    return ReferenceEquals(original, cleanup) ? original : new AggregateException(original, cleanup);
                }
                finally
                {
                    admission?.Dispose();
                    admission = null;
                    if (incoming.Value == null && input.Value == null) cache.replays.Remove(this);
                }
            }
            private void Finish()
            {
                admission.Dispose();
                admission = null;
                cache.replays.Remove(this);
            }
        }

        private readonly CudaAllocator allocator;
        private readonly CudaNativeCalls nativeCalls;
        private readonly Dictionary<string, Entry> entries = new(StringComparer.Ordinal);
        private readonly HashSet<string> blacklist = new(StringComparer.Ordinal);
        private readonly HashSet<string> seenOnce = new(StringComparer.Ordinal);
        private readonly List<ReplayReservation> replays = new();
        private TensorReference declinedInputs;
        private CaptureAttempt capture;
        private object retainedParent;
        private bool retirementRequested;
        private bool releaseBlocked;
        private bool disposed;
        private long useCounter;

        public long ReplayCount { get; private set; }
        public long CaptureCount { get; private set; }
        public CudaPrefillGraphCache(IAllocator allocator) : this(allocator, null) { }
        internal CudaPrefillGraphCache(IAllocator allocator, object retainedParent)
        {
            this.allocator = allocator as CudaAllocator;
            this.retainedParent = retainedParent;
            if (this.allocator != null)
                nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.Graph,
                    this.allocator.NativeCalls.Api, this.allocator.DeviceId);
        }
        public bool IsUsable => Enabled && allocator != null && !retirementRequested && !disposed && !releaseBlocked;

        private CudaOperationAdmission Enter()
        {
            if (!IsUsable) throw new ObjectDisposedException(nameof(CudaPrefillGraphCache));
            var admission = CudaOperationAdmission.Enter(this, new[] { allocator });
            try
            {
                if (!IsUsable) throw new ObjectDisposedException(nameof(CudaPrefillGraphCache));
                nativeCalls.ThrowIfQuarantined();
                return admission;
            }
            catch { admission.Dispose(); throw; }
        }

        public bool ShouldCapture(string key)
        {
            if (!IsUsable || capture != null || blacklist.Contains(key) || entries.ContainsKey(key)) return false;
            using var admission = Enter();
            return !seenOnce.Add(key);
        }
        public bool TryGetReplayInput(string key, out Tensor pinnedHidden)
            => TryGetReplayInput(key, 0, out pinnedHidden);
        public bool TryGetReplayInput(string key, int attendLen, out Tensor pinnedHidden)
        {
            pinnedHidden = null;
            if (!IsUsable || capture != null) return false;
            using var admission = Enter();
            if (!FindEntry(key, attendLen, out Entry entry)) return false;
            pinnedHidden = entry.Input.Value.CopyRef();
            return true;
        }
        private bool FindEntry(string key, int attendLen, out Entry entry)
        {
            if (!entries.TryGetValue(key, out entry)) return false;
            if (entry.MaxAttendLen > 0 && attendLen > entry.MaxAttendLen)
            {
                ReleaseEntry(entry);
                entries.Remove(key);
                entry = null;
                return false;
            }
            entry.LastUse = ++useCounter;
            return true;
        }

        internal bool TryReserveReplay(string key, int attendLen, Tensor incomingHidden, out ReplayReservation reservation)
        {
            reservation = null;
            if (!IsUsable || capture != null) return false;
            CudaOperationAdmission admission = Enter();
            ReplayReservation record = null;
            try
            {
                if (!FindEntry(key, attendLen, out Entry entry)) { admission.Dispose(); return false; }
                ValidateInput(incomingHidden);
                record = new ReplayReservation(this, entry, incomingHidden);
                replays.EnsureCapacity(replays.Count + 1);
                replays.Add(record);
                record.admission = admission;
                record.input.Value = entry.Input.Value.CopyRef();
                reservation = record;
                return true;
            }
            catch (Exception original)
            {
                if (record != null) throw record.CleanupAfterFailure(original);
                admission.Dispose();
                throw;
            }
        }

        public void Replay(string key)
        {
            using var admission = Enter();
            Launch(entries[key]);
            ReplayCount++;
        }
        internal void Replay(ReplayReservation reservation)
        {
            reservation.Validate();
            if (!ReferenceEquals(reservation.cache, this)) throw new InvalidOperationException("The replay belongs to another cache.");
            Launch(reservation.entry);
            reservation.launched = true;
            ReplayCount++;
        }
        private void Launch(Entry entry)
        {
            ValidateInput(entry.Input.Value);
            if (entry.Context != null)
                foreach (CudaDecodeDynParams.GraphReference reference in entry.Context.DynReferences) reference.Validate();
            using (var lease = nativeCalls.EnterEffect())
            {
                allocator.Context.BindCurrent(nativeCalls);
                entry.LaunchAttempted = true;
                nativeCalls.ThrowOnError(nativeCalls.cuGraphLaunch(entry.Exec, allocator.Stream.Handle));
            }
            nativeCalls.ThrowIfQuarantined();
            MarkDeviceModified(entry.Input.Value);
            foreach (TensorReference reference in entry.References) MarkDeviceModified(reference.Value);
        }
        private static void MarkDeviceModified(Tensor tensor)
        {
            if (tensor?.Storage is CudaStorage storage) storage.MarkDeviceModified();
        }
        private void ValidateInput(Tensor tensor)
        {
            if (tensor?.Storage is not CudaStorage storage || !ReferenceEquals(storage.AllocatorImpl, allocator))
                throw new InvalidOperationException("The graph input belongs to another actual allocator.");
            if (allocator.Context.IsDisposed || allocator.Stream.Handle == IntPtr.Zero
                || !allocator.Stream.MatchesContext(allocator.Context))
                throw new ObjectDisposedException(nameof(CudaPrefillGraphCache));
        }

        internal CaptureAttempt ReserveCapture(string key, Tensor ownedInput, IEnumerable<Tensor> knownKeepAlive,
            CudaDecodeDynParams dynParams = null)
        {
            if (capture != null)
            {
                var error = new InvalidOperationException("A capture is already active.");
                var declined = new TensorReference(ownedInput) { Next = declinedInputs };
                declinedInputs = declined;
                try { declined.Release(); declinedInputs = declined.Next; }
                catch (Exception cleanup) { releaseBlocked = true; throw new AggregateException(error, cleanup); }
                throw error;
            }
            var attempt = new CaptureAttempt(this, key);
            capture = attempt;
            attempt.original.Value = ownedInput;
            try
            {
                ValidateInput(ownedInput);
                attempt.admission = Enter();
                attempt.fallback.Value = ownedInput.CopyRef();
                attempt.entry.Input.Value = ownedInput.CopyRef();
                AddReferences(attempt.entry, knownKeepAlive);
                if (dynParams != null)
                {
                    attempt.entry.Context = new CudaGraphCapture.CaptureContext { Allocator = allocator, Owner = this };
                    attempt.entry.Context.DynReferences.EnsureCapacity(1);
                    attempt.entry.Context.DynReferences.Add(dynParams.AcquireGraphReference(allocator));
                }
                return attempt;
            }
            catch (Exception original)
            {
                AbortAfterFailure(attempt, original, out Exception preserved);
                throw preserved;
            }
        }
        private static void AddReferences(Entry entry, IEnumerable<Tensor> references)
        {
            if (references == null) return;
            foreach (Tensor tensor in references)
            {
                if (tensor == null) continue;
                var slot = new TensorReference();
                entry.References.Add(slot);
                slot.Value = tensor.CopyRef();
            }
        }

        internal bool RetainsInput(Tensor input)
        {
            if (ReferenceEquals(capture?.original.Value, input)) return true;
            for (TensorReference declined = declinedInputs; declined != null; declined = declined.Next)
                if (ReferenceEquals(declined.Value, input)) return true;
            return false;
        }

        public bool BeginCapture(string key)
        {
            if (!IsUsable || capture != null) return false;
            var attempt = new CaptureAttempt(this, key) { admission = Enter() };
            capture = attempt;
            try
            {
                if (BeginCapture(attempt)) return true;
                ReleaseEntry(attempt.entry);
                attempt.Finish();
                return false;
            }
            catch (Exception original)
            {
                AbortAfterFailure(attempt, original, out Exception preserved);
                throw preserved;
            }
        }
        internal bool BeginCapture(CaptureAttempt attempt)
        {
            attempt.Validate();
            attempt.entry.Context ??= new CudaGraphCapture.CaptureContext { Allocator = allocator, Owner = this };
            CudaGraphCapture.CaptureContext context = CudaGraphCapture.Begin(allocator, this, attempt.entry.Context);
            if (context == null)
            {
                ReleaseEntry(attempt.entry);
                attempt.safeFallback = true;
                return false;
            }
            int result;
            try
            {
                using var lease = nativeCalls.EnterEffect();
                allocator.Context.BindCurrent(nativeCalls);
                result = nativeCalls.cuStreamBeginCapture(allocator.Stream.Handle, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED);
                if (result == 0) { context.NativeBegun = true; context.NativeEntered = true; }
            }
            catch
            {
                attempt.beginThrew = true;
                CudaGraphCapture.End(context);
                throw;
            }
            if (result == 0) return true;
            CudaGraphCapture.End(context);
            blacklist.Add(attempt.Key);
            ReleaseEntry(attempt.entry);
            attempt.safeFallback = true;
            return false;
        }

        public bool EndCaptureAndLaunch(string key, Tensor pinnedHidden, IEnumerable<Tensor> keepAlive)
            => EndCaptureAndLaunch(key, pinnedHidden, keepAlive, 0);
        public bool EndCaptureAndLaunch(string key, Tensor pinnedHidden, IEnumerable<Tensor> keepAlive, int maxAttendLen)
        {
            CaptureAttempt attempt = capture;
            if (attempt == null || !string.Equals(attempt.Key, key, StringComparison.Ordinal)) return false;
            try
            {
                attempt.entry.Input.Value = pinnedHidden.CopyRef();
                AddReferences(attempt.entry, keepAlive);
                bool success = EndCaptureAndLaunch(attempt, null, maxAttendLen);
                attempt.Finish();
                return success;
            }
            catch (Exception original)
            {
                AbortAfterFailure(attempt, original, out Exception preserved);
                throw preserved;
            }
        }
        internal bool EndCaptureAndLaunch(CaptureAttempt attempt, Tensor actualLoopResult, int maxAttendLen)
        {
            attempt.Validate();
            if (actualLoopResult != null) attempt.RecordLoopResult(actualLoopResult);
            bool validGraph = EndNative(attempt.entry);
            if (!validGraph || (attempt.original.Value != null && !ReferenceEquals(actualLoopResult, attempt.original.Value)))
            {
                SettleNoLaunch(attempt);
                return false;
            }
            PrepareBlocks(attempt.entry);
            bool instantiated;
            using (var lease = nativeCalls.EnterEffect())
            {
                allocator.Context.BindCurrent(nativeCalls);
                int result = nativeCalls.cuGraphInstantiateWithFlags(out attempt.entry.Exec, attempt.entry.Graph, 0);
                instantiated = result == 0 && attempt.entry.Exec != IntPtr.Zero;
                nativeCalls.cuGraphDestroy(attempt.entry.Graph);
                attempt.entry.Graph = IntPtr.Zero;
            }
            if (!instantiated) { SettleNoLaunch(attempt); return false; }
            if (entries.Count >= Capacity) EvictLru();
            attempt.entry.MaxAttendLen = maxAttendLen;
            attempt.entry.LastUse = ++useCounter;
            entries[attempt.Key] = attempt.entry;
            attempt.published = true;
            Launch(attempt.entry);
            CaptureCount++;
            if (Log) Console.WriteLine($"[cuda-graph] captured {attempt.Key}");
            return true;
        }

        private bool EndNative(Entry entry)
        {
            CudaGraphCapture.CaptureContext context = entry.Context;
            if (context == null || !context.NativeBegun) return entry.Graph != IntPtr.Zero;
            using (var lease = nativeCalls.EnterEffect())
            {
                allocator.Context.BindCurrent(nativeCalls);
                int result;
                try { result = nativeCalls.cuStreamEndCapture(allocator.Stream.Handle, out entry.Graph); }
                catch (Exception failure)
                {
                    nativeCalls.PublishFailure(lease, failure, NativeRuntimeFailureStage.GraphRelease);
                    throw;
                }
                if (result != 0 && !(result == 901 && entry.Graph == IntPtr.Zero))
                {
                    var failure = new CudaException(result, "CUDA graph capture termination is not established.");
                    nativeCalls.PublishFailure(lease, failure, NativeRuntimeFailureStage.GraphRelease);
                    throw failure;
                }
                context.NativeBegun = false;
            }
            CudaGraphCapture.End(context);
            using (var completion = nativeCalls.EnterEffect())
            {
                allocator.Context.BindCurrent(nativeCalls);
                nativeCalls.cuCtxSynchronize();
            }
            return entry.Graph != IntPtr.Zero;
        }
        private void PrepareBlocks(Entry entry)
        {
            CudaGraphCapture.CaptureContext context = entry.Context;
            if (context == null) return;
            foreach (IntPtr pointer in context.TrackedBlocks)
            {
                var block = new DeviceBlock { Bytes = context.TrackedSizes[pointer] };
                entry.Blocks.Add(block);
                if (allocator.TryStealPooledBlock(pointer, block.Bytes)) block.Pointer = pointer;
            }
            for (CudaGraphCapture.StorageCapturePayload payload = context.Payloads; payload != null; payload = payload.Next)
                if (payload.PoolOwned)
                {
                    bool alreadyStolen = false;
                    foreach (DeviceBlock block in entry.Blocks)
                        if (block.Pointer == payload.Device) { alreadyStolen = true; break; }
                    if (!alreadyStolen)
                    {
                        var block = new DeviceBlock { Bytes = payload.Bytes };
                        entry.Blocks.Add(block);
                        if (allocator.TryStealPooledBlock(payload.Device, payload.Bytes)) block.Pointer = payload.Device;
                    }
                }
        }

        public void AbortCapture(string key) => AbortCapture(key, null);
        public void AbortCapture(string key, string reason)
        {
            if (capture == null) return;
            if (!string.Equals(capture.Key, key, StringComparison.Ordinal)) throw new InvalidOperationException("The capture key does not match.");
            if (Log && reason != null) Console.WriteLine($"[cuda-graph] abort reason: {reason}");
            CaptureAttempt attempt = capture;
            SettleNoLaunch(attempt);
            attempt.Finish();
        }
        internal bool AbortAfterFailure(CaptureAttempt attempt, Exception original, out Exception preservedFailure)
        {
            preservedFailure = original;
            try
            {
                if (!ReferenceEquals(capture, attempt)) throw new InvalidOperationException("The capture owner does not match.");
                if (attempt.admission != null) attempt.Validate();
                nativeCalls.ThrowIfQuarantined();
                if (attempt.entry.LaunchAttempted)
                {
                    attempt.fallback.Release();
                    attempt.original.Release();
                    attempt.incoming.Release();
                    if (attempt.result != null && !ReferenceEquals(attempt.result, attempt.original)) attempt.result.Release();
                    attempt.Finish();
                    return false;
                }
                SettleNoLaunch(attempt);
                if (attempt.beginThrew || attempt.entry.Context?.NativeEntered != true
                    || original is not (CudaGraphCaptureAbortedException or CudaException))
                {
                    attempt.original.Release(); attempt.fallback.Release(); attempt.incoming.Release();
                    if (attempt.result != null && !ReferenceEquals(attempt.result, attempt.original)) attempt.result.Release();
                    attempt.Finish();
                    return false;
                }
                return true;
            }
            catch (Exception cleanup)
            {
                releaseBlocked = true;
                if (!ReferenceEquals(original, cleanup)) preservedFailure = new AggregateException(original, cleanup);
                attempt.admission?.Dispose();
                attempt.admission = null;
                return false;
            }
        }
        private void SettleNoLaunch(CaptureAttempt attempt)
        {
            if (attempt.entry.LaunchAttempted) throw new InvalidOperationException("A graph launch was already attempted.");
            EndNative(attempt.entry);
            ReleaseEntry(attempt.entry);
            blacklist.Add(attempt.Key);
            attempt.safeFallback = true;
        }

        private void EvictLru()
        {
            string key = null;
            long oldest = long.MaxValue;
            foreach (KeyValuePair<string, Entry> pair in entries)
                if (pair.Value.LastUse < oldest) { key = pair.Key; oldest = pair.Value.LastUse; }
            if (key == null) return;
            ReleaseEntry(entries[key]);
            entries.Remove(key);
        }
        private void ReleaseNative(Entry entry)
        {
            if (entry.Graph == IntPtr.Zero && entry.Exec == IntPtr.Zero && entry.Context?.NativeEntered != true) return;
            if (entry.Context?.NativeBegun == true) EndNative(entry);
            using var lease = nativeCalls.EnterEffect();
            allocator.Context.BindCurrent(nativeCalls);
            allocator.Context.NativeCalls.ThrowIfQuarantined();
            nativeCalls.cuCtxSynchronize();
            if (entry.Exec != IntPtr.Zero) { nativeCalls.cuGraphExecDestroy(entry.Exec); entry.Exec = IntPtr.Zero; }
            if (entry.Graph != IntPtr.Zero) { nativeCalls.cuGraphDestroy(entry.Graph); entry.Graph = IntPtr.Zero; }
        }
        private void ReleaseEntry(Entry entry)
        {
            try { ReleaseEntryResources(entry); }
            catch { releaseBlocked = true; throw; }
        }
        private void ReleaseEntryResources(Entry entry)
        {
            ReleaseNative(entry);
            foreach (TensorReference reference in entry.References) reference.Release();
            entry.Input.Release();
            foreach (DeviceBlock block in entry.Blocks)
                if (block.Pointer != IntPtr.Zero)
                {
                    allocator.ReturnDeviceMemory(block.Pointer, block.Bytes);
                    block.Pointer = IntPtr.Zero;
                }
            CudaGraphCapture.CaptureContext context = entry.Context;
            if (context == null) return;
            for (CudaGraphCapture.StorageCapturePayload payload = context.Payloads; payload != null; payload = payload.Next)
            {
                if (payload.Device != IntPtr.Zero && !payload.PoolOwned)
                {
                    if (payload.PartialAllocation)
                    {
                        using var lease = nativeCalls.EnterEffect();
                        allocator.Context.BindCurrent(nativeCalls);
                        nativeCalls.cuMemFree(payload.Device);
                    }
                    else allocator.ReturnDeviceMemory(payload.Device, payload.Bytes);
                    payload.Device = IntPtr.Zero;
                }
                if (payload.Host != IntPtr.Zero)
                {
                    CudaStorage.FreeDonatedHostBuffer(payload.Host);
                    payload.Host = IntPtr.Zero;
                }
            }
            for (int i = 0; i < context.QuarantinedBlocks.Count; i++)
            {
                (IntPtr pointer, long bytes) = context.QuarantinedBlocks[i];
                if (pointer == IntPtr.Zero) continue;
                allocator.ReturnDeviceMemory(pointer, bytes);
                context.QuarantinedBlocks[i] = (IntPtr.Zero, bytes);
            }
            context.QuarantinedBlocks.Clear();
            for (int i = 0; i < context.DonatedHostBuffers.Count; i++)
            {
                CudaStorage.FreeDonatedHostBuffer(context.DonatedHostBuffers[i]);
                context.DonatedHostBuffers[i] = IntPtr.Zero;
            }
            context.DonatedHostBuffers.Clear();
            foreach (CudaDecodeDynParams.GraphReference reference in context.DynReferences) reference.Dispose();
            context.DynReferences.Clear();
        }

        internal void CollectDisposalOwnership(ICollection<Tensor> tensors)
        {
            var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
            void Add(TensorReference reference)
            {
                if (reference?.Value != null && seen.Add(reference.Value)) tensors.Add(reference.Value);
            }
            void AddEntry(Entry entry)
            {
                Add(entry.Input);
                foreach (TensorReference reference in entry.References) Add(reference);
            }
            foreach (Entry entry in entries.Values) AddEntry(entry);
            if (capture != null)
            {
                AddEntry(capture.entry);
                Add(capture.original); Add(capture.fallback); Add(capture.result);
                Add(capture.incoming);
            }
            foreach (ReplayReservation replay in replays) { Add(replay.input); Add(replay.incoming); }
            for (TensorReference declined = declinedInputs; declined != null; declined = declined.Next) Add(declined);
        }
        internal void ReleaseGraphsForRetirement()
        {
            nativeCalls?.ThrowIfQuarantined();
            if (releaseBlocked) throw new InvalidOperationException("Graph reference cleanup remains incomplete.");
            if (capture != null) ReleaseNative(capture.entry);
            foreach (Entry entry in entries.Values) ReleaseNative(entry);
        }

        internal void DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (disposed) return;
            if (!plan.Completes(allocator)) throw new InvalidOperationException("The parent plan does not cover graph completion.");
            restoration.Validate(nativeCalls.Api);
            retirementRequested = true;
            ReleaseAll();
        }
        private void ReleaseAll()
        {
            nativeCalls?.ThrowIfQuarantined();
            if (releaseBlocked) throw new InvalidOperationException("Graph reference cleanup remains incomplete.");
            if (capture != null)
            {
                ReleaseEntry(capture.entry);
                capture.original.Release(); capture.fallback.Release(); capture.result?.Release();
                capture.incoming.Release();
                capture = null;
            }
            foreach (ReplayReservation replay in replays) { replay.incoming.Release(); replay.input.Release(); }
            replays.Clear();
            while (declinedInputs != null) { declinedInputs.Release(); declinedInputs = declinedInputs.Next; }
            var keys = new List<string>(entries.Keys);
            foreach (string key in keys) { ReleaseEntry(entries[key]); entries.Remove(key); }
            if (nativeCalls != null)
            {
                using var lease = nativeCalls.EnterEffect();
                nativeCalls.ValidateSafeRelease(lease);
                nativeCalls.CompleteSafeRelease(lease);
            }
            retainedParent = null;
            disposed = true;
        }
        public void Dispose()
        {
            if (disposed) return;
            if (allocator == null) { retainedParent = null; disposed = true; return; }
            using var admission = CudaOperationAdmission.EnterCacheRetirement(allocator);
            retirementRequested = true;
            CudaContextRestoration restoration = CudaContextRestoration.Capture(nativeCalls);
            try { ReleaseAll(); }
            catch { restoration.MarkCleanupFailed(); throw; }
            restoration.Restore();
        }
    }
}
