using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Owns this node's CUDA allocators and coordinates reduce-to-zero collectives.
    /// Callers keep borrowed tensors undisposed until each collective completes.
    /// </summary>
    public sealed class TensorParallelGroup : ITensorParallelGroup, ICudaTensorParallelRetirement
    {
        private readonly CudaAllocator[] _allocators;
        private readonly CudaP2PCommunicator _communicator;
        private readonly CudaNativeCalls _nativeCalls;
        private readonly object _methodGate = new();
        private int _activeMethods;
        private bool _disposed;
        private bool _retirementRequested;
        private Exception _cleanupFailure;
        private CudaContextRestoration _constructionRestoration;

        private static readonly bool _forceHostAllReduce =
            string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_TP_HOST_ALLREDUCE"), "1", StringComparison.Ordinal);

        internal TensorParallelGroup(CudaAllocator[] allocators)
        {
            ArgumentNullException.ThrowIfNull(allocators);
            var transferred = (CudaAllocator[])allocators.Clone();
            if (transferred.Length == 0 || transferred.Any(a => a == null)
                || transferred.Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance).Count() != transferred.Length)
                throw new ArgumentException("A group requires non-null, distinct allocators.", nameof(allocators));
            if (transferred.Any(a => !ReferenceEquals(a.NativeCalls.Api, transferred[0].NativeCalls.Api)))
                throw new ArgumentException("A group requires one shared native API.", nameof(allocators));
            _allocators = transferred;
            Degree = transferred.Length;
            try
            {
                _nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.Worker,
                    transferred[0].NativeCalls.Api, transferred.Select(a => a.DeviceId).Distinct().ToArray());
            }
            catch (Exception original)
            {
                RollBackConstruction(original);
                throw;
            }
        }

        public TensorParallelGroup(int degree)
        {
            if (degree < 1)
                throw new ArgumentOutOfRangeException(nameof(degree), "TP degree must be >= 1.");
            try
            {
                int deviceCount = CudaDevice.GetDeviceCount();
                if (degree > deviceCount)
                    throw new InvalidOperationException(
                        $"Requested TP degree {degree} but only {deviceCount} CUDA device(s) available.");
                Degree = degree;
                _allocators = new CudaAllocator[degree];
                _nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.Worker,
                    CudaNativeApi.Instance, Enumerable.Range(0, degree).ToArray());
                _constructionRestoration = CudaContextRestoration.Capture(_nativeCalls);
                for (int i = 0; i < degree; i++)
                    _allocators[i] = new CudaAllocator(i);
                if (degree > 1)
                {
                    _communicator = new CudaP2PCommunicator(this, _allocators);
                    _communicator.InitializePeerAccess();
                }
                Console.WriteLine($"Tensor parallelism: {degree} GPUs " +
                    $"({string.Join(", ", Enumerable.Range(0, degree).Select(i => CudaDevice.GetDevice(i).Name))})");
                _constructionRestoration = null;
            }
            catch (Exception original)
            {
                RollBackConstruction(original);
                throw;
            }
        }

        public int Degree { get; }
        public bool IsActive => Degree > 1;
        public int GlobalDegree => Degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => GetCudaAllocator(rank);

        public CudaAllocator GetCudaAllocator(int rank)
        {
            if ((uint)rank >= (uint)Degree) throw new ArgumentOutOfRangeException(nameof(rank));
            return _allocators[rank];
        }

        private CudaCollectiveOperation EnterMethod()
        {
            ThrowIfRetiring();
            var operation = CudaCollectiveOperation.Enter(this, _allocators);
            try
            {
                lock (_methodGate)
                {
                    ThrowIfRetiring();
                    _activeMethods++;
                    operation.RecordMethod(this);
                }
                return operation;
            }
            catch (Exception original)
            {
                var failure = operation.SettleFailure(original);
                operation.Dispose();
                ExceptionDispatchInfo.Capture(failure).Throw();
                throw;
            }
        }

        internal void EndMethod()
        {
            lock (_methodGate) _activeMethods--;
        }

        internal CudaCollectiveOperation EnterCollective(Tensor[] tensors)
        {
            var operation = EnterMethod();
            try
            {
                operation.FreezeRanks(tensors);
                ThrowIfRetiring();
                operation.Check();
                return operation;
            }
            catch (Exception original)
            {
                var failure = operation.SettleFailure(original);
                operation.Dispose();
                ExceptionDispatchInfo.Capture(failure).Throw();
                throw;
            }
        }

        public void RunPerRank(Action<int> body)
        {
            using var operation = EnterMethod();
            try
            {
                for (int rank = 0; rank < Degree; rank++)
                {
                    ThrowIfRetiring();
                    operation.Check();
                    body(rank);
                    ThrowIfRetiring();
                    operation.Check();
                }
                operation.Complete();
            }
            catch (Exception original)
            {
                ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        /// <summary>In-place sum of equally shaped Float32 rank tensors.</summary>
        public void AllReduce(Tensor[] tensors)
        {
            ThrowIfRetiring();
            if (!IsActive) return;
            using var operation = EnterCollective(tensors);
            try
            {
                AllReduceAdmitted(operation);
                operation.Complete();
            }
            catch (Exception original)
            {
                ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        internal void AllReduceAdmitted(CudaCollectiveOperation operation)
        {
            ThrowIfRetiring();
            operation.ValidateAllocators(_allocators);
            if (!IsActive) return;
            if (_forceHostAllReduce) HostAllReduce(operation);
            else _communicator.AllReduce(operation);
            ThrowIfRetiring();
            operation.Check();
        }

        private void HostAllReduce(CudaCollectiveOperation operation)
        {
            for (int rank = 0; rank < Degree; rank++)
            {
                ThrowIfRetiring();
                operation.Check();
                _allocators[rank].Synchronize();
            }
            int n = operation.ElementCount;
            float[] acc = operation.Storages[0].GetElementsAsFloat(operation.StorageOffsets[0], n);
            operation.RetainBuffer(acc);
            for (int rank = 1; rank < Degree; rank++)
            {
                ThrowIfRetiring();
                operation.Check();
                float[] part = operation.Storages[rank].GetElementsAsFloat(operation.StorageOffsets[rank], n);
                for (int i = 0; i < n; i++) acc[i] += part[i];
            }
            for (int rank = 0; rank < Degree; rank++)
            {
                ThrowIfRetiring();
                operation.Check();
                operation.Storages[rank].SetElementsAsFloat(operation.StorageOffsets[rank], acc);
                operation.MarkPending(rank);
                operation.Storages[rank].EnsureDeviceCurrent();
            }
            operation.DrainPending();
        }

        public void Synchronize()
        {
            using var operation = EnterMethod();
            try
            {
                for (int rank = 0; rank < Degree; rank++)
                {
                    ThrowIfRetiring();
                    operation.Check();
                    operation.MarkPending(rank);
                    _allocators[rank].Synchronize();
                    operation.RecordDrained(rank);
                }
                operation.Complete();
            }
            catch (Exception original)
            {
                ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        public void Barrier() { ThrowIfRetiring(); }

        public void BroadcastControl(int op, int[] payload)
        {
            ThrowIfRetiring();
            throw new NotSupportedException("Control broadcast is only meaningful for multi-node distributed groups.");
        }

        public (int op, int[] payload) ReceiveControl()
        {
            ThrowIfRetiring();
            throw new NotSupportedException("Control receive is only meaningful for multi-node distributed groups.");
        }

        public void Dispose()
        {
            if (_disposed) return;
            lock (_methodGate) _retirementRequested = true;
            if (_cleanupFailure != null && !CudaP2PEffectOwner.IsHealthy(_nativeCalls))
                ExceptionDispatchInfo.Capture(_cleanupFailure).Throw();
            var plan = CudaRetirementPlan.PrepareGroup(this, this);
            var restoration = plan.CaptureRestoration();
            plan.AllowStorageRelease();
            ((ICudaTensorParallelRetirement)this).DisposeOwned(plan, restoration);
            plan.Complete();
            restoration?.Restore();
        }

        void ICudaTensorParallelRetirement.CollectOwnedAllocators(ICollection<IAllocator> allocators)
        {
            if (_allocators != null)
                foreach (CudaAllocator allocator in _allocators)
                    if (allocator != null) allocators.Add(allocator);
        }

        bool ICudaTensorParallelRetirement.OwnsCudaAllocators => true;

        void ICudaTensorParallelRetirement.DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (_disposed) return;
            lock (_methodGate) _retirementRequested = true;
            try
            {
                plan.ValidateGroupRelease(this);
                if (_nativeCalls != null)
                {
                    using var validation = _nativeCalls.EnterEffect();
                    _nativeCalls.ValidateSafeRelease(validation);
                }
                plan.Drain(restoration);
                _communicator?.DisposeOwned(plan, restoration);
                foreach (CudaAllocator allocator in _allocators)
                    if (allocator != null) allocator.DisposeOwned(plan, restoration);
                if (_nativeCalls != null)
                {
                    using var completion = _nativeCalls.EnterEffect();
                    _nativeCalls.CompleteSafeRelease(completion);
                }
                _disposed = true;
            }
            catch (Exception failure)
            {
                _cleanupFailure = failure;
                restoration?.MarkCleanupFailed();
                throw;
            }
        }

        private void RollBackConstruction(Exception original)
        {
            _retirementRequested = true;
            _cleanupFailure = original;
            if (_nativeCalls != null && !CudaP2PEffectOwner.IsHealthy(_nativeCalls)) return;
            if (_allocators == null || !_allocators.Any(a => a != null)) return;
            try
            {
                var plan = CudaRetirementPlan.PrepareGroup(this, this);
                var restoration = _constructionRestoration ?? plan.CaptureRestoration();
                plan.AllowStorageRelease();
                ((ICudaTensorParallelRetirement)this).DisposeOwned(plan, restoration);
                plan.Complete();
                restoration?.Restore();
            }
            catch (Exception cleanup)
            {
                _constructionRestoration?.MarkCleanupFailed();
                _cleanupFailure = new AggregateException(original, cleanup);
                ExceptionDispatchInfo.Capture(_cleanupFailure).Throw();
            }
        }

        internal void ThrowIfRetiring()
        {
            if (Volatile.Read(ref _retirementRequested))
                throw new ObjectDisposedException(nameof(TensorParallelGroup));
            _nativeCalls?.ThrowIfQuarantined();
        }
    }

    internal sealed class CudaP2PEffectOwner
    {
        internal readonly object Parent;
        internal readonly CudaAllocator[] Allocators;
        internal readonly CudaNativeCalls Calls;
        private bool _completed;

        internal CudaP2PEffectOwner(object parent, CudaAllocator[] allocators)
        {
            Parent = parent;
            Allocators = (CudaAllocator[])allocators.Clone();
            Calls = new CudaNativeCalls(this, NativeOwnerRole.Worker, Allocators[0].NativeCalls.Api,
                Allocators.Select(a => a.DeviceId).Distinct().ToArray());
        }

        internal static bool IsHealthy(CudaNativeCalls calls)
        {
            try { calls.ThrowIfQuarantined(); return true; }
            catch (NativeRuntimeQuarantinedException) { return false; }
        }

        internal void Complete()
        {
            if (_completed) return;
            using var lease = Calls.EnterEffect();
            Calls.ValidateSafeRelease(lease);
            Calls.CompleteSafeRelease(lease);
            _completed = true;
        }
    }

    internal sealed class CudaCollectiveOperation : IDisposable
    {
        private readonly object _parent;
        private readonly CudaAllocator[] _allocators;
        private readonly CudaStream[] _streams;
        private readonly CudaNativeCalls _calls;
        private CudaOperationAdmission _admission;
        private Tensor[] _tensors;
        private CudaStorage[] _storages;
        private long[] _storageOffsets;
        private long[][] _shapes;
        private int _elementCount;
        private long _byteCount;
        private float[] _hostBuffer;
        private float[] _resultBuffer;
        private readonly List<Array> _buffers = new();
        private readonly List<GCHandle> _pins = new();
        private readonly bool[] _pendingStreams;
        private readonly bool[] _pendingDefaultContexts;
        private readonly CudaP2PEffectOwner[] _rankEffects;
        private readonly Dictionary<(int Source, int Destination), CudaP2PEffectOwner> _peerEffects = new();
        private TensorParallelGroup _methodOwner;
        private bool _methodCounted;
        private bool _settled;
        private bool _released;
        private Exception _failure;

        private CudaCollectiveOperation(object parent, CudaAllocator[] allocators)
        {
            _parent = parent;
            _allocators = (CudaAllocator[])allocators.Clone();
            _calls = new CudaNativeCalls(this, NativeOwnerRole.Worker, _allocators[0].NativeCalls.Api,
                _allocators.Select(a => a.DeviceId).Distinct().ToArray());
            _streams = _allocators.Select(a => a.Stream).ToArray();
            _pendingStreams = new bool[_allocators.Length];
            _pendingDefaultContexts = new bool[_allocators.Length];
            _rankEffects = new CudaP2PEffectOwner[_allocators.Length];
        }

        internal static CudaCollectiveOperation Enter(object parent, CudaAllocator[] allocators)
        {
            var operation = new CudaCollectiveOperation(parent, allocators);
            operation._admission = CudaOperationAdmission.Enter(operation, operation._allocators);
            return operation;
        }

        internal ReadOnlySpan<CudaAllocator> Allocators => _allocators;
        internal ReadOnlySpan<CudaStorage> Storages => _storages;
        internal ReadOnlySpan<long> StorageOffsets => _storageOffsets;
        internal int ElementCount => _elementCount;
        internal long ByteCount => _byteCount;

        internal void RecordMethod(TensorParallelGroup owner)
        {
            _methodOwner = owner;
            _methodCounted = true;
        }

        internal void FreezeRanks(Tensor[] tensors)
        {
            Check();
            if (tensors == null || tensors.Length != _allocators.Length)
                throw new ArgumentException($"Expected {_allocators.Length} tensors, got {tensors?.Length ?? 0}.");
            _tensors = (Tensor[])tensors.Clone();
            _storages = new CudaStorage[_tensors.Length];
            _storageOffsets = new long[_tensors.Length];
            _shapes = new long[_tensors.Length][];
            for (int rank = 0; rank < _tensors.Length; rank++)
            {
                Tensor tensor = _tensors[rank] ?? throw new ArgumentException("Rank tensors must be non-null.", nameof(tensors));
                CudaStorage storage = tensor.Storage as CudaStorage
                    ?? throw new ArgumentException("Rank tensors require actual CUDA storage.", nameof(tensors));
                _storages[rank] = storage;
                if (!ReferenceEquals(storage.AllocatorImpl, _allocators[rank]))
                    throw new ArgumentException("Rank storage must use the exact rank allocator.", nameof(tensors));
                _admission.ValidateAllocator(storage.AllocatorImpl);
                _storageOffsets[rank] = tensor.StorageOffset;
                _shapes[rank] = tensor.Sizes.ToArray();
                if (storage.ElementType != DType.Float32)
                    throw new ArgumentException("AllReduce requires Float32 rank tensors.", nameof(tensors));
                int count = checked((int)storage.ElementCount);
                long bytes = storage.ByteLength;
                if (bytes != checked((long)count * sizeof(float)))
                    throw new ArgumentException("Rank storage extent must match Float32 elements.", nameof(tensors));
                if (rank == 0)
                {
                    _elementCount = count;
                    _byteCount = bytes;
                }
                else if (count != _elementCount || bytes != _byteCount
                    || !_shapes[0].AsSpan().SequenceEqual(_shapes[rank]))
                    throw new ArgumentException("Rank tensors require the same shape and storage extent.", nameof(tensors));
            }
            Check();
        }

        internal void ValidateAllocators(CudaAllocator[] allocators)
        {
            Check();
            if (allocators.Length != _allocators.Length)
                throw new InvalidOperationException("The collective requires its actual admitted rank set.");
            for (int rank = 0; rank < allocators.Length; rank++)
            {
                if (!ReferenceEquals(allocators[rank], _allocators[rank]))
                    throw new InvalidOperationException("The collective requires its actual admitted rank order.");
                _admission.ValidateAllocator(allocators[rank]);
            }
        }

        internal void Check()
        {
            if (_released) throw new ObjectDisposedException(nameof(CudaCollectiveOperation));
            _calls.ThrowIfQuarantined();
            foreach (CudaAllocator allocator in _allocators) _admission.ValidateAllocator(allocator);
        }

        internal void RetainBuffers(float[] host, float[] result)
        {
            _hostBuffer = host;
            _resultBuffer = result;
        }

        internal void RetainBuffer(Array buffer) => _buffers.Add(buffer);

        internal IntPtr PinBuffer(Array buffer)
        {
            RetainBuffer(buffer);
            var pin = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try { _pins.Add(pin); }
            catch { pin.Free(); throw; }
            return pin.AddrOfPinnedObject();
        }

        internal CudaP2PEffectOwner RankEffect(int rank)
            => _rankEffects[rank] ??= new CudaP2PEffectOwner(this, new[] { _allocators[rank] });

        internal CudaP2PEffectOwner PeerEffect(int source, int destination)
        {
            var key = (source, destination);
            if (!_peerEffects.TryGetValue(key, out var effect))
            {
                effect = new CudaP2PEffectOwner(this, new[] { _allocators[source], _allocators[destination] });
                _peerEffects.Add(key, effect);
            }
            return effect;
        }

        internal void MarkPending(int rank) => _pendingStreams[rank] = true;
        internal void MarkDefaultPending(int rank) => _pendingDefaultContexts[rank] = true;
        internal void RecordDrained(int rank)
        {
            _pendingStreams[rank] = false;
            _pendingDefaultContexts[rank] = false;
        }

        internal void DrainPending()
        {
            List<Exception> failures = null;
            for (int rank = 0; rank < _allocators.Length; rank++)
            {
                if (!_pendingStreams[rank] && !_pendingDefaultContexts[rank]) continue;
                if (!CudaP2PEffectOwner.IsHealthy(_allocators[rank].NativeCalls)) continue;
                try
                {
                    if (_pendingDefaultContexts[rank])
                    {
                        var effect = RankEffect(rank);
                        using (var lease = effect.Calls.EnterEffect())
                        {
                            _allocators[rank].Context.BindCurrent(effect.Calls);
                            effect.Calls.cuCtxSynchronize();
                        }
                        RecordDrained(rank);
                    }
                    else
                    {
                        _streams[rank].Synchronize();
                        _pendingStreams[rank] = false;
                    }
                }
                catch (Exception cleanup)
                {
                    (failures ??= new()).Add(cleanup);
                }
            }
            if (failures != null)
                throw failures.Count == 1 ? failures[0] : new AggregateException(failures);
        }

        private bool HasPending => _pendingStreams.Any(p => p) || _pendingDefaultContexts.Any(p => p);

        private void ValidateRelease()
        {
            using var lease = _calls.EnterEffect();
            _calls.ValidateSafeRelease(lease);
        }

        internal void Complete()
        {
            if (_settled) return;
            ValidateRelease();
            DrainPending();
            Check();
            FinishHealthy();
        }

        private void FinishHealthy()
        {
            if (HasPending) return;
            foreach (var phase in _rankEffects) phase?.Complete();
            foreach (var phase in _peerEffects.Values) phase.Complete();
            using (var lease = _calls.EnterEffect())
            {
                _calls.ValidateSafeRelease(lease);
                _calls.CompleteSafeRelease(lease);
            }
            for (int index = 0; index < _pins.Count; index++)
            {
                GCHandle pin = _pins[index];
                if (pin.IsAllocated) pin.Free();
                _pins[index] = default;
            }
            _pins.Clear();
            _buffers.Clear();
            GC.KeepAlive(_hostBuffer);
            GC.KeepAlive(_resultBuffer);
            _hostBuffer = null;
            _resultBuffer = null;
            _tensors = null;
            _storages = null;
            _storageOffsets = null;
            _shapes = null;
            _settled = true;
        }

        internal Exception SettleFailure(Exception original)
        {
            if (_failure != null) return _failure;
            _failure = original;
            if (_settled) return original;
            List<Exception> failures = null;
            try
            {
                if (CudaP2PEffectOwner.IsHealthy(_calls)) ValidateRelease();
                DrainPending();
            }
            catch (Exception cleanup)
            {
                (failures ??= new() { original }).Add(cleanup);
            }
            if (!HasPending && CudaP2PEffectOwner.IsHealthy(_calls))
            {
                try { FinishHealthy(); }
                catch (Exception cleanup) { (failures ??= new() { original }).Add(cleanup); }
            }
            if (failures != null) _failure = new AggregateException(failures);
            return _failure;
        }

        public void Dispose()
        {
            if (_released) return;
            _released = true;
            _admission?.Dispose();
            _admission = null;
            if (_methodCounted)
            {
                _methodCounted = false;
                _methodOwner.EndMethod();
            }
            GC.KeepAlive(_parent);
        }
    }
}
