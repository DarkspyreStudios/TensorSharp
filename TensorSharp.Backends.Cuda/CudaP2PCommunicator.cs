using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda
{
    /// <summary>Owned scratch for the existing reduce-to-zero and broadcast collective.</summary>
    internal sealed class CudaP2PCommunicator
    {
        private readonly TensorParallelGroup _owner;
        private readonly CudaAllocator[] _allocators;
        private readonly int _worldSize;
        private readonly bool[] _p2pEnabled;
        private readonly CudaNativeCalls _calls;
        private readonly List<CudaP2PEffectOwner> _initializationEffects = new();
        private ProbeOwner _probe;
        private StagingOwner _staging;
        private StagingOwner _stagingCandidate;
        private bool _initialized;
        private bool _released;
        private Exception _failure;

        internal CudaP2PCommunicator(TensorParallelGroup owner, CudaAllocator[] allocators)
        {
            _owner = owner ?? throw new ArgumentNullException(nameof(owner));
            _allocators = (CudaAllocator[])(allocators ?? throw new ArgumentNullException(nameof(allocators))).Clone();
            _worldSize = _allocators.Length;
            if (_worldSize < 2 || _allocators.Any(a => a == null)
                || _allocators.Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance).Count() != _worldSize)
                throw new ArgumentException("P2P requires distinct actual rank allocators.", nameof(allocators));
            if (_allocators.Any(a => !ReferenceEquals(a.NativeCalls.Api, _allocators[0].NativeCalls.Api)))
                throw new ArgumentException("P2P requires one shared native API.", nameof(allocators));
            _calls = new CudaNativeCalls(this, NativeOwnerRole.Worker, _allocators[0].NativeCalls.Api,
                _allocators.Select(a => a.DeviceId).Distinct().ToArray());
            _p2pEnabled = new bool[_worldSize * _worldSize];
        }

        internal void InitializePeerAccess()
        {
            if (_initialized) throw new InvalidOperationException("Peer access is already initialized.");
            using var admission = CudaOperationAdmission.Enter(this, _allocators);
            _owner.ThrowIfRetiring();
            if (CudaStorage.DisableP2P)
            {
                Console.WriteLine("  TP: P2P disabled by TENSORSHARP_TP_DISABLE_P2P; all cross-GPU transfers use host staging.");
                _initialized = true;
                return;
            }
            for (int source = 0; source < _worldSize; source++)
            {
                for (int destination = 0; destination < _worldSize; destination++)
                {
                    if (source == destination) continue;
                    _owner.ThrowIfRetiring();
                    var phase = new CudaP2PEffectOwner(this, new[] { _allocators[source], _allocators[destination] });
                    _initializationEffects.Add(phase);
                    using (var lease = phase.Calls.EnterEffect())
                    {
                        _allocators[source].Context.BindCurrent(phase.Calls);
                        phase.Calls.ThrowOnError(phase.Calls.cuDeviceCanAccessPeer(out int canAccess,
                            _allocators[source].DeviceId, _allocators[destination].DeviceId));
                        if (canAccess == 1)
                        {
                            int result = phase.Calls.cuCtxEnablePeerAccess(_allocators[destination].Context.Handle, 0);
                            if (result == 0 || result == 704)
                                _p2pEnabled[source * _worldSize + destination] = true;
                            else phase.Calls.ThrowOnError(result);
                        }
                    }
                    phase.Complete();
                }
            }
            for (int source = 0; source < _worldSize; source++)
            {
                for (int destination = 0; destination < _worldSize; destination++)
                {
                    if (source == destination || !CanAccessPeer(source, destination)) continue;
                    _owner.ThrowIfRetiring();
                    if (!VerifyP2PRoundTrip(source, destination))
                    {
                        Console.WriteLine(
                            $"  TP: P2P DMA self-test FAILED for GPU {source} → GPU {destination} " +
                            "(cuDeviceCanAccessPeer=1 but data is corrupt). " +
                            "Falling back to host-staged transfers for this pair.");
                        _p2pEnabled[source * _worldSize + destination] = false;
                        _p2pEnabled[destination * _worldSize + source] = false;
                        CudaStorage.MarkPeerAccessFailed(_allocators[source].DeviceId, _allocators[destination].DeviceId);
                    }
                }
            }
            _owner.ThrowIfRetiring();
            _calls.ThrowIfQuarantined();
            _initialized = true;
        }

        private bool VerifyP2PRoundTrip(int source, int destination)
        {
            _probe = new ProbeOwner(this, _allocators[source], _allocators[destination]);
            Exception original = null;
            bool valid = false;
            try { valid = _probe.Run(source, destination); }
            catch (Exception failure) { original = failure; _probe.Failure = failure; }
            try { _probe.Release(); }
            catch (Exception cleanup)
            {
                _failure = original == null ? cleanup : new AggregateException(original, cleanup);
                _probe.Failure = _failure;
                ExceptionDispatchInfo.Capture(_failure).Throw();
            }
            if (!_probe.Released)
            {
                if (original != null)
                {
                    _failure = original;
                    ExceptionDispatchInfo.Capture(original).Throw();
                }
                _calls.ThrowIfQuarantined();
                throw new InvalidOperationException("Probe ownership did not complete.");
            }
            _probe = null;
            return original == null && valid;
        }

        private bool CanAccessPeer(int source, int destination) => _p2pEnabled[source * _worldSize + destination];

        private void EnsureStagingBuffer(long bytes)
        {
            if (bytes == 0) return;
            if (_staging != null && _staging.Capacity >= bytes) return;
            if (_staging != null)
            {
                _staging.Release();
                _staging = null;
            }
            _stagingCandidate = new StagingOwner(this, _allocators[0]);
            try
            {
                _stagingCandidate.Allocate(bytes);
                _staging = _stagingCandidate;
                _stagingCandidate = null;
            }
            catch (Exception original)
            {
                try
                {
                    if (CudaP2PEffectOwner.IsHealthy(_stagingCandidate.Calls))
                    {
                        _stagingCandidate.Release();
                        _stagingCandidate = null;
                    }
                }
                catch (Exception cleanup)
                {
                    _failure = new AggregateException(original, cleanup);
                    ExceptionDispatchInfo.Capture(_failure).Throw();
                }
                throw;
            }
        }

        internal void AllReduce(CudaCollectiveOperation operation)
        {
            _owner.ThrowIfRetiring();
            operation.ValidateAllocators(_allocators);
            if (!_initialized || _released) throw new InvalidOperationException("Peer access is not available.");
            for (int rank = 0; rank < _worldSize; rank++)
            {
                _owner.ThrowIfRetiring();
                operation.Check();
                operation.MarkPending(rank);
                operation.Storages[rank].EnsureDeviceCurrent();
            }
            for (int rank = 0; rank < _worldSize; rank++)
            {
                _allocators[rank].Synchronize();
                operation.RecordDrained(rank);
            }
            for (int rank = 1; rank < _worldSize; rank++)
            {
                _owner.ThrowIfRetiring();
                operation.Check();
                if (CanAccessPeer(rank, 0) && _allocators[0].Kernels != null)
                {
                    EnsureStagingBuffer(operation.ByteCount);
                    IntPtr source = operation.Storages[rank].DevicePtrAtElement(0);
                    IntPtr destination = operation.Storages[0].DevicePtrAtElement(0);
                    var peer = operation.PeerEffect(rank, 0);
                    operation.MarkPending(0);
                    using (var lease = peer.Calls.EnterEffect())
                    {
                        _allocators[0].Context.BindCurrent(peer.Calls);
                        peer.Calls.ThrowOnError(peer.Calls.cuMemcpyPeerAsync(
                            _staging?.Pointer ?? IntPtr.Zero, _allocators[0].Context.Handle,
                            source, _allocators[rank].Context.Handle,
                            new UIntPtr((ulong)operation.ByteCount), _allocators[0].Stream.Handle));
                    }
                    var launch = operation.RankEffect(0);
                    operation.MarkPending(0);
                    using (var lease = launch.Calls.EnterEffect())
                    {
                        _allocators[0].Context.BindCurrent(launch.Calls);
                        _allocators[0].Kernels.LaunchBinaryF32(destination, _staging?.Pointer ?? IntPtr.Zero, destination,
                            operation.ElementCount, 0, _allocators[0].Stream.Handle);
                    }
                }
                else AllReduceViaHost(operation, rank, 0);
            }
            operation.DrainPending();
            _allocators[0].Synchronize();
            for (int rank = 1; rank < _worldSize; rank++)
            {
                _owner.ThrowIfRetiring();
                operation.Check();
                if (CanAccessPeer(0, rank))
                {
                    IntPtr source = operation.Storages[0].DevicePtrAtElement(0);
                    IntPtr destination = operation.Storages[rank].DevicePtrAtElement(0);
                    var peer = operation.PeerEffect(0, rank);
                    operation.MarkPending(rank);
                    using var lease = peer.Calls.EnterEffect();
                    _allocators[rank].Context.BindCurrent(peer.Calls);
                    peer.Calls.ThrowOnError(peer.Calls.cuMemcpyPeerAsync(
                        destination, _allocators[rank].Context.Handle,
                        source, _allocators[0].Context.Handle,
                        new UIntPtr((ulong)operation.ByteCount), _allocators[rank].Stream.Handle));
                }
                else BroadcastViaHost(operation, 0, rank);
            }
            operation.DrainPending();
            for (int rank = 0; rank < _worldSize; rank++)
            {
                _owner.ThrowIfRetiring();
                operation.Check();
                _allocators[rank].Synchronize();
            }
            for (int rank = 0; rank < _worldSize; rank++)
            {
                _owner.ThrowIfRetiring();
                operation.Check();
                operation.Storages[rank].MarkDeviceModified();
            }
        }

        private void AllReduceViaHost(CudaCollectiveOperation operation, int sourceRank, int destinationRank)
        {
            var source = new float[operation.ElementCount];
            var destination = new float[operation.ElementCount];
            IntPtr sourceHost = operation.PinBuffer(source);
            IntPtr destinationHost = operation.PinBuffer(destination);
            Download(operation, sourceRank, sourceHost);
            Download(operation, destinationRank, destinationHost);
            for (int element = 0; element < operation.ElementCount; element++)
                destination[element] += source[element];
            _owner.ThrowIfRetiring();
            operation.Check();
            Upload(operation, destinationRank, destinationHost);
        }

        private void BroadcastViaHost(CudaCollectiveOperation operation, int sourceRank, int destinationRank)
        {
            var host = new float[operation.ElementCount];
            IntPtr pointer = operation.PinBuffer(host);
            Download(operation, sourceRank, pointer);
            _owner.ThrowIfRetiring();
            operation.Check();
            Upload(operation, destinationRank, pointer);
        }

        private void Download(CudaCollectiveOperation operation, int rank, IntPtr hostPointer)
        {
            _owner.ThrowIfRetiring();
            operation.Check();
            IntPtr device = operation.Storages[rank].DevicePtrAtElement(0);
            var phase = operation.RankEffect(rank);
            using var lease = phase.Calls.EnterEffect();
            _allocators[rank].Context.BindCurrent(phase.Calls);
            phase.Calls.ThrowOnError(phase.Calls.cuMemcpyDtoH(hostPointer, device,
                new UIntPtr((ulong)operation.ByteCount)));
        }

        private void Upload(CudaCollectiveOperation operation, int rank, IntPtr hostPointer)
        {
            IntPtr device = operation.Storages[rank].DevicePtrAtElement(0);
            var phase = operation.RankEffect(rank);
            operation.MarkDefaultPending(rank);
            using var lease = phase.Calls.EnterEffect();
            _allocators[rank].Context.BindCurrent(phase.Calls);
            phase.Calls.ThrowOnError(phase.Calls.cuMemcpyHtoD(device, hostPointer,
                new UIntPtr((ulong)operation.ByteCount)));
        }

        internal void DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (_released) return;
            if (_failure != null && !CudaP2PEffectOwner.IsHealthy(_calls))
                ExceptionDispatchInfo.Capture(_failure).Throw();
            foreach (CudaAllocator allocator in _allocators)
                if (!plan.Owns(allocator)) throw new InvalidOperationException("P2P cleanup requires its actual owning plan.");
            try
            {
                _calls.ThrowIfQuarantined();
                using (var validation = _calls.EnterEffect()) _calls.ValidateSafeRelease(validation);
                _probe?.Release();
                if (_probe != null && !_probe.Released)
                {
                    if (_failure != null) ExceptionDispatchInfo.Capture(_failure).Throw();
                    _calls.ThrowIfQuarantined();
                }
                _probe = null;
                _stagingCandidate?.Release();
                _stagingCandidate = null;
                _staging?.Release();
                _staging = null;
                foreach (var phase in _initializationEffects) phase.Complete();
                using (var completion = _calls.EnterEffect()) _calls.CompleteSafeRelease(completion);
                _released = true;
            }
            catch (Exception failure)
            {
                _failure = failure;
                restoration?.MarkCleanupFailed();
                throw;
            }
        }

        private sealed class ProbeOwner
        {
            private readonly CudaP2PCommunicator _parent;
            private readonly CudaAllocator _source;
            private readonly CudaAllocator _destination;
            private readonly CudaNativeCalls _calls;
            private ProbeBufferOwner _sourceBuffer;
            private ProbeBufferOwner _destinationBuffer;
            private CudaP2PEffectOwner _pair;
            private byte[] _pattern;
            private byte[] _readback;
            private GCHandle _patternPin;
            private GCHandle _readbackPin;
            private bool _sourceDefaultPending;
            private bool _destinationDefaultPending;
            private bool _destinationPeerPending;
            internal Exception Failure;
            internal bool Released { get; private set; }

            internal ProbeOwner(CudaP2PCommunicator parent, CudaAllocator source, CudaAllocator destination)
            {
                _parent = parent;
                _source = source;
                _destination = destination;
                _calls = new CudaNativeCalls(this, NativeOwnerRole.Worker, source.NativeCalls.Api,
                    new[] { source.DeviceId, destination.DeviceId }.Distinct().ToArray());
            }

            internal bool Run(int sourceRank, int destinationRank)
            {
                const int bytes = 4096;
                _pattern = new byte[bytes];
                _readback = new byte[bytes];
                for (int element = 0; element < bytes; element++)
                    _pattern[element] = (byte)((element * 7 + sourceRank * 31 + destinationRank * 17) & 0xFF);
                _patternPin = GCHandle.Alloc(_pattern, GCHandleType.Pinned);
                _readbackPin = GCHandle.Alloc(_readback, GCHandleType.Pinned);
                _sourceBuffer = new ProbeBufferOwner(this, _source);
                _sourceBuffer.Allocate(bytes);
                _sourceDefaultPending = true;
                using (var lease = _sourceBuffer.Calls.EnterEffect())
                {
                    _source.Context.BindCurrent(_sourceBuffer.Calls);
                    _sourceBuffer.Calls.ThrowOnError(_sourceBuffer.Calls.cuMemcpyHtoD(
                        _sourceBuffer.Pointer, _patternPin.AddrOfPinnedObject(), new UIntPtr(bytes)));
                }
                _destinationBuffer = new ProbeBufferOwner(this, _destination);
                _destinationBuffer.Allocate(bytes);
                _destinationDefaultPending = true;
                using (var lease = _destinationBuffer.Calls.EnterEffect())
                {
                    _destination.Context.BindCurrent(_destinationBuffer.Calls);
                    _destinationBuffer.Calls.ThrowOnError(_destinationBuffer.Calls.cuMemsetD8(
                        _destinationBuffer.Pointer, 0, new UIntPtr(bytes)));
                }
                _sourceBuffer.DrainDefaultWork();
                _sourceDefaultPending = false;
                _destinationBuffer.DrainDefaultWork();
                _destinationDefaultPending = false;
                _pair = new CudaP2PEffectOwner(this, new[] { _source, _destination });
                _destinationPeerPending = true;
                using (var lease = _pair.Calls.EnterEffect())
                {
                    _destination.Context.BindCurrent(_pair.Calls);
                    _pair.Calls.ThrowOnError(_pair.Calls.cuMemcpyPeerAsync(
                        _destinationBuffer.Pointer, _destination.Context.Handle,
                        _sourceBuffer.Pointer, _source.Context.Handle,
                        new UIntPtr(bytes), _destination.Stream.Handle));
                }
                _destination.Stream.Synchronize();
                _destinationPeerPending = false;
                using (var lease = _destinationBuffer.Calls.EnterEffect())
                {
                    _destination.Context.BindCurrent(_destinationBuffer.Calls);
                    _destinationBuffer.Calls.ThrowOnError(_destinationBuffer.Calls.cuMemcpyDtoH(
                        _readbackPin.AddrOfPinnedObject(), _destinationBuffer.Pointer, new UIntPtr(bytes)));
                }
                for (int element = 0; element < bytes; element++)
                    if (_readback[element] != _pattern[element]) return false;
                return true;
            }

            internal void Release()
            {
                if (Released) return;
                bool healthy = CudaP2PEffectOwner.IsHealthy(_calls);
                if (healthy)
                {
                    using var validation = _calls.EnterEffect();
                    _calls.ValidateSafeRelease(validation);
                }
                List<Exception> failures = null;
                if (_sourceDefaultPending && CudaP2PEffectOwner.IsHealthy(_source.NativeCalls))
                {
                    try { _sourceBuffer.DrainDefaultWork(); _sourceDefaultPending = false; }
                    catch (Exception cleanup) { (failures ??= new()).Add(cleanup); }
                }
                if (_destinationDefaultPending && CudaP2PEffectOwner.IsHealthy(_destination.NativeCalls))
                {
                    try { _destinationBuffer.DrainDefaultWork(); _destinationDefaultPending = false; }
                    catch (Exception cleanup) { (failures ??= new()).Add(cleanup); }
                }
                if (_destinationPeerPending && CudaP2PEffectOwner.IsHealthy(_destination.NativeCalls))
                {
                    try { _destination.Stream.Synchronize(); _destinationPeerPending = false; }
                    catch (Exception cleanup) { (failures ??= new()).Add(cleanup); }
                }
                if (failures != null)
                    throw failures.Count == 1 ? failures[0] : new AggregateException(failures);
                if (_sourceDefaultPending || _destinationDefaultPending || _destinationPeerPending
                    || !CudaP2PEffectOwner.IsHealthy(_calls)) return;
                _sourceBuffer?.Release();
                _destinationBuffer?.Release();
                _pair?.Complete();
                if (_patternPin.IsAllocated) { _patternPin.Free(); _patternPin = default; }
                if (_readbackPin.IsAllocated) { _readbackPin.Free(); _readbackPin = default; }
                using (var completion = _calls.EnterEffect()) _calls.CompleteSafeRelease(completion);
                Released = true;
                _pattern = null;
                _readback = null;
                GC.KeepAlive(_parent);
            }

            private sealed class ProbeBufferOwner
            {
                private readonly ProbeOwner _parent;
                private readonly CudaAllocator _allocator;
                internal readonly CudaNativeCalls Calls;
                internal IntPtr Pointer;
                internal long Capacity;

                internal ProbeBufferOwner(ProbeOwner parent, CudaAllocator allocator)
                {
                    _parent = parent;
                    _allocator = allocator;
                    Calls = new CudaNativeCalls(this, NativeOwnerRole.Storage, allocator.NativeCalls.Api, allocator.DeviceId);
                }

                internal void Allocate(long bytes)
                {
                    using var lease = Calls.EnterEffect();
                    _allocator.Context.BindCurrent(Calls);
                    Calls.ThrowOnError(Calls.cuMemAlloc(out Pointer, new UIntPtr((ulong)bytes)));
                    Capacity = bytes;
                }

                internal void DrainDefaultWork()
                {
                    using var lease = Calls.EnterEffect();
                    _allocator.Context.BindCurrent(Calls);
                    Calls.cuCtxSynchronize();
                }

                internal void Release()
                {
                    using var lease = Calls.EnterEffect();
                    Calls.ValidateSafeRelease(lease);
                    if (Pointer != IntPtr.Zero)
                    {
                        _allocator.Context.BindCurrent(Calls);
                        Calls.cuMemFree(Pointer);
                        Pointer = IntPtr.Zero;
                        Capacity = 0;
                    }
                    Calls.CompleteSafeRelease(lease);
                    GC.KeepAlive(_parent);
                }
            }
        }

        private sealed class StagingOwner
        {
            private readonly CudaP2PCommunicator _parent;
            private readonly CudaAllocator _allocator;
            internal readonly CudaNativeCalls Calls;
            internal IntPtr Pointer;
            internal long Capacity;

            internal StagingOwner(CudaP2PCommunicator parent, CudaAllocator allocator)
            {
                _parent = parent;
                _allocator = allocator;
                Calls = new CudaNativeCalls(this, NativeOwnerRole.Storage, allocator.NativeCalls.Api, allocator.DeviceId);
            }

            internal void Allocate(long bytes)
            {
                using var lease = Calls.EnterEffect();
                _allocator.Context.BindCurrent(Calls);
                Calls.ThrowOnError(Calls.cuMemAlloc(out Pointer, new UIntPtr((ulong)bytes)));
                Capacity = bytes;
            }

            internal void Release()
            {
                using var lease = Calls.EnterEffect();
                Calls.ValidateSafeRelease(lease);
                if (Pointer != IntPtr.Zero)
                {
                    _allocator.Context.BindCurrent(Calls);
                    Calls.cuMemFree(Pointer);
                    Pointer = IntPtr.Zero;
                    Capacity = 0;
                }
                Calls.CompleteSafeRelease(lease);
                GC.KeepAlive(_parent);
            }
        }
    }
}
