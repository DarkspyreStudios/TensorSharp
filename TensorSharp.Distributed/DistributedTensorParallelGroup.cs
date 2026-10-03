using System;
using System.Net;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cuda;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Hierarchical local reduce, TCP exchange and local broadcast.
    /// CUDA callers keep borrowed rank tensors undisposed until completion.
    /// </summary>
    public sealed class DistributedTensorParallelGroup : ITensorParallelGroup, INestedTensorParallelGroup, ICudaTensorParallelRetirement
    {
        public ITensorParallelGroup LocalGroup => _localGroup;
        private readonly ITensorParallelGroup _localGroup;
        private readonly TcpCommunicator _tcp;
        private readonly int _nodeId;
        private readonly int _nodeCount;
        private readonly CudaAllocator[] _cudaAllocators;
        private readonly CudaNativeCalls _nativeCalls;
        private bool _disposed;
        private bool _retirementRequested;
        private Exception _cleanupFailure;
        private float[] _hostBuffer = Array.Empty<float>();
        private float[] _resultBuffer = Array.Empty<float>();
        private GCHandle _hostPin;
        private GCHandle _resultPin;
        private int _bufferSize;

        internal DistributedTensorParallelGroup(TensorParallelGroup localGroup, float[] hostBuffer, float[] resultBuffer)
        {
            ArgumentNullException.ThrowIfNull(localGroup);
            ArgumentNullException.ThrowIfNull(hostBuffer);
            ArgumentNullException.ThrowIfNull(resultBuffer);
            if (ReferenceEquals(hostBuffer, resultBuffer) || hostBuffer.Length != resultBuffer.Length)
                throw new ArgumentException("Owned transfer buffers must be distinct and equal in length.");
            try
            {
                _hostPin = GCHandle.Alloc(hostBuffer, GCHandleType.Pinned);
                _resultPin = GCHandle.Alloc(resultBuffer, GCHandleType.Pinned);
            }
            catch
            {
                if (_hostPin.IsAllocated) { _hostPin.Free(); _hostPin = default; }
                if (_resultPin.IsAllocated) { _resultPin.Free(); _resultPin = default; }
                throw;
            }
            _localGroup = localGroup;
            _nodeCount = 2;
            _hostBuffer = hostBuffer;
            _resultBuffer = resultBuffer;
            _bufferSize = hostBuffer.Length;
            try
            {
                _cudaAllocators = CollectCudaAllocators(localGroup);
                _nativeCalls = CreateNativeCalls();
            }
            catch (Exception original)
            {
                RollBackConstruction(original);
                throw;
            }
        }

        public DistributedTensorParallelGroup(int localDegree, int nodeId, IPEndPoint[] peerEndpoints)
            : this(new TensorParallelGroup(localDegree), nodeId, peerEndpoints, true)
        {
        }

        public DistributedTensorParallelGroup(ITensorParallelGroup localGroup, int nodeId, IPEndPoint[] peerEndpoints)
            : this(localGroup, nodeId, peerEndpoints, false)
        {
        }

        private DistributedTensorParallelGroup(ITensorParallelGroup localGroup, int nodeId,
            IPEndPoint[] peerEndpoints, bool localAlreadyOwned)
        {
            if (localAlreadyOwned) _localGroup = localGroup;
            try
            {
                _nodeId = nodeId;
                _nodeCount = peerEndpoints.Length;
                if (localGroup == null) throw new ArgumentNullException(nameof(localGroup));
                if (_nodeCount < 2)
                    throw new ArgumentException("Distributed TP requires at least 2 nodes.", nameof(peerEndpoints));
                _localGroup = localGroup;
                int localDegree = localGroup.Degree;
                _cudaAllocators = CollectCudaAllocators(localGroup);
                _nativeCalls = CreateNativeCalls();
                _tcp = new TcpCommunicator(nodeId, peerEndpoints);
                Console.WriteLine($"Distributed tensor parallelism: node {nodeId}/{_nodeCount}, " +
                    $"{localDegree} local GPU(s), {_nodeCount * localDegree} total across cluster.");
                _nativeCalls?.ThrowIfQuarantined();
            }
            catch (Exception original)
            {
                if (_localGroup != null) RollBackConstruction(original);
                throw;
            }
        }

        private static CudaAllocator[] CollectCudaAllocators(ITensorParallelGroup localGroup)
        {
            if (localGroup is not ICudaTensorParallelRetirement local || !local.OwnsCudaAllocators)
                return Array.Empty<CudaAllocator>();
            var collected = new List<IAllocator>();
            local.CollectOwnedAllocators(collected);
            if (collected.Count == 0 || collected.Any(a => a is not CudaAllocator))
                throw new InvalidOperationException("CUDA dispatch requires actual local CUDA allocators.");
            var allocators = collected.Cast<CudaAllocator>().Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance).ToArray();
            if (allocators.Any(a => !ReferenceEquals(a.NativeCalls.Api, allocators[0].NativeCalls.Api)))
                throw new InvalidOperationException("CUDA dispatch requires one actual native API.");
            return allocators;
        }

        private CudaNativeCalls CreateNativeCalls() => _cudaAllocators.Length == 0 ? null
            : new CudaNativeCalls(this, NativeOwnerRole.Worker, _cudaAllocators[0].NativeCalls.Api,
                _cudaAllocators.Select(a => a.DeviceId).Distinct().ToArray());

        public int Degree => _localGroup.Degree;
        public bool IsActive => true;
        public int GlobalDegree => Degree * _nodeCount;
        public int GlobalRankOffset => _nodeId * Degree;
        public int NodeCount => _nodeCount;
        public IAllocator GetAllocator(int rank) => _localGroup.GetAllocator(rank);

        private CudaCollectiveOperation EnterCudaOperation()
        {
            ThrowIfRetiring();
            if (_nativeCalls == null) return null;
            var operation = CudaCollectiveOperation.Enter(this, _cudaAllocators);
            try
            {
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
            if (_nativeCalls == null)
            {
                ThrowIfRetiring();
                _localGroup.RunPerRank(rank => { ThrowIfRetiring(); body(rank); });
                return;
            }
            using var operation = EnterCudaOperation();
            try
            {
                _localGroup.RunPerRank(rank =>
                {
                    ThrowIfRetiring();
                    operation?.Check();
                    body(rank);
                    ThrowIfRetiring();
                    operation?.Check();
                });
                ThrowIfRetiring();
                operation?.Complete();
            }
            catch (Exception original)
            {
                if (operation != null) ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        public void AllReduce(Tensor[] tensors)
        {
            ThrowIfRetiring();
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.");
            if (_nativeCalls == null)
            {
                AllReduceNonCuda(tensors);
                return;
            }
            using var operation = EnterCudaOperation();
            try
            {
                operation.FreezeRanks(tensors);
                AllReduceAdmitted(operation);
                operation.Complete();
            }
            catch (Exception original)
            {
                ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        private void AllReduceAdmitted(CudaCollectiveOperation operation)
        {
            ThrowIfRetiring();
            operation.ValidateAllocators(_cudaAllocators);
            if (_localGroup is TensorParallelGroup local) local.AllReduceAdmitted(operation);
            else if (_localGroup is DistributedTensorParallelGroup nested) nested.AllReduceAdmitted(operation);
            else throw new InvalidOperationException("CUDA collective dispatch requires its actual known local group.");
            ThrowIfRetiring();
            operation.Check();
            int count = operation.ElementCount;
            EnsureBuffers(count);
            float[] host = _hostBuffer;
            float[] result = _resultBuffer;
            operation.RetainBuffers(host, result);
            float[] localData = operation.Storages[0].GetElementsAsFloat(operation.StorageOffsets[0], count);
            Array.Copy(localData, host, count);
            ThrowIfRetiring();
            operation.Check();
            _tcp.AllReduce(host, count);
            ThrowIfRetiring();
            operation.Check();
            Array.Copy(host, result, count);
            for (int rank = 0; rank < _cudaAllocators.Length; rank++)
            {
                ThrowIfRetiring();
                operation.Check();
                operation.Storages[rank].SetElementsAsFloat(operation.StorageOffsets[rank], result);
                operation.MarkPending(rank);
                operation.Storages[rank].EnsureDeviceCurrent();
            }
            // EnsureDeviceCurrent queues CUDA HtoD work even for a single local rank.
            operation.DrainPending();
            ThrowIfRetiring();
            operation.Check();
        }

        private void AllReduceNonCuda(Tensor[] tensors)
        {
            _localGroup.AllReduce(tensors);
            int count = (int)tensors[0].Storage.ElementCount;
            EnsureBuffers(count);
            var hostData = tensors[0].GetElementsAsFloat(count);
            Array.Copy(hostData, _hostBuffer, count);
            _tcp.AllReduce(_hostBuffer, count);
            Array.Copy(_hostBuffer, _resultBuffer, count);
            for (int rank = 0; rank < Degree; rank++)
            {
                tensors[rank].SetElementsAsFloat(_resultBuffer);
                tensors[rank].EnsureDeviceCurrent();
            }
            if (Degree > 1) _localGroup.Synchronize();
        }

        public void CrossNodeAllReduce(float[] buffer, int count)
        {
            ThrowIfRetiring();
            ArgumentNullException.ThrowIfNull(buffer);
            if (count > 0) _tcp.AllReduce(buffer, count);
        }

        public void Synchronize()
        {
            if (_nativeCalls == null)
            {
                ThrowIfRetiring();
                _localGroup.Synchronize();
                return;
            }
            using var operation = EnterCudaOperation();
            try
            {
                _localGroup.Synchronize();
                ThrowIfRetiring();
                operation?.Check();
                operation?.Complete();
            }
            catch (Exception original)
            {
                if (operation != null) ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        public void Barrier()
        {
            if (_nativeCalls == null)
            {
                ThrowIfRetiring();
                _localGroup.Synchronize();
                _tcp.Barrier();
                return;
            }
            using var operation = EnterCudaOperation();
            try
            {
                _localGroup.Synchronize();
                ThrowIfRetiring();
                operation?.Check();
                _tcp.Barrier();
                ThrowIfRetiring();
                operation?.Complete();
            }
            catch (Exception original)
            {
                if (operation != null) ExceptionDispatchInfo.Capture(operation.SettleFailure(original)).Throw();
                throw;
            }
        }

        public void BroadcastControl(int op, int[] payload)
        {
            ThrowIfRetiring();
            _tcp.BroadcastControl(op, payload);
        }

        public (int op, int[] payload) ReceiveControl()
        {
            ThrowIfRetiring();
            var control = _tcp.ReceiveControl();
            ThrowIfRetiring();
            return control;
        }

        private void EnsureBuffers(int count)
        {
            if (_bufferSize == count) return;
            if (_hostPin.IsAllocated) { _hostPin.Free(); _hostPin = default; }
            if (_resultPin.IsAllocated) { _resultPin.Free(); _resultPin = default; }
            _bufferSize = 0;
            _hostBuffer = new float[count];
            _resultBuffer = new float[count];
            _hostPin = GCHandle.Alloc(_hostBuffer, GCHandleType.Pinned);
            _resultPin = GCHandle.Alloc(_resultBuffer, GCHandleType.Pinned);
            _bufferSize = count;
        }

        public void Dispose()
        {
            if (_disposed) return;
            Volatile.Write(ref _retirementRequested, true);
            if (_cleanupFailure != null && _nativeCalls != null && !CudaP2PEffectOwner.IsHealthy(_nativeCalls))
                ExceptionDispatchInfo.Capture(_cleanupFailure).Throw();
            if (((ICudaTensorParallelRetirement)this).OwnsCudaAllocators)
            {
                var plan = CudaRetirementPlan.PrepareGroup(this, this);
                var restoration = plan.CaptureRestoration();
                plan.AllowStorageRelease();
                ((ICudaTensorParallelRetirement)this).DisposeOwned(plan, restoration);
                plan.Complete();
                restoration?.Restore();
                return;
            }
            DisposeTransport();
            _localGroup?.Dispose();
            _disposed = true;
        }

        void ICudaTensorParallelRetirement.CollectOwnedAllocators(ICollection<IAllocator> allocators)
        {
            if (_localGroup is ICudaTensorParallelRetirement local) local.CollectOwnedAllocators(allocators);
        }

        bool ICudaTensorParallelRetirement.OwnsCudaAllocators =>
            _localGroup is ICudaTensorParallelRetirement local && local.OwnsCudaAllocators;

        void ICudaTensorParallelRetirement.DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration)
        {
            if (_disposed) return;
            if (_localGroup is not ICudaTensorParallelRetirement local)
                throw new InvalidOperationException("CUDA retirement requires an actual local CUDA group.");
            Volatile.Write(ref _retirementRequested, true);
            if (_cleanupFailure != null && _nativeCalls != null && !CudaP2PEffectOwner.IsHealthy(_nativeCalls))
                ExceptionDispatchInfo.Capture(_cleanupFailure).Throw();
            try
            {
                plan.ValidateGroupRelease(this);
                if (_nativeCalls != null)
                {
                    _nativeCalls.ThrowIfQuarantined();
                    using var validation = _nativeCalls.EnterEffect();
                    _nativeCalls.ValidateSafeRelease(validation);
                }
                plan.Drain(restoration);
                DisposeTransport();
                local.DisposeOwned(plan, restoration);
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
            Volatile.Write(ref _retirementRequested, true);
            _cleanupFailure = original;
            if (_nativeCalls != null && !CudaP2PEffectOwner.IsHealthy(_nativeCalls)) return;
            try
            {
                if (_localGroup is ICudaTensorParallelRetirement local && local.OwnsCudaAllocators)
                {
                    var plan = CudaRetirementPlan.PrepareGroup(this, this);
                    var restoration = plan.CaptureRestoration();
                    plan.AllowStorageRelease();
                    ((ICudaTensorParallelRetirement)this).DisposeOwned(plan, restoration);
                    plan.Complete();
                    restoration?.Restore();
                }
                else
                {
                    DisposeTransport();
                    _localGroup?.Dispose();
                }
            }
            catch (Exception cleanup)
            {
                _cleanupFailure = new AggregateException(original, cleanup);
                ExceptionDispatchInfo.Capture(_cleanupFailure).Throw();
            }
        }

        private void DisposeTransport()
        {
            if (_hostPin.IsAllocated) { _hostPin.Free(); _hostPin = default; }
            if (_resultPin.IsAllocated) { _resultPin.Free(); _resultPin = default; }
            _tcp?.Dispose();
        }

        private void ThrowIfRetiring()
        {
            if (Volatile.Read(ref _retirementRequested))
                throw new ObjectDisposedException(nameof(DistributedTensorParallelGroup));
            _nativeCalls?.ThrowIfQuarantined();
        }
    }
}
