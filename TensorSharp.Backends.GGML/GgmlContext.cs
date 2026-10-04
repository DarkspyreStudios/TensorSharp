// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Reflection;

namespace TensorSharp.GGML
{
    public sealed class GgmlContext : IDisposable
    {
        private readonly object lifetimeGate = new();
        private GgmlNativeLoader.OwnedResourceLease runtimeLease;
        private NativeOwnerRegistration registration;
        private volatile Exception cleanupFailure;
        private int activeStorages;
        private bool disposed;
        private bool cleaning;
        private bool disposing;
        internal GgmlMemoryPool MemoryPool { get; }

        /// <summary>
        /// Return every pooled-but-unused host block to the operating system and report
        /// how many bytes that was. The pool keeps freed blocks so the next allocation
        /// is cheap; on a device that is about to be killed for memory, cheap is not
        /// the point.
        /// </summary>
        public long ReleasePooledMemory()
        {
            lock (lifetimeGate)
            {
                ThrowIfDisposed();
                cleaning = true;
            }
            GgmlNativeLoader.ResourceCleanupReservation reservation = null;
            try
            {
                reservation = GgmlNativeLoader.ReserveResourceCleanup(this);
                GgmlBasicOps.HostReadBarrier();
                long bytes = MemoryPool.Trim(GgmlNative.InvalidateHostBuffer);
                reservation.Complete();
                return bytes;
            }
            catch (Exception error)
            {
                if (reservation != null || GgmlNativeLoader.IsUnsafeCleanupRefusal(error)) RetainCleanupFailure(error);
                throw;
            }
            finally
            {
                reservation?.Dispose();
                lock (lifetimeGate) cleaning = false;
            }
        }

        internal StorageLease AcquireStorageLease(object storage, NativeOwnerRegistration storageRegistration)
        {
            lock (lifetimeGate)
            {
                ThrowIfDisposed();
                var lease = GgmlNativeLoader.AcquireOwnedLease(GgmlRuntimeResourceKind.Tensor, storage, storageRegistration);
                activeStorages++;
                return new StorageLease(this, lease);
            }
        }

        internal sealed class StorageLease(GgmlContext context, GgmlNativeLoader.OwnedResourceLease lease) : IDisposable
        {
            private bool released;
            internal void RetainCleanupFailure(object owner, Exception error) => lease.RetainCleanupFailure(owner, error);
            internal void CompleteRelease(object owner, NativeOwnerRegistration registration)
            {
                lease.CompleteRelease(owner, registration);
                Dispose();
            }
            public void Dispose()
            {
                lock (context.lifetimeGate)
                {
                    if (released || lease.HasCleanupFailure) return;
                    released = true;
                    context.activeStorages--;
                    lease.Dispose();
                }
            }
        }

        internal void ThrowIfDisposed()
        {
            lock (lifetimeGate)
            {
                if (disposed || disposing) throw new ObjectDisposedException(nameof(GgmlContext));
                if (cleanupFailure != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(cleanupFailure).Throw();
                if (cleaning) throw new InvalidOperationException("The context is performing resource cleanup.");
                registration?.ThrowIfQuarantined();
            }
        }

        internal void RetainCleanupFailure(Exception error)
        {
            cleanupFailure ??= error;
            runtimeLease?.RetainCleanupFailure(this, error);
        }

        public void Dispose()
        {
            lock (lifetimeGate)
            {
                if (disposed) return;
                ThrowIfDisposed();
                if (activeStorages != 0) throw new InvalidOperationException("Dispose the context's tensor storages before disposing their context.");
                disposing = true;
            }
            GgmlNativeLoader.ResourceCleanupReservation reservation = null;
            try
            {
                reservation = GgmlNativeLoader.ReserveResourceCleanup(this);
                GgmlBasicOps.HostReadBarrier();
                MemoryPool.Trim(GgmlNative.InvalidateHostBuffer);
                runtimeLease.CompleteRelease(this, registration);
                lock (lifetimeGate) disposed = true;
                reservation.Complete();
                GC.SuppressFinalize(this);
            }
            catch (Exception error)
            {
                if (reservation != null || GgmlNativeLoader.IsUnsafeCleanupRefusal(error)) RetainCleanupFailure(error);
                throw;
            }
            finally
            {
                reservation?.Dispose();
                lock (lifetimeGate) disposing = false;
            }
        }

        ~GgmlContext()
        {
            // Storages retain their context. Only an unreachable, storage-free context can reach this path.
            try
            {
                if (MemoryPool != null && runtimeLease != null) Dispose();
            }
            catch (Exception error)
            {
                RetainCleanupFailure(error);
            }
        }

        public GgmlContext(int[] deviceIds, GgmlBackendType backendType)
            : this(deviceIds, backendType, enableCollectives: true)
        {
        }

        /// <param name="enableCollectives">
        /// False for a LAYER SPLIT: bring up one backend per GPU but create no
        /// cross-device collective. Each GPU runs a contiguous run of layers and
        /// the only thing that crosses a boundary is the residual, handed over
        /// through host memory, so there is nothing to AllReduce.
        /// </param>
        public GgmlContext(int[] deviceIds, GgmlBackendType backendType, bool enableCollectives)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            DeviceIds = (int[])deviceIds.Clone();
            DeviceId = deviceIds[0];
            BackendType = backendType;
            registration = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Allocator);
            runtimeLease = GgmlNativeLoader.AcquireOwnedLease(GgmlRuntimeResourceKind.Context, this, registration, backendType);
            try
            {
                MemoryPool = new GgmlMemoryPool(this);
                MemoryPool.EnsureInitialBlocks();
                GgmlNative.EnsureAvailable(backendType);

                if (deviceIds.Length > 1)
                {
                    // Several GPUs: bring up one ggml backend per GPU. Ops then select a
                    // rank with GgmlNative.SetActiveDevice; tensors carry their rank
                    // through GgmlAllocator.DeviceId. Used both by tensor parallelism
                    // (every GPU holds a shard of every weight, collectives on) and by a
                    // layer split (each GPU holds a run of whole layers, collectives off).
                    if (backendType != GgmlBackendType.Cuda && backendType != GgmlBackendType.Vulkan)
                    {
                        throw new NotSupportedException(
                            $"The GGML {backendType} backend exposes a single device; multi-GPU requires the CUDA or Vulkan backend.");
                    }
                    // The native side needs to know whether the ranks will be driven
                    // concurrently: that decides whether ggml-cuda's graph capture
                    // (which is process-wide and breaks under concurrent CUDA calls)
                    // has to be turned off for the run.
                    if (enableCollectives)
                    {
                        GgmlNative.TensorParallelInit(backendType, DeviceIds, GgmlTensorParallelGroup.ParallelRanks);
                        HasDeviceAllReduce = GgmlNative.TensorParallelHasDeviceAllReduce();
                    }
                    else
                    {
                        GgmlNative.MultiDeviceInit(backendType, DeviceIds);
                        HasDeviceAllReduce = false;
                    }
                }
                OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());

                // On Metal, default to async (lazy) GPU dispatch. This is the same model
                // llama.cpp uses: per-op `ggml_metal_graph_compute` commits its command
                // buffer and returns immediately; only the final `ggml_backend_synchronize`
                // (or a host-side data read via TensorComputePrimitives.GetFloatPointer)
                // actually waits on the GPU. With this enabled we avoid one
                // `[cmd_buf waitUntilCompleted]` (~30-100µs of driver/IPC round-trip on
                // M-series Macs) per op submitted, which dominates prefill on long
                // prompts where TensorSharp's per-op driving model would otherwise
                // submit hundreds of command buffers serially.
                //
                // Set TS_GGML_ASYNC_COMPUTE=0 to disable and fall back to the legacy
                // eager-sync behaviour for debugging.
                var disableAsync = Environment.GetEnvironmentVariable("TS_GGML_ASYNC_COMPUTE");
                bool enableAsync = backendType == GgmlBackendType.Metal &&
                                   !string.Equals(disableAsync, "0", StringComparison.Ordinal);
                GgmlNative.SetAsyncCompute(enableAsync);
            }
            catch (Exception operation)
            {
                try
                {
                    if (MemoryPool != null)
                    {
                        if (!GgmlNativeLoader.NativeOwnershipMayExist) MemoryPool.TrimHostOnly();
                        else
                        {
                            using var reservation = GgmlNativeLoader.ReserveResourceCleanup(this);
                            MemoryPool.Trim(GgmlNative.InvalidateHostBuffer);
                            reservation.Complete();
                        }
                    }
                    runtimeLease.CompleteRelease(this, registration);
                    disposed = true;
                    GC.SuppressFinalize(this);
                }
                catch (Exception cleanup)
                {
                    var failure = new AggregateException(operation, cleanup);
                    RetainCleanupFailure(failure);
                    throw failure;
                }
                throw;
            }
        }

        /// <summary>Physical device index backing rank 0.</summary>
        public int DeviceId { get; }

        /// <summary>Physical device index per rank; length == the TP degree.</summary>
        public int[] DeviceIds { get; }

        /// <summary>Number of GPUs this context spans (1 = no tensor parallelism).</summary>
        public int Degree => DeviceIds.Length;

        /// <summary>
        /// True when cross-GPU AllReduce runs entirely on the devices (NCCL or
        /// P2P via ggml's CUDA comm backend) rather than through host memory.
        /// </summary>
        public bool HasDeviceAllReduce { get; }

        public GgmlBackendType BackendType { get; }
    }
}
