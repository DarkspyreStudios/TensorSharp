using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed unsafe class CudaStorage : Storage
    {
        private readonly object sync = new object();
        private IntPtr hostBuffer;
        private IntPtr deviceBuffer;
        private long deviceAllocationBytes;
        private bool hostDirty;
        private bool deviceDirty;
        private readonly CudaNativeCalls nativeCalls;
        private readonly CudaGraphCapture.StorageCapturePayload capturePayload;
        private bool allocationIsRental;
        internal CudaStorageReservation Reservation { get; }

        public CudaStorage(CudaAllocator allocator, DType elementType, long elementCount)
            : this(allocator, elementType, elementCount, null) { }

        internal CudaStorage(CudaAllocator allocator, DType elementType, long elementCount,
            CudaStorageReservation reservation)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            capturePayload = new CudaGraphCapture.StorageCapturePayload(this);
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));

            Reservation = reservation ?? allocator.Census.Reserve();
            try
            {
                allocator.Census.Attach(Reservation, this);
                nativeCalls = new CudaNativeCalls(this, NativeOwnerRole.Storage,
                    allocator.NativeCalls.Api, allocator.DeviceId);
                allocator.Census.StartNative(Reservation);
                allocator.RentDeviceMemory(this, ByteLength);
            }
            catch (Exception original)
            {
                try { Destroy(); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }

        internal override object ReferenceMutationGate => AllocatorImpl?.Census.Gate;

        internal override void ValidateReferenceAddition()
        {
            AllocatorImpl.Census.ValidateAddition();
            if (Reservation.Releasing || Reservation.State == CudaStorageReservationState.Released)
                throw new ObjectDisposedException(nameof(CudaStorage));
        }

        internal override bool IsRetainedFinalizerFailure(Exception error)
            => nativeCalls != null && nativeCalls.IsRetainedFailure(error);

        internal IntPtr AllocateDeviceMemory(long allocationBytes)
        {
            CudaGraphCapture.CaptureContext capture = CudaGraphCapture.PrepareStorageTransfer(this, capturePayload);
            if (TryAllocateDeviceMemory(allocationBytes) == 2)
            {
                if (CudaGraphCapture.CanTransferStorage(capture, capturePayload))
                    throw new CudaGraphCaptureAbortedException("Device allocation requires reclamation during graph capture.");
                // Finalizers can need the caller's native gates. Nested OOM unwinds
                // with its allocation error before any collection or retry.
                if (!AllocatorImpl.TryReclaimOutsideEffects()) nativeCalls.ThrowOnError(2);
                lock (AllocatorImpl.Census.Gate) AllocatorImpl.Census.ValidateAddition();
                int result = TryAllocateDeviceMemory(allocationBytes);
                nativeCalls.ThrowOnError(result);
            }
            return deviceBuffer;
        }

        private int TryAllocateDeviceMemory(long allocationBytes)
        {
            using var lease = nativeCalls.EnterEffect();
            AllocatorImpl.Context.BindCurrent(nativeCalls);
            deviceAllocationBytes = allocationBytes;
            int result = nativeCalls.cuMemAlloc(out deviceBuffer, new UIntPtr((ulong)allocationBytes));
            if (result == 2)
            {
                if (CudaGraphCapture.CanTransferStorage(capturePayload.Context, capturePayload)) return result;
                if (deviceBuffer != IntPtr.Zero)
                {
                    nativeCalls.cuCtxSynchronize();
                    nativeCalls.cuMemFree(deviceBuffer);
                    deviceBuffer = IntPtr.Zero;
                }
                deviceAllocationBytes = 0;
            }
            else nativeCalls.ThrowOnError(result);
            return result;
        }

        internal void AcceptDeviceMemory(IntPtr pointer, long allocationBytes)
        {
            deviceBuffer = pointer;
            deviceAllocationBytes = allocationBytes;
            allocationIsRental = true;
        }

        internal CudaAllocator AllocatorImpl { get; }

        internal IntPtr DeviceBuffer => deviceBuffer;

        public int DeviceId => AllocatorImpl.DeviceId;

        protected override void Destroy()
        {
            if (Reservation == null || Reservation.State == CudaStorageReservationState.Released) return;
            CudaGraphCapture.CaptureContext capture = CudaGraphCapture.PrepareStorageTransfer(this, capturePayload);
            bool graphsPending = AllocatorImpl.Census.BeginRelease(Reservation);
            if (nativeCalls == null)
            {
                AllocatorImpl.Census.Complete(Reservation);
                return;
            }
            if (capture != null)
            {
                using (var transferLease = nativeCalls.EnterEffect())
                {
                    nativeCalls.ValidateSafeRelease(transferLease);
                    try
                    {
                        if (graphsPending)
                            throw new InvalidOperationException("CUDA storage release cannot transfer pending parent graph ownership.");
                        if (!CudaGraphCapture.CanTransferStorage(capture, capturePayload))
                            throw new InvalidOperationException("The prepared storage capture is no longer active.");
                        capturePayload.Device = deviceBuffer;
                        capturePayload.Bytes = deviceAllocationBytes;
                        capturePayload.PartialAllocation = !allocationIsRental;
                        capturePayload.Host = hostBuffer;
                        capturePayload.Transferred = true;
                        deviceBuffer = IntPtr.Zero;
                        deviceAllocationBytes = 0;
                        hostBuffer = IntPtr.Zero;
                        nativeCalls.CompleteSafeRelease(transferLease);
                    }
                    catch (Exception cleanup)
                    {
                        nativeCalls.PublishFailure(transferLease, cleanup, NativeRuntimeFailureStage.StorageRelease);
                        throw;
                    }
                }
                AllocatorImpl.Census.Complete(Reservation);
                AllocatorImpl.ReturnCapturedPayloadToPool(capturePayload);
                return;
            }
            using (var lease = nativeCalls.EnterEffect())
                lock (sync)
                {
                    nativeCalls.ValidateSafeRelease(lease);
                    try
                    {
                        if (graphsPending && (deviceBuffer != IntPtr.Zero || hostBuffer != IntPtr.Zero))
                            throw new InvalidOperationException("CUDA storage release cannot prove pending parent graph ownership safe.");
                        if (deviceBuffer != IntPtr.Zero || hostBuffer != IntPtr.Zero)
                        {
                            AllocatorImpl.Context.BindCurrent(nativeCalls);
                            nativeCalls.cuCtxSynchronize();
                        }
                        if (deviceBuffer != IntPtr.Zero)
                        {
                            AllocatorImpl.ReturnDeviceMemory(deviceBuffer, deviceAllocationBytes);
                            deviceBuffer = IntPtr.Zero;
                            deviceAllocationBytes = 0;
                        }
                        if (hostBuffer != IntPtr.Zero)
                        {
                            CudaGraphCapture.OnHostBufferOrphaned(AllocatorImpl, hostBuffer, out bool donated);
                            if (!donated) NativeMemory.AlignedFree(hostBuffer.ToPointer());
                            hostBuffer = IntPtr.Zero;
                        }
                        nativeCalls.CompleteSafeRelease(lease);
                    }
                    catch (Exception cleanup)
                    {
                        nativeCalls.PublishFailure(lease, cleanup, NativeRuntimeFailureStage.StorageRelease);
                        throw;
                    }
                }
            AllocatorImpl.Census.Complete(Reservation);
        }

        /// <summary>Free a host mirror previously donated to a CUDA graph cache
        /// entry by <see cref="Destroy"/> during capture.</summary>
        internal static void FreeDonatedHostBuffer(IntPtr hostBuffer)
        {
            if (hostBuffer != IntPtr.Zero)
                NativeMemory.AlignedFree(hostBuffer.ToPointer());
        }

        public override string LocationDescription()
        {
            return $"CUDA:{DeviceId}";
        }

        public override IntPtr PtrAtElement(long index)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            ValidateElementRange(index, 0);

            // Existing TensorSharp model code may mutate through raw pointers. Treat
            // every pointer checkout as a possible host-side write so the next CUDA
            // dispatch refreshes device memory before using this storage as input.
            SyncHostFromDevice();
            hostDirty = true;
            return HostPtrAtElementUnchecked(index);
        }

        internal IntPtr DevicePtrAtElement(long index)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            ValidateElementRange(index, 0);
            return AddBytes(deviceBuffer, checked(index * ElementType.Size()));
        }

        public override void EnsureDeviceCurrent()
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                if (!hostDirty)
                    return;

                CudaGraphCapture.OnCapturedHostUpload(AllocatorImpl, ByteLength);
                long t0 = CudaProfileCounters.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                AllocatorImpl.Context.BindCurrent(nativeCalls);
                nativeCalls.ThrowOnError(nativeCalls.cuMemcpyHtoDAsync(
                    deviceBuffer,
                    hostBuffer,
                    new UIntPtr((ulong)ByteLength),
                    AllocatorImpl.Stream.Handle));
                hostDirty = false;
                deviceDirty = false;
                if (CudaProfileCounters.Enabled)
                    CudaProfileCounters.RecordSync("HtoD(async)", ByteLength, System.Diagnostics.Stopwatch.GetTimestamp() - t0);
            }
        }

        internal void MarkDeviceModified()
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            deviceDirty = true;
            hostDirty = false;
        }

        internal void SyncHostFromDevice()
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                // Fast path: device hasn't been written AND the host mirror already
                // exists → the host copy is current, nothing to do.
                // When hostBuffer is still null (device-only storage that was never
                // checked out to the host) we MUST fall through and allocate + copy,
                // otherwise GetElementAsFloat / PtrAtElement dereference a null
                // hostBuffer → NullReferenceException.
                if (!deviceDirty && hostBuffer != IntPtr.Zero)
                    return;

                long t0 = CudaProfileCounters.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                EnsureHostBuffer();
                AllocatorImpl.Context.BindCurrent(nativeCalls);
                nativeCalls.ThrowOnError(nativeCalls.cuMemcpyDtoHAsync(
                    hostBuffer,
                    deviceBuffer,
                    new UIntPtr((ulong)ByteLength),
                    AllocatorImpl.Stream.Handle));
                AllocatorImpl.Stream.Synchronize();
                deviceDirty = false;
                hostDirty = false;
                if (CudaProfileCounters.Enabled)
                    CudaProfileCounters.RecordSync("DtoH(sync)", ByteLength, System.Diagnostics.Stopwatch.GetTimestamp() - t0);
            }
        }

        internal void CopyDeviceFrom(CudaStorage src)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (src.ByteLength != ByteLength)
                throw new ArgumentException("CUDA device copy requires equal byte lengths.", nameof(src));

            CopyDeviceFrom(src, 0, 0, ByteLength);
        }

        // Cached device-pair peer accessibility. On topologies without P2P
        // (vGPU, no-NVLink consumer cards) a direct device-to-device copy is
        // invalid and must go through host memory instead.
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(int, int), bool> _peerAccessCache = new();

        /// <summary>
        /// Force all cross-GPU transfers through host staging, bypassing P2P
        /// entirely. Set TENSORSHARP_TP_DISABLE_P2P=1 to make peer-capable
        /// hardware take exactly the code path that no-peer hardware (A16 vGPU
        /// profiles, consumer cards) takes. Slower, but useful for isolating
        /// whether a multi-GPU defect lives in the P2P DMA path.
        /// </summary>
        internal static readonly bool DisableP2P =
            string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DISABLE_P2P"), "1", StringComparison.Ordinal);

        /// <summary>
        /// Called by <see cref="CudaP2PCommunicator"/> when the P2P DMA self-test
        /// detects that a device pair reports peer-accessible but the actual
        /// cuMemcpyPeer round-trip produces corrupt data (seen on some L4 PCIe
        /// topologies where IOMMU, BAR1 sizing, or switch configuration silently
        /// breaks P2P DMA). Forces the pair through host staging permanently.
        /// </summary>
        internal static void MarkPeerAccessFailed(int deviceA, int deviceB)
        {
            _peerAccessCache[(deviceA, deviceB)] = false;
            _peerAccessCache[(deviceB, deviceA)] = false;
        }

        private static bool CanAccessPeer(CudaNativeCalls calls, int srcDevice, int dstDevice)
        {
            if (srcDevice == dstDevice)
                return true;
            if (DisableP2P)
                return false;
            return _peerAccessCache.GetOrAdd((srcDevice, dstDevice), key =>
            {
                try
                {
                    int fwd = 0, rev = 0;
                    calls.cuDeviceCanAccessPeer(out fwd, key.Item1, key.Item2);
                    calls.cuDeviceCanAccessPeer(out rev, key.Item2, key.Item1);
                    return fwd == 1 && rev == 1;
                }
                catch
                {
                    calls.ThrowIfQuarantined();
                    return false; // conservative: stage through host
                }
            });
        }

        internal void CopyDeviceFrom(CudaStorage src, long destinationByteOffset, long sourceByteOffset, long byteCount)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (destinationByteOffset < 0 || sourceByteOffset < 0 || byteCount < 0 ||
                destinationByteOffset + byteCount > ByteLength ||
                sourceByteOffset + byteCount > src.ByteLength)
            {
                throw new ArgumentOutOfRangeException(nameof(byteCount));
            }

            if (byteCount == 0)
                return;

            if (TryCopyDeviceFrom(src, destinationByteOffset, sourceByteOffset, byteCount, null))
                return;

            // A real fallback needs staging. Re-admit both actual owners after this allocation.
            byte[] stage = new byte[byteCount];
            TryCopyDeviceFrom(src, destinationByteOffset, sourceByteOffset, byteCount, stage);
        }

        private bool TryCopyDeviceFrom(CudaStorage src, long destinationByteOffset,
            long sourceByteOffset, long byteCount, byte[] stage)
        {
            using var transfer = new DeviceTransfer(this, src);
            CudaNativeCalls calls = transfer.Calls;
            ThrowIfDisposed();
            src.ThrowIfDisposed();
            bool sameAllocator = ReferenceEquals(AllocatorImpl, src.AllocatorImpl);
            bool peer = !sameAllocator && CanAccessPeer(calls, src.DeviceId, DeviceId);
            if (!sameAllocator && !peer && stage == null)
                return false;
            src.EnsureDeviceCurrent();
            AllocatorImpl.Context.BindCurrent(calls);
            IntPtr dst = AddBytes(deviceBuffer, destinationByteOffset);
            IntPtr source = AddBytes(src.deviceBuffer, sourceByteOffset);
            if (sameAllocator)
            {
                calls.ThrowOnError(calls.cuMemcpyDtoDAsync(
                    dst,
                    source,
                    new UIntPtr((ulong)byteCount),
                    AllocatorImpl.Stream.Handle));
            }
            else
            {
                // Cross-GPU copy. Direct device-to-device (cuMemcpyDtoD, and even
                // cuMemcpyPeer) requires the two devices to be peer-accessible.
                // On GPUs without P2P — virtualized vGPU profiles (A16-16Q) or
                // consumer/workstation cards without NVLink — that raises CUDA
                // error 700 (illegal memory access) or silently transfers wrong
                // data. Use a direct peer copy only when the devices can access
                // each other; otherwise stage explicitly through host memory,
                // which works on every topology.
                src.SynchronizeDeviceWork();
                if (peer)
                {
                    AllocatorImpl.Context.BindCurrent(calls);
                    calls.ThrowOnError(calls.cuMemcpyPeerAsync(
                        dst, AllocatorImpl.Context.Handle,
                        source, src.AllocatorImpl.Context.Handle,
                        new UIntPtr((ulong)byteCount),
                        AllocatorImpl.Stream.Handle));
                    // The copy runs on OUR stream but reads the SOURCE device's
                    // buffer, and nothing orders the source stream against it:
                    // the caller may dispose the source tensor immediately, its
                    // buffer goes back to the pool, and the source rank's next
                    // kernel writes it WHILE the peer DMA is still reading — at
                    // tensor-parallel prefill sizes (MBs in flight) that
                    // corrupted most of the transfer (measured ~96% of bytes on
                    // 2x RTX 2000 Ada; the repeated-token TP garbage). Block
                    // until the copy completes, matching the synchronous
                    // host-staged fallback below.
                    AllocatorImpl.Stream.Synchronize();
                }
                else
                {
                    unsafe
                    {
                        fixed (byte* stagePtr = stage)
                        {
                            src.AllocatorImpl.Context.BindCurrent(calls);
                            calls.ThrowOnError(calls.cuMemcpyDtoH((IntPtr)stagePtr, source, new UIntPtr((ulong)byteCount)));
                            AllocatorImpl.Context.BindCurrent(calls);
                            calls.ThrowOnError(calls.cuMemcpyHtoD(dst, (IntPtr)stagePtr, new UIntPtr((ulong)byteCount)));
                        }
                    }
                }
            }

            MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// Device-to-device copy of <paramref name="rows"/> rows of
        /// <paramref name="innerBytes"/> contiguous bytes each, with independent
        /// source/destination row pitches, in a SINGLE strided kernel launch.
        /// Replaces the per-row <c>cuMemcpyDtoDAsync</c> loop the strided tensor
        /// copy used to issue (one driver submission per row — tens of thousands
        /// per prefill forward on WDDM). Same allocator only.
        /// </summary>
        internal bool TryCopyDeviceFrom2D(
            CudaStorage src,
            long destByteOffset, long srcByteOffset,
            long rows, long innerBytes,
            long destPitchBytes, long srcPitchBytes)
        {
            if (src == null || !ReferenceEquals(AllocatorImpl, src.AllocatorImpl))
                return false;
            var kernels = AllocatorImpl.Kernels;
            if (kernels == null)
                return false;
            if (rows <= 0 || innerBytes <= 0)
                return true;

            // Bounds: the last row's inner span must stay inside both buffers.
            long srcLast = checked(srcByteOffset + (rows - 1) * srcPitchBytes + innerBytes);
            long dstLast = checked(destByteOffset + (rows - 1) * destPitchBytes + innerBytes);
            if (srcByteOffset < 0 || destByteOffset < 0 || srcLast > src.ByteLength || dstLast > ByteLength)
                return false;

            using var transfer = new DeviceTransfer(this, src);
            ThrowIfDisposed();
            src.ThrowIfDisposed();
            src.EnsureDeviceCurrent();
            AllocatorImpl.Context.BindCurrent(transfer.Calls);
            IntPtr dst = AddBytes(deviceBuffer, destByteOffset);
            IntPtr source = AddBytes(src.deviceBuffer, srcByteOffset);
            kernels.LaunchCopy2DBytes(
                source, dst, rows, innerBytes, srcPitchBytes, destPitchBytes,
                AllocatorImpl.Stream.Handle);
            MarkDeviceModified();
            return true;
        }

        private sealed class DeviceTransfer : IDisposable
        {
            private readonly CudaStorage destination;
            private readonly CudaStorage source;
            private readonly NativeEffectLease lease;
            internal readonly CudaNativeCalls Calls;

            internal DeviceTransfer(CudaStorage destination, CudaStorage source)
            {
                destination.ThrowIfDisposed();
                source.ThrowIfDisposed();
                this.destination = destination;
                this.source = source;
                if (!ReferenceEquals(destination.nativeCalls.Api, source.nativeCalls.Api))
                    throw new InvalidOperationException("CUDA storage transfer requires one native API owner.");
                Calls = new CudaNativeCalls(this, NativeOwnerRole.Storage, destination.nativeCalls.Api,
                    destination.DeviceId, source.DeviceId);
                lease = Calls.EnterEffect();
            }

            public void Dispose()
            {
                try
                {
                    try { Calls.CompleteSafeRelease(lease); }
                    catch (NativeRuntimeQuarantinedException refusal) when (Calls.IsRetainedFailure(refusal)) { }
                }
                finally
                {
                    lease.Dispose();
                    GC.KeepAlive(destination);
                    GC.KeepAlive(source);
                }
            }
        }

        internal void SynchronizeDeviceWork()
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            AllocatorImpl.Context.BindCurrent(nativeCalls);
            AllocatorImpl.Stream.Synchronize();
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            SyncHostFromDevice();
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            int[] array = new int[length];
            int* source = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            EnsureHostBuffer();
            int* target = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        public override float GetElementAsFloat(long index)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            SyncHostFromDevice();
            ValidateElementRange(index, 1);

            return ElementType switch
            {
                DType.Float32 => ((float*)hostBuffer.ToPointer())[index],
                DType.Float64 => (float)((double*)hostBuffer.ToPointer())[index],
                DType.Int32 => ((int*)hostBuffer.ToPointer())[index],
                DType.UInt8 => ((byte*)hostBuffer.ToPointer())[index],
                DType.Float16 => (float)BitConverter.UInt16BitsToHalf(((ushort*)hostBuffer.ToPointer())[index]),
                _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
            };
        }

        public override float[] GetElementsAsFloat(long index, int length)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            SyncHostFromDevice();
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            float[] array = new float[length];
            float* source = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        public override void SetElementAsFloat(long index, float value)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            ValidateElementRange(index, 1);
            EnsureHostBuffer();
            switch (ElementType)
            {
                case DType.Float32:
                    ((float*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Float64:
                    ((double*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Int32:
                    ((int*)hostBuffer.ToPointer())[index] = (int)value;
                    break;
                case DType.UInt8:
                    ((byte*)hostBuffer.ToPointer())[index] = (byte)value;
                    break;
                case DType.Float16:
                    ((ushort*)hostBuffer.ToPointer())[index] = BitConverter.HalfToUInt16Bits((System.Half)value);
                    break;
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }

            hostDirty = true;
            deviceDirty = false;
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            EnsureHostBuffer();
            float* target = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotSupportedException("CUDA storage currently supports TensorSharp Float32/Float64/Int32/UInt8 host access.");
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            ThrowIfDisposed();
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));

            ValidateByteRange(storageIndex, byteCount);
            EnsureHostBuffer();
            Buffer.MemoryCopy(src.ToPointer(), HostPtrAtElementUnchecked(storageIndex).ToPointer(), byteCount, byteCount);
            hostDirty = true;
            deviceDirty = false;
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            ThrowIfDisposed();
            using var lease = nativeCalls.EnterEffect();
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));

            SyncHostFromDevice();
            ValidateByteRange(storageIndex, byteCount);
            Buffer.MemoryCopy(HostPtrAtElementUnchecked(storageIndex).ToPointer(), dst.ToPointer(), byteCount, byteCount);
        }

        private IntPtr HostPtrAtElementUnchecked(long index)
        {
            EnsureHostBuffer();
            return AddBytes(hostBuffer, checked(index * ElementType.Size()));
        }

        private void EnsureHostBuffer()
        {
            if (hostBuffer != IntPtr.Zero)
                return;

            long allocationSize = Math.Max(ByteLength, 1);
            hostBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)allocationSize, 64);
            if (hostBuffer == IntPtr.Zero)
                throw new OutOfMemoryException($"Failed to allocate {allocationSize} bytes of CUDA host mirror memory.");
        }

        private static IntPtr AddBytes(IntPtr pointer, long byteOffset)
        {
            return new IntPtr(pointer.ToInt64() + byteOffset);
        }

        private void ValidateElementRange(long index, long length)
        {
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            long byteOffset = checked(storageIndex * ElementType.Size());
            if (storageIndex < 0 || byteCount < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        private void ThrowIfDisposed()
        {
            if (deviceBuffer == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStorage));
        }
    }
}
