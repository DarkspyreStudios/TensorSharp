using System;
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed unsafe class MlxStorage : Storage
    {
        private readonly object sync = new();
        private IntPtr buffer;
        private MlxNative.MlxArray deviceArray;
        private bool hostDirty = true;
        private bool deviceDirty;
        private readonly NativeOwnerRegistration nativeOwner;
        private ReleaseAdmission releaseAdmission;

        public MlxStorage(MlxAllocator allocator, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));

            NativeOwnerRegistration registration = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Storage);
            registration.AttachMlxSharedRuntime();
            nativeOwner = registration;
        }

        internal MlxAllocator AllocatorImpl { get; }

        public int DeviceId => AllocatorImpl.DeviceId;

        internal override object ReferenceMutationGate
        {
            get
            {
                ValidateReferenceMutationBoundary();
                return sync;
            }
        }
        internal override void ValidateReferenceAddition() => nativeOwner.ThrowIfQuarantined();
        internal override bool IsRetainedFinalizerFailure(Exception error)
        {
            if (nativeOwner == null) return false;
            if (NativeQuarantineAuthority.IsRetainedFailure(nativeOwner, error)) return true;
            if (error is AggregateException aggregate)
                foreach (Exception inner in aggregate.InnerExceptions)
                    if (IsRetainedFinalizerFailure(inner)) return true;
            return error.InnerException != null && IsRetainedFinalizerFailure(error.InnerException);
        }

        internal override IDisposable AdmitFinalRelease()
        {
            if (nativeOwner == null) return null;
            ValidateReferenceMutationBoundary();
            lock (sync)
            {
                var admission = new ReleaseAdmission(this);
                releaseAdmission = admission;
                return admission;
            }
        }

        private void ValidateReferenceMutationBoundary()
        {
            if (!MlxWorker.Shared.IsOnWorkerThread)
                NativeQuarantineAuthority.ValidateMlxWorkerDispatch(nativeOwner);
        }

        protected override void Destroy()
        {
            if (nativeOwner == null) return;
            NativeMlxReleaseReservation reservation = releaseAdmission?.Reservation
                ?? throw new InvalidOperationException("MLX storage release requires its pre-decrement admission.");
            bool published = false;
            try
            {
                MlxWorker.Shared.Invoke(() =>
                {
                    using NativeEffectLease effect = reservation.EnterEffect();
                    effect.ValidateMlxRelease(this, reservation);
                    NativeRuntimeFailureStage stage = NativeRuntimeFailureStage.StorageRelease;
                    try
                    {
                        // Preserve the host mirror while native release is unproven.
                        ReleaseDeviceArray(ref deviceArray, ref stage,
                            static () =>
                            {
                                MlxNative.InstallCurrentErrorHandler();
                                MlxNative.SynchronizeAllUsedStreams();
                            }, MlxNative.FreeArray);
                        if (buffer != IntPtr.Zero)
                        {
                            NativeMemory.AlignedFree(buffer.ToPointer());
                            buffer = IntPtr.Zero;
                        }
                        effect.CompleteSafeRelease(this);
                    }
                    catch (Exception cleanup)
                    {
                        effect.PublishFailure(this, cleanup, stage);
                        published = true;
                        throw;
                    }
                });
            }
            catch (Exception error) when (!published && !IsRetainedFinalizerFailure(error))
            {
                // Queue rejection also leaves the consumed owner's native graph intact.
                try
                {
                    using NativeEffectLease effect = reservation.EnterEffect();
                    effect.PublishFailure(this, error, NativeRuntimeFailureStage.StorageRelease);
                }
                catch (Exception publication) { throw new AggregateException(error, publication); }
                throw;
            }
        }

        internal static void ReleaseDeviceArray(ref MlxNative.MlxArray array, ref NativeRuntimeFailureStage stage,
            Action synchronize, Action<MlxNative.MlxArray> release)
        {
            if (!array.IsValid) return;
            stage = NativeRuntimeFailureStage.Synchronization;
            synchronize();
            stage = NativeRuntimeFailureStage.StorageRelease;
            release(array);
            array = default;
        }

        private sealed class ReleaseAdmission(MlxStorage owner) : IDisposable
        {
            internal NativeMlxReleaseReservation Reservation { get; } = owner.nativeOwner.ReserveMlxRelease(owner);

            public void Dispose()
            {
                lock (owner.sync)
                {
                    Reservation.Dispose();
                    if (ReferenceEquals(owner.releaseAdmission, this)) owner.releaseAdmission = null;
                }
            }
        }

        public override string LocationDescription()
        {
            return $"MLX:{DeviceId}";
        }

        private T InvokeStorage<T>(Func<T> operation)
        {
            // Dispatch before taking the same gate used by final-reference admission.
            return MlxWorker.Shared.Invoke(() =>
            {
                lock (sync)
                {
                    ThrowIfDestroyed();
                    nativeOwner.ThrowIfQuarantined();
                    return operation();
                }
            });
        }

        private void InvokeStorage(Action operation)
            => InvokeStorage(() =>
            {
                operation();
                return 0;
            });

        public override IntPtr PtrAtElement(long index)
        {
            return InvokeStorage(() =>
            {
                ValidateElementRange(index, 0);
                EnsureHostReadableCore();
                hostDirty = true;
                return AddBytes(buffer, checked(index * ElementType.Size()));
            });
        }

        public override void EnsureHostReadable()
            => InvokeStorage(EnsureHostReadableCore);

        private void EnsureHostReadableCore()
        {
            EnsureHostBufferAllocated();
            if (!deviceDirty || !deviceArray.IsValid || ByteLength == 0)
                return;

            MlxNative.CopyArrayToHost(deviceArray, ElementType, buffer, ByteLength);
            deviceDirty = false;
            hostDirty = false;
        }

        internal MlxNative.MlxArray CreateArrayView(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));

            return InvokeStorage(() =>
            {
                if (!ReferenceEquals(tensor.GetLiveOwnedStorageForDisposal(), this))
                    throw new ObjectDisposedException(nameof(Tensor));
                EnsureDeviceCurrentCore();
                return MlxNative.AsStrided(deviceArray, ToIntArray(tensor.Sizes), ToLongArray(tensor.Strides), tensor.StorageOffset);
            });
        }

        internal static void SetDeviceResult(Tensor tensor, ref MlxNative.MlxArray output)
        {
            MlxStorage storage = (MlxStorage)tensor.Storage;
            MlxNative.MlxArray incoming = output;
            try
            {
                storage.InvokeStorage(() =>
                {
                    if (!ReferenceEquals(tensor.GetLiveOwnedStorageForDisposal(), storage))
                        throw new ObjectDisposedException(nameof(Tensor));
                    if (tensor.StorageOffset == 0 && storage.ElementCount == tensor.ElementCount())
                        storage.ReplaceDeviceArray(ref incoming);
                    else
                        storage.UpdateDeviceSlice(tensor, ref incoming);
                });
            }
            finally { output = incoming; }
        }

        private sealed class DeviceArrayChange(MlxStorage owner) : MlxNativeResources
        {
            // Retain the actual storage graph after its operation delegate unwinds.
            internal readonly MlxStorage Storage = owner;
            internal override NativeRuntimeFailureStage CleanupFailureStage => NativeRuntimeFailureStage.StorageRelease;
            internal MlxNative.MlxArray Incoming;
            internal MlxNative.MlxArray Temporary;
            internal MlxNative.MlxArray Replacement;
            internal bool IncomingTaken;
            internal bool Prepared;
        }

        internal void ReplaceDeviceArray(ref MlxNative.MlxArray array)
        {
            ThrowIfDestroyed();
            nativeOwner.ThrowIfQuarantined();
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            var resources = new DeviceArrayChange(this) { Incoming = array };
            try
            {
                InvokeStorage(() =>
                {
                    MlxWorker.Shared.InvokeNative(resources, effect =>
                    {
                        ThrowIfDestroyed();
                        nativeOwner.ThrowIfQuarantined();
                        resources.IncomingTaken = true;
                        // Slice updates require the storage array to stay one-dimensional.
                        resources.Replacement = MlxNative.Reshape(resources.Incoming, new[] { (int)ElementCount });
                        resources.Prepared = true;
                        return 0;
                    }, () => CompleteDeviceArrayChange(resources, fromDevice: true));
                });
            }
            finally
            {
                if (resources.IncomingTaken) array = default;
            }
        }

        internal void UpdateDeviceSlice(Tensor tensor, ref MlxNative.MlxArray update)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));
            if (!update.IsValid)
                throw new ArgumentException("MLX update array is empty.", nameof(update));
            if (!tensor.IsContiguous())
                throw new NotSupportedException("MLX slice updates require contiguous tensor views.");
            if (tensor.StorageOffset < 0 || tensor.StorageOffset > int.MaxValue)
                throw new NotSupportedException("MLX slice offsets larger than Int32.MaxValue are not supported yet.");
            if (tensor.ElementCount() > int.MaxValue)
                throw new NotSupportedException("MLX slice lengths larger than Int32.MaxValue are not supported yet.");

            var resources = new DeviceArrayChange(this) { Incoming = update };
            try
            {
                InvokeStorage(() =>
                {
                    if (!ReferenceEquals(tensor.GetLiveOwnedStorageForDisposal(), this))
                        throw new ObjectDisposedException(nameof(Tensor));
                    MlxWorker.Shared.InvokeNative(resources, effect =>
                    {
                        ThrowIfDestroyed();
                        nativeOwner.ThrowIfQuarantined();
                        resources.IncomingTaken = true;
                        EnsureDeviceCurrentCore();
                        int length = (int)tensor.ElementCount();
                        int start = (int)tensor.StorageOffset;
                        resources.Temporary = MlxNative.Reshape(resources.Incoming, new[] { length });
                        resources.Replacement = MlxNative.SliceUpdate(deviceArray, resources.Temporary, start, checked(start + length));
                        resources.Prepared = true;
                        return 0;
                    }, () => CompleteDeviceArrayChange(resources, fromDevice: true));
                });
            }
            finally
            {
                if (resources.IncomingTaken) update = default;
            }
        }

        public override void EnsureDeviceCurrent()
            => InvokeStorage(EnsureDeviceCurrentCore);

        private void EnsureDeviceCurrentCore()
        {
            ThrowIfDestroyed();
            nativeOwner.ThrowIfQuarantined();
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            if (deviceArray.IsValid && !hostDirty)
                return;

            var resources = new DeviceArrayChange(this);
            MlxWorker.Shared.InvokeNative(resources, effect =>
            {
                if (buffer != IntPtr.Zero)
                    resources.Replacement = MlxNative.NewArrayFromHost(buffer, new[] { (int)ElementCount }, ElementType);
                else
                    resources.Replacement = MlxNative.Full(new[] { (int)ElementCount }, 0f, ElementType);
                resources.Prepared = true;
                return 0;
            }, () => CompleteDeviceArrayChange(resources, fromDevice: false));
        }

        private void CompleteDeviceArrayChange(DeviceArrayChange resources, bool fromDevice)
        {
            MlxNative.FreeArrayReference(ref resources.Temporary);
            if (resources.IncomingTaken) MlxNative.FreeArrayReference(ref resources.Incoming);
            if (!resources.Prepared)
            {
                MlxNative.FreeArrayReference(ref resources.Replacement);
                return;
            }

            MlxNative.FreeArrayReference(ref deviceArray);
            deviceArray = resources.Replacement;
            resources.Replacement = default;
            hostDirty = false;
            deviceDirty = fromDevice;
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            ValidateElementRange(index, length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            int[] result = new int[length];
            return InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                int* src = (int*)AddBytes(buffer, checked(index * ElementType.Size()));
                for (int i = 0; i < length; i++)
                    result[i] = src[i];
                return result;
            });
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                hostDirty = true;
                int* dst = (int*)AddBytes(buffer, checked(index * ElementType.Size()));
                for (int i = 0; i < value.Length; i++)
                    dst[i] = value[i];
            });
        }

        public override float GetElementAsFloat(long index)
        {
            ValidateElementRange(index, 1);
            return InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                return ElementType switch
                {
                    DType.Float32 => ((float*)buffer)[index],
                    DType.Float64 => (float)((double*)buffer)[index],
                    DType.Float16 => (float)((half*)buffer)[index],
                    DType.Int32 => ((int*)buffer)[index],
                    DType.UInt8 => ((byte*)buffer)[index],
                    _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
                };
            });
        }

        /// <summary>
        /// Bulk host read. Written as ONE <see cref="EnsureHostReadable"/> plus a
        /// single typed copy loop rather than <c>length</c> calls to
        /// <see cref="GetElementAsFloat"/>: every one of those calls re-entered
        /// <c>ThrowIfDestroyed()</c> + <c>lock (sync)</c>, and the hot caller here is
        /// ModelBase.TensorToFloatArray on the LM head, i.e. 202,048 locked
        /// per-element reads for EVERY decoded token of Muse-Glimmer.
        ///
        /// Like the per-element path this is a pure read: it must NOT raise
        /// <c>hostDirty</c> (which is why it uses <c>buffer</c> directly instead of
        /// <see cref="PtrAtElement"/>, whose checkout implies a pending host write).
        /// </summary>
        public override float[] GetElementsAsFloat(long index, int length)
        {
            ValidateElementRange(index, length);
            float[] result = new float[length];
            if (length == 0)
                return result;

            return InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                IntPtr src = AddBytes(buffer, checked(index * ElementType.Size()));
                switch (ElementType)
                {
                    case DType.Float32:
                        new ReadOnlySpan<float>((float*)src, length).CopyTo(result);
                        break;
                    case DType.Float64:
                        {
                            double* typed = (double*)src;
                            for (int i = 0; i < length; i++)
                                result[i] = (float)typed[i];
                            break;
                        }
                    case DType.Float16:
                        {
                            half* typed = (half*)src;
                            for (int i = 0; i < length; i++)
                                result[i] = typed[i];
                            break;
                        }
                    case DType.Int32:
                        {
                            int* typed = (int*)src;
                            for (int i = 0; i < length; i++)
                                result[i] = typed[i];
                            break;
                        }
                    case DType.UInt8:
                        {
                            byte* typed = (byte*)src;
                            for (int i = 0; i < length; i++)
                                result[i] = typed[i];
                            break;
                        }
                    default:
                        throw new NotSupportedException("Element type " + ElementType + " not supported");
                }

                return result;
            });
        }

        public override void SetElementAsFloat(long index, float value)
        {
            ValidateElementRange(index, 1);
            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                switch (ElementType)
                {
                    case DType.Float32:
                        ((float*)buffer)[index] = value;
                        break;
                    case DType.Float64:
                        ((double*)buffer)[index] = value;
                        break;
                    case DType.Float16:
                        ((half*)buffer)[index] = value;
                        break;
                    case DType.Int32:
                        ((int*)buffer)[index] = (int)value;
                        break;
                    case DType.UInt8:
                        ((byte*)buffer)[index] = (byte)value;
                        break;
                    default:
                        throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
                hostDirty = true;
            });
        }

        /// <summary>
        /// Bulk host write, the mirror image of
        /// <see cref="GetElementsAsFloat"/>: one <see cref="EnsureHostReadable"/>
        /// (needed because a partial write must not discard the rest of a
        /// device-resident buffer) and one typed copy loop. The per-element
        /// spelling cost one <c>ThrowIfDestroyed()</c> + <c>lock (sync)</c> per
        /// element, which the Muse-Glimmer mmproj upload path hits ~1.85e9 times
        /// when its ~1.9B parameters are dequantized to F32.
        ///
        /// <c>hostDirty</c> is raised exactly once at the end, matching what the
        /// per-element path did on every write, so the next
        /// <see cref="EnsureDeviceCurrent"/> re-uploads.
        /// </summary>
        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            int length = value.Length;
            if (length == 0)
                return;

            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                IntPtr dst = AddBytes(buffer, checked(index * ElementType.Size()));
                switch (ElementType)
                {
                    case DType.Float32:
                        value.AsSpan().CopyTo(new Span<float>((float*)dst, length));
                        break;
                    case DType.Float64:
                        {
                            double* typed = (double*)dst;
                            for (int i = 0; i < length; i++)
                                typed[i] = value[i];
                            break;
                        }
                    case DType.Float16:
                        {
                            half* typed = (half*)dst;
                            for (int i = 0; i < length; i++)
                                typed[i] = value[i];
                            break;
                        }
                    case DType.Int32:
                        {
                            int* typed = (int*)dst;
                            for (int i = 0; i < length; i++)
                                typed[i] = (int)value[i];
                            break;
                        }
                    case DType.UInt8:
                        {
                            byte* typed = (byte*)dst;
                            for (int i = 0; i < length; i++)
                                typed[i] = (byte)value[i];
                            break;
                        }
                    default:
                        // Thrown before hostDirty is raised, as in the per-element path.
                        throw new NotSupportedException("Element type " + ElementType + " not supported");
                }

                hostDirty = true;
            });
        }

        public override void SetElementsAsHalf(long index, half[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Float16)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                hostDirty = true;
                half* dst = (half*)AddBytes(buffer, checked(index * ElementType.Size()));
                for (int i = 0; i < value.Length; i++)
                    dst[i] = value[i];
                hostDirty = true;
            });
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));
            ValidateByteRange(storageIndex, byteCount);
            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                Buffer.MemoryCopy(src.ToPointer(), AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), byteCount, byteCount);
                hostDirty = true;
            });
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));
            ValidateByteRange(storageIndex, byteCount);
            InvokeStorage(() =>
            {
                EnsureHostReadableCore();
                Buffer.MemoryCopy(AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), dst.ToPointer(), byteCount, byteCount);
            });
        }

        private static int[] ToIntArray(ReadOnlySpan<long> values)
        {
            int[] result = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > int.MaxValue)
                    throw new NotSupportedException("MLX tensor dimensions larger than Int32.MaxValue are not supported yet.");
                result[i] = (int)values[i];
            }

            return result;
        }

        private static long[] ToLongArray(ReadOnlySpan<long> values)
        {
            long[] result = new long[values.Length];
            values.CopyTo(result);
            return result;
        }

        private void EnsureHostBufferAllocated()
        {
            if (buffer != IntPtr.Zero)
                return;

            nuint allocationBytes = (nuint)Math.Max(ByteLength, 1);
            buffer = (IntPtr)NativeMemory.AlignedAlloc(allocationBytes, 64);
            NativeMemory.Clear(buffer.ToPointer(), allocationBytes);
        }

        private void ValidateElementRange(long index, long length)
        {
            ThrowIfDestroyed();
            nativeOwner.ThrowIfQuarantined();
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            ThrowIfDestroyed();
            nativeOwner.ThrowIfQuarantined();
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            long byteOffset = checked(storageIndex * ElementType.Size());
            if (byteOffset < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        private static IntPtr AddBytes(IntPtr ptr, long bytes)
        {
            return new IntPtr(ptr.ToInt64() + bytes);
        }
    }
}
