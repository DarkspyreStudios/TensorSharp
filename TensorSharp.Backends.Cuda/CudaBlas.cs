using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal static class CudaBlas
    {
        public static bool TryAddmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
            => TryAddmmCore(result, beta, src, alpha, m1, m2, false);

        public static bool TryAddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
            => TryAddmmCore(result, beta, src, alpha, m1, m2, true);

        private static bool TryAddmmCore(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2, bool batched)
        {
            using var operands = BlasOperands.TryCreate(result, src, m1, m2);
            if (operands == null) return false;
            try
            {
                var output = operands.Layouts[0];
                var source = operands.Layouts[1];
                var left = operands.Layouts[2];
                var right = operands.Layouts[3];
                if (operands.Storages.Any(storage => storage.ElementType != DType.Float32)) return false;
                int dimensions = batched ? 3 : 2;
                if (operands.Layouts.Any(layout => layout.Sizes.Length != dimensions)) return false;

                int first = batched ? 1 : 0;
                int batch = batched ? checked((int)left.Sizes[0]) : 1;
                int rows = checked((int)left.Sizes[first]);
                int shared = checked((int)left.Sizes[first + 1]);
                int cols = checked((int)right.Sizes[first + 1]);
                if (right.Sizes[first] != shared || output.Sizes[first] != rows || output.Sizes[first + 1] != cols
                    || source.Sizes[first] != rows || source.Sizes[first + 1] != cols
                    || batched && (right.Sizes[0] != batch || output.Sizes[0] != batch || source.Sizes[0] != batch))
                    return false;
                if (!IsRowMajorMatrix(left, batched) || !IsRowMajorMatrix(output, batched)) return false;
                if (!TryGetRightOperand(right, batched, out int transa, out int lda)) return false;

                CudaStorage resultStorage = operands.Storages[0];
                CudaStorage srcStorage = operands.Storages[1];
                if (beta != 0.0f)
                {
                    if (ReferenceEquals(srcStorage, resultStorage) && source.Offset == output.Offset)
                        resultStorage.EnsureDeviceCurrent();
                    else if (source.Contiguous && output.Contiguous && source.ElementCount == output.ElementCount)
                        resultStorage.CopyDeviceFrom(srcStorage);
                    else return false;
                }

                operands.Check();
                resultStorage.EnsureDeviceCurrent();
                operands.Storages[2].EnsureDeviceCurrent();
                operands.Storages[3].EnsureDeviceCurrent();
                operands.Check();

                // Preparation can enter graph/storage owners; only pointer access and submission share this short effect.
                using (var callEffect = operands.Calls.EnterEffect())
                using (var handleEffect = operands.BlasCalls.EnterEffect())
                {
                    operands.Check();
                    operands.Allocator.Context.BindCurrent(operands.BlasCalls);
                    IntPtr handle = operands.Blas.Handle;
                    if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaCublasHandle));
                    operands.BlasCalls.cublasSetStream(handle, operands.Stream.Handle).ThrowOnCublasError();
                    IntPtr aPtr = operands.Storages[3].DevicePtrAtElement(right.Offset);
                    IntPtr bPtr = operands.Storages[2].DevicePtrAtElement(left.Offset);
                    IntPtr cPtr = resultStorage.DevicePtrAtElement(output.Offset);
                    if (batched)
                        operands.BlasCalls.cublasSgemmStridedBatched(handle, transa, CublasApi.CUBLAS_OP_N,
                            cols, rows, shared, ref alpha, aPtr, lda, checked(right.Strides[0]),
                            bPtr, shared, checked(left.Strides[0]), ref beta, cPtr, cols,
                            checked(output.Strides[0]), batch).ThrowOnCublasError();
                    else
                        operands.BlasCalls.cublasSgemm(handle, transa, CublasApi.CUBLAS_OP_N,
                            cols, rows, shared, ref alpha, aPtr, lda, bPtr, shared,
                            ref beta, cPtr, cols).ThrowOnCublasError();
                }
                operands.Check();
                resultStorage.MarkDeviceModified();
                operands.Check();
                return true;
            }
            catch (Exception original)
            {
                operands.Failure = original;
                throw;
            }
        }

        private static bool IsRowMajorMatrix(Layout layout, bool batched)
        {
            int first = batched ? 1 : 0;
            return layout.Strides[first + 1] == 1 && layout.Strides[first] == layout.Sizes[first + 1]
                && (!batched || layout.Strides[0] == layout.Sizes[1] * layout.Sizes[2]);
        }

        private static bool TryGetRightOperand(Layout layout, bool batched, out int transa, out int lda)
        {
            int first = batched ? 1 : 0;
            int rows = checked((int)layout.Sizes[first]);
            int cols = checked((int)layout.Sizes[first + 1]);
            if (layout.Strides[first + 1] == 1 && layout.Strides[first] == cols)
            {
                transa = CublasApi.CUBLAS_OP_N;
                lda = cols;
                return true;
            }
            if (layout.Strides[first] == 1 && layout.Strides[first + 1] == rows)
            {
                transa = CublasApi.CUBLAS_OP_T;
                lda = rows;
                return true;
            }
            transa = 0;
            lda = 0;
            return false;
        }

        private sealed class Layout
        {
            internal readonly long[] Sizes;
            internal readonly long[] Strides;
            internal readonly long Offset;
            internal readonly long ElementCount;
            internal readonly bool Contiguous;

            internal Layout(Tensor tensor)
            {
                Sizes = tensor.Sizes.ToArray();
                Strides = tensor.Strides.ToArray();
                Offset = tensor.StorageOffset;
                ElementCount = tensor.ElementCount();
                Contiguous = tensor.IsContiguous();
            }
        }

        private sealed class BlasOperands : IDisposable
        {
            private readonly Tensor[] _tensors;
            private readonly CudaAllocator[] _allocators;
            private CudaOperationAdmission _admission;
            private bool _released;
            internal readonly CudaStorage[] Storages;
            internal readonly Layout[] Layouts;
            internal readonly CudaAllocator Allocator;
            internal readonly CudaCublasHandle Blas;
            internal readonly CudaStream Stream;
            internal readonly CudaNativeCalls Calls;
            internal readonly CudaNativeCalls BlasCalls;
            internal Exception Failure;

            private BlasOperands(Tensor[] tensors, CudaStorage[] storages)
            {
                _tensors = tensors;
                Storages = storages;
                _allocators = storages.Select(storage => storage.AllocatorImpl)
                    .Distinct<CudaAllocator>(ReferenceEqualityComparer.Instance).ToArray();
                Allocator = storages[0].AllocatorImpl;
                Blas = Allocator.Blas;
                Stream = Allocator.Stream;
                Calls = new CudaNativeCalls(this, NativeOwnerRole.Worker, Allocator.NativeCalls.Api,
                    _allocators.Select(allocator => allocator.DeviceId).Distinct().ToArray());
                try
                {
                    _admission = CudaOperationAdmission.Enter(this, _allocators);
                    BlasCalls = Blas.NativeCalls;
                    Layouts = tensors.Select(tensor => new Layout(tensor)).ToArray();
                    Check();
                }
                catch (Exception original)
                {
                    Failure = original;
                    Dispose();
                    throw;
                }
            }

            internal static BlasOperands TryCreate(Tensor result, Tensor src, Tensor m1, Tensor m2)
            {
                var tensors = new[] { result, src, m1, m2 };
                var storages = new CudaStorage[tensors.Length];
                for (int index = 0; index < tensors.Length; index++)
                {
                    storages[index] = tensors[index]?.Storage as CudaStorage;
                    if (storages[index] == null) return null;
                }
                return new BlasOperands(tensors, storages);
            }

            internal void Check()
            {
                Calls.ThrowIfQuarantined();
                foreach (CudaAllocator allocator in _allocators) _admission.ValidateAllocator(allocator);
            }

            public void Dispose()
            {
                if (_released) return;
                _released = true;
                List<Exception> failures = null;
                try
                {
                    using var effect = Calls.EnterEffect();
                    Calls.CompleteSafeRelease(effect);
                }
                catch (NativeRuntimeQuarantinedException refusal) when (Calls.IsRetainedFailure(refusal)) { }
                catch (Exception cleanup) { (failures ??= new()).Add(cleanup); }
                try { _admission?.Dispose(); }
                catch (Exception cleanup) { (failures ??= new()).Add(cleanup); }
                GC.KeepAlive(_tensors);
                if (failures == null) return;
                if (Failure != null) failures.Insert(0, Failure);
                if (failures.Count == 1) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failures[0]).Throw();
                throw new AggregateException(failures);
            }
        }
    }
}
