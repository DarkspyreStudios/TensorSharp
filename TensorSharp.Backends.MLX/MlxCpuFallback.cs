using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using TensorSharp.Cpu;
using TensorSharp.Core;

namespace TensorSharp.MLX
{
    internal enum MlxFallbackReturnKind
    {
        Void,
        Tensor,
        Raw
    }

    internal static class MlxCpuFallback
    {
        private static readonly CpuAllocator CpuAllocator = new(BlasEnum.DotNet);

        // Ops that have already reported their CPU fallback (once per op name per process).
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> WarnedOps =
            new(StringComparer.Ordinal);

        private static void WarnFallback(string opName, string consequence)
        {
            if (!WarnedOps.TryAdd(opName, true))
                return;

            try
            {
                Console.Error.WriteLine(
                    $"[mlx] op '{opName}' has no MLX GPU implementation for these arguments; it runs as {consequence}. " +
                    "Reported once per op.");
            }
            catch
            {
                // Diagnostics must never break op dispatch.
            }
        }

        public static object Invoke(string opName, MlxFallbackReturnKind returnKind, int[] modifiedTensorIndexes, object[] args)
        {
            var resources = new FallbackResources(args);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                resources.CaptureArguments();
                object result = InvokeCore(opName, returnKind, modifiedTensorIndexes, resources);
                resources.OperationSucceeded = true;
                return result;
            }, resources.Release);
        }

        private static object InvokeCore(string opName, MlxFallbackReturnKind returnKind,
            int[] modifiedTensorIndexes, FallbackResources resources)
        {
            object[] args = resources.OperationArgs;
            if (string.Equals(opName, "SiLUMulSplit", StringComparison.Ordinal))
            {
                WarnFallback(opName, "an unfused two-op decomposition (slower than the fused kernel)");
                Tensor result = SiLUMulSplit((Tensor)args[0], (Tensor)args[1], (int)args[2], resources);
                return args[0] is Tensor ? resources.OriginalArgs[0] : result;
            }

            if (string.Equals(opName, "scaled_dot_product_attention", StringComparison.Ordinal))
            {
                WarnFallback(opName, "element-by-element CPU attention (orders of magnitude slower)");
                Tensor result = ScaledDotProductAttention((Tensor)args[0], (Tensor)args[1], (Tensor)args[2],
                    (Tensor)args[3], (Tensor)args[4], (float)args[5], resources);
                return args[0] is Tensor ? resources.OriginalArgs[0] : result;
            }

            WarnFallback(opName, "an element-by-element CPU fallback with a GPU-to-host round-trip (orders of magnitude slower)");

            object returnValue = InvokeCpu(opName, resources);
            foreach (int modifiedTensorIndex in modifiedTensorIndexes)
            {
                if (modifiedTensorIndex >= 0 &&
                    modifiedTensorIndex < args.Length &&
                    args[modifiedTensorIndex] is Tensor modifiedTensor &&
                    resources.MappedTensors.TryGetValue(modifiedTensor, out Tensor cpuTensor))
                {
                    CopyLogical(modifiedTensor, cpuTensor);
                }
            }

            if (returnKind == MlxFallbackReturnKind.Raw)
                return returnValue;

            if (returnKind == MlxFallbackReturnKind.Void)
                return null;

            if (args.Length > 0 && args[0] is Tensor)
                return resources.OriginalArgs[0];

            if (returnValue is Tensor cpuReturn)
            {
                resources.PendingResult = CreateMlxLike(cpuReturn, args);
                CopyLogical(resources.PendingResult, cpuReturn);
                return resources.PendingResult;
            }

            return null;
        }

        private static object InvokeCpu(string opName, FallbackResources resources)
        {
            object[] args = resources.OperationArgs;
            object[] cpuArgs = resources.CpuArgs;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] is Tensor tensor)
                {
                    if (!resources.MappedTensors.TryGetValue(tensor, out Tensor cpuTensor))
                    {
                        resources.PendingCpu = new Tensor(CpuAllocator, tensor.ElementType, tensor.Sizes);
                        CopyLogical(resources.PendingCpu, tensor);
                        resources.MappedTensors.Add(tensor, resources.PendingCpu);
                        cpuTensor = resources.PendingCpu;
                        resources.PendingCpu = null;
                    }

                    cpuArgs[i] = cpuTensor;
                }
                else
                {
                    cpuArgs[i] = args[i];
                }
            }

            object result = OpRegistry.Invoke(opName, cpuArgs);
            resources.CpuReturn = result as Tensor;
            return result;
        }

        private static Tensor CreateMlxLike(Tensor source, object[] originalArgs)
        {
            IAllocator allocator = null;
            foreach (object arg in originalArgs)
            {
                if (arg is Tensor tensor && tensor.Storage is MlxStorage)
                {
                    allocator = tensor.Allocator;
                    break;
                }
            }

            allocator ??= new MlxAllocator();
            return new Tensor(allocator, source.ElementType, source.Sizes);
        }

        internal static void CopyLogical(Tensor destination, Tensor source)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (destination.ElementType != source.ElementType)
                throw new InvalidOperationException("Source and destination tensors must have the same element type.");
            if (destination.ElementCount() != source.ElementCount())
                throw new InvalidOperationException("Source and destination tensors must have the same number of elements.");
            if (destination.DimensionCount != source.DimensionCount)
                throw new InvalidOperationException("Source and destination tensors must have the same rank.");

            for (int i = 0; i < source.DimensionCount; i++)
            {
                if (destination.Sizes[i] != source.Sizes[i])
                    throw new InvalidOperationException("Source and destination tensors must have the same shape.");
            }

            if (source.DimensionCount == 0)
            {
                CopyElement(destination, destination.StorageOffset, source, source.StorageOffset);
                return;
            }

            CopyRecursive(destination, source, 0, destination.StorageOffset, source.StorageOffset);
        }

        private static void CopyRecursive(Tensor destination, Tensor source, int dimension, long destinationOffset, long sourceOffset)
        {
            if (dimension == source.DimensionCount)
            {
                CopyElement(destination, destinationOffset, source, sourceOffset);
                return;
            }

            long size = source.Sizes[dimension];
            long sourceStride = source.Strides[dimension];
            long destinationStride = destination.Strides[dimension];
            for (long i = 0; i < size; i++)
            {
                CopyRecursive(
                    destination,
                    source,
                    dimension + 1,
                    destinationOffset + i * destinationStride,
                    sourceOffset + i * sourceStride);
            }
        }

        private static unsafe void CopyElement(Tensor destination, long destinationOffset, Tensor source, long sourceOffset)
        {
            long byteCount = source.ElementType.Size();
            byte* tmp = stackalloc byte[(int)byteCount];
            source.Storage.CopyFromStorage((IntPtr)tmp, sourceOffset, byteCount);
            destination.Storage.CopyToStorage(destinationOffset, (IntPtr)tmp, byteCount);
        }

        private static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim, FallbackResources resources)
        {
            if (gateUp == null)
                throw new ArgumentNullException(nameof(gateUp));
            if (halfDim <= 0 || gateUp.DimensionCount != 2 || gateUp.Sizes[1] < halfDim * 2L)
                throw new ArgumentException("SiLUMulSplit expects a [tokens, 2*halfDim] tensor.", nameof(gateUp));

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            if (result == null) resources.PendingResult = writeTarget;
            Tensor gate = resources.GateView = gateUp.Narrow(1, 0, halfDim);
            Tensor up = resources.UpView = gateUp.Narrow(1, halfDim, halfDim);
            Ops.Copy(writeTarget, gate);
            Ops.SiLUMul(writeTarget, writeTarget, up);
            return writeTarget;
        }

        private static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask,
            float scale, FallbackResources resources)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (key == null)
                throw new ArgumentNullException(nameof(key));
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (query.ElementType != DType.Float32 || key.ElementType != DType.Float32 || value.ElementType != DType.Float32)
                throw new NotSupportedException("MLX fallback scaled-dot-product attention currently supports Float32 tensors only.");
            if (query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4)
                throw new ArgumentException("Scaled-dot-product attention expects query/key/value tensors with shape [batch, heads, seq, dim].");
            if (query.Sizes[0] != key.Sizes[0] || query.Sizes[0] != value.Sizes[0] ||
                query.Sizes[1] != key.Sizes[1] || query.Sizes[1] != value.Sizes[1] ||
                key.Sizes[2] != value.Sizes[2] || query.Sizes[3] != key.Sizes[3])
            {
                throw new InvalidOperationException("Scaled-dot-product attention tensor shapes are incompatible.");
            }

            long batch = query.Sizes[0];
            long heads = query.Sizes[1];
            long queryLen = query.Sizes[2];
            long keyLen = key.Sizes[2];
            long headDim = query.Sizes[3];
            long valueDim = value.Sizes[3];
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, DType.Float32, false, batch, heads, queryLen, valueDim);
            if (result == null) resources.PendingResult = writeTarget;

            if (keyLen > int.MaxValue)
                throw new NotSupportedException("MLX fallback attention does not support key lengths above Int32.MaxValue.");

            float[] scores = new float[(int)keyLen];
            for (long b = 0; b < batch; b++)
            {
                for (long h = 0; h < heads; h++)
                {
                    for (long q = 0; q < queryLen; q++)
                    {
                        float max = float.NegativeInfinity;
                        for (long k = 0; k < keyLen; k++)
                        {
                            float dot = 0.0f;
                            for (long d = 0; d < headDim; d++)
                                dot += query.GetElementAsFloat(b, h, q, d) * key.GetElementAsFloat(b, h, k, d);

                            float score = dot * scale + ReadAttentionMask(mask, b, h, q, k);
                            scores[k] = score;
                            if (score > max)
                                max = score;
                        }

                        float sum = 0.0f;
                        for (long k = 0; k < keyLen; k++)
                        {
                            float exp = MathF.Exp(scores[k] - max);
                            scores[k] = exp;
                            sum += exp;
                        }

                        float invSum = sum == 0.0f ? 0.0f : 1.0f / sum;
                        for (long d = 0; d < valueDim; d++)
                        {
                            float acc = 0.0f;
                            for (long k = 0; k < keyLen; k++)
                                acc += scores[k] * invSum * value.GetElementAsFloat(b, h, k, d);
                            writeTarget.SetElementAsFloat(acc, b, h, q, d);
                        }
                    }
                }
            }

            return writeTarget;
        }

        private static float ReadAttentionMask(Tensor mask, long b, long h, long q, long k)
        {
            if (mask == null)
                return 0.0f;
            if (mask.ElementType != DType.Float32)
                throw new NotSupportedException("Attention mask must be Float32.");

            return mask.DimensionCount switch
            {
                2 => mask.GetElementAsFloat(q, k),
                3 => mask.GetElementAsFloat(b, q, k),
                4 => mask.GetElementAsFloat(b, h, q, k),
                _ => throw new NotSupportedException("Attention mask must be rank 2, 3, or 4."),
            };
        }

        private sealed class FallbackResources : MlxNativeResources
        {
            internal readonly object[] OriginalArgs;
            internal readonly object[] OperationArgs;
            internal readonly object[] CpuArgs;
            private readonly Tensor[] inputViews;
            internal readonly Dictionary<Tensor, Tensor> MappedTensors;
            private readonly Dictionary<Tensor, bool> releaseReceipts;
            internal Tensor PendingCpu;
            internal Tensor CpuReturn;
            internal Tensor PendingResult;
            internal Tensor GateView;
            internal Tensor UpView;
            internal bool OperationSucceeded;

            internal FallbackResources(object[] args)
            {
                OriginalArgs = (object[])args.Clone();
                OperationArgs = new object[args.Length];
                CpuArgs = new object[args.Length];
                inputViews = new Tensor[args.Length];
                MappedTensors = new Dictionary<Tensor, Tensor>(args.Length, ReferenceEqualityComparer.Instance);
                releaseReceipts = new Dictionary<Tensor, bool>(checked(args.Length * 2 + 5), ReferenceEqualityComparer.Instance);
            }

            internal override bool CleanupRequiresOrdinaryEffect => false;
            internal override NativeRuntimeFailureStage CleanupFailureStage => NativeRuntimeFailureStage.StorageRelease;
            internal override bool HasNativeResources
            {
                get
                {
                    if (PendingCpu != null || CpuReturn != null || PendingResult != null || GateView != null
                        || UpView != null || MappedTensors.Count != 0) return true;
                    foreach (Tensor view in inputViews)
                        if (view != null) return true;
                    return false;
                }
            }

            internal void CaptureArguments()
            {
                for (int i = 0; i < OriginalArgs.Length; i++)
                {
                    if (OriginalArgs[i] is not Tensor tensor)
                    {
                        OperationArgs[i] = OriginalArgs[i];
                        continue;
                    }
                    bool reused = false;
                    for (int prior = 0; prior < i; prior++)
                        if (ReferenceEquals(OriginalArgs[prior], tensor))
                        {
                            OperationArgs[i] = OperationArgs[prior];
                            reused = true;
                            break;
                        }
                    if (reused) continue;
                    if (tensor.GetLiveOwnedStorageForDisposal() == null)
                        throw new ObjectDisposedException(nameof(Tensor));
                    inputViews[i] = tensor.CopyRef();
                    OperationArgs[i] = inputViews[i];
                }
            }

            internal void Release()
            {
                Exception failure = null;
                ReleaseTensor(ref PendingCpu, ref failure);
                bool mappedReleased = true;
                foreach (Tensor mapped in MappedTensors.Values)
                {
                    Tensor owned = mapped;
                    ReleaseTensor(ref owned, ref failure);
                    mappedReleased &= owned == null;
                }
                if (mappedReleased) MappedTensors.Clear();
                ReleaseTensor(ref CpuReturn, ref failure);
                ReleaseTensor(ref GateView, ref failure);
                ReleaseTensor(ref UpView, ref failure);
                for (int i = 0; i < inputViews.Length; i++) ReleaseTensor(ref inputViews[i], ref failure);
                if (!OperationSucceeded) ReleaseTensor(ref PendingResult, ref failure);
                Array.Clear(OperationArgs);
                Array.Clear(CpuArgs);
                if (failure != null) ExceptionDispatchInfo.Capture(failure).Throw();
                // Only checked temporary release transfers the result held by the worker's return value.
                PendingResult = null;
            }

            private void ReleaseTensor(ref Tensor tensor, ref Exception failure)
            {
                if (tensor == null) return;
                if (releaseReceipts.TryGetValue(tensor, out bool released))
                {
                    if (released) tensor = null;
                    return;
                }
                releaseReceipts.Add(tensor, false);
                try
                {
                    tensor.Dispose();
                    releaseReceipts[tensor] = true;
                    tensor = null;
                }
                catch (Exception error) { failure = MlxNative.JoinNativeErrors(failure, error); }
            }
        }
    }
}
