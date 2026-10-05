using System;
using TensorSharp.Core;

namespace TensorSharp.MLX
{
    [OpsClass]
    public sealed class MlxBasicOps
    {
        [RegisterOpStorageType("fill", typeof(MlxStorage))]
        public static void Fill(Tensor result, float value)
        {
            if (!CanUseNativeWriteTarget(result))
            {
                FallbackVoid("fill", result, value);
                return;
            }

            var resources = new MlxArrayResources(1, result);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray output = ref resources.Arrays[0];
                output = MlxNative.Full(ToIntArray(result.Sizes), value, result.ElementType);
                MlxStorage.SetDeviceResult(result, ref output);
            }, resources.Release);
        }

        [RegisterOpStorageType("copy", typeof(MlxStorage))]
        public static void Copy(Tensor result, Tensor src)
        {
            if (!CanUseNativeCopy(result, src))
            {
                FallbackVoid("copy", result, src);
                return;
            }

            var resources = new MlxArrayResources(3, result, src);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray casted = ref resources.Arrays[1];
                ref MlxNative.MlxArray contiguous = ref resources.Arrays[2];
                srcView = GetView(src);
                MlxNative.MlxArray copySource = srcView;
                if (src.ElementType != result.ElementType)
                {
                    casted = MlxNative.Astype(srcView, result.ElementType);
                    copySource = casted;
                }

                contiguous = MlxNative.Contiguous(copySource);
                MlxStorage.SetDeviceResult(result, ref contiguous);
            }, resources.Release);
        }

        [RegisterOpStorageType("addmm", typeof(MlxStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src, m1, m2))
                return FallbackTensor("addmm", writeTarget, beta, src, alpha, m1, m2);

            var resources = new MlxArrayResources(4, writeTarget, src, m1, m2);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray m1View = ref resources.Arrays[1];
                ref MlxNative.MlxArray m2View = ref resources.Arrays[2];
                ref MlxNative.MlxArray output = ref resources.Arrays[3];
                srcView = GetView(src);
                m1View = GetView(m1);
                m2View = GetView(m2);
                output = MlxNative.Addmm(srcView, m1View, m2View, alpha, beta);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("addmmbatch", typeof(MlxStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src, m1, m2))
                return FallbackTensor("addmmbatch", writeTarget, beta, src, alpha, m1, m2);

            var resources = new MlxArrayResources(4, writeTarget, src, m1, m2);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray m1View = ref resources.Arrays[1];
                ref MlxNative.MlxArray m2View = ref resources.Arrays[2];
                ref MlxNative.MlxArray output = ref resources.Arrays[3];
                srcView = GetView(src);
                m1View = GetView(m1);
                m2View = GetView(m2);
                output = MlxNative.Addmm(srcView, m1View, m2View, alpha, beta);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("abs", typeof(MlxStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => Unary("abs", result, src, MlxNative.MlxUnaryOp.Abs);

        [RegisterOpStorageType("neg", typeof(MlxStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => Unary("neg", result, src, MlxNative.MlxUnaryOp.Neg);

        [RegisterOpStorageType("exp", typeof(MlxStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => Unary("exp", result, src, MlxNative.MlxUnaryOp.Exp);

        [RegisterOpStorageType("log", typeof(MlxStorage))]
        public static Tensor Log(Tensor result, Tensor src) => Unary("log", result, src, MlxNative.MlxUnaryOp.Log);

        [RegisterOpStorageType("log1p", typeof(MlxStorage))]
        public static Tensor Log1p(Tensor result, Tensor src) => Unary("log1p", result, src, MlxNative.MlxUnaryOp.Log1p);

        [RegisterOpStorageType("floor", typeof(MlxStorage))]
        public static Tensor Floor(Tensor result, Tensor src) => Unary("floor", result, src, MlxNative.MlxUnaryOp.Floor);

        [RegisterOpStorageType("ceil", typeof(MlxStorage))]
        public static Tensor Ceil(Tensor result, Tensor src) => Unary("ceil", result, src, MlxNative.MlxUnaryOp.Ceil);

        [RegisterOpStorageType("cos", typeof(MlxStorage))]
        public static Tensor Cos(Tensor result, Tensor src) => Unary("cos", result, src, MlxNative.MlxUnaryOp.Cos);

        [RegisterOpStorageType("tanh", typeof(MlxStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => Unary("tanh", result, src, MlxNative.MlxUnaryOp.Tanh);

        [RegisterOpStorageType("sigmoid", typeof(MlxStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => Unary("sigmoid", result, src, MlxNative.MlxUnaryOp.Sigmoid);

        [RegisterOpStorageType("relu", typeof(MlxStorage))]
        public static Tensor Relu(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("relu", writeTarget, src);

            var resources = new MlxArrayResources(3, writeTarget, src);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray zero = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                srcView = GetView(src);
                zero = MlxNative.NewScalar(0.0f);
                output = MlxNative.Binary(MlxNative.MlxBinaryOp.Maximum, srcView, zero);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("SiLU", typeof(MlxStorage))]
        public static Tensor SiLU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("SiLU", writeTarget, src);

            var resources = new MlxArrayResources(3, writeTarget, src);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray sigmoid = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                srcView = GetView(src);
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.SiLU(srcView);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, srcView);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, srcView, sigmoid);
                }
                MlxStorage.SetDeviceResult(writeTarget, ref output);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("GELU", typeof(MlxStorage))]
        public static Tensor GELU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("GELU", writeTarget, src);

            var resources = new MlxArrayResources(2, writeTarget, src);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray output = ref resources.Arrays[1];
                srcView = GetView(src);
                if (!MlxCompiledOps.Disabled)
                    output = MlxCompiledOps.GeluTanh(srcView);
                else
                    output = Gelu(srcView);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("addt", typeof(MlxStorage))]
        public static Tensor AddTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("addt", result, lhs, rhs, MlxNative.MlxBinaryOp.Add);

        [RegisterOpStorageType("subt", typeof(MlxStorage))]
        public static Tensor SubTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("subt", result, lhs, rhs, MlxNative.MlxBinaryOp.Sub);

        [RegisterOpStorageType("mult", typeof(MlxStorage))]
        public static Tensor MulTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("mult", result, lhs, rhs, MlxNative.MlxBinaryOp.Mul);

        [RegisterOpStorageType("divt", typeof(MlxStorage))]
        public static Tensor DivTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("divt", result, lhs, rhs, MlxNative.MlxBinaryOp.Div);

        [RegisterOpStorageType("addv", typeof(MlxStorage))]
        public static Tensor AddValue(Tensor result, Tensor lhs, float rhs) => Scalar("addv", result, lhs, rhs, MlxNative.MlxBinaryOp.Add, false);

        [RegisterOpStorageType("subv", typeof(MlxStorage))]
        public static Tensor SubValue(Tensor result, Tensor lhs, float rhs) => Scalar("subv", result, lhs, rhs, MlxNative.MlxBinaryOp.Sub, false);

        [RegisterOpStorageType("rsubv", typeof(MlxStorage))]
        public static Tensor RSubValue(Tensor result, float lhs, Tensor rhs) => Scalar("rsubv", result, rhs, lhs, MlxNative.MlxBinaryOp.Sub, true);

        [RegisterOpStorageType("mulv", typeof(MlxStorage))]
        public static Tensor MulValue(Tensor result, Tensor lhs, float rhs) => Scalar("mulv", result, lhs, rhs, MlxNative.MlxBinaryOp.Mul, false);

        [RegisterOpStorageType("divv", typeof(MlxStorage))]
        public static Tensor DivValue(Tensor result, Tensor lhs, float rhs) => Scalar("divv", result, lhs, rhs, MlxNative.MlxBinaryOp.Div, false);

        [RegisterOpStorageType("rdivv", typeof(MlxStorage))]
        public static Tensor RDivValue(Tensor result, float lhs, Tensor rhs) => Scalar("rdivv", result, rhs, lhs, MlxNative.MlxBinaryOp.Div, true);

        [RegisterOpStorageType("SiLUMul", typeof(MlxStorage))]
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(gate, up))
                return FallbackTensor("SiLUMul", writeTarget, gate, up);

            var resources = new MlxArrayResources(5, writeTarget, gate, up);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray gateView = ref resources.Arrays[0];
                ref MlxNative.MlxArray upView = ref resources.Arrays[1];
                ref MlxNative.MlxArray sigmoid = ref resources.Arrays[2];
                ref MlxNative.MlxArray silu = ref resources.Arrays[3];
                ref MlxNative.MlxArray output = ref resources.Arrays[4];
                gateView = GetView(gate);
                upView = GetView(up);
                if (!MlxCompiledOps.Disabled)
                {
                    // Single fused kernel: silu(gate) * up.
                    output = MlxCompiledOps.SwiGLU(gateView, upView);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gateView);
                    silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gateView, sigmoid);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, upView);
                }
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("SiLUMulSplit", typeof(MlxStorage))]
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            if (gateUp == null)
                throw new ArgumentNullException(nameof(gateUp));

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(gateUp)
                || gateUp.DimensionCount != 2
                || halfDim <= 0
                || gateUp.Sizes[0] > int.MaxValue
                || gateUp.Sizes[1] < halfDim * 2L)
            {
                return FallbackTensor("SiLUMulSplit", writeTarget, gateUp, halfDim);
            }

            int rows = checked((int)gateUp.Sizes[0]);
            var resources = new MlxArrayResources(6, writeTarget, gateUp);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray gateUpView = ref resources.Arrays[0];
                ref MlxNative.MlxArray gate = ref resources.Arrays[1];
                ref MlxNative.MlxArray up = ref resources.Arrays[2];
                ref MlxNative.MlxArray sigmoid = ref resources.Arrays[3];
                ref MlxNative.MlxArray silu = ref resources.Arrays[4];
                ref MlxNative.MlxArray output = ref resources.Arrays[5];
                gateUpView = GetView(gateUp);
                gate = MlxNative.Slice(gateUpView, new[] { 0, 0 }, new[] { rows, halfDim }, new[] { 1, 1 });
                up = MlxNative.Slice(gateUpView, new[] { 0, halfDim }, new[] { rows, halfDim * 2 }, new[] { 1, 1 });
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.SwiGLU(gate, up);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gate);
                    silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gate, sigmoid);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, up);
                }
                MlxStorage.SetDeviceResult(writeTarget, ref output);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("GELUMul", typeof(MlxStorage))]
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(gate, up))
                return FallbackTensor("GELUMul", writeTarget, gate, up);

            var resources = new MlxArrayResources(4, writeTarget, gate, up);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray gateView = ref resources.Arrays[0];
                ref MlxNative.MlxArray upView = ref resources.Arrays[1];
                ref MlxNative.MlxArray gelu = ref resources.Arrays[2];
                ref MlxNative.MlxArray output = ref resources.Arrays[3];
                gateView = GetView(gate);
                upView = GetView(up);
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.GeGLU(gateView, upView);
                }
                else
                {
                    gelu = Gelu(gateView);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gelu, upView);
                }
                MlxStorage.SetDeviceResult(writeTarget, ref output);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("SigmoidMul", typeof(MlxStorage))]
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(x, gate))
                return FallbackTensor("SigmoidMul", writeTarget, x, gate);

            var resources = new MlxArrayResources(4, writeTarget, x, gate);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray xView = ref resources.Arrays[0];
                ref MlxNative.MlxArray gateView = ref resources.Arrays[1];
                ref MlxNative.MlxArray sigmoid = ref resources.Arrays[2];
                ref MlxNative.MlxArray output = ref resources.Arrays[3];
                xView = GetView(x);
                gateView = GetView(gate);
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.SigmoidMul(xView, gateView);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gateView);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, xView, sigmoid);
                }
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("softmax", typeof(MlxStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("softmax", writeTarget, src);

            var resources = new MlxArrayResources(2, writeTarget, src);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray output = ref resources.Arrays[1];
                srcView = GetView(src);
                output = MlxNative.SoftmaxLastAxis(srcView);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("rope", typeof(MlxStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (seqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            long cols = LastDimension(src);
            if (cols <= 0 || src.ElementCount() % cols != 0 || src.ElementCount() / cols > int.MaxValue)
                return FallbackTensor("rope", result, src, seqLen, rowOffset);

            int rows = (int)(src.ElementCount() / cols);
            int[] positions = new int[rows];
            for (int row = 0; row < positions.Length; row++)
                positions[row] = row % seqLen + rowOffset;

            using Tensor positionTensor = Tensor.FromArray(src.Allocator, positions);
            return RoPEEx(result, src, positionTensor, (int)cols, 0, 0, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false, false);
        }

        [RegisterOpStorageType("rope_ex", typeof(MlxStorage))]
        public static Tensor RoPEEx(
            Tensor result,
            Tensor src,
            Tensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(src)
                || !IsMlxInt32(positions)
                || !src.IsContiguous()
                || !positions.IsContiguous()
                || extFactor != 0.0f
                || attnFactor != 1.0f
                || betaFast != 0.0f
                || betaSlow != 0.0f
                || ropeDim <= 0
                || (ropeDim & 1) != 0)
            {
                return FallbackTensor(
                    "rope_ex",
                    writeTarget,
                    src,
                    positions,
                    ropeDim,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow,
                    addToResult,
                    invertPositions);
            }

            long cols = LastDimension(src);
            long rowsLong = src.ElementCount() / cols;
            if (cols <= 0
                || src.ElementCount() % cols != 0
                || cols > int.MaxValue
                || rowsLong > int.MaxValue
                || positions.ElementCount() != rowsLong
                || ropeDim > cols)
            {
                return FallbackTensor(
                    "rope_ex",
                    writeTarget,
                    src,
                    positions,
                    ropeDim,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow,
                    addToResult,
                    invertPositions);
            }

            bool traditional = (mode & 2) == 0;
            int rows = (int)rowsLong;
            int features = (int)cols;
            var resources = new MlxArrayResources(7, writeTarget, src, positions, result);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray resultView = ref resources.Arrays[0];
                ref MlxNative.MlxArray srcView = ref resources.Arrays[1];
                ref MlxNative.MlxArray positionsView = ref resources.Arrays[2];
                ref MlxNative.MlxArray flattened = ref resources.Arrays[3];
                ref MlxNative.MlxArray roped = ref resources.Arrays[4];
                ref MlxNative.MlxArray reshaped = ref resources.Arrays[5];
                ref MlxNative.MlxArray combined = ref resources.Arrays[6];
                srcView = GetView(src);
                positionsView = GetView(positions);
                flattened = MlxNative.Reshape(srcView, new[] { rows, 1, 1, features });
                roped = MlxNative.FastRopeDynamic(flattened, ropeDim, traditional, freqBase, freqScale, positionsView);
                reshaped = MlxNative.Reshape(roped, ToIntArray(src.Sizes));

                if (addToResult && result != null)
                {
                    resultView = GetView(result);
                    combined = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, reshaped, resultView);
                    MlxStorage.SetDeviceResult(writeTarget, ref combined);
                }
                else
                {
                    MlxStorage.SetDeviceResult(writeTarget, ref reshaped);
                }
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("scaled_dot_product_attention", typeof(MlxStorage))]
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            long[] outputSizes = new[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, query.ElementType, false, outputSizes);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(query, key, value)
                || !AreOptionalFloat32(mask)
                || query.DimensionCount != 4
                || key.DimensionCount != 4
                || value.DimensionCount != 4
                || (mask != null && mask.DimensionCount != 4))
            {
                return FallbackTensor("scaled_dot_product_attention", writeTarget, query, key, value, mask, scale);
            }

            var resources = new MlxArrayResources(10, writeTarget, query, key, value, mask);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray queryView = ref resources.Arrays[0];
                ref MlxNative.MlxArray keyView = ref resources.Arrays[1];
                ref MlxNative.MlxArray valueView = ref resources.Arrays[2];
                ref MlxNative.MlxArray maskView = ref resources.Arrays[3];
                ref MlxNative.MlxArray queryHeadMajor = ref resources.Arrays[4];
                ref MlxNative.MlxArray keyHeadMajor = ref resources.Arrays[5];
                ref MlxNative.MlxArray valueHeadMajor = ref resources.Arrays[6];
                ref MlxNative.MlxArray attentionHeadMajor = ref resources.Arrays[7];
                ref MlxNative.MlxArray attentionSeqMajor = ref resources.Arrays[8];
                ref MlxNative.MlxArray contiguous = ref resources.Arrays[9];
                int[] headMajorAxes = { 0, 2, 1, 3 };
                queryView = GetView(query);
                keyView = GetView(key);
                valueView = GetView(value);
                maskView = GetOptionalView(mask);

                queryHeadMajor = MlxNative.Transpose(queryView, headMajorAxes);
                keyHeadMajor = MlxNative.Transpose(keyView, headMajorAxes);
                valueHeadMajor = MlxNative.Transpose(valueView, headMajorAxes);
                attentionHeadMajor = MlxNative.FastScaledDotProductAttention(
                    queryHeadMajor,
                    keyHeadMajor,
                    valueHeadMajor,
                    scale,
                    mask == null ? string.Empty : "array",
                    maskView);
                attentionSeqMajor = MlxNative.Transpose(attentionHeadMajor, headMajorAxes);
                contiguous = MlxNative.Contiguous(attentionSeqMajor);
                MlxStorage.SetDeviceResult(writeTarget, ref contiguous);
                return writeTarget;
            }, resources.Release);
        }

        [RegisterOpStorageType("indexselect", typeof(MlxStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indices, bool isAdd)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, indices.Sizes[0], src.Sizes[1]);
            if (isAdd
                || !CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(src)
                || !IsMlxInt32(indices)
                || src.DimensionCount != 2
                || indices.DimensionCount != 1)
            {
                return FallbackTensor("indexselect", writeTarget, src, indices, isAdd);
            }

            var resources = new MlxArrayResources(4, writeTarget, src, indices);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray indicesView = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                ref MlxNative.MlxArray contiguous = ref resources.Arrays[3];
                srcView = GetView(src);
                indicesView = GetView(indices);
                output = MlxNative.TakeAxis(srcView, indicesView, 0);
                contiguous = MlxNative.Contiguous(output);
                MlxStorage.SetDeviceResult(writeTarget, ref contiguous);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("repeat_interleave", typeof(MlxStorage))]
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dim < 0 || dim >= src.DimensionCount)
                throw new ArgumentOutOfRangeException(nameof(dim));
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            long[] resultSizes = src.Sizes.ToArray();
            resultSizes[dim] *= repeats;
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src.Allocator, src.ElementType, false, resultSizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("repeat_interleave", writeTarget, src, repeats, dim);

            var resources = new MlxArrayResources(3, writeTarget, src);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray output = ref resources.Arrays[1];
                ref MlxNative.MlxArray contiguous = ref resources.Arrays[2];
                srcView = GetView(src);
                output = MlxNative.RepeatAxis(srcView, repeats, dim);
                contiguous = MlxNative.Contiguous(output);
                MlxStorage.SetDeviceResult(writeTarget, ref contiguous);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("layernorm", typeof(MlxStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src) || !AreOptionalFloat32(alpha, beta))
                return FallbackTensor("layernorm", writeTarget, src, alpha, beta, eps);

            var resources = new MlxArrayResources(4, writeTarget, src, alpha, beta);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray alphaView = ref resources.Arrays[1];
                ref MlxNative.MlxArray betaView = ref resources.Arrays[2];
                ref MlxNative.MlxArray output = ref resources.Arrays[3];
                srcView = GetView(src);
                alphaView = GetOptionalView(alpha);
                betaView = GetOptionalView(beta);
                output = MlxNative.FastLayerNorm(srcView, alphaView, betaView, eps);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("rmsnorm", typeof(MlxStorage))]
        public static Tensor RmsNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (beta != null || !CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src) || !AreOptionalFloat32(alpha))
                return FallbackTensor("rmsnorm", writeTarget, src, alpha, beta, eps);

            var resources = new MlxArrayResources(3, writeTarget, src, alpha);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray alphaView = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                srcView = GetView(src);
                alphaView = GetOptionalView(alpha);
                output = MlxNative.FastRmsNorm(srcView, alphaView, eps);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        [RegisterOpStorageType("add_causal_mask", typeof(MlxStorage))]
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (seqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            long cols = LastDimension(tensor);
            long rowsLong = cols == 0 ? 0 : tensor.ElementCount() / cols;
            if (!CanUseNativeWriteTarget(tensor)
                || !tensor.IsContiguous()
                || cols <= 0
                || tensor.ElementCount() % cols != 0
                || cols > int.MaxValue
                || rowsLong > int.MaxValue)
            {
                FallbackVoid("add_causal_mask", tensor, seqLen, startPos, maskedValue);
                return;
            }

            int rows = (int)rowsLong;
            int keyLength = (int)cols;
            var resources = new MlxArrayResources(15, tensor);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray tensorView = ref resources.Arrays[0];
                ref MlxNative.MlxArray scores2d = ref resources.Arrays[1];
                ref MlxNative.MlxArray rowRange = ref resources.Arrays[2];
                ref MlxNative.MlxArray row2d = ref resources.Arrays[3];
                ref MlxNative.MlxArray seqScalar = ref resources.Arrays[4];
                ref MlxNative.MlxArray rowInSequence = ref resources.Arrays[5];
                ref MlxNative.MlxArray startScalar = ref resources.Arrays[6];
                ref MlxNative.MlxArray threshold = ref resources.Arrays[7];
                ref MlxNative.MlxArray colRange = ref resources.Arrays[8];
                ref MlxNative.MlxArray col2d = ref resources.Arrays[9];
                ref MlxNative.MlxArray futureMask = ref resources.Arrays[10];
                ref MlxNative.MlxArray replacement = ref resources.Arrays[11];
                ref MlxNative.MlxArray addScalar = ref resources.Arrays[12];
                ref MlxNative.MlxArray maskedScores = ref resources.Arrays[13];
                ref MlxNative.MlxArray reshaped = ref resources.Arrays[14];
                tensorView = GetView(tensor);
                scores2d = MlxNative.Reshape(tensorView, new[] { rows, keyLength });
                rowRange = MlxNative.Arange(0, rows, 1, DType.Int32);
                row2d = MlxNative.Reshape(rowRange, new[] { rows, 1 });
                seqScalar = MlxNative.NewScalar(seqLen);
                rowInSequence = MlxNative.Remainder(row2d, seqScalar);
                startScalar = MlxNative.NewScalar(startPos);
                threshold = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, rowInSequence, startScalar);
                colRange = MlxNative.Arange(0, keyLength, 1, DType.Int32);
                col2d = MlxNative.Reshape(colRange, new[] { 1, keyLength });
                futureMask = MlxNative.Greater(col2d, threshold);

                if (float.IsNegativeInfinity(maskedValue))
                {
                    replacement = MlxNative.Full(new[] { rows, keyLength }, float.NegativeInfinity, DType.Float32);
                }
                else
                {
                    addScalar = MlxNative.NewScalar(maskedValue);
                    replacement = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, scores2d, addScalar);
                }

                maskedScores = MlxNative.Where(futureMask, replacement, scores2d);
                reshaped = MlxNative.Reshape(maskedScores, ToIntArray(tensor.Sizes));
                MlxStorage.SetDeviceResult(tensor, ref reshaped);
            }, resources.Release);
        }

        private static Tensor Unary(string opName, Tensor result, Tensor src, MlxNative.MlxUnaryOp op)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor(opName, writeTarget, src);

            // The sub-graph stays on the worker; native calls retain separate admission.
            var resources = new MlxArrayResources(2, writeTarget, src);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray srcView = ref resources.Arrays[0];
                ref MlxNative.MlxArray output = ref resources.Arrays[1];
                srcView = GetView(src);
                output = MlxNative.Unary(op, srcView);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        private static Tensor Binary(string opName, Tensor result, Tensor lhs, Tensor rhs, MlxNative.MlxBinaryOp op)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(lhs, rhs))
                return FallbackTensor(opName, writeTarget, lhs, rhs);

            var resources = new MlxArrayResources(3, writeTarget, lhs, rhs);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray lhsView = ref resources.Arrays[0];
                ref MlxNative.MlxArray rhsView = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                lhsView = GetView(lhs);
                rhsView = GetView(rhs);
                output = MlxNative.Binary(op, lhsView, rhsView);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        private static Tensor Scalar(string opName, Tensor result, Tensor tensor, float scalar, MlxNative.MlxBinaryOp op, bool scalarIsLhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, tensor, false, tensor.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(tensor))
                return scalarIsLhs
                    ? FallbackTensor(opName, writeTarget, scalar, tensor)
                    : FallbackTensor(opName, writeTarget, tensor, scalar);

            var resources = new MlxArrayResources(3, writeTarget, tensor);
            MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray tensorView = ref resources.Arrays[0];
                ref MlxNative.MlxArray scalarArray = ref resources.Arrays[1];
                ref MlxNative.MlxArray output = ref resources.Arrays[2];
                tensorView = GetView(tensor);
                scalarArray = MlxNative.NewScalar(scalar);
                output = scalarIsLhs
                    ? MlxNative.Binary(op, scalarArray, tensorView)
                    : MlxNative.Binary(op, tensorView, scalarArray);
                MlxStorage.SetDeviceResult(writeTarget, ref output);
            }, resources.Release);
            return writeTarget;
        }

        private static MlxNative.MlxArray Gelu(MlxNative.MlxArray input)
        {
            var resources = new MlxArrayResources(13);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray coeffCubic = ref resources.Arrays[0];
                ref MlxNative.MlxArray coeffInner = ref resources.Arrays[1];
                ref MlxNative.MlxArray one = ref resources.Arrays[2];
                ref MlxNative.MlxArray half = ref resources.Arrays[3];
                ref MlxNative.MlxArray squared = ref resources.Arrays[4];
                ref MlxNative.MlxArray cubed = ref resources.Arrays[5];
                ref MlxNative.MlxArray scaledCubic = ref resources.Arrays[6];
                ref MlxNative.MlxArray inner = ref resources.Arrays[7];
                ref MlxNative.MlxArray scaledInner = ref resources.Arrays[8];
                ref MlxNative.MlxArray tanh = ref resources.Arrays[9];
                ref MlxNative.MlxArray onePlusTanh = ref resources.Arrays[10];
                ref MlxNative.MlxArray halfInput = ref resources.Arrays[11];
                ref MlxNative.MlxArray output = ref resources.Arrays[12];
                coeffCubic = MlxNative.NewScalar(0.044715f);
                coeffInner = MlxNative.NewScalar(0.7978845608f);
                one = MlxNative.NewScalar(1.0f);
                half = MlxNative.NewScalar(0.5f);

                squared = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, input);
                cubed = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, squared, input);
                scaledCubic = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, cubed, coeffCubic);
                inner = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, input, scaledCubic);
                scaledInner = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inner, coeffInner);
                tanh = MlxNative.Unary(MlxNative.MlxUnaryOp.Tanh, scaledInner);
                onePlusTanh = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, one, tanh);
                halfInput = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, half);
                output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, halfInput, onePlusTanh);
                resources.ReturnedIndex = 12;
                return output;
            }, resources.Release);
        }

        private static bool CanUseNativeCopy(Tensor result, Tensor src)
        {
            return result != null &&
                src != null &&
                result.Storage is MlxStorage &&
                src.Storage is MlxStorage &&
                result.IsContiguous();
        }

        private static bool CanUseNativeWriteTarget(Tensor tensor)
        {
            return tensor != null &&
                tensor.Storage is MlxStorage &&
                tensor.ElementType == DType.Float32 &&
                tensor.IsContiguous();
        }

        private static bool AreFloat32(params Tensor[] tensors)
        {
            foreach (Tensor tensor in tensors)
            {
                if (tensor == null || tensor.Storage is not MlxStorage || tensor.ElementType != DType.Float32)
                    return false;
            }

            return true;
        }

        private static bool AreOptionalFloat32(params Tensor[] tensors)
        {
            foreach (Tensor tensor in tensors)
            {
                if (tensor != null && (tensor.Storage is not MlxStorage || tensor.ElementType != DType.Float32))
                    return false;
            }

            return true;
        }

        private static bool IsMlxInt32(Tensor tensor)
        {
            return tensor != null && tensor.Storage is MlxStorage && tensor.ElementType == DType.Int32;
        }

        private static MlxNative.MlxArray GetView(Tensor tensor)
        {
            return ((MlxStorage)tensor.Storage).CreateArrayView(tensor);
        }

        private static MlxNative.MlxArray GetOptionalView(Tensor tensor)
        {
            return tensor == null ? default : GetView(tensor);
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

        private static long LastDimension(Tensor tensor)
        {
            return tensor.DimensionCount == 0 ? 1 : tensor.Sizes[tensor.DimensionCount - 1];
        }

        private static Tensor FallbackTensor(string opName, params object[] args)
        {
            return (Tensor)MlxCpuFallback.Invoke(opName, MlxFallbackReturnKind.Tensor, new[] { 0 }, args);
        }

        private static void FallbackVoid(string opName, params object[] args)
        {
            MlxCpuFallback.Invoke(opName, MlxFallbackReturnKind.Void, new[] { 0 }, args);
        }
    }
}
