using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

internal sealed class CudaNativeCalls
{
    private readonly NativeOwnerRegistration _registration;
    internal ICudaNativeApi Api { get; }

    internal CudaNativeCalls(object actualOwner, NativeOwnerRole role, ICudaNativeApi api, params int[] deviceOrdinals)
    {
        ArgumentNullException.ThrowIfNull(api);
        ArgumentNullException.ThrowIfNull(deviceOrdinals);
        deviceOrdinals = (int[])deviceOrdinals.Clone();
        foreach (int ordinal in deviceOrdinals) ArgumentOutOfRangeException.ThrowIfNegative(ordinal);
        Api = api;
        _registration = NativeQuarantineAuthority.Register(actualOwner, role);
        if (deviceOrdinals.Length == 0) _registration.AttachCudaUnresolvedRuntime();
        else foreach (int ordinal in deviceOrdinals) _registration.AttachCudaPrimaryDevice(ordinal);
    }

    internal NativeEffectLease EnterEffect() => _registration.EnterEffect();
    internal void ThrowIfQuarantined() => _registration.ThrowIfQuarantined();
    internal void ValidateSafeRelease(NativeEffectLease lease) => lease.ValidateSafeRelease(_registration.Owner);
    internal void CompleteSafeRelease(NativeEffectLease lease) => lease.CompleteSafeRelease(_registration.Owner);
    internal NativeRuntimeFailure PublishFailure(NativeEffectLease lease, Exception error, NativeRuntimeFailureStage stage)
        => lease.PublishFailure(_registration.Owner, error, stage);

    internal void ThrowOnError(int code)
    {
        if (code == 0) return;
        using var lease = EnterEffect();
        throw DriverError(code);
    }

    private Exception DriverError(int code)
    {
        string message = "Unknown CUDA error";
        try
        {
            if (Api.cuGetErrorString(code, out IntPtr text) == 0 && text != IntPtr.Zero)
                message = Marshal.PtrToStringAnsi(text) ?? message;
        }
        catch (Exception diagnosticFailure)
        {
            return new AggregateException(new CudaException(code, message), diagnosticFailure);
        }
        return new CudaException(code, message);
    }

    private void RejectUnsafeResult(NativeEffectLease lease, int result, NativeRuntimeFailureStage stage, bool cublas)
    {
        if (result == 0) return;
        Exception error = cublas ? CublasError(result) : DriverError(result);
        PublishFailure(lease, error, stage);
        throw error;
    }

    private static CudaException CublasError(int result)
    {
        try { result.ThrowOnCublasError(); }
        catch (CudaException error) { return error; }
        throw new InvalidOperationException("The cuBLAS error decoder did not reject failure.");
    }

    internal int cuInit(uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuInit(flags);
        return result;
    }

    internal int cuDeviceGet(out int device, int ordinal)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceGet(out device, ordinal);
        return result;
    }

    internal int cuDeviceGetCount(out int count)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceGetCount(out count);
        return result;
    }

    internal int cuDeviceGetName(byte[] name, int len, int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceGetName(name, len, device);
        return result;
    }

    internal int cuDeviceTotalMem(out UIntPtr bytes, int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceTotalMem(out bytes, device);
        return result;
    }

    internal int cuMemGetInfo(out UIntPtr free, out UIntPtr total)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemGetInfo(out free, out total);
        return result;
    }

    internal int cuDeviceGetAttribute(out int value, int attribute, int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceGetAttribute(out value, attribute, device);
        return result;
    }

    internal int cuCtxCreate(out IntPtr ctx, uint flags, int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuCtxCreate(out ctx, flags, device);
        return result;
    }

    internal int cuCtxDestroy(IntPtr ctx)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuCtxDestroy(ctx); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.ContextRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.ContextRelease, false);
        return result;
    }

    internal int cuCtxSetCurrent(IntPtr ctx)
    {
        using var lease = EnterEffect();
        int result = Api.cuCtxSetCurrent(ctx);
        return result;
    }

    internal int cuCtxGetCurrent(out IntPtr ctx)
    {
        using var lease = EnterEffect();
        int result = Api.cuCtxGetCurrent(out ctx);
        return result;
    }

    internal int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuDevicePrimaryCtxRetain(out ctx, device);
        return result;
    }

    internal int cuDevicePrimaryCtxRelease(int device)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuDevicePrimaryCtxRelease(device); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.ContextRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.ContextRelease, false);
        return result;
    }

    internal int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemAlloc(out devicePtr, byteSize);
        return result;
    }

    internal int cuMemFree(IntPtr devicePtr)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuMemFree(devicePtr); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.StorageRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.StorageRelease, false);
        return result;
    }

    internal int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemHostAlloc(out hostPtr, byteSize, flags);
        return result;
    }

    internal int cuMemFreeHost(IntPtr hostPtr)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuMemFreeHost(hostPtr); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.StorageRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.StorageRelease, false);
        return result;
    }

    internal int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyHtoD(dstDevice, srcHost, byteCount);
        return result;
    }

    internal int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyHtoDAsync(dstDevice, srcHost, byteCount, stream);
        return result;
    }

    internal int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyDtoH(dstHost, srcDevice, byteCount);
        return result;
    }

    internal int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyDtoHAsync(dstHost, srcDevice, byteCount, stream);
        return result;
    }

    internal int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyDtoD(dstDevice, srcDevice, byteCount);
        return result;
    }

    internal int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyDtoDAsync(dstDevice, srcDevice, byteCount, stream);
        return result;
    }

    internal int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemsetD8(dstDevice, value, count);
        return result;
    }

    internal int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemsetD8Async(dstDevice, value, count, stream);
        return result;
    }

    internal int cuModuleLoadData(out IntPtr module, IntPtr image)
    {
        using var lease = EnterEffect();
        int result = Api.cuModuleLoadData(out module, image);
        return result;
    }

    internal int cuModuleGetFunction(out IntPtr function, IntPtr module, string name)
    {
        using var lease = EnterEffect();
        int result = Api.cuModuleGetFunction(out function, module, name);
        return result;
    }

    internal int cuModuleUnload(IntPtr module)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuModuleUnload(module); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.ContextRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.ContextRelease, false);
        return result;
    }

    internal int cuFuncSetAttribute(IntPtr function, int attribute, int value)
    {
        using var lease = EnterEffect();
        int result = Api.cuFuncSetAttribute(function, attribute, value);
        return result;
    }

    internal int cuLaunchKernel(IntPtr function, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra)
    {
        using var lease = EnterEffect();
        int result = Api.cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
        return result;
    }

    internal int cuStreamCreate(out IntPtr stream, uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuStreamCreate(out stream, flags);
        return result;
    }

    internal int cuStreamDestroy(IntPtr stream)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuStreamDestroy(stream); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.WorkerRetirement);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.WorkerRetirement, false);
        return result;
    }

    internal int cuStreamSynchronize(IntPtr stream)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuStreamSynchronize(stream); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.Synchronization);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.Synchronization, false);
        return result;
    }

    internal int cuStreamBeginCapture(IntPtr stream, int mode)
    {
        using var lease = EnterEffect();
        int result = Api.cuStreamBeginCapture(stream, mode);
        return result;
    }

    internal int cuStreamEndCapture(IntPtr stream, out IntPtr graph)
    {
        using var lease = EnterEffect();
        int result = Api.cuStreamEndCapture(stream, out graph);
        return result;
    }

    internal int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuGraphInstantiateWithFlags(out graphExec, graph, flags);
        return result;
    }

    internal int cuGraphLaunch(IntPtr graphExec, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cuGraphLaunch(graphExec, stream);
        return result;
    }

    internal int cuGraphExecDestroy(IntPtr graphExec)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuGraphExecDestroy(graphExec); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.GraphRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.GraphRelease, false);
        return result;
    }

    internal int cuGraphDestroy(IntPtr graph)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuGraphDestroy(graph); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.GraphRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.GraphRelease, false);
        return result;
    }

    internal int cuGetErrorString(int error, out IntPtr str)
    {
        using var lease = EnterEffect();
        int result = Api.cuGetErrorString(error, out str);
        return result;
    }

    internal int cuEventCreate(out IntPtr phEvent, uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuEventCreate(out phEvent, flags);
        return result;
    }

    internal int cuEventDestroy(IntPtr hEvent)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuEventDestroy(hEvent); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.WorkerRetirement);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.WorkerRetirement, false);
        return result;
    }

    internal int cuEventRecord(IntPtr hEvent, IntPtr hStream)
    {
        using var lease = EnterEffect();
        int result = Api.cuEventRecord(hEvent, hStream);
        return result;
    }

    internal int cuEventSynchronize(IntPtr hEvent)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cuEventSynchronize(hEvent); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.Synchronization);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.Synchronization, false);
        return result;
    }

    internal int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuStreamWaitEvent(hStream, hEvent, flags);
        return result;
    }

    internal int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev)
    {
        using var lease = EnterEffect();
        int result = Api.cuDeviceCanAccessPeer(out canAccessPeer, dev, peerDev);
        return result;
    }

    internal int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags)
    {
        using var lease = EnterEffect();
        int result = Api.cuCtxEnablePeerAccess(peerContext, flags);
        return result;
    }

    internal int cuMemcpyPeerAsync(IntPtr dstDevice, IntPtr dstContext, IntPtr srcDevice, IntPtr srcContext, UIntPtr byteCount, IntPtr hStream)
    {
        using var lease = EnterEffect();
        int result = Api.cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, byteCount, hStream);
        return result;
    }

    internal int cublasCreate(out IntPtr handle)
    {
        using var lease = EnterEffect();
        int result = Api.cublasCreate(out handle);
        return result;
    }

    internal int cublasDestroy(IntPtr handle)
    {
        using var lease = EnterEffect();
        int result;
        try { result = Api.cublasDestroy(handle); }
        catch (Exception failure)
        {
            PublishFailure(lease, failure, NativeRuntimeFailureStage.ContextRelease);
            throw;
        }
        RejectUnsafeResult(lease, result, NativeRuntimeFailureStage.ContextRelease, true);
        return result;
    }

    internal int cublasSetStream(IntPtr handle, IntPtr stream)
    {
        using var lease = EnterEffect();
        int result = Api.cublasSetStream(handle, stream);
        return result;
    }

    internal int cublasSetMathMode(IntPtr handle, int mode)
    {
        using var lease = EnterEffect();
        int result = Api.cublasSetMathMode(handle, mode);
        return result;
    }

    internal int cublasSgemm(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc)
    {
        using var lease = EnterEffect();
        int result = Api.cublasSgemm(handle, transa, transb, m, n, k, ref alpha, a, lda, b, ldb, ref beta, c, ldc);
        return result;
    }

    internal int cublasSgemmStridedBatched(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, ref float beta, IntPtr c, int ldc, long strideC, int batchCount)
    {
        using var lease = EnterEffect();
        int result = Api.cublasSgemmStridedBatched(handle, transa, transb, m, n, k, ref alpha, a, lda, strideA, b, ldb, strideB, ref beta, c, ldc, strideC, batchCount);
        return result;
    }

    internal int cublasGemmEx(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int aType, int lda, IntPtr b, int bType, int ldb, ref float beta, IntPtr c, int cType, int ldc, int computeType, int algo)
    {
        using var lease = EnterEffect();
        int result = Api.cublasGemmEx(handle, transa, transb, m, n, k, ref alpha, a, aType, lda, b, bType, ldb, ref beta, c, cType, ldc, computeType, algo);
        return result;
    }

    internal int cuCtxGetDevice(out int device)
    {
        using var lease = EnterEffect();
        int result = Api.cuCtxGetDevice(out device);
        return result;
    }
}
