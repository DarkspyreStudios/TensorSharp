using System;

namespace TensorSharp.Cuda.Interop;

// The immutable production adapter forwards the finite raw import signatures.
internal sealed class CudaNativeApi : ICudaNativeApi
{
    internal static readonly CudaNativeApi Instance = new();
    private CudaNativeApi() { }

    public int cuInit(uint flags)
        => CudaDriverApi.cuInit(flags);

    public int cuDeviceGet(out int device, int ordinal)
        => CudaDriverApi.cuDeviceGet(out device, ordinal);

    public int cuDeviceGetCount(out int count)
        => CudaDriverApi.cuDeviceGetCount(out count);

    public int cuDeviceGetName(byte[] name, int len, int device)
        => CudaDriverApi.cuDeviceGetName(name, len, device);

    public int cuDeviceTotalMem(out UIntPtr bytes, int device)
        => CudaDriverApi.cuDeviceTotalMem(out bytes, device);

    public int cuMemGetInfo(out UIntPtr free, out UIntPtr total)
        => CudaDriverApi.cuMemGetInfo(out free, out total);

    public int cuDeviceGetAttribute(out int value, int attribute, int device)
        => CudaDriverApi.cuDeviceGetAttribute(out value, attribute, device);

    public int cuCtxCreate(out IntPtr ctx, uint flags, int device)
        => CudaDriverApi.cuCtxCreate(out ctx, flags, device);

    public int cuCtxDestroy(IntPtr ctx)
        => CudaDriverApi.cuCtxDestroy(ctx);

    public int cuCtxSetCurrent(IntPtr ctx)
        => CudaDriverApi.cuCtxSetCurrent(ctx);

    public int cuCtxGetCurrent(out IntPtr ctx)
        => CudaDriverApi.cuCtxGetCurrent(out ctx);

    public int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device)
        => CudaDriverApi.cuDevicePrimaryCtxRetain(out ctx, device);

    public int cuDevicePrimaryCtxRelease(int device)
        => CudaDriverApi.cuDevicePrimaryCtxRelease(device);

    public int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize)
        => CudaDriverApi.cuMemAlloc(out devicePtr, byteSize);

    public int cuMemFree(IntPtr devicePtr)
        => CudaDriverApi.cuMemFree(devicePtr);

    public int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags)
        => CudaDriverApi.cuMemHostAlloc(out hostPtr, byteSize, flags);

    public int cuMemFreeHost(IntPtr hostPtr)
        => CudaDriverApi.cuMemFreeHost(hostPtr);

    public int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount)
        => CudaDriverApi.cuMemcpyHtoD(dstDevice, srcHost, byteCount);

    public int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream)
        => CudaDriverApi.cuMemcpyHtoDAsync(dstDevice, srcHost, byteCount, stream);

    public int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount)
        => CudaDriverApi.cuMemcpyDtoH(dstHost, srcDevice, byteCount);

    public int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream)
        => CudaDriverApi.cuMemcpyDtoHAsync(dstHost, srcDevice, byteCount, stream);

    public int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount)
        => CudaDriverApi.cuMemcpyDtoD(dstDevice, srcDevice, byteCount);

    public int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream)
        => CudaDriverApi.cuMemcpyDtoDAsync(dstDevice, srcDevice, byteCount, stream);

    public int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count)
        => CudaDriverApi.cuMemsetD8(dstDevice, value, count);

    public int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream)
        => CudaDriverApi.cuMemsetD8Async(dstDevice, value, count, stream);

    public int cuModuleLoadData(out IntPtr module, IntPtr image)
        => CudaDriverApi.cuModuleLoadData(out module, image);

    public int cuModuleGetFunction(out IntPtr function, IntPtr module, string name)
        => CudaDriverApi.cuModuleGetFunction(out function, module, name);

    public int cuModuleUnload(IntPtr module)
        => CudaDriverApi.cuModuleUnload(module);

    public int cuFuncSetAttribute(IntPtr function, int attribute, int value)
        => CudaDriverApi.cuFuncSetAttribute(function, attribute, value);

    public int cuLaunchKernel(IntPtr function, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra)
        => CudaDriverApi.cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);

    public int cuStreamCreate(out IntPtr stream, uint flags)
        => CudaDriverApi.cuStreamCreate(out stream, flags);

    public int cuStreamDestroy(IntPtr stream)
        => CudaDriverApi.cuStreamDestroy(stream);

    public int cuStreamSynchronize(IntPtr stream)
        => CudaDriverApi.cuStreamSynchronize(stream);

    public int cuStreamBeginCapture(IntPtr stream, int mode)
        => CudaDriverApi.cuStreamBeginCapture(stream, mode);

    public int cuStreamEndCapture(IntPtr stream, out IntPtr graph)
        => CudaDriverApi.cuStreamEndCapture(stream, out graph);

    public int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags)
        => CudaDriverApi.cuGraphInstantiateWithFlags(out graphExec, graph, flags);

    public int cuGraphLaunch(IntPtr graphExec, IntPtr stream)
        => CudaDriverApi.cuGraphLaunch(graphExec, stream);

    public int cuGraphExecDestroy(IntPtr graphExec)
        => CudaDriverApi.cuGraphExecDestroy(graphExec);

    public int cuGraphDestroy(IntPtr graph)
        => CudaDriverApi.cuGraphDestroy(graph);

    public int cuGetErrorString(int error, out IntPtr str)
        => CudaDriverApi.cuGetErrorString(error, out str);

    public int cuEventCreate(out IntPtr phEvent, uint flags)
        => CudaDriverApi.cuEventCreate(out phEvent, flags);

    public int cuEventDestroy(IntPtr hEvent)
        => CudaDriverApi.cuEventDestroy(hEvent);

    public int cuEventRecord(IntPtr hEvent, IntPtr hStream)
        => CudaDriverApi.cuEventRecord(hEvent, hStream);

    public int cuEventSynchronize(IntPtr hEvent)
        => CudaDriverApi.cuEventSynchronize(hEvent);

    public int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags)
        => CudaDriverApi.cuStreamWaitEvent(hStream, hEvent, flags);

    public int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev)
        => CudaDriverApi.cuDeviceCanAccessPeer(out canAccessPeer, dev, peerDev);

    public int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags)
        => CudaDriverApi.cuCtxEnablePeerAccess(peerContext, flags);

    public int cuMemcpyPeerAsync(IntPtr dstDevice, IntPtr dstContext, IntPtr srcDevice, IntPtr srcContext, UIntPtr byteCount, IntPtr hStream)
        => CudaDriverApi.cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, byteCount, hStream);

    public int cublasCreate(out IntPtr handle)
        => CublasApi.cublasCreate(out handle);

    public int cublasDestroy(IntPtr handle)
        => CublasApi.cublasDestroy(handle);

    public int cublasSetStream(IntPtr handle, IntPtr stream)
        => CublasApi.cublasSetStream(handle, stream);

    public int cublasSetMathMode(IntPtr handle, int mode)
        => CublasApi.cublasSetMathMode(handle, mode);

    public int cublasSgemm(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc)
        => CublasApi.cublasSgemm(handle, transa, transb, m, n, k, ref alpha, a, lda, b, ldb, ref beta, c, ldc);

    public int cublasSgemmStridedBatched(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, ref float beta, IntPtr c, int ldc, long strideC, int batchCount)
        => CublasApi.cublasSgemmStridedBatched(handle, transa, transb, m, n, k, ref alpha, a, lda, strideA, b, ldb, strideB, ref beta, c, ldc, strideC, batchCount);

    public int cublasGemmEx(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int aType, int lda, IntPtr b, int bType, int ldb, ref float beta, IntPtr c, int cType, int ldc, int computeType, int algo)
        => CublasApi.cublasGemmEx(handle, transa, transb, m, n, k, ref alpha, a, aType, lda, b, bType, ldb, ref beta, c, cType, ldc, computeType, algo);

    public int cuCtxGetDevice(out int device)
        => CudaDriverApi.cuCtxGetDevice(out device);

    public int cuCtxSynchronize() => CudaDriverApi.cuCtxSynchronize();
}
