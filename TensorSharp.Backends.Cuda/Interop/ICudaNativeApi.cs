using System;

namespace TensorSharp.Cuda.Interop;

// Instance-local native calls. Callers borrow this interface; it owns no native resources.
internal interface ICudaNativeApi
{
    int cuInit(uint flags);
    int cuDeviceGet(out int device, int ordinal);
    int cuDeviceGetCount(out int count);
    int cuDeviceGetName(byte[] name, int len, int device);
    int cuDeviceTotalMem(out UIntPtr bytes, int device);
    int cuMemGetInfo(out UIntPtr free, out UIntPtr total);
    int cuDeviceGetAttribute(out int value, int attribute, int device);
    int cuCtxCreate(out IntPtr ctx, uint flags, int device);
    int cuCtxDestroy(IntPtr ctx);
    int cuCtxSetCurrent(IntPtr ctx);
    int cuCtxGetCurrent(out IntPtr ctx);
    int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);
    int cuDevicePrimaryCtxRelease(int device);
    int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);
    int cuMemFree(IntPtr devicePtr);
    int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags);
    int cuMemFreeHost(IntPtr hostPtr);
    int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);
    int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);
    int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);
    int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);
    int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);
    int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);
    int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);
    int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream);
    int cuModuleLoadData(out IntPtr module, IntPtr image);
    int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);
    int cuModuleUnload(IntPtr module);
    int cuFuncSetAttribute(IntPtr function, int attribute, int value);
    int cuLaunchKernel(IntPtr function, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra);
    int cuStreamCreate(out IntPtr stream, uint flags);
    int cuStreamDestroy(IntPtr stream);
    int cuStreamSynchronize(IntPtr stream);
    int cuStreamBeginCapture(IntPtr stream, int mode);
    int cuStreamEndCapture(IntPtr stream, out IntPtr graph);
    int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags);
    int cuGraphLaunch(IntPtr graphExec, IntPtr stream);
    int cuGraphExecDestroy(IntPtr graphExec);
    int cuGraphDestroy(IntPtr graph);
    int cuGetErrorString(int error, out IntPtr str);
    int cuEventCreate(out IntPtr phEvent, uint flags);
    int cuEventDestroy(IntPtr hEvent);
    int cuEventRecord(IntPtr hEvent, IntPtr hStream);
    int cuEventSynchronize(IntPtr hEvent);
    int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);
    int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev);
    int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags);
    int cuMemcpyPeerAsync(IntPtr dstDevice, IntPtr dstContext, IntPtr srcDevice, IntPtr srcContext, UIntPtr byteCount, IntPtr hStream);
    int cublasCreate(out IntPtr handle);
    int cublasDestroy(IntPtr handle);
    int cublasSetStream(IntPtr handle, IntPtr stream);
    int cublasSetMathMode(IntPtr handle, int mode);
    int cublasSgemm(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc);
    int cublasSgemmStridedBatched(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, ref float beta, IntPtr c, int ldc, long strideC, int batchCount);
    int cublasGemmEx(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int aType, int lda, IntPtr b, int bType, int ldb, ref float beta, IntPtr c, int cType, int ldc, int computeType, int algo);
    int cuCtxGetDevice(out int device);
    int cuCtxSynchronize();
}
