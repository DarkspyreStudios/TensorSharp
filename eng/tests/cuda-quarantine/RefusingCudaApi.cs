using TensorSharp.Cuda.Interop;

namespace TensorSharp.CudaQuarantineFixture;

// Unused calls refuse; no controlled API call reaches a CUDA library.
internal abstract class RefusingCudaApi : ICudaNativeApi
{
    public virtual int cuCtxSynchronize() => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuInit(uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceGet(out int device, int ordinal) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceGetCount(out int count) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceGetName(byte[] name, int len, int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceTotalMem(out UIntPtr bytes, int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemGetInfo(out UIntPtr free, out UIntPtr total) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceGetAttribute(out int value, int attribute, int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxCreate(out IntPtr ctx, uint flags, int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxDestroy(IntPtr ctx) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxSetCurrent(IntPtr ctx) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxGetCurrent(out IntPtr ctx) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDevicePrimaryCtxRelease(int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemFree(IntPtr devicePtr) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemFreeHost(IntPtr hostPtr) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuModuleLoadData(out IntPtr module, IntPtr image) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuModuleGetFunction(out IntPtr function, IntPtr module, string name) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuModuleUnload(IntPtr module) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuFuncSetAttribute(IntPtr function, int attribute, int value) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuLaunchKernel(IntPtr function, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamCreate(out IntPtr stream, uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamDestroy(IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamSynchronize(IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamBeginCapture(IntPtr stream, int mode) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamEndCapture(IntPtr stream, out IntPtr graph) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuGraphLaunch(IntPtr graphExec, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuGraphExecDestroy(IntPtr graphExec) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuGraphDestroy(IntPtr graph) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuGetErrorString(int error, out IntPtr str) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuEventCreate(out IntPtr phEvent, uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuEventDestroy(IntPtr hEvent) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuEventRecord(IntPtr hEvent, IntPtr hStream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuEventSynchronize(IntPtr hEvent) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuMemcpyPeerAsync(IntPtr dstDevice, IntPtr dstContext, IntPtr srcDevice, IntPtr srcContext, UIntPtr byteCount, IntPtr hStream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasCreate(out IntPtr handle) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasDestroy(IntPtr handle) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasSetStream(IntPtr handle, IntPtr stream) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasSetMathMode(IntPtr handle, int mode) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasSgemm(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasSgemmStridedBatched(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, ref float beta, IntPtr c, int ldc, long strideC, int batchCount) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cublasGemmEx(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr a, int aType, int lda, IntPtr b, int bType, int ldb, ref float beta, IntPtr c, int cType, int ldc, int computeType, int algo) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
    public virtual int cuCtxGetDevice(out int device) => throw new NotSupportedException("Unconfigured controlled CUDA call.");
}
