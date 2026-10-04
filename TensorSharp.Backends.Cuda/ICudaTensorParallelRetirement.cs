using System.Collections.Generic;

namespace TensorSharp.Cuda;

internal interface ICudaTensorParallelRetirement
{
    bool OwnsCudaAllocators { get; }
    void CollectOwnedAllocators(ICollection<IAllocator> allocators);
    void DisposeOwned(CudaRetirementPlan plan, CudaContextRestoration restoration);
}
