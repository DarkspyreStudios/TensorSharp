using System;

namespace TensorSharp.Models;

internal static class TensorParallelRankRestoration
{
    internal static void Restore(int rank, Exception? operationFailure, Action<int> restoreRank)
    {
        try { restoreRank(rank); }
        catch (Exception restoreFailure) when (operationFailure != null)
        {
            if (!ReferenceEquals(operationFailure, restoreFailure))
                throw new AggregateException("Tensor-parallel execution and rank restoration failed.",
                    operationFailure, restoreFailure);
        }
    }
}
