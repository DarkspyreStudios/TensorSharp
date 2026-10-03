using System.Collections.Generic;

namespace TensorSharp.Models;

internal static class ModelDisposalOwnership
{
    internal static void Add(ICollection<Tensor> tensors, params Tensor[] values)
        => AddRange(tensors, values);

    internal static void AddRange(ICollection<Tensor> tensors, IEnumerable<Tensor> values)
    {
        if (values == null) return;
        foreach (Tensor tensor in values)
            if (tensor != null) tensors.Add(tensor);
    }

    internal static void AddRows(ICollection<Tensor> tensors, IEnumerable<Tensor[]> rows)
    {
        if (rows == null) return;
        foreach (Tensor[] row in rows) AddRange(tensors, row);
    }
}
