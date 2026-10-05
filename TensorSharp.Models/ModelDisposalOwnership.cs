using System.Collections.Generic;

namespace TensorSharp.Models;

internal static class ModelDisposalOwnership
{
    internal static void PublishConstructionWeight(Dictionary<string, Tensor> weights, string name,
        ref Tensor incoming, List<Tensor> displaced)
    {
        if (weights.TryGetValue(name, out Tensor previous))
        {
            displaced.EnsureCapacity(checked(displaced.Count + 1));
            displaced.Add(previous);
        }
        weights[name] = incoming;
        incoming = null;
    }

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
