using System.Collections.Generic;

namespace TensorSharp.Models;

internal static class ModelDisposalOwnership
{
    internal static bool IsHealthyStorageRefusal(System.Exception failure, IEnumerable<Tensor> ownedTensors)
    {
        if (failure is not NativeMlxCallbackBusyException busy) return false;
        foreach (Tensor tensor in ownedTensors)
            if (tensor.GetLiveOwnedStorageForDisposal() is { } storage && busy.IsFor(storage)) return true;
        return false;
    }

    internal static void ReleasePublishedWeight(Dictionary<string, Tensor> weights, string name)
    {
        weights[name].Dispose();
        weights.Remove(name);
    }

    internal static void ReleaseMlxWeightCaches(IEnumerable<QuantizedWeight> weights,
        IEnumerable<StackedExpertWeights> stackedWeights, System.Action<System.IntPtr> release)
    {
        foreach (QuantizedWeight weight in weights) release(weight.CacheKey);
        // Expert cache handles can differ from the stacked buffer's pointer identity.
        foreach (StackedExpertWeights stacked in stackedWeights) release(stacked.Data);
    }

    internal static void RetainPendingWeight(ref Tensor incoming, List<Tensor> displaced)
    {
        if (incoming == null) return;
        // Capacity is reserved while the pending tensor is still reachable from its owner.
        displaced.EnsureCapacity(checked(displaced.Count + 1));
        displaced.Add(incoming);
        incoming = null;
    }

    internal static Tensor NewConstructionContiguous(Tensor source, ref Tensor incoming,
        List<Tensor> displaced)
    {
        RetainPendingWeight(ref incoming, displaced);
        // NewContiguous cannot expose its destination when Copy throws before returning it.
        Tensor result = incoming = new Tensor(source.Allocator, source.ElementType, source.Sizes);
        Ops.Copy(result, source);
        return result;
    }

    internal static void PublishConstructionWeight<TKey>(Dictionary<TKey, Tensor> weights, TKey name,
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

    internal static void PublishConstructionWeights<TKey>(Dictionary<TKey, (Tensor First, Tensor Second)> weights,
        TKey name, Tensor first, ref Tensor second, List<Tensor> displaced)
    {
        if (weights.TryGetValue(name, out var previous))
        {
            displaced.EnsureCapacity(checked(displaced.Count + 2));
            displaced.Add(previous.First);
            displaced.Add(previous.Second);
        }
        weights[name] = (first, second);
        second = null;
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
