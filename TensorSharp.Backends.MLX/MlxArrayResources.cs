using System;

namespace TensorSharp.MLX
{
    internal sealed class MlxArrayResources : MlxNativeResources
    {
        internal readonly MlxNative.MlxArray[] Arrays;
        internal int ReturnedIndex = -1;
        internal bool HasStoredResult;
        private readonly Storage[] storages;
        private readonly MlxNative.MlxArray[] borrowedInputs = Array.Empty<MlxNative.MlxArray>();

        internal override bool HasNativeResources
        {
            get
            {
                foreach (MlxNative.MlxArray array in Arrays)
                    if (array.IsValid) return true;
                return false;
            }
        }

        internal MlxArrayResources(int referenceCount, params Tensor[] tensors)
        {
            Arrays = new MlxNative.MlxArray[referenceCount];
            storages = new Storage[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
                storages[i] = tensors[i]?.Storage;
        }

        internal MlxArrayResources(int referenceCount, MlxNative.MlxArray[] inputs)
            : this(referenceCount)
        {
            borrowedInputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        }

        internal void Release()
        {
            for (int i = 0; i < Arrays.Length; i++)
                if (i != ReturnedIndex)
                    MlxNative.FreeArrayReference(ref Arrays[i]);
            GC.KeepAlive(storages);
            GC.KeepAlive(borrowedInputs);
        }
    }
}
