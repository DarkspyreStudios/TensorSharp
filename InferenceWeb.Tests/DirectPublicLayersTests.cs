using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class DirectPublicLayersTests
{
    [Fact]
    public void DirectRuntimeTypes_ArePublic()
    {
        Assert.True(typeof(DirectContext).IsPublic);
        Assert.True(typeof(DirectLinear).IsPublic);
        Assert.True(typeof(DirectOps).IsPublic);
        Assert.True(typeof(DirectEmbedding).IsPublic);
        Assert.True(typeof(DirectConv2D).IsPublic);
        Assert.True(typeof(DirectImageOps).IsPublic);
    }

    [Fact]
    public void TensorStoreFactories_LoadLinearAndEmbeddingWeights()
    {
        var store = new MemoryTensorStore()
            .Add("embedding", [3, 2], [1f, 2f, 3f, 4f, 5f, 6f])
            .Add("linear.weight", [2, 2], [1f, 0f, 0f, 2f])
            .Add("linear.bias", [2], [0.5f, -1f]);
        using var context = new DirectContext(new CpuAllocator(BlasEnum.DotNet));
        using var embedding = DirectEmbedding.FromTensorStore(context, store, "embedding");
        using var linear = DirectLinear.FromTensorStore(
            context,
            store,
            "linear.weight",
            "linear.bias");

        using var embedded = embedding.Forward([2, 0]);
        using var projected = linear.Forward(embedded);

        Assert.Equal([5.5f, 11f, 1.5f, 3f], DirectOps.ToArray(projected));
    }

    [Fact]
    public void Conv2D_UsesPyTorchWeightLayoutAndChannelsLastActivations()
    {
        using var context = new DirectContext(new CpuAllocator(BlasEnum.DotNet));
        // Two output channels over one 2x2 input channel. The first sums each
        // patch; the second selects its top-left value and adds a bias.
        using var convolution = DirectConv2D.FromFloats(
            context,
            [
                1f, 1f, 1f, 1f,
                1f, 0f, 0f, 0f,
            ],
            outputChannels: 2,
            inputChannels: 1,
            kernelHeight: 2,
            kernelWidth: 2,
            bias: [0f, 10f]);
        using var input = context.FromFloats(
            [
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
            ],
            1, 3, 3, 1);

        using var output = convolution.Forward(input, workspaceBytes: 16);

        Assert.Equal([1, 2, 2, 2], output.Sizes.ToArray());
        Assert.Equal(
            [12f, 11f, 16f, 12f, 24f, 14f, 28f, 15f],
            DirectOps.ToArray(output));
    }

    [Fact]
    public void GroupNorm_NormalizesEachGroupAndAppliesChannelAffine()
    {
        using var context = new DirectContext(new CpuAllocator(BlasEnum.DotNet));
        using var input = context.FromFloats([1f, 3f, 2f, 6f], 1, 1, 1, 4);
        using var gain = context.FromFloats([1f, 2f, 1f, 0.5f], 4);
        using var bias = context.FromFloats([0f, 1f, -1f, 2f], 4);

        using var output = DirectImageOps.GroupNorm(
            context,
            input,
            groups: 2,
            gain,
            bias,
            epsilon: 1e-12f);

        AssertClose([-1f, 3f, -2f, 2.5f], DirectOps.ToArray(output));
    }

    [Fact]
    public void NearestUpsample2D_RepeatsSpatialValues()
    {
        using var context = new DirectContext(new CpuAllocator(BlasEnum.DotNet));
        using var input = context.FromFloats([1f, 2f, 3f, 4f], 1, 2, 2, 1);

        using var output = DirectImageOps.NearestUpsample2D(context, input, 2);

        Assert.Equal([1, 4, 4, 1], output.Sizes.ToArray());
        Assert.Equal(
            [
                1f, 1f, 2f, 2f,
                1f, 1f, 2f, 2f,
                3f, 3f, 4f, 4f,
                3f, 3f, 4f, 4f,
            ],
            DirectOps.ToArray(output));
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    private sealed class MemoryTensorStore : IFloatTensorStore
    {
        private readonly Dictionary<string, (long[] Shape, float[] Values)> _values =
            new(StringComparer.Ordinal);

        public MemoryTensorStore Add(string name, long[] shape, float[] values)
        {
            _values.Add(name, (shape, values));
            return this;
        }

        public bool HasTensor(string name) => _values.ContainsKey(name);

        public float[] ReadFloat32(string name) => (float[])_values[name].Values.Clone();

        public long[] TensorShape(string name) => _values.TryGetValue(name, out var value)
            ? (long[])value.Shape.Clone()
            : [];
    }
}
