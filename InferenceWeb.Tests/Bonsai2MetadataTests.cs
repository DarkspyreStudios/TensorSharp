// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Bonsai2MetadataTests
{
    [Theory]
    [InlineData(GgmlTensorType.PQ2_0, 128, 34)]
    [InlineData(GgmlTensorType.PTQ1_0, 128, 28)]
    [InlineData(GgmlTensorType.Q2_0, 64, 18)]
    public void QuantizedRowLayoutsMatchBonsaiAndUpstreamFormats(GgmlTensorType type, int blockSize, int bytes)
    {
        using var fixture = new Header();
        Assert.Equal(blockSize, GgufFile.GetBlockSize(type));
        Assert.Equal(bytes, GgufFile.GetTypeSize(type));
        var tensor = new GgufTensorInfo { Type = type, Shape = new ulong[] { 5120, 6144 } };
        Assert.Equal(5120L / blockSize * bytes * 6144, fixture.File.GetTensorByteCount(tensor));
    }

    [Fact]
    public void OrdinaryGgufHasNoBonsaiTransform()
    {
        using var fixture = new Header();
        Assert.Null(BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Fact]
    public void CustomQuantWithoutRotationMetadataFailsBeforeLoadingWeights()
    {
        using var fixture = new Header();
        fixture.AddTensor("output.weight");
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Fact]
    public void ReadsExplicitSignedRotationAndInverseEmbedding()
    {
        using var fixture = ValidHeader();
        var metadata = BonsaiHadamardMetadata.Read(fixture.File)!;
        Assert.Equal(1024, metadata.BlockSize);
        Assert.True(metadata.GdnVGrouped);
        Assert.Contains("output.weight", metadata.WeightNames);
        Assert.Contains("token_embd.weight", metadata.InverseWeightNames);
        Assert.Equal(-1f, metadata.SignsByWidth[1024][0]);
        Assert.Equal(1f, metadata.SignsByWidth[1024][1]);
        Assert.Equal("qwen35", ModelArchitectureRegistry.Resolve("qwen35", fixture.File).Id);
    }

    [Theory]
    [InlineData("version", 2u)]
    [InlineData("transform", "unnormalized")]
    [InlineData("axis", "output")]
    [InlineData("sign_mode", "implicit")]
    public void UnknownTransformSemanticsFail(string key, object value)
    {
        using var fixture = ValidHeader();
        fixture.File.Metadata["prism.hadamard." + key] = value;
        Assert.Throws<NotSupportedException>(() => BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(2048)]
    public void InvalidBlockSizeFails(int blockSize)
    {
        using var fixture = ValidHeader();
        fixture.File.Metadata["prism.hadamard.block_size"] = (uint)blockSize;
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Fact]
    public void InvalidSignsAndMissingSignEntriesFail()
    {
        using var fixture = ValidHeader();
        var signs = (sbyte[])fixture.File.Metadata["prism.hadamard.sign_values"];
        signs[52] = 0;
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
        signs[52] = 1;
        fixture.File.Metadata["prism.hadamard.sign_values"] = signs[..^1];
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Fact]
    public void MissingAndDuplicateWeightDeclarationsFail()
    {
        using var fixture = ValidHeader();
        fixture.File.Metadata["prism.hadamard.weight_names"] = new[] { "output.weight", "output.weight" };
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
        fixture.File.Metadata["prism.hadamard.weight_names"] = Array.Empty<string>();
        Assert.Throws<InvalidDataException>(() => BonsaiHadamardMetadata.Read(fixture.File));
    }

    [Fact]
    public void FusionCannotMixRotatedAndUnrotatedProjectionInputs()
    {
        using var fixture = ValidHeader();
        var metadata = BonsaiHadamardMetadata.Read(fixture.File)!;
        Assert.True(metadata.CanFuseProjections("output.weight", "output.weight"));
        Assert.True(metadata.CanFuseProjections("ssm_alpha.weight", "ssm_beta.weight"));
        Assert.False(metadata.CanFuseProjections("output.weight", "ssm_alpha.weight"));
        Assert.False(metadata.CanFuseProjections("ssm_beta.weight", "output.weight"));
        Assert.False(metadata.CanFuseProjections("token_embd.weight", "token_embd.weight"));
    }

    [Theory]
    [InlineData(4)]
    [InlineData(1024)]
    public void InverseEmbeddingMatchesIndependentDenseHadamardAcrossBlocks(int blockSize)
    {
        int width = blockSize * 2;
        float[] input = Enumerable.Range(0, width).Select(i => MathF.Sin(i * 0.71f)).ToArray();
        float[] signs = Enumerable.Range(0, width).Select(i => i % 3 == 0 ? -1f : 1f).ToArray();
        float[] actual = (float[])input.Clone();
        BonsaiHadamardMetadata.InverseTransform(actual, signs, blockSize);
        for (int i = 0; i < width; i++)
        {
            double sum = 0;
            int start = i / blockSize * blockSize;
            for (int j = 0; j < blockSize; j++)
            {
                int parity = System.Numerics.BitOperations.PopCount((uint)((i % blockSize) & j)) & 1;
                sum += (parity == 0 ? 1 : -1) * input[start + j];
            }
            double expected = sum / Math.Sqrt(blockSize) * signs[i];
            Assert.True(Math.Abs(expected - actual[i]) < 2e-5, $"index {i}: {actual[i]} != {expected}");
        }
    }

    [Fact]
    public void BonsaiProjectorDiscoveryPrefersBf16AndFallsBackToQ8()
    {
        string directory = Path.Combine(Path.GetTempPath(), "bonsai-projector-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string model = Path.Combine(directory, "Ternary-Bonsai-2-27B-PQ2_0.gguf");
            string q8 = Path.Combine(directory, "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf");
            string bf16 = Path.Combine(directory, "Ternary-Bonsai-2-27B-mmproj-BF16.gguf");
            File.WriteAllBytes(q8, Array.Empty<byte>());
            Assert.Equal(q8, ModelArchitectureRegistry.FindCompanionProjector("qwen35", model));
            File.WriteAllBytes(bf16, Array.Empty<byte>());
            Assert.Equal(bf16, ModelArchitectureRegistry.FindCompanionProjector("qwen35", model));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [ModelFact("TENSORSHARP_BONSAI2_MODEL")]
    public void RealBonsai2HeaderAndTransformAreConsistent()
    {
        using var file = new GgufFile(Environment.GetEnvironmentVariable("TENSORSHARP_BONSAI2_MODEL")!);
        var metadata = BonsaiHadamardMetadata.Read(file)!;
        Assert.Equal(401, metadata.WeightNames.Count);
        Assert.Single(metadata.InverseWeightNames);
        Assert.Equal(new[] { 5120, 6144, 17408 }, metadata.SignsByWidth.Keys.OrderBy(x => x));
        Assert.Equal("qwen35", ModelArchitectureRegistry.Resolve(file.GetString("general.architecture"), file).Id);
        foreach (var tensor in file.Tensors.Values)
            Assert.True(file.GetTensorByteCount(tensor) > 0);
    }

    private static Header ValidHeader()
    {
        var header = new Header();
        header.AddTensor("output.weight");
        header.AddTensor("token_embd.weight");
        var metadata = header.File.Metadata;
        metadata["prism.hadamard.version"] = 1u;
        metadata["prism.hadamard.block_size"] = 1024u;
        metadata["prism.hadamard.transform"] = "normalized-sylvester-walsh-hadamard";
        metadata["prism.hadamard.axis"] = "input-last-dimension";
        metadata["prism.hadamard.sign_mode"] = "explicit";
        metadata["prism.hadamard.sign_widths"] = new[] { 1024 };
        metadata["prism.hadamard.sign_values"] = Enumerable.Range(0, 1024).Select(i => (sbyte)(i % 3 == 0 ? -1 : 1)).ToArray();
        metadata["prism.hadamard.weight_names"] = new[] { "output.weight" };
        metadata["prism.hadamard.inverse_weight_names"] = new[] { "token_embd.weight" };
        metadata["prism.hadamard.gdn_v_grouped"] = true;
        return header;
    }

    private sealed class Header : IDisposable
    {
        private readonly string _path = Path.GetTempFileName();
        public GgufFile File { get; }

        public Header()
        {
            using (var writer = new BinaryWriter(System.IO.File.Create(_path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(0UL);
            }
            File = new GgufFile(_path);
        }

        public void AddTensor(string name) => File.Tensors.Add(name, new GgufTensorInfo
        {
            Name = name, Type = GgmlTensorType.PQ2_0, Shape = new ulong[] { 1024, 2 },
        });

        public void Dispose()
        {
            File.Dispose();
            System.IO.File.Delete(_path);
        }
    }
}
