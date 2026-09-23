using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class ModelMetadataTests
{
    [Fact]
    public void GgufInspectionReadsOnlyTheHeaderAndLeavesCallerStreamOpen()
    {
        byte[] header = GgufHeader();
        using var source = new HeaderOnlyStream(header, 1024);
        GgufMetadata metadata = GgufFile.InspectMetadata(source);
        Assert.Equal(3U, metadata.Version);
        Assert.Equal("qwen3", metadata.Values["general.architecture"]);
        Assert.Equal(new ulong[] { 2 }, metadata.Tensors["weight"].Shape);
        Assert.Equal(1024, metadata.FileLength);
        Assert.True(source.CanRead);
        Assert.Equal(header.Length, source.BytesRead);
    }

    [Fact]
    public void SafetensorsInspectionDoesNotNeedTensorBytesAndChecksDeclaredStorageBounds()
    {
        const string json = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        byte[] header = [.. BitConverter.GetBytes((ulong)Encoding.UTF8.GetByteCount(json)), .. Encoding.UTF8.GetBytes(json)];
        using var source = new HeaderOnlyStream(header, header.Length + 8);
        SafetensorsMetadata metadata = SafetensorsFile.InspectMetadata(source);
        Assert.Equal(SafetensorDtype.F32, metadata.Tensors["weight"].Dtype);
        Assert.Equal(header.Length, metadata.DataOffset);
        Assert.Equal(header.Length, source.BytesRead);
        Assert.True(source.CanRead);
        using var truncated = new HeaderOnlyStream(header, header.Length + 4);
        Assert.Throws<InvalidDataException>(() => SafetensorsFile.InspectMetadata(truncated));
    }

    [Theory]
    [InlineData("string")]
    [InlineData("array")]
    [InlineData("tensor-count")]
    [InlineData("rank")]
    public void OversizedGgufAllocationsAreRejectedBeforeAllocation(string invalid)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
        {
            writer.Write(0x46554747U); writer.Write(3U);
            writer.Write(invalid == "tensor-count" ? ulong.MaxValue : invalid == "rank" ? 1UL : 0UL);
            writer.Write(invalid is "string" or "array" ? 1UL : 0UL);
            if (invalid == "string") writer.Write(ulong.MaxValue);
            if (invalid == "array") { WriteString(writer, "values"); writer.Write(9U); writer.Write(8U); writer.Write(ulong.MaxValue); }
            if (invalid == "rank") { WriteString(writer, "weight"); writer.Write(uint.MaxValue); }
            writer.Write(new byte[100]);
        }
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => GgufFile.InspectMetadata(stream));
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void ExplicitBudgetBoundsInspectionEvenWhenSourceClaimsALargeFile()
    {
        using var source = new HeaderOnlyStream(GgufHeader(), long.MaxValue);
        Assert.Throws<InvalidDataException>(() => GgufFile.InspectMetadata(source, 24));
        Assert.True(source.BytesRead <= 24);
        byte[] huge = BitConverter.GetBytes(ulong.MaxValue);
        using var safetensors = new HeaderOnlyStream(huge, long.MaxValue);
        Assert.Throws<InvalidDataException>(() => SafetensorsFile.InspectMetadata(safetensors));
        Assert.Equal(8, safetensors.BytesRead);
    }

    [Fact]
    public void DuplicateTensorNamesAreRejectedByInspectionAndNormalLoadParser()
    {
        byte[] bytes = GgufHeader(duplicate: true);
        using var source = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => GgufFile.InspectMetadata(source));
        string path = Path.Combine(Path.GetTempPath(), $"metadata-{Guid.NewGuid():N}.gguf");
        try { File.WriteAllBytes(path, bytes); Assert.Throws<InvalidDataException>(() => new GgufFile(path)); }
        finally { File.Delete(path); }
    }

    private static byte[] GgufHeader(bool duplicate = false)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, true);
        writer.Write(0x46554747U); writer.Write(3U); writer.Write(duplicate ? 2UL : 1UL); writer.Write(1UL);
        WriteString(writer, "general.architecture"); writer.Write(8U); WriteString(writer, "qwen3");
        for (int index = 0; index < (duplicate ? 2 : 1); index++)
        { WriteString(writer, "weight"); writer.Write(1U); writer.Write(2UL); writer.Write(0U); writer.Write(0UL); }
        return stream.ToArray();
    }
    private static void WriteString(BinaryWriter writer, string value)
    { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }

    private sealed class HeaderOnlyStream(byte[] header, long fileLength) : MemoryStream(header)
    {
        private readonly byte[] _header = header;
        public int BytesRead { get; private set; }
        public override long Length => fileLength;
        public override int Read(Span<byte> buffer)
        {
            if (Position + buffer.Length > _header.Length) throw new InvalidOperationException("The parser attempted to read tensor data.");
            _header.AsSpan((int)Position, buffer.Length).CopyTo(buffer);
            Position += buffer.Length; BytesRead += buffer.Length; return buffer.Length;
        }
        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
    }
}
