// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class TorchStateDictionaryFileTests
{
    [Fact]
    public void InspectsRestrictedMetadataAndLeavesCallerStreamOpen()
    {
        string path = CreateCheckpoint();
        try
        {
            using Stream source = File.OpenRead(path);
            TorchStateDictionaryMetadata metadata = TorchStateDictionaryFile.InspectMetadata(source);
            Assert.Equal(new long[] { 2, 2 }, metadata.Tensors["weight"].Shape);
            Assert.Equal(TorchStorageDtype.BFloat16, metadata.Tensors["config.uncond_text"].Dtype);
            Assert.Equal(2, metadata.IntegerValues["patch"]);
            Assert.True(source.CanRead);
            Assert.Throws<InvalidDataException>(() => TorchStateDictionaryFile.InspectMetadata(source, 16));
            Assert.True(source.CanRead);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void InspectionUsesTheSamePickleWhitelistAsLoading()
    {
        string path = CreateCheckpoint("dangerous.module Callable");
        try
        {
            using Stream source = File.OpenRead(path);
            Assert.Throws<InvalidDataException>(() => TorchStateDictionaryFile.InspectMetadata(source));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ReadsWhitelistedDenseTensorsAndConfigMetadata()
    {
        string path = CreateCheckpoint();
        try
        {
            using var checkpoint = new TorchStateDictionaryFile(path);

            Assert.Equal(new long[] { 2, 2 }, checkpoint.TensorShape("weight"));
            Assert.Equal(new[] { 1.5f, -2.0f, 3.25f, 4.0f }, checkpoint.ReadFloat32("weight"));
            Assert.Equal(new[] { 1.0f, -2.0f }, checkpoint.ReadFloat32("config.uncond_text"));
            Assert.Equal(2, checkpoint.IntegerMetadata["patch"]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RejectsUnknownPickleGlobalsWithoutExecutingThem()
    {
        string path = CreateCheckpoint("dangerous.module Callable");
        try
        {
            InvalidDataException exception = Assert.Throws<InvalidDataException>(
                () => new TorchStateDictionaryFile(path));
            Assert.Contains("is not allowed", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string CreateCheckpoint(string storageGlobal = "torch FloatStorage")
    {
        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"torch-state-{Guid.NewGuid():N}.pt");
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        WriteEntry(archive, "fixture/byteorder", Encoding.ASCII.GetBytes("little"));
        WriteEntry(archive, "fixture/data.pkl", BuildPickle(storageGlobal));

        byte[] floats = new byte[4 * sizeof(float)];
        float[] values = [1.5f, -2.0f, 3.25f, 4.0f];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteInt32LittleEndian(floats.AsSpan(index * 4), BitConverter.SingleToInt32Bits(values[index]));
        WriteEntry(archive, "fixture/data/0", floats);

        byte[] bfloat16 = new byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(bfloat16, 0x3F80);
        BinaryPrimitives.WriteUInt16LittleEndian(bfloat16.AsSpan(2), 0xC000);
        WriteEntry(archive, "fixture/data/1", bfloat16);
        return path;
    }

    private static byte[] BuildPickle(string storageGlobal)
    {
        using var stream = new MemoryStream();
        stream.Write([0x80, 0x02, (byte)'}', (byte)'(']);
        WriteUnicode(stream, "model");
        stream.WriteByte((byte)'}');
        stream.WriteByte((byte)'(');
        WriteUnicode(stream, "weight");
        WriteTensor(stream, storageGlobal, "0", 4, [2, 2], [2, 1]);
        stream.WriteByte((byte)'u');
        WriteUnicode(stream, "config");
        stream.WriteByte((byte)'}');
        stream.WriteByte((byte)'(');
        WriteUnicode(stream, "patch");
        WriteSmallInteger(stream, 2);
        WriteUnicode(stream, "uncond_text");
        WriteTensor(stream, "torch BFloat16Storage", "1", 2, [2], [1]);
        stream.WriteByte((byte)'u');
        stream.WriteByte((byte)'u');
        stream.WriteByte((byte)'.');
        return stream.ToArray();
    }

    private static void WriteTensor(
        Stream stream,
        string storageGlobal,
        string storageKey,
        int storageElements,
        int[] shape,
        int[] stride)
    {
        WriteGlobal(stream, "torch._utils _rebuild_tensor_v2");
        stream.WriteByte((byte)'(');
        stream.WriteByte((byte)'(');
        WriteUnicode(stream, "storage");
        WriteGlobal(stream, storageGlobal);
        WriteUnicode(stream, storageKey);
        WriteUnicode(stream, "cpu");
        WriteSmallInteger(stream, storageElements);
        stream.WriteByte((byte)'t');
        stream.WriteByte((byte)'Q');
        WriteSmallInteger(stream, 0);
        WriteTuple(stream, shape);
        WriteTuple(stream, stride);
        stream.WriteByte(0x89);
        WriteGlobal(stream, "collections OrderedDict");
        stream.WriteByte((byte)')');
        stream.WriteByte((byte)'R');
        stream.WriteByte((byte)'t');
        stream.WriteByte((byte)'R');
    }

    private static void WriteTuple(Stream stream, int[] values)
    {
        foreach (int value in values)
            WriteSmallInteger(stream, value);
        stream.WriteByte(values.Length switch
        {
            1 => (byte)0x85,
            2 => (byte)0x86,
            3 => (byte)0x87,
            _ => throw new ArgumentOutOfRangeException(nameof(values)),
        });
    }

    private static void WriteSmallInteger(Stream stream, int value)
    {
        stream.WriteByte((byte)'K');
        stream.WriteByte(checked((byte)value));
    }

    private static void WriteUnicode(Stream stream, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        stream.WriteByte((byte)'X');
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(length, bytes.Length);
        stream.Write(length);
        stream.Write(bytes);
    }

    private static void WriteGlobal(Stream stream, string value)
    {
        string[] parts = value.Split(' ', 2);
        byte[] bytes = Encoding.ASCII.GetBytes($"c{parts[0]}\n{parts[1]}\n");
        stream.Write(bytes);
    }

    private static void WriteEntry(ZipArchive archive, string name, byte[] contents)
    {
        ZipArchiveEntry entry = archive.CreateEntry(name, CompressionLevel.NoCompression);
        using Stream output = entry.Open();
        output.Write(contents);
    }
}
