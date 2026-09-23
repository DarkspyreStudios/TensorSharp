using System;
using System.Collections.Generic;
using System.IO;

namespace TensorSharp.Runtime;

/// <summary>Metadata from one GGUF artifact. No tensor data, mappings, native resources or stream ownership.</summary>
public sealed record GgufMetadata(uint Version, IReadOnlyDictionary<string, object> Values,
    IReadOnlyDictionary<string, GgufTensorInfo> Tensors, long DataOffset, long FileLength);

/// <summary>Metadata from one safetensors artifact; tensor offsets are validated against the declared stream length.</summary>
public sealed record SafetensorsMetadata(IReadOnlyDictionary<string, string> Values,
    IReadOnlyDictionary<string, SafetensorTensorInfo> Tensors, long DataOffset, long FileLength);

/// <summary>Restricted PyTorch ZIP metadata. No storage entries are decompressed or read as tensor data.</summary>
public sealed record TorchStateDictionaryMetadata(IReadOnlyDictionary<string, long> IntegerValues,
    IReadOnlyDictionary<string, TorchTensorInfo> Tensors);

/// <summary>Bounds total bytes read, including after seeks. Leaves the caller-owned seekable source open.</summary>
internal sealed class MetadataReadStream : Stream
{
    private readonly Stream _source;
    public long Remaining { get; private set; }

    public MetadataReadStream(Stream source, int maximumBytes)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumBytes);
        if (!source.CanRead || !source.CanSeek) throw new ArgumentException("Metadata inspection requires a readable seekable stream.", nameof(source));
        _source = source;
        _source.Position = 0;
        Remaining = maximumBytes;
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _source.Length;
    public override long Position { get => _source.Position; set => _source.Position = value; }
    public override long Seek(long offset, SeekOrigin origin) => _source.Seek(offset, origin);
    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
    public override int Read(Span<byte> buffer)
    {
        if (buffer.Length == 0) return 0;
        if (Remaining == 0) throw new InvalidDataException("Model metadata exceeds the inspection byte budget.");
        int read = _source.Read(buffer[..(int)Math.Min(buffer.Length, Remaining)]);
        Remaining -= read;
        return read;
    }
    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    internal static void RequireElements(BinaryReader reader, ulong count, int minimumBytes)
    {
        long remaining = reader.BaseStream.Length - reader.BaseStream.Position;
        if (reader.BaseStream is MetadataReadStream bounded) remaining = Math.Min(remaining, bounded.Remaining);
        if (count > int.MaxValue || remaining < 0 || count > (ulong)(remaining / minimumBytes))
            throw new InvalidDataException("Model metadata length exceeds its remaining file or inspection budget.");
    }
}
