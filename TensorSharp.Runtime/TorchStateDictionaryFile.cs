// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>Storage types accepted by <see cref="TorchStateDictionaryFile"/>.</summary>
    public enum TorchStorageDtype
    {
        Float32,
        BFloat16,
        Int64,
    }

    /// <summary>Metadata for one dense, contiguous tensor in a PyTorch ZIP checkpoint.</summary>
    public sealed class TorchTensorInfo
    {
        public string Name { get; init; } = string.Empty;
        public string StorageKey { get; init; } = string.Empty;
        public TorchStorageDtype Dtype { get; init; }
        public long StorageOffset { get; init; }
        public long[] Shape { get; init; } = Array.Empty<long>();

        public long NumElements
        {
            get
            {
                long result = 1;
                foreach (long dimension in Shape)
                    result = checked(result * dimension);
                return result;
            }
        }
    }

    /// <summary>
    /// Reads a deliberately restricted subset of PyTorch's ZIP checkpoint format as a named
    /// tensor store. The metadata pickle is interpreted by a whitelist-only stack machine: it
    /// accepts primitive containers, persistent storage references, <c>OrderedDict</c>, and
    /// <c>torch._utils._rebuild_tensor_v2</c>. It never imports modules or invokes serialized code.
    /// </summary>
    /// <remarks>
    /// This is not a general-purpose pickle reader. It rejects unknown opcodes, globals, reducers,
    /// non-CPU storage, non-contiguous views, and storage types other than F32, BF16, and I64.
    /// </remarks>
    public sealed class TorchStateDictionaryFile : IFloatTensorStore, IDisposable
    {
        private readonly Stream _stream;
        private readonly ZipArchive _archive;
        private readonly Dictionary<string, ZipArchiveEntry> _storageEntries = new(StringComparer.Ordinal);
        private readonly Dictionary<string, TorchTensorInfo> _tensors = new(StringComparer.Ordinal);
        private bool _disposed;

        public TorchStateDictionaryFile(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);
            Path = path;
            FileStream? stream = null;
            ZipArchive? archive = null;

            try
            {
                stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: false);
                _stream = stream;
                _archive = archive;
                ParseArchive();
            }
            catch
            {
                archive?.Dispose();
                stream?.Dispose();
                throw;
            }
        }

        public string Path { get; }

        private TorchStateDictionaryFile(MetadataReadStream stream)
        {
            Path = string.Empty;
            _stream = stream;
            _archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
            try { ParseArchive(); }
            catch { _archive.Dispose(); throw; }
        }

        /// <summary>Inspects restricted ZIP/pickle metadata over a bounded seekable source. Storage entries are not read.
        /// Leaves the caller's source open; no temporary file, tensor allocation or repository code execution occurs.</summary>
        public static TorchStateDictionaryMetadata InspectMetadata(Stream source, int maximumBytes = 16 * 1024 * 1024)
        {
            using var bounded = new MetadataReadStream(source, maximumBytes);
            using var file = new TorchStateDictionaryFile(bounded);
            return new(file.IntegerMetadata, new System.Collections.ObjectModel.ReadOnlyDictionary<string, TorchTensorInfo>(file._tensors));
        }

        public IReadOnlyDictionary<string, TorchTensorInfo> Tensors => _tensors;

        public IReadOnlyDictionary<string, long> IntegerMetadata { get; private set; }
            = new Dictionary<string, long>(StringComparer.Ordinal);

        public bool HasTensor(string name) => _tensors.ContainsKey(name);

        public long[] TensorShape(string name) =>
            _tensors.TryGetValue(name, out TorchTensorInfo? info)
                ? (long[])info.Shape.Clone()
                : Array.Empty<long>();

        public TorchTensorInfo GetInfo(string name) =>
            _tensors.TryGetValue(name, out TorchTensorInfo? info)
                ? info
                : throw new KeyNotFoundException($"PyTorch checkpoint tensor not found: {name}");

        public float[] ReadFloat32(string name)
        {
            ThrowIfDisposed();
            TorchTensorInfo info = GetInfo(name);
            int elementCount = checked((int)info.NumElements);
            var result = new float[elementCount];
            ZipArchiveEntry entry = _storageEntries[info.StorageKey];
            int elementSize = DtypeSize(info.Dtype);
            long byteOffset = checked(info.StorageOffset * elementSize);
            int byteCount = checked(elementCount * elementSize);

            using Stream input = entry.Open();
            SkipExactly(input, byteOffset);

            byte[] bytes = new byte[byteCount];
            input.ReadExactly(bytes);
            ConvertToFloat32(info.Dtype, bytes, result);
            return result;
        }

        private void ParseArchive()
        {
            ZipArchiveEntry[] metadataEntries = _archive.Entries
                .Where(entry => entry.FullName.EndsWith("/data.pkl", StringComparison.Ordinal))
                .ToArray();
            if (metadataEntries.Length != 1)
                throw new InvalidDataException("A PyTorch ZIP checkpoint must contain exactly one '<root>/data.pkl' entry.");

            ZipArchiveEntry metadataEntry = metadataEntries[0];
            string root = metadataEntry.FullName[..^"data.pkl".Length];
            ValidateByteOrder(root);

            foreach (ZipArchiveEntry entry in _archive.Entries)
            {
                string dataPrefix = root + "data/";
                if (!entry.FullName.StartsWith(dataPrefix, StringComparison.Ordinal))
                    continue;

                string key = entry.FullName[dataPrefix.Length..];
                if (!IsStorageKey(key) || !_storageEntries.TryAdd(key, entry))
                    throw new InvalidDataException($"Invalid or duplicate PyTorch storage entry '{entry.FullName}'.");
            }

            if (metadataEntry.Length > 16 * 1024 * 1024)
                throw new InvalidDataException("PyTorch checkpoint metadata exceeds the 16 MiB safety limit.");

            byte[] pickle = new byte[checked((int)metadataEntry.Length)];
            using (Stream metadata = metadataEntry.Open())
                metadata.ReadExactly(pickle);

            object rootValue = new RestrictedPickleReader(pickle).Read();
            if (rootValue is not Dictionary<string, object> rootDictionary)
                throw new InvalidDataException("PyTorch checkpoint root must be a dictionary.");

            Dictionary<string, object> modelDictionary = rootDictionary.TryGetValue("model", out object? model)
                ? RequireDictionary(model, "model")
                : rootDictionary;

            AddTensors(modelDictionary, prefix: string.Empty);

            var integerMetadata = new Dictionary<string, long>(StringComparer.Ordinal);
            if (rootDictionary.TryGetValue("config", out object? configValue))
            {
                Dictionary<string, object> config = RequireDictionary(configValue, "config");
                foreach ((string key, object value) in config)
                {
                    if (value is long integer)
                        integerMetadata[key] = integer;
                    else if (value is TensorReference tensor)
                        AddTensor("config." + key, tensor);
                    else
                        throw new InvalidDataException($"Unsupported PyTorch config value '{key}'.");
                }
            }

            IntegerMetadata = integerMetadata;
            if (_tensors.Count == 0)
                throw new InvalidDataException("PyTorch checkpoint does not contain any tensors.");
        }

        private void AddTensors(Dictionary<string, object> dictionary, string prefix)
        {
            foreach ((string name, object value) in dictionary)
            {
                string qualifiedName = prefix + name;
                if (value is TensorReference tensor)
                    AddTensor(qualifiedName, tensor);
                else if (value is Dictionary<string, object> nested)
                    AddTensors(nested, qualifiedName + ".");
                else
                    throw new InvalidDataException($"Unsupported value for PyTorch state entry '{qualifiedName}'.");
            }
        }

        private void AddTensor(string name, TensorReference tensor)
        {
            if (!_storageEntries.TryGetValue(tensor.Storage.Key, out ZipArchiveEntry? storageEntry))
                throw new InvalidDataException($"PyTorch tensor '{name}' references missing storage '{tensor.Storage.Key}'.");
            if (tensor.StorageOffset < 0)
                throw new InvalidDataException($"PyTorch tensor '{name}' has a negative storage offset.");
            if (!IsContiguous(tensor.Shape, tensor.Stride))
                throw new NotSupportedException($"PyTorch tensor '{name}' is not a contiguous row-major view.");

            long elementCount = ElementCount(tensor.Shape, name);
            long storageEnd = checked(tensor.StorageOffset + elementCount);
            if (storageEnd > tensor.Storage.ElementCount)
                throw new InvalidDataException($"PyTorch tensor '{name}' exceeds its declared storage bounds.");

            long requiredBytes = checked(tensor.Storage.ElementCount * DtypeSize(tensor.Storage.Dtype));
            if (storageEntry.Length != requiredBytes)
            {
                throw new InvalidDataException(
                    $"PyTorch storage '{tensor.Storage.Key}' is {storageEntry.Length} bytes; {requiredBytes} were declared.");
            }

            var info = new TorchTensorInfo
            {
                Name = name,
                StorageKey = tensor.Storage.Key,
                Dtype = tensor.Storage.Dtype,
                StorageOffset = tensor.StorageOffset,
                Shape = (long[])tensor.Shape.Clone(),
            };
            if (!_tensors.TryAdd(name, info))
                throw new InvalidDataException($"Duplicate PyTorch tensor name '{name}'.");
        }

        private void ValidateByteOrder(string root)
        {
            ZipArchiveEntry? byteOrder = _archive.GetEntry(root + "byteorder");
            if (byteOrder == null)
                return;

            using Stream input = byteOrder.Open();
            using var reader = new StreamReader(input, Encoding.ASCII, detectEncodingFromByteOrderMarks: false);
            string value = reader.ReadToEnd().Trim();
            if (!string.Equals(value, "little", StringComparison.Ordinal))
                throw new NotSupportedException($"PyTorch checkpoint byte order '{value}' is not supported.");
        }

        private static Dictionary<string, object> RequireDictionary(object value, string name) =>
            value as Dictionary<string, object>
            ?? throw new InvalidDataException($"PyTorch checkpoint '{name}' must be a dictionary.");

        private static bool IsStorageKey(string key) =>
            key.Length > 0 && key.Length <= 128 && key.All(character => char.IsAsciiLetterOrDigit(character) || character is '_' or '-');

        private static long ElementCount(long[] shape, string name)
        {
            long result = 1;
            foreach (long dimension in shape)
            {
                if (dimension < 0)
                    throw new InvalidDataException($"PyTorch tensor '{name}' has a negative dimension.");
                result = checked(result * dimension);
            }
            return result;
        }

        private static bool IsContiguous(long[] shape, long[] stride)
        {
            if (shape.Length != stride.Length)
                return false;

            long expected = 1;
            for (int index = shape.Length - 1; index >= 0; index--)
            {
                if (shape[index] > 1 && stride[index] != expected)
                    return false;
                expected = checked(expected * shape[index]);
            }
            return true;
        }

        private static int DtypeSize(TorchStorageDtype dtype) => dtype switch
        {
            TorchStorageDtype.Float32 => 4,
            TorchStorageDtype.BFloat16 => 2,
            TorchStorageDtype.Int64 => 8,
            _ => throw new NotSupportedException($"Unsupported PyTorch storage type {dtype}."),
        };

        private static void ConvertToFloat32(TorchStorageDtype dtype, ReadOnlySpan<byte> source, Span<float> destination)
        {
            int elementSize = DtypeSize(dtype);
            if (source.Length != checked(destination.Length * elementSize))
                throw new InvalidDataException("PyTorch storage read returned an unexpected byte count.");

            for (int index = 0; index < destination.Length; index++)
            {
                ReadOnlySpan<byte> element = source.Slice(index * elementSize, elementSize);
                destination[index] = dtype switch
                {
                    TorchStorageDtype.Float32 => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(element)),
                    TorchStorageDtype.BFloat16 => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(element) << 16),
                    TorchStorageDtype.Int64 => BinaryPrimitives.ReadInt64LittleEndian(element),
                    _ => throw new NotSupportedException($"Unsupported PyTorch storage type {dtype}."),
                };
            }
        }

        private static void SkipExactly(Stream stream, long byteCount)
        {
            if (byteCount == 0)
                return;
            if (stream.CanSeek)
            {
                if (stream.Seek(byteCount, SeekOrigin.Current) < byteCount)
                    throw new EndOfStreamException();
                return;
            }

            Span<byte> discard = stackalloc byte[4096];
            while (byteCount > 0)
            {
                int read = stream.Read(discard[..(int)Math.Min(discard.Length, byteCount)]);
                if (read == 0)
                    throw new EndOfStreamException();
                byteCount -= read;
            }
        }

        private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _archive.Dispose();
            _stream.Dispose();
        }

        private sealed record GlobalReference(string Module, string Name);

        private sealed record StorageReference(string Key, TorchStorageDtype Dtype, long ElementCount);

        private sealed record TensorReference(
            StorageReference Storage,
            long StorageOffset,
            long[] Shape,
            long[] Stride);

        private sealed class RestrictedPickleReader
        {
            private static readonly object Marker = new();
            private readonly ReadOnlyMemory<byte> _data;
            private readonly List<object> _stack = new();
            private readonly Dictionary<int, object> _memo = new();
            private int _position;

            internal RestrictedPickleReader(ReadOnlyMemory<byte> data) => _data = data;

            internal object Read()
            {
                bool protocolSeen = false;
                while (_position < _data.Length)
                {
                    byte opcode = ReadByte();
                    switch (opcode)
                    {
                        case 0x80: // PROTO
                            if (protocolSeen || ReadByte() != 2)
                                throw Invalid("Only pickle protocol 2 is supported.");
                            protocolSeen = true;
                            break;
                        case (byte)'}': // EMPTY_DICT
                            Push(new Dictionary<string, object>(StringComparer.Ordinal));
                            break;
                        case (byte)'(': // MARK
                            Push(Marker);
                            break;
                        case (byte)'X': // BINUNICODE
                            Push(ReadUnicode());
                            break;
                        case (byte)'c': // GLOBAL
                            Push(ReadGlobal());
                            break;
                        case (byte)'q': // BINPUT
                            Memoize(ReadByte());
                            break;
                        case (byte)'r': // LONG_BINPUT
                            Memoize(ReadInt32());
                            break;
                        case (byte)'h': // BINGET
                            Push(Recall(ReadByte()));
                            break;
                        case (byte)'j': // LONG_BINGET
                            Push(Recall(ReadInt32()));
                            break;
                        case (byte)'J': // BININT
                            Push((long)ReadInt32());
                            break;
                        case (byte)'K': // BININT1
                            Push((long)ReadByte());
                            break;
                        case (byte)'M': // BININT2
                            Push((long)ReadUInt16());
                            break;
                        case (byte)'\x89': // NEWFALSE
                            Push(false);
                            break;
                        case (byte)'\x88': // NEWTRUE
                            Push(true);
                            break;
                        case (byte)')': // EMPTY_TUPLE
                            Push(Array.Empty<object>());
                            break;
                        case (byte)'t': // TUPLE
                            Push(PopMarkedItems());
                            break;
                        case (byte)'\x85': // TUPLE1
                            Push(new[] { Pop() });
                            break;
                        case (byte)'\x86': // TUPLE2
                            Push(PopFixedTuple(2));
                            break;
                        case (byte)'\x87': // TUPLE3
                            Push(PopFixedTuple(3));
                            break;
                        case (byte)'Q': // BINPERSID
                            Push(BuildStorageReference(Pop()));
                            break;
                        case (byte)'R': // REDUCE
                            {
                                object arguments = Pop();
                                object callable = Pop();
                                Push(Reduce(callable, arguments));
                                break;
                            }
                        case (byte)'u': // SETITEMS
                            SetItems();
                            break;
                        case (byte)'.': // STOP
                            if (_position != _data.Length || _stack.Count != 1)
                                throw Invalid("Unexpected trailing data or stack state at STOP.");
                            return Pop();
                        default:
                            throw Invalid($"Pickle opcode 0x{opcode:X2} is not allowed.");
                    }
                }

                throw Invalid("Pickle stream ended before STOP.");
            }

            private GlobalReference ReadGlobal()
            {
                string module = ReadLine();
                string name = ReadLine();
                bool allowed = (module, name) switch
                {
                    ("torch._utils", "_rebuild_tensor_v2") => true,
                    ("collections", "OrderedDict") => true,
                    ("torch", "FloatStorage") => true,
                    ("torch", "BFloat16Storage") => true,
                    ("torch", "LongStorage") => true,
                    _ => false,
                };
                if (!allowed)
                    throw Invalid($"Pickle global '{module}.{name}' is not allowed.");
                return new GlobalReference(module, name);
            }

            private object Reduce(object callable, object arguments)
            {
                if (callable is not GlobalReference global || arguments is not object[] tuple)
                    throw Invalid("REDUCE requires an allowed global and tuple arguments.");

                if (global is { Module: "collections", Name: "OrderedDict" })
                {
                    if (tuple.Length != 0)
                        throw Invalid("OrderedDict reducer must have no arguments.");
                    return new Dictionary<string, object>(StringComparer.Ordinal);
                }

                if (global is not { Module: "torch._utils", Name: "_rebuild_tensor_v2" } || tuple.Length != 6)
                    throw Invalid($"Reducer '{global.Module}.{global.Name}' is not allowed.");
                if (tuple[0] is not StorageReference storage || tuple[1] is not long offset
                    || tuple[2] is not object[] shapeValues || tuple[3] is not object[] strideValues
                    || tuple[4] is not bool || tuple[5] is not Dictionary<string, object>)
                {
                    throw Invalid("_rebuild_tensor_v2 arguments do not match the supported tensor form.");
                }

                return new TensorReference(storage, offset, ToLongArray(shapeValues), ToLongArray(strideValues));
            }

            private StorageReference BuildStorageReference(object value)
            {
                if (value is not object[] tuple || tuple.Length != 5
                    || tuple[0] is not string storageTag || storageTag != "storage"
                    || tuple[1] is not GlobalReference type
                    || tuple[2] is not string key
                    || tuple[3] is not string location || location != "cpu"
                    || tuple[4] is not long elementCount || elementCount < 0)
                {
                    throw Invalid("Persistent ID is not a supported CPU tensor storage reference.");
                }

                TorchStorageDtype dtype = (type.Module, type.Name) switch
                {
                    ("torch", "FloatStorage") => TorchStorageDtype.Float32,
                    ("torch", "BFloat16Storage") => TorchStorageDtype.BFloat16,
                    ("torch", "LongStorage") => TorchStorageDtype.Int64,
                    _ => throw Invalid($"Storage global '{type.Module}.{type.Name}' is not supported."),
                };
                if (!IsStorageKey(key))
                    throw Invalid($"Storage key '{key}' is invalid.");
                return new StorageReference(key, dtype, elementCount);
            }

            private void SetItems()
            {
                object[] items = PopMarkedItems();
                if (items.Length % 2 != 0 || _stack.Count == 0 || _stack[^1] is not Dictionary<string, object> dictionary)
                    throw Invalid("SETITEMS requires a dictionary and key/value pairs.");

                for (int index = 0; index < items.Length; index += 2)
                {
                    if (items[index] is not string key || !dictionary.TryAdd(key, items[index + 1]))
                        throw Invalid("SETITEMS keys must be unique strings.");
                }
            }

            private object[] PopMarkedItems()
            {
                int markerIndex = _stack.LastIndexOf(Marker);
                if (markerIndex < 0)
                    throw Invalid("Pickle MARK is missing.");
                object[] values = _stack.Skip(markerIndex + 1).ToArray();
                _stack.RemoveRange(markerIndex, _stack.Count - markerIndex);
                return values;
            }

            private object[] PopFixedTuple(int count)
            {
                if (_stack.Count < count)
                    throw Invalid("Tuple opcode underflowed the pickle stack.");
                var tuple = new object[count];
                for (int index = count - 1; index >= 0; index--)
                    tuple[index] = Pop();
                return tuple;
            }

            private static long[] ToLongArray(object[] values)
            {
                var result = new long[values.Length];
                for (int index = 0; index < values.Length; index++)
                    result[index] = values[index] is long value
                        ? value
                        : throw new InvalidDataException("Tensor shape and stride values must be integers.");
                return result;
            }

            private string ReadUnicode()
            {
                int length = ReadInt32();
                if (length < 0 || length > _data.Length - _position)
                    throw Invalid("BINUNICODE length is out of range.");
                string value = Encoding.UTF8.GetString(_data.Span.Slice(_position, length));
                _position += length;
                return value;
            }

            private string ReadLine()
            {
                ReadOnlySpan<byte> remaining = _data.Span[_position..];
                int newline = remaining.IndexOf((byte)'\n');
                if (newline < 0)
                    throw Invalid("GLOBAL line is unterminated.");
                string value = Encoding.UTF8.GetString(remaining[..newline]);
                _position += newline + 1;
                return value;
            }

            private byte ReadByte()
            {
                if (_position >= _data.Length)
                    throw Invalid("Unexpected end of pickle stream.");
                return _data.Span[_position++];
            }

            private ushort ReadUInt16()
            {
                if (_position > _data.Length - 2)
                    throw Invalid("Unexpected end of pickle stream.");
                ushort value = BinaryPrimitives.ReadUInt16LittleEndian(_data.Span[_position..]);
                _position += 2;
                return value;
            }

            private int ReadInt32()
            {
                if (_position > _data.Length - 4)
                    throw Invalid("Unexpected end of pickle stream.");
                int value = BinaryPrimitives.ReadInt32LittleEndian(_data.Span[_position..]);
                _position += 4;
                return value;
            }

            private void Memoize(int index)
            {
                if (index < 0 || _stack.Count == 0 || !_memo.TryAdd(index, _stack[^1]))
                    throw Invalid($"Invalid or duplicate pickle memo index {index}.");
            }

            private object Recall(int index) => _memo.TryGetValue(index, out object? value)
                ? value
                : throw Invalid($"Unknown pickle memo index {index}.");

            private void Push(object value) => _stack.Add(value);

            private object Pop()
            {
                if (_stack.Count == 0)
                    throw Invalid("Pickle stack underflow.");
                int index = _stack.Count - 1;
                object value = _stack[index];
                _stack.RemoveAt(index);
                return value;
            }

            private InvalidDataException Invalid(string message) =>
                new($"Invalid or unsupported PyTorch checkpoint metadata at byte {_position}: {message}");
        }
    }
}
