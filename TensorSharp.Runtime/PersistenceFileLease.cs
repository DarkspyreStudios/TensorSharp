// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Keeps a persistence artifact available to TensorSharp's path-based native and
    /// memory-mapped readers. File-backed stores are used in place. Other stores are
    /// projected to a TensorSharp-owned temporary file and removed with the lease.
    /// </summary>
    internal sealed class PersistenceFileLease : IDisposable, IAsyncDisposable
    {
        private const int CopyBufferSize = 81920;

        private Stream? _retainedSource;
        private string? _temporaryFile;

        private PersistenceFileLease(string filePath, Stream? retainedSource, string? temporaryFile)
        {
            FilePath = filePath;
            _retainedSource = retainedSource;
            _temporaryFile = temporaryFile;
        }

        internal string FilePath { get; }

        internal bool IsTemporary => _temporaryFile != null;

        internal static async Task<PersistenceFileLease> AcquireAsync(
            PersistenceFileReference source,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(source);
            cancellationToken.ThrowIfCancellationRequested();

            Stream? stream = await source.Store.OpenReadAsync(
                source.StoreName,
                source.AssetName,
                cancellationToken).ConfigureAwait(false);

            if (stream == null)
            {
                throw new FileNotFoundException(
                    $"Persistence artifact '{source}' does not exist.",
                    source.AssetName);
            }

            if (!stream.CanRead)
            {
                await stream.DisposeAsync().ConfigureAwait(false);
                throw new InvalidDataException($"Persistence artifact '{source}' returned an unreadable stream.");
            }

            if (stream is FileStream fileStream && !string.IsNullOrWhiteSpace(fileStream.Name))
                return new PersistenceFileLease(fileStream.Name, fileStream, temporaryFile: null);

            string temporaryDirectory = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "TensorSharp",
                "persistence");
            Directory.CreateDirectory(temporaryDirectory);

            string extension = SafeExtension(source.AssetName);
            string temporaryFile = System.IO.Path.Combine(
                temporaryDirectory,
                $"{Guid.NewGuid():N}{extension}");

            try
            {
                var options = new FileStreamOptions
                {
                    Access = FileAccess.Write,
                    BufferSize = CopyBufferSize,
                    Mode = FileMode.CreateNew,
                    Options = FileOptions.Asynchronous | FileOptions.SequentialScan,
                    Share = FileShare.None,
                };

                await using (var output = new FileStream(temporaryFile, options))
                {
                    await stream.CopyToAsync(output, CopyBufferSize, cancellationToken).ConfigureAwait(false);
                    await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                }

                await stream.DisposeAsync().ConfigureAwait(false);
                stream = null;
                return new PersistenceFileLease(temporaryFile, retainedSource: null, temporaryFile);
            }
            catch
            {
                if (stream != null)
                    await stream.DisposeAsync().ConfigureAwait(false);
                File.Delete(temporaryFile);
                throw;
            }
        }

        private static string SafeExtension(string assetName)
        {
            string extension = System.IO.Path.GetExtension(assetName);
            if (extension.Length is 0 or > 16)
                return string.Empty;

            return extension.IndexOfAny(System.IO.Path.GetInvalidFileNameChars()) >= 0
                ? string.Empty
                : extension;
        }

        public void Dispose()
        {
            Stream? source = Interlocked.Exchange(ref _retainedSource, null);
            source?.Dispose();

            string? temporaryFile = Interlocked.Exchange(ref _temporaryFile, null);
            if (temporaryFile != null)
                File.Delete(temporaryFile);
        }

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
