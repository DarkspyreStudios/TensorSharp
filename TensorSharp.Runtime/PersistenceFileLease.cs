// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Runtime
{
    /// <summary>Owns either retained file-store streams or one named temporary projection.
    /// All files stay available together until the model or reader releases this lease.</summary>
    internal sealed class PersistenceFileLease : IDisposable, IAsyncDisposable
    {
        private const int CopyBufferSize = 81920;
        private List<Stream>? _retainedSources;
        private string? _temporaryDirectory;

        private PersistenceFileLease(string filePath, IReadOnlyList<string> filePaths, List<Stream>? sources, string? directory)
        {
            FilePath = filePath;
            FilePaths = filePaths;
            _retainedSources = sources;
            _temporaryDirectory = directory;
        }

        internal string FilePath { get; }
        internal IReadOnlyList<string> FilePaths { get; }
        internal bool IsTemporary => _temporaryDirectory != null;

        internal static Task<PersistenceFileLease> AcquireAsync(PersistenceFileReference source, CancellationToken cancellationToken = default) =>
            AcquireAsync(PersistenceFileSet.Single(source), cancellationToken);

        internal static async Task<PersistenceFileLease> AcquireAsync(PersistenceFileSet source, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(source);
            var streams = new Dictionary<string, Stream>(StringComparer.Ordinal);
            string? directory = null;
            try
            {
                foreach (var pair in source.Files)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    PersistenceFileReference artifact = pair.Value;
                    Stream stream = await artifact.Store.OpenReadAsync(artifact.StoreName, artifact.AssetName, cancellationToken).ConfigureAwait(false)
                        ?? throw new FileNotFoundException("A persistence artifact does not exist.", pair.Key);
                    streams.Add(pair.Key, stream);
                    if (!stream.CanRead)
                        throw new InvalidDataException("A persistence artifact returned an unreadable stream.");
                }
                cancellationToken.ThrowIfCancellationRequested();
                string? root = InPlaceRoot(source, streams);
                if (root != null)
                {
                    return new PersistenceFileLease(Path.Combine(root, source.PrimaryPath),
                        source.Files.Keys.Select(name => Path.Combine(root, name)).ToArray(), streams.Values.ToList(), null);
                }

                directory = Directory.CreateTempSubdirectory("TensorSharp-persistence-").FullName;
                foreach (var pair in streams)
                {
                    string path = Path.Combine(directory, pair.Key);
                    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                    await using var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                        CopyBufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
                    await pair.Value.CopyToAsync(output, CopyBufferSize, cancellationToken).ConfigureAwait(false);
                    await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                }
                cancellationToken.ThrowIfCancellationRequested();
                foreach (Stream stream in streams.Values) await stream.DisposeAsync().ConfigureAwait(false);
                streams.Clear();
                return new PersistenceFileLease(Path.Combine(directory, source.PrimaryPath),
                    source.Files.Keys.Select(name => Path.Combine(directory, name)).ToArray(), null, directory);
            }
            catch
            {
                try { foreach (Stream stream in streams.Values) await stream.DisposeAsync().ConfigureAwait(false); }
                finally { if (directory != null) Directory.Delete(directory, recursive: true); }
                throw;
            }
        }

        private static string? InPlaceRoot(PersistenceFileSet source, Dictionary<string, Stream> streams)
        {
            if (streams[source.PrimaryPath] is not FileStream primary) return null;
            string? root = Path.GetFullPath(primary.Name);
            foreach (string _ in source.PrimaryPath.Split('/'))
            {
                root = Path.GetDirectoryName(root);
                if (root == null) return null;
            }
            StringComparison comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            return streams.All(pair => pair.Value is FileStream file
                && string.Equals(Path.GetFullPath(file.Name), Path.Combine(root, pair.Key), comparison)) ? root : null;
        }

        public void Dispose()
        {
            List<Stream>? sources = Interlocked.Exchange(ref _retainedSources, null);
            try { if (sources != null) foreach (Stream source in sources) source.Dispose(); }
            finally
            {
                string? directory = Interlocked.Exchange(ref _temporaryDirectory, null);
                if (directory != null) Directory.Delete(directory, recursive: true);
            }
        }

        public ValueTask DisposeAsync() { Dispose(); return ValueTask.CompletedTask; }
    }
}
