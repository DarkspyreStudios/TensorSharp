using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;

namespace TensorSharp.Runtime
{
    /// <summary>A complete, named set of persistence artifacts consumed as one model.
    /// Names are portable relative paths, independent of the store's asset keys.</summary>
    public sealed class PersistenceFileSet
    {
        public string PrimaryPath { get; }
        public IReadOnlyDictionary<string, PersistenceFileReference> Files { get; }

        public PersistenceFileSet(string primaryPath, IReadOnlyDictionary<string, PersistenceFileReference> files)
        {
            ArgumentNullException.ThrowIfNull(files);
            var snapshot = new Dictionary<string, PersistenceFileReference>(StringComparer.Ordinal);
            var portableNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var pair in files)
            {
                ValidatePath(pair.Key);
                ArgumentNullException.ThrowIfNull(pair.Value);
                if (!portableNames.Add(pair.Key))
                    throw new ArgumentException("Artifact paths must be unique on every supported filesystem.", nameof(files));
                snapshot.Add(pair.Key, pair.Value);
            }
            ValidatePath(primaryPath);
            if (!snapshot.ContainsKey(primaryPath))
                throw new ArgumentException("The primary artifact must belong to the file set.", nameof(primaryPath));
            foreach (string path in snapshot.Keys)
            {
                for (int index = path.IndexOf('/'); index >= 0; index = path.IndexOf('/', index + 1))
                    if (portableNames.Contains(path[..index]))
                        throw new ArgumentException("Artifact file and directory paths overlap.", nameof(files));
            }
            PrimaryPath = primaryPath;
            Files = new ReadOnlyDictionary<string, PersistenceFileReference>(snapshot);
        }

        internal static PersistenceFileSet Single(PersistenceFileReference source)
        {
            ArgumentNullException.ThrowIfNull(source);
            string name = Path.GetFileName(source.AssetName.Replace('\\', '/'));
            return new PersistenceFileSet(name, new Dictionary<string, PersistenceFileReference> { [name] = source });
        }

        private static void ValidatePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || path.Any(c => char.IsControl(c) || "\\:<>\"|?*".Contains(c))
                || path.Split('/').Any(part => part.Length == 0 || part is "." or ".." || part.EndsWith(' ') || part.EndsWith('.') || IsDeviceName(part)))
                throw new ArgumentException("Artifact names must be portable relative file paths.", nameof(path));
        }

        private static bool IsDeviceName(string part)
        {
            string stem = part.Split('.')[0].ToUpperInvariant();
            return stem is "CON" or "PRN" or "AUX" or "NUL"
                || (stem.Length == 4 && (stem.StartsWith("COM", StringComparison.Ordinal) || stem.StartsWith("LPT", StringComparison.Ordinal))
                    && "123456789¹²³".Contains(stem[3]));
        }
    }
}
