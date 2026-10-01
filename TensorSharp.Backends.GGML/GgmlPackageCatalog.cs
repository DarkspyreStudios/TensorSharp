#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.GGML;

public static partial class GgmlNativeLoader
{
    private const string PackageCatalogError = "The installed GGML package catalog is invalid or conflicting.";
    private const string PackageCatalogSchema = "tensorsharp-native-artifacts/1";

    /// <summary>
    /// Resolves current-build package catalogs at fixed deployed locations without loading native code.
    /// Missing catalogs yield no candidates. Invalid or competing catalogs refuse the entire resolution.
    /// </summary>
    /// <remarks>Call explicitly during discovery or initialization, not from a constructor or metadata getter.</remarks>
    public static Task<IReadOnlyList<GgmlNativeCandidate>> ResolvePackageCandidatesAsync(
        CancellationToken cancellationToken = default)
    {
        string? assemblyDirectory = Path.GetDirectoryName(typeof(GgmlNativeLoader).Assembly.Location);
        string[] roots = string.IsNullOrEmpty(assemblyDirectory)
            ? [AppContext.BaseDirectory]
            : [AppContext.BaseDirectory, assemblyDirectory];
        return ResolvePackageCandidatesCoreAsync(roots, RuntimeIdentifier, cancellationToken);
    }

    private static async Task<IReadOnlyList<GgmlNativeCandidate>> ResolvePackageCandidatesCoreAsync(
        IReadOnlyList<string> roots, string rid, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            string[] variants = rid switch
            {
                "osx-arm64" => ["metal"],
                "linux-x64" or "linux-arm64" or "win-x64" => ["cuda13", "vulkan", "cpu"],
                "win-arm64" => ["vulkan", "cpu"],
                _ => []
            };
            if (variants.Length == 0)
                return Array.AsReadOnly(Array.Empty<GgmlNativeCandidate>());
            string baseline = rid == "osx-arm64" ? "metal" : "cpu";
            string entry = rid.StartsWith("win-", StringComparison.Ordinal) ? "GgmlOps.dll" :
                rid.StartsWith("osx-", StringComparison.Ordinal) ? "libGgmlOps.dylib" : "libGgmlOps.so";
            var bindings = new Dictionary<string, GgmlNativeCandidate>(StringComparer.Ordinal);
            var seenRoots = new HashSet<string>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
            foreach (string location in roots)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!Path.IsPathFullyQualified(location))
                    throw new InvalidDataException();
                string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(location));
                if (!seenRoots.Add(root))
                    continue;
                foreach (string variant in variants)
                {
                    bool isBaseline = variant == baseline;
                    string path = Path.Combine(root, "ggml", (isBaseline ? "baseline" : variant) + ".artifact.json");
                    EnsureCatalogPathUnlinked(path);
                    FileAttributes attributes;
                    try
                    {
                        attributes = File.GetAttributes(path);
                    }
                    catch (Exception missing) when (missing is FileNotFoundException or DirectoryNotFoundException)
                    {
                        continue;
                    }
                    if ((attributes & (FileAttributes.Directory | FileAttributes.ReparsePoint | FileAttributes.Device)) != 0)
                        throw new InvalidDataException();
                    if (bindings.ContainsKey(variant))
                        throw new InvalidDataException();
                    RequireCatalogFile(path);
                    using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                        4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
                    using JsonDocument document = await JsonDocument.ParseAsync(stream,
                        cancellationToken: cancellationToken).ConfigureAwait(false);
                    RejectDuplicateCatalogProperties(document.RootElement);
                    string directory = isBaseline ? Path.Combine(root, "runtimes", rid, "native") : Path.Combine(root, "ggml", variant);
                    GgmlNativeCandidate candidate = ParsePackageCatalog(document.RootElement, directory, rid, variant, entry);
                    foreach (GgmlNativeFile file in candidate.Files)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        string filePath = Path.Combine(directory, file.Path.Replace('/', Path.DirectorySeparatorChar));
                        FileInfo fileInfo = RequireCatalogFile(filePath);
                        if (fileInfo.Length != file.Size)
                            throw new InvalidDataException();
                        using FileStream bytes = new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read,
                            4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
                        string digest = Convert.ToHexStringLower(await SHA256.HashDataAsync(bytes, cancellationToken).ConfigureAwait(false));
                        if (!string.Equals(digest, file.Sha256, StringComparison.Ordinal))
                            throw new InvalidDataException();
                    }
                    bindings.Add(variant, candidate);
                }
            }

            // Each artifact advertises one primary backend plus CPU; these are one file binding,
            // not independently loadable libraries. The owner still stops after any possible load.
            var candidates = new List<GgmlNativeCandidate>();
            foreach (string variant in variants.Where(v => v != "cpu"))
                if (bindings.TryGetValue(variant, out GgmlNativeCandidate? candidate))
                    candidates.Add(candidate);
            if (bindings.TryGetValue(baseline, out GgmlNativeCandidate? cpu))
                candidates.Add(cpu with { Backend = GgmlBackendType.Cpu });
            foreach (string variant in variants.Where(v => v != baseline))
                if (bindings.TryGetValue(variant, out GgmlNativeCandidate? candidate))
                    candidates.Add(candidate with { Backend = GgmlBackendType.Cpu });
            cancellationToken.ThrowIfCancellationRequested();
            return candidates.AsReadOnly();
        }
        catch (Exception error) when (error is InvalidDataException or IOException or UnauthorizedAccessException or JsonException or
            ArgumentException or NotSupportedException or InvalidOperationException or FormatException)
        {
            // Do not expose deployed paths, raw JSON or supplier exception text in discovery diagnostics.
            throw new InvalidDataException(PackageCatalogError);
        }
    }

    private static GgmlNativeCandidate ParsePackageCatalog(JsonElement catalog, string directory, string rid, string variant, string entry)
    {
        RequireCatalogObject(catalog, "schema", "driverId", "rid", "variant", "version", "tensorSharpBuild",
            "nativeAbi", "tensorSharp", "ggml", "backends", "entryLibrary", "files", "components");
        RequireCatalogValue(catalog, "schema", PackageCatalogSchema);
        RequireCatalogValue(catalog, "driverId", "ggml");
        RequireCatalogValue(catalog, "rid", rid);
        RequireCatalogValue(catalog, "variant", variant);
        RequireCatalogValue(catalog, "version", TensorSharpBuild);
        RequireCatalogValue(catalog, "tensorSharpBuild", TensorSharpBuild);
        RequireCatalogValue(catalog, "nativeAbi", NativeAbi);
        RequireCatalogValue(catalog, "entryLibrary", entry);
        if (!CatalogHex(NativeAbi, 64) || !CatalogHex(GgmlCommit, 40) || string.IsNullOrWhiteSpace(TensorSharpBuild))
            throw new InvalidDataException();

        JsonElement supplier = catalog.GetProperty("tensorSharp"), upstream = catalog.GetProperty("ggml");
        RequireCatalogObject(supplier, "packageVersion", "packageCommit", "nativeSourceCommit");
        RequireCatalogValue(supplier, "packageVersion", TensorSharpBuild);
        string source = CatalogText(supplier, "nativeSourceCommit");
        if (!CatalogHex(source, 40) || !CatalogHex(CatalogText(supplier, "packageCommit"), 40))
            throw new InvalidDataException();
        RequireCatalogObject(upstream, "version", "commit");
        _ = CatalogText(upstream, "version");
        RequireCatalogValue(upstream, "commit", GgmlCommit);
        JsonElement backendList = catalog.GetProperty("backends");
        string primary = variant == "cuda13" ? "cuda" : variant;
        string[] expected = variant == "cpu" ? ["cpu"] : [primary, "cpu"];
        if (backendList.ValueKind != JsonValueKind.Array || !backendList.EnumerateArray().Select(e => e.GetString()).SequenceEqual(expected))
            throw new InvalidDataException();

        JsonElement fileList = catalog.GetProperty("files");
        if (fileList.ValueKind != JsonValueKind.Array || fileList.GetArrayLength() == 0)
            throw new InvalidDataException();
        var files = new Dictionary<string, GgmlNativeFile>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonElement item in fileList.EnumerateArray())
        {
            GgmlNativeFile file = ParseCatalogFile(item);
            if (!files.TryAdd(file.Path, file))
                throw new InvalidDataException();
        }
        if (!files.TryGetValue(entry, out GgmlNativeFile? entryFile) || entryFile.Path != entry)
            throw new InvalidDataException();
        ValidateCatalogComponents(catalog.GetProperty("components"), files, source, entry);
        GgmlBackendType backend = variant switch
        {
            "cpu" => GgmlBackendType.Cpu,
            "metal" => GgmlBackendType.Metal,
            "vulkan" => GgmlBackendType.Vulkan,
            "cuda13" => GgmlBackendType.Cuda,
            _ => throw new InvalidDataException()
        };
        return new GgmlNativeCandidate(directory, TensorSharpBuild, rid, variant, backend,
            Array.AsReadOnly(files.Values.ToArray()))
        { NativeAbi = NativeAbi };
    }

    private static void ValidateCatalogComponents(JsonElement components, Dictionary<string, GgmlNativeFile> files, string source, string entry)
    {
        if (components.ValueKind != JsonValueKind.Array || components.GetArrayLength() == 0)
            throw new InvalidDataException();
        var ids = new HashSet<string>(StringComparer.Ordinal);
        var covered = new HashSet<string>(StringComparer.Ordinal);
        var binaryOwners = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (JsonElement component in components.EnumerateArray())
        {
            RequireCatalogObject(component, "id", "name", "kind", "supplier", "version", "sourceId", "binaryFiles", "evidenceFiles");
            string id = CatalogText(component, "id"), kind = CatalogText(component, "kind");
            if (!id.All(c => c is >= 'a' and <= 'z' or >= '0' and <= '9' or '.' or '_' or '-') ||
                !(id[0] is >= 'a' and <= 'z' or >= '0' and <= '9') || !ids.Add(id))
                throw new InvalidDataException();
            _ = CatalogText(component, "name");
            _ = CatalogText(component, "supplier");
            _ = CatalogText(component, "version");
            _ = CatalogText(component, "sourceId");
            bool core = id is "tensorsharp" or "ggml";
            if (core)
            {
                RequireCatalogValue(component, "kind", id == "tensorsharp" ? "bridge" : "static");
                RequireCatalogValue(component, "version", id == "tensorsharp" ? TensorSharpBuild : GgmlCommit);
                RequireCatalogValue(component, "sourceId", "git:" + (id == "tensorsharp" ? source : GgmlCommit));
            }
            else if (kind != "redistributed")
                throw new InvalidDataException();
            foreach (string field in new[] { "binaryFiles", "evidenceFiles" })
            {
                JsonElement references = component.GetProperty(field);
                if (references.ValueKind != JsonValueKind.Array || references.GetArrayLength() == 0 || (core && references.GetArrayLength() != 1))
                    throw new InvalidDataException();
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (JsonElement reference in references.EnumerateArray())
                {
                    GgmlNativeFile file = ParseCatalogFile(reference);
                    bool binary = field == "binaryFiles";
                    if (!seen.Add(file.Path) || !files.TryGetValue(file.Path, out GgmlNativeFile? actual) || actual != file ||
                        (binary ? file.Path.Contains('/') : !file.Path.StartsWith("licenses/", StringComparison.Ordinal)))
                        throw new InvalidDataException();
                    if (core && file.Path != (binary ? entry : id == "tensorsharp" ? "licenses/TensorSharp-LICENSE.txt" : "licenses/ggml-LICENSE.txt"))
                        throw new InvalidDataException();
                    if (!core && file.Path is "licenses/TensorSharp-LICENSE.txt" or "licenses/ggml-LICENSE.txt")
                        throw new InvalidDataException();
                    if (binary)
                    {
                        if (binaryOwners.TryGetValue(file.Path, out string? owner) &&
                            !(core && owner is "tensorsharp" or "ggml"))
                            throw new InvalidDataException();
                        binaryOwners[file.Path] = id;
                    }
                    covered.Add(file.Path);
                }
            }
        }
        if (!ids.Contains("tensorsharp") || !ids.Contains("ggml") || !covered.SetEquals(files.Keys))
            throw new InvalidDataException();
    }

    private static GgmlNativeFile ParseCatalogFile(JsonElement item)
    {
        RequireCatalogObject(item, "path", "size", "sha256");
        string path = CatalogText(item, "path"), hash = CatalogText(item, "sha256");
        if (CheckRelativePath(path) != null || path.Any(c => c is '<' or '>' or '"' or '|' or '?' or '*') ||
            path.Split('/').Any(p => p.EndsWith('.') || p.EndsWith(' ')) ||
            !item.GetProperty("size").TryGetInt64(out long size) || size <= 0 || !CatalogHex(hash, 64))
            throw new InvalidDataException();
        return new(path, size, hash);
    }

    private static void RequireCatalogObject(JsonElement value, params string[] fields)
    {
        if (value.ValueKind != JsonValueKind.Object ||
            !value.EnumerateObject().Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal)
                .SequenceEqual(fields.OrderBy(n => n, StringComparer.Ordinal)))
            throw new InvalidDataException();
    }

    private static string CatalogText(JsonElement value, string field)
    {
        JsonElement item = value.GetProperty(field);
        string? text = item.ValueKind == JsonValueKind.String ? item.GetString() : null;
        if (string.IsNullOrEmpty(text) || text.Trim() != text || text.Any(char.IsControl))
            throw new InvalidDataException();
        return text;
    }

    private static void RequireCatalogValue(JsonElement value, string field, string expected)
    {
        if (!string.Equals(CatalogText(value, field), expected, StringComparison.Ordinal))
            throw new InvalidDataException();
    }

    private static bool CatalogHex(string value, int length) => value.Length == length &&
        value.All(c => c is >= 'a' and <= 'f' or >= '0' and <= '9');

    private static void RejectDuplicateCatalogProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (JsonProperty property in value.EnumerateObject())
            {
                if (!names.Add(property.Name))
                    throw new InvalidDataException();
                RejectDuplicateCatalogProperties(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (JsonElement item in value.EnumerateArray())
                RejectDuplicateCatalogProperties(item);
    }

    private static void EnsureCatalogPathUnlinked(string path)
    {
        if (new FileInfo(path).LinkTarget != null)
            throw new InvalidDataException();
        for (DirectoryInfo? directory = new FileInfo(path).Directory; directory != null; directory = directory.Parent)
            if (directory.LinkTarget != null)
                throw new InvalidDataException();
    }

    private static FileInfo RequireCatalogFile(string path)
    {
        EnsureCatalogPathUnlinked(path);
        var info = new FileInfo(path);
        if (!info.Exists || info.Length <= 0 || (info.Attributes & (FileAttributes.Directory | FileAttributes.ReparsePoint | FileAttributes.Device)) != 0)
            throw new InvalidDataException();
        return info;
    }
}
