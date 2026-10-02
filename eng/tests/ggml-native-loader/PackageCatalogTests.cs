using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using TensorSharp.GGML;
using Xunit;

namespace TensorSharp.NativeLoader.Tests;

public sealed class PackageCatalogTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ggml-catalog-" + Guid.NewGuid().ToString("N"));

    public PackageCatalogTests() => Directory.CreateDirectory(root);

    [Fact]
    public async Task PublicResolverHasExactSignatureAndDoesNotLoadNativeCode()
    {
        MethodInfo? method = typeof(GgmlNativeLoader).GetMethod("ResolvePackageCandidatesAsync", [typeof(CancellationToken)]);
        Assert.NotNull(method);
        Assert.Equal(typeof(Task<IReadOnlyList<GgmlNativeCandidate>>), method.ReturnType);
        Assert.True(method.GetParameters()[0].HasDefaultValue);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        var task = (Task<IReadOnlyList<GgmlNativeCandidate>>)method.Invoke(null, [cancelled.Token])!;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.Null(GgmlNativeLoader.Current);
        Assert.Equal(GgmlRuntimeState.Unconfigured, GgmlNativeLoader.State);
    }

    [Fact]
    public async Task MissingCatalogsDoNotDiscoverLooseLibrariesOrAmbientPaths()
    {
        File.WriteAllText(Path.Combine(root, "libGgmlOps.so"), "not native");
        Directory.CreateDirectory(Path.Combine(root, "ggml", "cuda13"));
        File.WriteAllText(Path.Combine(root, "ggml", "cuda13", "libGgmlOps.so"), "not native");
        Assert.Empty(await Resolve("linux-x64", root));
        Assert.Null(GgmlNativeLoader.Current);
    }

    [Fact]
    public async Task PublicResolverUsesFixedDeployedBaselineAndReturnsValidatedFilesWithoutNativeEntry()
    {
        string rid = GgmlNativeLoader.RuntimeIdentifier;
        Assert.Contains(rid, new[] { "osx-arm64", "linux-x64", "linux-arm64", "win-x64", "win-arm64" });
        Write(rid, rid == "osx-arm64" ? "metal" : "cpu", true);
        string deployedCatalogs = Path.Combine(AppContext.BaseDirectory, "ggml");
        string deployedRuntime = Path.Combine(AppContext.BaseDirectory, "runtimes");
        Assert.False(Directory.Exists(deployedCatalogs));
        Assert.False(Directory.Exists(deployedRuntime));
        Directory.Move(Path.Combine(root, "ggml"), deployedCatalogs);
        try
        {
            Directory.Move(Path.Combine(root, "runtimes"), deployedRuntime);
            MethodInfo method = typeof(GgmlNativeLoader).GetMethod("ResolvePackageCandidatesAsync", [typeof(CancellationToken)])!;
            var task = (Task<IReadOnlyList<GgmlNativeCandidate>>)method.Invoke(null, [CancellationToken.None])!;
            IReadOnlyList<GgmlNativeCandidate> candidates = await task;
            Assert.NotEmpty(candidates);
            Assert.All(candidates, c => Assert.Null(GgmlNativeLoader.Check(c)));
            Assert.All(candidates, c => Assert.Equal(Path.Combine(deployedRuntime, rid, "native"), c.Directory));
            Assert.Null(GgmlNativeLoader.Current);
            Assert.Equal(GgmlRuntimeState.Unconfigured, GgmlNativeLoader.State);
        }
        finally
        {
            Directory.Delete(deployedCatalogs, true);
            if (Directory.Exists(deployedRuntime))
                Directory.Delete(deployedRuntime, true);
        }
    }

    [Theory]
    [InlineData("linux-x64")]
    [InlineData("linux-arm64")]
    [InlineData("win-x64")]
    public async Task CatalogsExpandIntoPrimaryThenBaselineAndOptionalCpuCandidates(string rid)
    {
        Write(rid, "cpu", true);
        Write(rid, "vulkan", false);
        Write(rid, "cuda13", false);
        IReadOnlyList<GgmlNativeCandidate> candidates = await Resolve(rid, root, root);
        Assert.Equal(new[] { GgmlBackendType.Cuda, GgmlBackendType.Vulkan, GgmlBackendType.Cpu,
            GgmlBackendType.Cpu, GgmlBackendType.Cpu }, candidates.Select(c => c.Backend));
        Assert.Equal(new[] { "cuda13", "vulkan", "cpu", "cuda13", "vulkan" }, candidates.Select(c => c.Variant));
        Assert.All(candidates, c => Assert.Equal(GgmlNativeLoader.NativeAbi, c.NativeAbi));
        Assert.All(candidates, c => Assert.Contains(c.Files, f => f.Path == "licenses/ggml-LICENSE.txt"));
        Assert.IsAssignableFrom<IReadOnlyList<GgmlNativeCandidate>>(candidates);
        Assert.True(((ICollection<GgmlNativeCandidate>)candidates).IsReadOnly);
        Assert.True(((ICollection<GgmlNativeFile>)candidates[0].Files).IsReadOnly);
    }

    [Fact]
    public async Task WindowsArm64HasCpuAndVulkanAndMacBaselineHasMetalThenCpu()
    {
        Write("win-arm64", "cpu", true);
        Write("win-arm64", "vulkan", false);
        Assert.Equal(new[] { GgmlBackendType.Vulkan, GgmlBackendType.Cpu, GgmlBackendType.Cpu },
            (await Resolve("win-arm64", root)).Select(c => c.Backend));
        Directory.Delete(Path.Combine(root, "ggml"), true);
        Directory.Delete(Path.Combine(root, "runtimes"), true);
        Write("osx-arm64", "metal", true);
        Assert.Equal(new[] { GgmlBackendType.Metal, GgmlBackendType.Cpu },
            (await Resolve("osx-arm64", root)).Select(c => c.Backend));
    }

    [Theory]
    [InlineData("schema")]
    [InlineData("driverId")]
    [InlineData("rid")]
    [InlineData("variant")]
    [InlineData("version")]
    [InlineData("tensorSharpBuild")]
    [InlineData("nativeAbi")]
    [InlineData("entryLibrary")]
    public async Task MissingOrConflictingIdentityRefusesTheWholeCatalog(string field)
    {
        JsonObject catalog = Write("linux-x64", "cpu", true);
        catalog[field] = "wrong";
        Save(catalog, true);
        await Refused("linux-x64");
        catalog.Remove(field);
        Save(catalog, true);
        await Refused("linux-x64");
    }

    [Fact]
    public async Task SourceUpstreamBackendAndUnknownFieldsAreFailClosed()
    {
        foreach (Action<JsonObject> mutate in new Action<JsonObject>[] {
            c => c["ggml"]!["commit"] = new string('0', 40),
            c => c["tensorSharp"]!["packageVersion"] = "older",
            c => c["tensorSharp"]!["packageCommit"] = "main",
            c => c["tensorSharp"]!["nativeSourceCommit"] = "main",
            c => c["backends"] = new JsonArray("cuda", "cpu"),
            c => c["directory"] = "/untrusted",
            c => c["files"] = new JsonArray(),
            c => c["components"] = new JsonArray() })
        {
            JsonObject catalog = Write("linux-x64", "cpu", true);
            mutate(catalog);
            Save(catalog, true);
            await Refused("linux-x64");
        }
    }

    [Fact]
    public async Task DuplicateJsonKeysMalformedJsonAndNonobjectRootsRefuse()
    {
        JsonObject catalog = Write("linux-x64", "cpu", true);
        string json = catalog.ToJsonString();
        foreach (string invalid in new[] { "[]", "{", json.Replace("\"driverId\":\"ggml\"", "\"driverId\":\"ggml\",\"driverId\":\"ggml\""),
            json.Replace("\"commit\":", "\"commit\":\"ignored\",\"commit\":") })
        {
            File.WriteAllText(CatalogPath(true, "cpu"), invalid);
            await Refused("linux-x64");
        }
    }

    [Fact]
    public async Task DuplicateBindingsAcrossDistinctRootsRefuseEvenWithEqualBytes()
    {
        Write("linux-x64", "cpu", true);
        string second = Path.Combine(root, "second");
        Directory.CreateDirectory(Path.Combine(second, "ggml"));
        File.Copy(CatalogPath(true, "cpu"), Path.Combine(second, "ggml", "baseline.artifact.json"));
        await Assert.ThrowsAsync<InvalidDataException>(() => Resolve("linux-x64", root, second));
    }

    [Fact]
    public async Task ActualFilesAndComponentReferencesMustMatchTheWholeClosure()
    {
        foreach (Action<JsonObject> mutate in new Action<JsonObject>[] {
            c => c["files"]![0]!["sha256"] = new string('0', 64),
            c => c["files"]![0]!["size"] = 999,
            c => c["files"]![0]!["path"] = "../escape",
            c => c["files"]!.AsArray().Add(c["files"]![0]!.DeepClone()),
            c => c["components"]![0]!["binaryFiles"]![0]!["sha256"] = new string('0', 64),
            c => c["components"]!.AsArray().RemoveAt(1) })
        {
            JsonObject catalog = Write("linux-x64", "cpu", true);
            mutate(catalog);
            Save(catalog, true);
            await Refused("linux-x64");
        }
        Write("linux-x64", "cpu", true);
        File.Delete(Path.Combine(root, "runtimes", "linux-x64", "native", "libGgmlOps.so"));
        await Refused("linux-x64");
    }

    [Fact]
    public async Task LinkedCatalogPayloadAndDirectoryRefuse()
    {
        Write("linux-x64", "cpu", true);
        string catalog = CatalogPath(true, "cpu");
        File.Move(catalog, catalog + ".target");
        File.CreateSymbolicLink(catalog, catalog + ".target");
        await Refused("linux-x64");
        File.Delete(catalog);
        File.Move(catalog + ".target", catalog);
        string entry = Path.Combine(root, "runtimes", "linux-x64", "native", "libGgmlOps.so");
        File.Move(entry, entry + ".target");
        File.CreateSymbolicLink(entry, entry + ".target");
        await Refused("linux-x64");
        File.Delete(entry);
        File.Move(entry + ".target", entry);
        string native = Path.GetDirectoryName(entry)!;
        Directory.Move(native, native + ".target");
        Directory.CreateSymbolicLink(native, native + ".target");
        await Refused("linux-x64");
    }

    [Fact]
    public async Task MissingBaselineStillUsesAdvertisedOptionalCpuWithoutInventingAnArtifact()
    {
        Write("linux-arm64", "cuda13", false);
        IReadOnlyList<GgmlNativeCandidate> candidates = await Resolve("linux-arm64", root);
        Assert.Equal(new[] { GgmlBackendType.Cuda, GgmlBackendType.Cpu }, candidates.Select(c => c.Backend));
        Assert.Equal(candidates[0].Directory, candidates[1].Directory);
        Assert.Equal(candidates[0].Files, candidates[1].Files);
    }

    [Theory]
    [InlineData("")]
    [InlineData("../escaped")]
    [InlineData("nested/../escaped")]
    [InlineData("nested//escaped")]
    [InlineData("nested/./escaped")]
    [InlineData("/absolute")]
    [InlineData("C:/absolute")]
    [InlineData("nested\\escaped")]
    [InlineData("trailing.")]
    [InlineData("trailing ")]
    [InlineData("wild*card")]
    public async Task CatalogFilePathsRefuseNonportableNames(string path)
    {
        JsonObject catalog = Write("linux-x64", "cpu", true);
        catalog["files"]![0]!["path"] = path;
        Save(catalog, true);
        await Refused("linux-x64");
    }

    [Fact]
    public async Task CatalogRequiresPositiveOrdinaryFilesAndExactCaseEntry()
    {
        foreach (Action<JsonObject> mutate in new Action<JsonObject>[] {
            c => c["files"]![0]!["size"] = 0,
            c => c["files"]![0]!["size"] = -1,
            c => c["files"]![0]!["size"] = "7",
            c => c["files"]![0]!["sha256"] = new string('A', 64),
            c => c["files"]![0]!["path"] = "LIBGGmlOps.so",
            c => c["files"]!.AsArray().Add(new JsonObject { ["path"] = "unmapped.txt", ["size"] = 1, ["sha256"] = new string('a', 64) }) })
        {
            JsonObject catalog = Write("linux-x64", "cpu", true);
            mutate(catalog);
            Save(catalog, true);
            await Refused("linux-x64");
        }
        Write("linux-x64", "cpu", true);
        string catalogPath = CatalogPath(true, "cpu");
        File.Delete(catalogPath);
        Directory.CreateDirectory(catalogPath);
        await Refused("linux-x64");
    }

    [Fact]
    public async Task UnsupportedRidAndUnadmittedVariantAreNeverDiscovered()
    {
        Write("win-arm64", "cpu", true);
        Write("win-arm64", "cuda13", false);
        Assert.Single(await Resolve("win-arm64", root));
        Assert.Empty(await Resolve("osx-x64", root));
    }

    [Fact]
    public async Task RealCollectibleSupplierCatalogGenerationRetiresAfterReturnedDataIsReleased()
    {
        Write("linux-x64", "cpu", true);
        WeakReference[] roots = await ReadCollectibleCatalogAndUnload(root);
        for (int attempt = 0; attempt < 20 && roots.Any(reference => reference.IsAlive); attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        Assert.All(roots, reference => Assert.False(reference.IsAlive));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static async Task<WeakReference[]> ReadCollectibleCatalogAndUnload(string root)
    {
        var context = new AssemblyLoadContext("managed-package-catalog-only", isCollectible: true);
        Assembly assembly = context.LoadFromAssemblyPath(typeof(GgmlNativeLoader).Assembly.Location);
        Type loader = assembly.GetType(typeof(GgmlNativeLoader).FullName!)!;
        Assert.Same(context, AssemblyLoadContext.GetLoadContext(loader.Assembly));
        MethodInfo method = loader.GetMethod("ResolvePackageCandidatesCoreAsync", BindingFlags.Static | BindingFlags.NonPublic)!;
        var operation = (Task)method.Invoke(null, [new[] { root }, "linux-x64", CancellationToken.None])!;
        await operation;
        object candidates = operation.GetType().GetProperty("Result")!.GetValue(operation)!;
        object candidate = ((System.Collections.IEnumerable)candidates).Cast<object>().Single();
        Assert.Same(assembly, candidate.GetType().Assembly);
        Assert.Null(loader.GetProperty("Current")!.GetValue(null));
        WeakReference[] references = [new(context), new(assembly), new(loader), new(candidate), new(candidates), new(operation)];
        context.Unload();
        return references;
    }

    private async Task Refused(string rid)
    {
        InvalidDataException error = await Assert.ThrowsAsync<InvalidDataException>(() => Resolve(rid, root));
        Assert.Equal("The installed GGML package catalog is invalid or conflicting.", error.Message);
        Assert.DoesNotContain(root, error.ToString());
        Assert.Null(GgmlNativeLoader.Current);
    }

    private static Task<IReadOnlyList<GgmlNativeCandidate>> Resolve(string rid, params string[] roots)
    {
        MethodInfo? method = typeof(GgmlNativeLoader).GetMethod("ResolvePackageCandidatesCoreAsync", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        return (Task<IReadOnlyList<GgmlNativeCandidate>>)method.Invoke(null, [roots, rid, CancellationToken.None])!;
    }

    private JsonObject Write(string rid, string variant, bool baseline)
    {
        string entry = rid.StartsWith("win-") ? "GgmlOps.dll" : rid.StartsWith("osx-") ? "libGgmlOps.dylib" : "libGgmlOps.so";
        string directory = baseline ? Path.Combine(root, "runtimes", rid, "native") : Path.Combine(root, "ggml", variant);
        Directory.CreateDirectory(Path.Combine(directory, "licenses"));
        JsonObject FileRecord(string name)
        {
            string path = Path.Combine(directory, name);
            File.WriteAllText(path, "controlled filesystem evidence only");
            return new JsonObject
            {
                ["path"] = name,
                ["size"] = new FileInfo(path).Length,
                ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)))
            };
        }
        JsonObject binary = FileRecord(entry), tsLicense = FileRecord("licenses/TensorSharp-LICENSE.txt"), ggmlLicense = FileRecord("licenses/ggml-LICENSE.txt");
        JsonObject Component(string id, string kind, string source, JsonObject evidence) => new()
        {
            ["id"] = id,
            ["name"] = id,
            ["kind"] = kind,
            ["supplier"] = id,
            ["version"] = id == "ggml" ? GgmlNativeLoader.GgmlCommit : GgmlNativeLoader.TensorSharpBuild,
            ["sourceId"] = "git:" + source,
            ["binaryFiles"] = new JsonArray(binary.DeepClone()),
            ["evidenceFiles"] = new JsonArray(evidence.DeepClone())
        };
        var catalog = new JsonObject
        {
            ["schema"] = "tensorsharp-native-artifacts/1",
            ["driverId"] = "ggml",
            ["rid"] = rid,
            ["variant"] = variant,
            ["version"] = GgmlNativeLoader.TensorSharpBuild,
            ["tensorSharpBuild"] = GgmlNativeLoader.TensorSharpBuild,
            ["nativeAbi"] = GgmlNativeLoader.NativeAbi,
            ["tensorSharp"] = new JsonObject
            {
                ["packageVersion"] = GgmlNativeLoader.TensorSharpBuild,
                ["packageCommit"] = new string('a', 40),
                ["nativeSourceCommit"] = new string('b', 40)
            },
            ["ggml"] = new JsonObject { ["version"] = "0.9.0", ["commit"] = GgmlNativeLoader.GgmlCommit },
            ["backends"] = variant == "cpu" ? new JsonArray("cpu") : new JsonArray(variant == "cuda13" ? "cuda" : variant, "cpu"),
            ["entryLibrary"] = entry,
            ["files"] = new JsonArray(binary.DeepClone(), tsLicense.DeepClone(), ggmlLicense.DeepClone()),
            ["components"] = new JsonArray(Component("tensorsharp", "bridge", new string('b', 40), tsLicense),
                Component("ggml", "static", GgmlNativeLoader.GgmlCommit, ggmlLicense))
        };
        Save(catalog, baseline);
        return catalog;
    }

    private string CatalogPath(bool baseline, string variant) => Path.Combine(root, "ggml", (baseline ? "baseline" : variant) + ".artifact.json");

    private void Save(JsonObject catalog, bool baseline)
    {
        Directory.CreateDirectory(Path.Combine(root, "ggml"));
        File.WriteAllText(CatalogPath(baseline, baseline ? "cpu" : catalog["variant"]!.GetValue<string>()), catalog.ToJsonString());
    }

    public void Dispose() => Directory.Delete(root, true);
}
