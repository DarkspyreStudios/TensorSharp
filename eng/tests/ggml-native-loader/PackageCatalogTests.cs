using System.Reflection;
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
            return new JsonObject { ["path"] = name, ["size"] = new FileInfo(path).Length,
                ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))) };
        }
        JsonObject binary = FileRecord(entry), tsLicense = FileRecord("licenses/TensorSharp-LICENSE.txt"), ggmlLicense = FileRecord("licenses/ggml-LICENSE.txt");
        JsonObject Component(string id, string kind, string source, JsonObject evidence) => new() {
            ["id"] = id, ["name"] = id, ["kind"] = kind, ["supplier"] = id,
            ["version"] = id == "ggml" ? GgmlNativeLoader.GgmlCommit : GgmlNativeLoader.TensorSharpBuild,
            ["sourceId"] = "git:" + source, ["binaryFiles"] = new JsonArray(binary.DeepClone()),
            ["evidenceFiles"] = new JsonArray(evidence.DeepClone()) };
        var catalog = new JsonObject {
            ["schema"] = "tensorsharp-native-artifacts/1", ["driverId"] = "ggml", ["rid"] = rid,
            ["variant"] = variant, ["version"] = GgmlNativeLoader.TensorSharpBuild,
            ["tensorSharpBuild"] = GgmlNativeLoader.TensorSharpBuild, ["nativeAbi"] = GgmlNativeLoader.NativeAbi,
            ["tensorSharp"] = new JsonObject { ["packageVersion"] = GgmlNativeLoader.TensorSharpBuild,
                ["packageCommit"] = new string('a', 40), ["nativeSourceCommit"] = new string('b', 40) },
            ["ggml"] = new JsonObject { ["version"] = "0.9.0", ["commit"] = GgmlNativeLoader.GgmlCommit },
            ["backends"] = variant == "cpu" ? new JsonArray("cpu") : new JsonArray(variant == "cuda13" ? "cuda" : variant, "cpu"),
            ["entryLibrary"] = entry, ["files"] = new JsonArray(binary.DeepClone(), tsLicense.DeepClone(), ggmlLicense.DeepClone()),
            ["components"] = new JsonArray(Component("tensorsharp", "bridge", new string('b', 40), tsLicense),
                Component("ggml", "static", GgmlNativeLoader.GgmlCommit, ggmlLicense)) };
        Save(catalog, baseline);
        return catalog;
    }

    private string CatalogPath(bool baseline, string variant) => Path.Combine(root, "ggml", (baseline ? "baseline" : variant) + ".artifact.json");

    private void Save(JsonObject catalog, bool baseline)
    {
        Directory.CreateDirectory(Path.Combine(root, "ggml"));
        File.WriteAllText(CatalogPath(baseline, catalog["variant"]!.GetValue<string>()), catalog.ToJsonString());
    }

    public void Dispose() => Directory.Delete(root, true);
}
