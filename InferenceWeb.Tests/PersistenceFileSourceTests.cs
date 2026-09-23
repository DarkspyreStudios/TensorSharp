using System.Text;
using Darkspyre.Persistence;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class PersistenceFileSourceTests
{
    [Fact]
    public async Task MixedStoresProjectTogetherWithoutDeletingTheOriginal()
    {
        string directory = Directory.CreateTempSubdirectory("ts-mixed-").FullName;
        string original = Path.Combine(directory, "opaque-asset.dat");
        await File.WriteAllBytesAsync(original, SplitGguf(0, "first", 1));
        try
        {
            using GgufFile reader = await GgufFile.OpenAsync(Set(File.OpenRead(original), new ObservedStream(SplitGguf(1, "second", 2))));
            Assert.DoesNotContain(original, reader.FilePaths);
            Assert.Equal(2, reader.Tensors.Count);
            reader.Dispose();
            Assert.True(File.Exists(original));
            Assert.All(reader.FilePaths, path => Assert.False(File.Exists(path)));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public async Task UnlistedFileStoreSiblingIsNotLoaded()
    {
        string directory = Directory.CreateTempSubdirectory("ts-unlisted-").FullName;
        string first = Path.Combine(directory, "model-00001-of-00002.gguf");
        string second = Path.Combine(directory, "model-00002-of-00002.gguf");
        await File.WriteAllBytesAsync(first, SplitGguf(0, "first", 1));
        await File.WriteAllBytesAsync(second, SplitGguf(1, "second", 2));
        try
        {
            var source = new PersistenceFileReference(new SingleAssetStore(() => File.OpenRead(first)), "models", Path.GetFileName(first));
            await Assert.ThrowsAsync<InvalidDataException>(() => GgufFile.OpenAsync(source));
            Assert.True(File.Exists(second));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void NamedSetIsImmutableAndRejectsCaseCollisionsOrOverlappingPaths()
    {
        var source = new PersistenceFileReference(new SingleAssetStore(() => new MemoryStream()), "models", "opaque");
        var files = new Dictionary<string, PersistenceFileReference> { ["model.gguf"] = source };
        var set = new PersistenceFileSet("model.gguf", files);
        files.Clear();
        Assert.Single(set.Files);
        Assert.Throws<ArgumentException>(() => new PersistenceFileSet("model.gguf", new Dictionary<string, PersistenceFileReference>
        { ["model.gguf"] = source, ["MODEL.gguf"] = source }));
        Assert.Throws<ArgumentException>(() => new PersistenceFileSet("a", new Dictionary<string, PersistenceFileReference>
        { ["a"] = source, ["a/model.gguf"] = source }));
    }

    [Fact]
    public async Task NamedMemoryShardsRemainTogetherAndReadFromTheCorrectOwner()
    {
        var first = new ObservedStream(SplitGguf(0, "first", 1.5f));
        var second = new ObservedStream(SplitGguf(1, "second", 2.5f));
        GgufFile reader = await GgufFile.OpenAsync(Set(first, second));
        string[] paths = reader.FilePaths.ToArray();
        try
        {
            Assert.True(reader.IsSplit);
            Assert.Equal(new[] { "model-00001-of-00002.gguf", "model-00002-of-00002.gguf" }, paths.Select(Path.GetFileName));
            Assert.Single(paths.Select(Path.GetDirectoryName).Distinct());
            Assert.Equal(1.5f, BitConverter.ToSingle(reader.ReadTensorData(reader.Tensors["first"])));
            Assert.Equal(2.5f, BitConverter.ToSingle(reader.ReadTensorData(reader.Tensors["second"])));
            reader.ThrowIfTruncated();
            Assert.True(first.Disposed);
            Assert.True(second.Disposed);
        }
        finally { reader.Dispose(); }
        Assert.All(paths, path => Assert.False(File.Exists(path)));
        Assert.False(Directory.Exists(Path.GetDirectoryName(paths[0])));
    }

    [Fact]
    public async Task CompleteFileStoreSetIsRetainedInPlace()
    {
        string directory = Directory.CreateTempSubdirectory("ts-split-").FullName;
        string firstPath = Path.Combine(directory, "model-00001-of-00002.gguf");
        string secondPath = Path.Combine(directory, "model-00002-of-00002.gguf");
        await File.WriteAllBytesAsync(firstPath, SplitGguf(0, "first", 1));
        await File.WriteAllBytesAsync(secondPath, SplitGguf(1, "second", 2));
        var first = new FileStream(firstPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var second = new FileStream(secondPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            using (GgufFile reader = await GgufFile.OpenAsync(Set(first, second)))
            {
                Assert.Equal(new[] { firstPath, secondPath }, reader.FilePaths);
                Assert.True(first.CanRead);
                Assert.True(second.CanRead);
            }
            Assert.False(first.CanRead);
            Assert.False(second.CanRead);
            Assert.True(File.Exists(firstPath));
            Assert.True(File.Exists(secondPath));
        }
        finally { first.Dispose(); second.Dispose(); Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("misnamed")]
    [InlineData("wrong-index")]
    [InlineData("duplicate-tensor")]
    public async Task IncompleteOrInconsistentShardsFailAndCleanUp(string defect)
    {
        var first = new ObservedStream(SplitGguf(0, "first", 1));
        var second = new ObservedStream(SplitGguf(defect == "wrong-index" ? 0 : 1,
            defect == "duplicate-tensor" ? "first" : "second", 2));
        PersistenceFileSet set = Set(first, second, defect == "misnamed" ? "other.gguf" : "model-00002-of-00002.gguf");
        if (defect == "missing") set = new(set.PrimaryPath, set.Files.Where(pair => pair.Key == set.PrimaryPath).ToDictionary());
        await Assert.ThrowsAsync<InvalidDataException>(() => GgufFile.OpenAsync(set));
        Assert.True(first.Disposed);
        Assert.NotNull(first.ProjectedPath);
        Assert.False(File.Exists(first.ProjectedPath));
        Assert.False(Directory.Exists(Path.GetDirectoryName(first.ProjectedPath)));
        if (defect != "missing") Assert.True(second.Disposed);
        second.Dispose();
    }

    [Fact]
    public async Task CancellationDuringProjectionReleasesEveryOpenedSourceAndTemporaryFile()
    {
        using var cancellation = new CancellationTokenSource();
        var first = new ObservedStream(SplitGguf(0, "first", 1));
        var second = new ObservedStream(SplitGguf(1, "second", 2), cancellation.Cancel);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => GgufFile.OpenAsync(Set(first, second), cancellation.Token));
        Assert.True(first.Disposed);
        Assert.True(second.Disposed);
        Assert.NotNull(first.ProjectedPath);
        Assert.False(Directory.Exists(Path.GetDirectoryName(first.ProjectedPath)));
    }

    [Theory]
    [InlineData("../model.gguf")]
    [InlineData("/model.gguf")]
    [InlineData("directory/../model.gguf")]
    [InlineData("C:\\model.gguf")]
    [InlineData("directory//model.gguf")]
    [InlineData("aux.gguf")]
    public void NamedSetsRejectNonPortablePathsBeforeOpeningStores(string path)
    {
        var source = new PersistenceFileReference(new SingleAssetStore(() => throw new InvalidOperationException()), "models", "opaque-key");
        Assert.Throws<ArgumentException>(() => new PersistenceFileSet(path, new Dictionary<string, PersistenceFileReference> { [path] = source }));
    }

    [Fact]
    public async Task ModelServiceRetainsTheWholeProjectionUntilUnload()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars();
        environment.Set("TS_DSV4_DSPARK", null);
        string[] paths = [];
        using var service = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) =>
        {
            using var reader = new GgufFile(path);
            paths = reader.FilePaths.ToArray();
            Assert.Equal(2, reader.Tensors.Count);
            return new FakeModel(path);
        });
        await service.LoadModelAsync(Set(new ObservedStream(SplitGguf(0, "first", 1)),
            new ObservedStream(SplitGguf(1, "second", 2))), null, "cpu");
        Assert.Equal(2, paths.Length);
        Assert.All(paths, path => Assert.True(File.Exists(path)));
        service.UnloadModel();
        Assert.All(paths, path => Assert.False(File.Exists(path)));
    }

    private static PersistenceFileSet Set(Stream first, Stream second, string secondName = "model-00002-of-00002.gguf") =>
        new("model-00001-of-00002.gguf", new Dictionary<string, PersistenceFileReference>
        {
            ["model-00001-of-00002.gguf"] = new(new SingleAssetStore(() => first), "models", "opaque-first"),
            [secondName] = new(new SingleAssetStore(() => second), "other-store", "opaque-second")
        });

    private static byte[] SplitGguf(int index, string tensor, float value)
    {
        using var content = new MemoryStream();
        using var writer = new BinaryWriter(content, Encoding.UTF8, leaveOpen: true);
        void WriteString(string text) { byte[] bytes = Encoding.UTF8.GetBytes(text); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
        writer.Write(0x46554747u); writer.Write(3u); writer.Write(1UL); writer.Write(2UL);
        WriteString("split.count"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)2);
        WriteString("split.no"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)index);
        WriteString(tensor); writer.Write(1u); writer.Write(1UL); writer.Write((uint)GgmlTensorType.F32); writer.Write(0UL);
        while (content.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(value);
        return content.ToArray();
    }

    private sealed class ObservedStream(byte[] content, Action? beforeCopy = null) : MemoryStream(content, writable: false)
    {
        public string? ProjectedPath { get; private set; }
        public bool Disposed { get; private set; }
        public override Task CopyToAsync(Stream destination, int bufferSize, CancellationToken cancellationToken)
        {
            ProjectedPath = Assert.IsType<FileStream>(destination).Name;
            beforeCopy?.Invoke();
            return base.CopyToAsync(destination, bufferSize, cancellationToken);
        }
        protected override void Dispose(bool disposing) { Disposed = true; base.Dispose(disposing); }
    }

    [Fact]
    public async Task FileBackedStoreIsUsedInPlace()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"ts-persist-file-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "model.gguf");
        await File.WriteAllBytesAsync(path, MinimalGguf());

        try
        {
            var source = new PersistenceFileReference(
                new SingleAssetStore(() => new FileStream(
                    path,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.Read | FileShare.Delete)),
                "models",
                "model.gguf");

            using (GgufFile gguf = await GgufFile.OpenAsync(source))
            {
                Assert.Equal(Path.GetFullPath(path), Path.GetFullPath(Assert.Single(gguf.FilePaths)));
            }

            Assert.True(File.Exists(path));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task NonFileGgufStreamIsProjectedAndDeletedWithReader()
    {
        var source = new PersistenceFileReference(
            new SingleAssetStore(() => new MemoryStream(MinimalGguf(), writable: false)),
            "models",
            "model.gguf");

        GgufFile gguf = await GgufFile.OpenAsync(source);
        string projectedPath = Assert.Single(gguf.FilePaths);
        Assert.True(File.Exists(projectedPath));

        gguf.Dispose();

        Assert.False(File.Exists(projectedPath));
    }

    [Fact]
    public async Task NonFileSafetensorsStreamIsProjectedAndDeletedWithReader()
    {
        var source = new PersistenceFileReference(
            new SingleAssetStore(() => new MemoryStream(MinimalSafetensors(), writable: false)),
            "models",
            "model.safetensors");

        SafetensorsFile safetensors = await SafetensorsFile.OpenAsync(source);
        string projectedPath = safetensors.Path;
        Assert.Equal(1.5f, Assert.Single(safetensors.ReadFloat32("value")));
        Assert.True(File.Exists(projectedPath));

        safetensors.Dispose();

        Assert.False(File.Exists(projectedPath));
    }

    [Fact]
    public async Task ModelServiceOwnsProjectionUntilModelUnload()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars();
        environment.Set("TS_DSV4_DSPARK", null);

        string? loadedPath = null;
        using var service = new ModelService(
            NullLogger<ModelService>.Instance,
            (path, _, _, _) =>
            {
                loadedPath = path;
                return new FakeModel(path);
            });
        var source = new PersistenceFileReference(
            new SingleAssetStore(() => new MemoryStream(MinimalGguf(), writable: false)),
            "models",
            "model.gguf");

        await service.LoadModelAsync(source, mmProj: null, backendStr: "cpu");

        Assert.NotNull(loadedPath);
        Assert.True(File.Exists(loadedPath));

        service.UnloadModel();

        Assert.False(File.Exists(loadedPath));
    }

    [Fact]
    public async Task FailedReplacementRetainsPreviousProjectionForRollback()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars();
        environment.Set("TS_DSV4_DSPARK", null);

        var loadedPaths = new List<string>();
        int call = 0;
        using var service = new ModelService(
            NullLogger<ModelService>.Instance,
            (path, _, _, _) =>
            {
                loadedPaths.Add(path);
                call++;
                return call == 2
                    ? throw new InvalidOperationException("replacement failed")
                    : new FakeModel(path);
            });
        var first = new PersistenceFileReference(
            new SingleAssetStore(() => new MemoryStream(MinimalGguf(), writable: false)),
            "models",
            "first.gguf");
        var replacement = new PersistenceFileReference(
            new SingleAssetStore(() => new MemoryStream(MinimalGguf(), writable: false)),
            "models",
            "replacement.gguf");

        await service.LoadModelAsync(first, mmProj: null, backendStr: "cpu");
        string firstPath = Assert.Single(loadedPaths);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => service.LoadModelAsync(replacement, mmProj: null, backendStr: "cpu"));

        Assert.Equal("replacement failed", exception.Message);
        Assert.Equal(3, loadedPaths.Count);
        Assert.Equal(firstPath, loadedPaths[2]);
        Assert.True(File.Exists(firstPath));
        Assert.False(File.Exists(loadedPaths[1]));
        Assert.True(service.IsLoaded);

        service.UnloadModel();
        Assert.False(File.Exists(firstPath));
    }

    private static byte[] MinimalGguf()
    {
        using var content = new MemoryStream();
        using (var writer = new BinaryWriter(content, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x46554747u);
            writer.Write(3u);
            writer.Write(0UL);
            writer.Write(0UL);
            writer.Write(new byte[8]);
        }

        return content.ToArray();
    }

    private static byte[] MinimalSafetensors()
    {
        byte[] header = Encoding.UTF8.GetBytes(
            "{\"value\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
        using var content = new MemoryStream();
        using (var writer = new BinaryWriter(content, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write((ulong)header.Length);
            writer.Write(header);
            writer.Write(1.5f);
        }

        return content.ToArray();
    }

    private sealed class SingleAssetStore(Func<Stream> open) : IPersistenceStore
    {
        public Task<bool> ExistsAsync(
            string storeName,
            string assetName,
            CancellationToken cancellationToken = default) => Task.FromResult(true);

        public Task<Stream?> OpenReadAsync(
            string storeName,
            string assetName,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult<Stream?>(open());
        }

        public Task<IReadOnlyList<string>> EnumerateAsync(
            string storeName,
            string assetNamePrefix,
            CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<string>>([assetNamePrefix]);

        public Task SaveAsync(
            string storeName,
            string assetName,
            Stream content,
            CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task DeleteAsync(
            string storeName,
            string assetName,
            CancellationToken cancellationToken = default) => throw new NotSupportedException();
    }

    private sealed class FakeModel(string ggufPath) : ModelBase(ggufPath, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];

        protected override void ResetKVCacheCore()
        {
        }
    }
}
