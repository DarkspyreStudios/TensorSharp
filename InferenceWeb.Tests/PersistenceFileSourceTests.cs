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
