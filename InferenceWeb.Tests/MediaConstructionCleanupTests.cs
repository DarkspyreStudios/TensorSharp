using System.Reflection;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MediaConstructionCleanupTests
{
    [Fact]
    public void FileOnlyFailureRetainsRecoveryWithoutRepeatingReleasedResources()
    {
        string directory = Path.Combine(Path.GetTempPath(), "media-file-recovery-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "resource.bin");
            File.WriteAllBytes(path, [1]);
            using var file = File.OpenRead(path);
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            using var tensor = new Tensor(allocator, DType.Float32, 1);
            tensor.SetElementsAsFloat([9]);
            var original = new InvalidDataException("Media construction failed.");
            var fileFailure = new IOException("The input file could not be closed.");
            bool refuseFileClose = true;
            int resourceReleases = 0;
            var cleanup = new MediaConstructionCleanup(new object(), allocator,
                tensors => tensors.Add(tensor), () =>
                {
                    resourceReleases++;
                    tensor.Dispose();
                }, () =>
                {
                    if (refuseFileClose) throw fileFailure;
                    file.Dispose();
                }, _ => Assert.Fail("A file-only failure must not fence native resources."));

            NativeConstructionCleanupException failure = Assert.Throws<NativeConstructionCleanupException>(() =>
                cleanup.RollBackConstruction(original));
            Assert.Same(original, failure.InnerExceptions[0]);
            Assert.Same(fileFailure, failure.InnerExceptions[1]);
            Assert.Same(cleanup.Cleanup, failure.Cleanup);
            Assert.False(failure.Cleanup.IsReleased);
            Assert.Equal(IntPtr.Zero, Assert.IsType<CpuStorage>(tensor.Storage).buffer);
            Assert.Throws<IOException>(() => File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None).Dispose());

            refuseFileClose = false;
            failure.Cleanup.Dispose();
            failure.Cleanup.Dispose();
            Assert.True(failure.Cleanup.IsReleased);
            Assert.Equal(1, resourceReleases);
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            using var borrowed = new Tensor(allocator, DType.Float32, 1);
            borrowed.SetElementsAsFloat([7]);
            Assert.Equal([7f], borrowed.GetElementsAsFloat(1));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ConditionerPreservesFailedTextValidationAndReleasesItsInputFile()
    {
        string directory = Path.Combine(Path.GetTempPath(), "conditioner-construction-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "text.gguf");
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0ul);
                writer.Write(0ul);
                writer.Write(0ul);
            }

            var failure = Assert.Throws<NotSupportedException>(() =>
                new QwenImage21Conditioner(path, path, BackendType.Cpu));
            Assert.Contains("Qwen3-VL-8B text encoder", failure.Message);
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void HealthyStorageAdmissionRefusalRetainsConstructionRecoveryForExplicitRelease()
    {
        string directory = Path.Combine(Path.GetTempPath(), "media-recovery-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "resource.bin");
            File.WriteAllBytes(path, [1]);
            using var file = File.OpenRead(path);
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            using var tensor = new Tensor(allocator, DType.Float32, 1);
            tensor.SetElementsAsFloat([9]);
            var original = new InvalidDataException("Media construction failed.");
            // Stand in for MLX's expensive admission seam with its actual, storage-bound refusal.
            // Reflection is test-only; the production recipe uses the existing Core type directly.
            var busy = (Exception)Activator.CreateInstance(
                typeof(Tensor).Assembly.GetType("TensorSharp.NativeMlxCallbackBusyException", throwOnError: true)!,
                BindingFlags.Instance | BindingFlags.NonPublic, binder: null, args: [tensor.Storage], culture: null)!;
            Exception? refusal = busy;
            Exception? retainedUnsafe = null;
            var cleanup = new MediaConstructionCleanup(new object(), allocator,
                tensors => tensors.Add(tensor), () =>
                {
                    if (refusal is not null) throw refusal;
                    tensor.Dispose();
                }, file.Dispose, error => retainedUnsafe = error);

            NativeConstructionCleanupException failure = Assert.Throws<NativeConstructionCleanupException>(() =>
                cleanup.RollBackConstruction(original));
            Assert.Same(original, failure.InnerExceptions[0]);
            Assert.Same(busy, failure.InnerExceptions[1]);
            Assert.Same(cleanup.Cleanup, failure.Cleanup);
            Assert.False(failure.Cleanup.IsReleased);
            Assert.Null(retainedUnsafe);
            NativeConstructionCleanupHandle? pending = failure.Cleanup;
            Assert.Throws<InvalidOperationException>(() =>
                MediaConstructionCleanup.RequireReleasedConstruction(ref pending));
            Assert.Same(failure.Cleanup, pending);
            Assert.True(tensor.Storage.IsOwnerExclusive());
            Assert.Equal([9f], tensor.GetElementsAsFloat(1));
            Assert.Throws<IOException>(() => File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None).Dispose());

            refusal = null;
            failure.Cleanup.Dispose();
            Assert.True(failure.Cleanup.IsReleased);
            MediaConstructionCleanup.RequireReleasedConstruction(ref pending);
            Assert.Null(pending);
            Assert.Equal(IntPtr.Zero, Assert.IsType<CpuStorage>(tensor.Storage).buffer);
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData("qwen", false)]
    [InlineData("qwen", true)]
    [InlineData("mistral", false)]
    [InlineData("mistral", true)]
    [InlineData("glm", false)]
    [InlineData("glm", true)]
    [InlineData("gemma-audio", false)]
    [InlineData("gemma-audio", true)]
    [InlineData("gemma-vision", false)]
    [InlineData("gemma-vision", true)]
    [InlineData("gemma-vision-safetensors", false)]
    [InlineData("gemma-vision-safetensors", true)]
    public void FailedMediaWeightPopulationReleasesTheWeightAndFileButNotTheBorrowedAllocator(string encoder, bool parentOwned)
    {
        string directory = Path.Combine(Path.GetTempPath(), "vision-construction-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            bool safetensors = encoder == "gemma-vision-safetensors";
            string path = Path.Combine(directory, safetensors ? "projector.safetensors" : "projector.gguf");
            if (safetensors) WriteSafetensorsProjector(path);
            else WriteProjector(path, encoder == "gemma-audio" ? "a.weight" : "v.weight");
            var failure = new InvalidOperationException("Weight population failed.");
            var allocator = new PopulationFailureAllocator(failure);
            using var borrowed = new Tensor(allocator, DType.Float32, 1);
            borrowed.SetElementsAsFloat([7]);

            NativeConstructionCleanupHandle? reserved = null;
            Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
                CreateEncoder(encoder, path, allocator, parentOwned ? cleanup => reserved = cleanup : null)));

            if (parentOwned)
            {
                Assert.NotNull(reserved);
                Assert.True(reserved.IsReleased);
            }

            Assert.Equal(1, allocator.FailedWeight!.Releases);
            Assert.Equal(IntPtr.Zero, allocator.FailedWeight.buffer);
            Assert.False(allocator.Disposed);
            Assert.Equal([7f], borrowed.GetElementsAsFloat(1));
            using var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static IDisposable CreateEncoder(string encoder, string path, IAllocator allocator) => encoder switch
    {
        "qwen" => new Qwen35VisionEncoder(path, allocator),
        "mistral" => new Mistral3VisionEncoder(path, allocator),
        "glm" => new GlmNextVisionEncoder(path, allocator),
        "gemma-audio" => new Gemma4AudioEncoder(path, allocator),
        "gemma-vision" or "gemma-vision-safetensors" => new Gemma4VisionEncoder(path, allocator),
        _ => throw new ArgumentOutOfRangeException(nameof(encoder))
    };

    private static IDisposable CreateEncoder(string encoder, string path, IAllocator allocator,
        Action<NativeConstructionCleanupHandle>? reserve)
    {
        if (reserve is null) return CreateEncoder(encoder, path, allocator);
        return encoder switch
        {
            "qwen" => new Qwen35VisionEncoder(path, allocator, false,
                child => reserve(child.ConstructionCleanup), (_, _) => Assert.Fail("Healthy rollback must not fence the parent.")),
            "mistral" => new Mistral3VisionEncoder(path, allocator,
                child => reserve(child.ConstructionCleanup), (_, _) => Assert.Fail("Healthy rollback must not fence the parent.")),
            "glm" => new GlmNextVisionEncoder(path, allocator,
                child => reserve(child.ConstructionCleanup), (_, _) => Assert.Fail("Healthy rollback must not fence the parent.")),
            "gemma-audio" => new Gemma4AudioEncoder(path, allocator,
                child => reserve(child.ConstructionCleanup), (_, _) => Assert.Fail("Healthy rollback must not fence the parent.")),
            "gemma-vision" or "gemma-vision-safetensors" => new Gemma4VisionEncoder(path, allocator,
                child => reserve(child.ConstructionCleanup), (_, _) => Assert.Fail("Healthy rollback must not fence the parent.")),
            _ => throw new ArgumentOutOfRangeException(nameof(encoder))
        };
    }

    private static void WriteProjector(string path, string weightName)
    {
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(1ul);
        writer.Write(0ul);
        byte[] name = Encoding.UTF8.GetBytes(weightName);
        writer.Write((ulong)name.Length);
        writer.Write(name);
        writer.Write(1u);
        writer.Write(1ul);
        writer.Write((uint)GgmlTensorType.F32);
        writer.Write(0ul);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(1f);
    }

    private static void WriteSafetensorsProjector(string path)
    {
        const string tower = "model.encoder.vision_tower.";
        (string Name, int[] Shape)[] tensors =
        [
            (tower + "patch_embedder.input_proj.weight", [1, 3]),
            (tower + "encoder.layers.0.input_layernorm.weight", [1]),
            (tower + "encoder.layers.0.self_attn.q_norm.weight", [1]),
            (tower + "encoder.layers.0.mlp.gate_proj.linear.weight", [1, 1]),
            ("model.encoder.embed_vision.embedding_projection.weight", [1, 1])
        ];
        using var header = new MemoryStream();
        int offset = 0;
        using (var json = new Utf8JsonWriter(header))
        {
            json.WriteStartObject();
            foreach (var (name, shape) in tensors)
            {
                json.WriteStartObject(name);
                json.WriteString("dtype", "F32");
                json.WriteStartArray("shape");
                int count = 1;
                foreach (int dimension in shape)
                {
                    json.WriteNumberValue(dimension);
                    count *= dimension;
                }
                json.WriteEndArray();
                json.WriteStartArray("data_offsets");
                json.WriteNumberValue(offset);
                offset += count * sizeof(float);
                json.WriteNumberValue(offset);
                json.WriteEndArray();
                json.WriteEndObject();
            }
            json.WriteEndObject();
        }
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write((ulong)header.Length);
        writer.Write(header.ToArray());
        for (int index = 0; index < offset / sizeof(float); index++) writer.Write(1f);
    }

    private sealed class PopulationFailureAllocator(Exception failure) : IAllocator, IDisposable
    {
        private bool _borrowedAllocated;
        public FailureStorage? FailedWeight { get; private set; }
        public bool Disposed { get; private set; }
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count)
        {
            if (_borrowedAllocated) return FailedWeight = new FailureStorage(this, type, count, failure);
            _borrowedAllocated = true;
            return new CpuStorage(this, type, count);
        }
        public void Dispose() => Disposed = true;
    }

    private sealed class FailureStorage(IAllocator allocator, DType type, long count, Exception failure)
        : CpuStorage(allocator, type, count)
    {
        public int Releases { get; private set; }
        public override void SetElementsAsFloat(long index, float[] value) => throw failure;
        protected override void Destroy()
        {
            base.Destroy();
            Releases++;
        }
    }
}
