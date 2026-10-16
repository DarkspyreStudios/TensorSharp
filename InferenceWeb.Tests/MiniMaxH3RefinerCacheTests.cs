using System.Text;
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3RefinerCacheTests : IDisposable
{
    private const int TextDim = 6;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"h3-refiner-{Guid.NewGuid():N}.gguf");
    private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);

    public MiniMaxH3RefinerCacheTests() => WriteModel(_path);
    public void Dispose() => File.Delete(_path);

    private MiniMaxH3DirectDiT Direct(GgufFile model) =>
        new(model, MiniMaxH3Config.Detect(model.Tensors), _allocator);

    private static float[] Text(int rows = 3) =>
        Enumerable.Range(0, rows * TextDim).Select(i => MathF.Sin(i * .31f)).ToArray();

    [Fact]
    public void RefinedTextReusesEqualBuffersAndSurvivesBorrowDisposal()
    {
        using var model = new GgufFile(_path);
        using var dit = Direct(model);
        float[] text = Text();
        IntPtr address;
        float[] expected;
        using (var first = dit.RefineText(text, 3))
        using (var second = dit.RefineText((float[])text.Clone(), 3))
        {
            address = CpuNativeHelpers.GetBufferStart(first);
            expected = first.GetElementsAsFloat(24);
            Assert.Equal(address, CpuNativeHelpers.GetBufferStart(second));
            Assert.Equal(expected, second.GetElementsAsFloat(24));
        }
        using var third = dit.RefineText(text, 3);
        Assert.Equal(address, CpuNativeHelpers.GetBufferStart(third));
        Assert.Equal(expected, third.GetElementsAsFloat(24));
    }

    [Fact]
    public void TextMutationAndShapeChangeInvalidateTheSnapshot()
    {
        using var model = new GgufFile(_path);
        using var dit = Direct(model);
        float[] text = Text();
        using var original = dit.RefineText(text, 3);
        text[1] += .75f;
        using var changed = dit.RefineText(text, 3);
        using var uncached = Direct(model);
        using var expected = uncached.RefineText(text, 3);
        Assert.NotEqual(CpuNativeHelpers.GetBufferStart(original), CpuNativeHelpers.GetBufferStart(changed));
        Assert.False(original.GetElementsAsFloat(24).SequenceEqual(changed.GetElementsAsFloat(24)));
        Assert.Equal(expected.GetElementsAsFloat(24), changed.GetElementsAsFloat(24));
        using var shorter = dit.RefineText(text[..(2 * TextDim)], 2);
        using var expectedShorter = uncached.RefineText(text[..(2 * TextDim)], 2);
        Assert.Equal(expectedShorter.GetElementsAsFloat(16), shorter.GetElementsAsFloat(16));
    }

    [Fact]
    public void ResidencyReleaseDropsCachedTokensAndSignedZeroIsComparedExactly()
    {
        using var model = new GgufFile(_path);
        using var dit = Direct(model);
        float[] text = Text();
        using var positiveZero = dit.RefineText(text, 3);
        text[0] = -0f;
        using var negativeZero = dit.RefineText(text, 3);
        Assert.NotEqual(CpuNativeHelpers.GetBufferStart(positiveZero), CpuNativeHelpers.GetBufferStart(negativeZero));
        dit.ReleaseDeviceResidency();
        using var afterRelease = dit.RefineText(text, 3);
        Assert.NotEqual(CpuNativeHelpers.GetBufferStart(negativeZero), CpuNativeHelpers.GetBufferStart(afterRelease));
        Assert.Equal(negativeZero.GetElementsAsFloat(24), afterRelease.GetElementsAsFloat(24));
    }

    [Fact]
    public void PackedTrunkCannotMutateRefinedTextAcrossDiffusionSteps()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        var shape = new MiniMaxH3Shape
        {
            Width = 32, Height = 32, Frames = 1, Fps = 24,
            LatentWidth = 2, LatentHeight = 2, LatentFrames = 1, AudioLatentFrames = 1,
        };
        float[] text = Text(), video = Enumerable.Range(0, 96).Select(i => MathF.Sin(i * .17f)).ToArray();
        float[] audio = Enumerable.Range(0, 64).Select(i => MathF.Cos(i * .23f)).ToArray();
        (float[] Video, float[] Audio) Forward(MiniMaxH3DiT d, float sigma) =>
            d.Forward(video, 1, audio, 2, text, 3, MiniMaxH3Layout.Build(3, shape, sigma), sigma);
        var first = Forward(dit, .75f);
        var repeat = Forward(dit, .75f);
        Assert.Equal(first.Video, repeat.Video);
        Assert.Equal(first.Audio, repeat.Audio);
        video[0] += .25f;
        audio[0] -= .25f;
        var next = Forward(dit, .25f);
        using var fresh = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        var expected = Forward(fresh, .25f);
        Assert.Equal(expected.Video, next.Video);
        Assert.Equal(expected.Audio, next.Audio);
    }

    [Theory]
    [InlineData("forward")]
    [InlineData("release")]
    [InlineData("dispose")]
    public void DirectCallsDoNotWaitOnTheNativeArenaMonitor(string operation)
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        object gate = typeof(MiniMaxH3DiT).GetField("_nativeGraphLock",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(dit)!;
        var shape = new MiniMaxH3Shape
        {
            Width = 32, Height = 32, Frames = 1, Fps = 24,
            LatentWidth = 2, LatentHeight = 2, LatentFrames = 1, AudioLatentFrames = 1,
        };
        Task pending;
        bool completed;
        // A backend worker must be able to finish a direct call while another
        // thread owns this native-only monitor. Otherwise a synchronous MLX
        // dispatch can form a lock/worker cycle. CPU exercises the same branch
        // deterministically without requiring a GPU or timing worker queues.
        lock (gate)
        {
            pending = Task.Run(() =>
            {
                switch (operation)
                {
                    case "forward":
                        dit.Forward(new float[96], 1, new float[64], 2, Text(), 3,
                            MiniMaxH3Layout.Build(3, shape, .75f), .75f);
                        break;
                    case "release": dit.ReleaseDeviceResidency(); break;
                    case "dispose": dit.Dispose(); break;
                }
            });
            completed = pending.Wait(TimeSpan.FromSeconds(5));
        }
        pending.GetAwaiter().GetResult();
        Assert.True(completed, "Direct execution waited on the native graph arena monitor.");
    }

    [Fact]
    public void NativeForwardStillWaitsOnItsArenaMonitor()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.GgmlCuda, null);
        object gate = typeof(MiniMaxH3DiT).GetField("_nativeGraphLock",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(dit)!;
        using var started = new ManualResetEventSlim();
        Task pending;
        bool completed;
        lock (gate)
        {
            pending = Task.Run(() =>
            {
                started.Set();
                // Validation fails before any native allocation or GPU call.
                Assert.Throws<ArgumentNullException>(() => dit.Forward(
                    null, 0, null, 0, null, 0, null, .5f));
            });
            Assert.True(started.Wait(TimeSpan.FromSeconds(5)));
            completed = pending.Wait(TimeSpan.FromMilliseconds(100));
        }
        pending.GetAwaiter().GetResult();
        Assert.False(completed, "Native execution bypassed its graph arena monitor.");
    }

    [Fact]
    public void RotaryTablesReuseEqualPositionsAcrossTimestepsWithoutAllocating()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        var shape = MiniMaxH3Geometry.Resolve(64, 64, 22, 24);
        var firstLayout = MiniMaxH3Layout.Build(3, shape, .75f);
        var nextLayout = MiniMaxH3Layout.Build(3, shape, .25f);
        Assert.Equal(firstLayout.Positions, nextLayout.Positions);
        var first = dit.GetRopeForForward(firstLayout.Positions);
        var expected = dit.BuildRope(nextLayout.Positions);
        Assert.Equal(expected.Cos, first.Cos);
        Assert.Equal(expected.Sin, first.Sin);

        // A new layout owns different arrays even when its packed positions are
        // unchanged. The hit must also retain device-table cache identity.
        var next = dit.GetRopeForForward(nextLayout.Positions);
        Assert.Same(first.Cos, next.Cos);
        Assert.Same(first.Sin, next.Sin);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int repeat = 0; repeat < 20; repeat++)
            dit.GetRopeForForward(nextLayout.Positions);
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    [Fact]
    public void RotaryPositionMutationLengthAndSignedZeroInvalidateTheSnapshot()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        float[] positions = [0, 0, 0, 1, .5f, 2, 2, 1, 3];
        var first = dit.GetRopeForForward(positions);
        positions[4] += 2;
        var changed = dit.GetRopeForForward(positions);
        Assert.NotSame(first.Cos, changed.Cos);
        Assert.Equal(dit.BuildRope(positions).Cos, changed.Cos);
        Assert.Equal(dit.BuildRope(positions).Sin, changed.Sin);

        positions[0] = -0f;
        var negativeZero = dit.GetRopeForForward(positions);
        Assert.NotSame(changed.Sin, negativeZero.Sin);
        Assert.Equal(dit.BuildRope(positions).Sin.Select(BitConverter.SingleToInt32Bits),
            negativeZero.Sin.Select(BitConverter.SingleToInt32Bits));

        float[] shorter = positions[..6];
        var resized = dit.GetRopeForForward(shorter);
        Assert.NotSame(negativeZero.Cos, resized.Cos);
        Assert.Equal(dit.BuildRope(shorter).Cos, resized.Cos);
        Assert.Equal(dit.BuildRope(shorter).Sin, resized.Sin);
    }

    [Fact]
    public void PublicRotaryTablesCannotMutateForwardCache()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        float[] positions = [0, 0, 0, 1, .5f, 2];
        var cached = dit.GetRopeForForward(positions);
        var publicTables = dit.BuildRope(positions);
        Assert.NotSame(cached.Cos, publicTables.Cos);
        Assert.NotSame(cached.Sin, publicTables.Sin);
        Array.Fill(publicTables.Cos, -123f);
        Array.Fill(publicTables.Sin, 456f);
        var next = dit.GetRopeForForward((float[])positions.Clone());
        Assert.Same(cached.Cos, next.Cos);
        Assert.Same(cached.Sin, next.Sin);
        Assert.Equal(dit.BuildRope(positions).Cos, next.Cos);
        Assert.Equal(dit.BuildRope(positions).Sin, next.Sin);
    }

    [Fact]
    public async Task ConcurrentRotaryCacheReadersReceiveMatchingCompleteTables()
    {
        using var dit = new MiniMaxH3DiT(_path, BackendType.Cpu, _allocator);
        float[][] positions = Enumerable.Range(0, 8).Select(caseIndex =>
            Enumerable.Range(0, (caseIndex + 1) * 3).Select(i =>
                i == 0 ? (caseIndex % 2 == 0 ? 0f : -0f) : (i + caseIndex * 7) * .17f).ToArray()).ToArray();
        var expected = positions.Select(dit.BuildRope).ToArray();
        using var start = new Barrier(4);
        Task[] workers = Enumerable.Range(0, 4).Select(worker => Task.Factory.StartNew(() =>
        {
            Assert.True(start.SignalAndWait(TimeSpan.FromSeconds(10)));
            for (int iteration = 0; iteration < 4096; iteration++)
            {
                int index = (iteration + worker * 3) % positions.Length;
                // Mix cache replacements and immediate hits while other callers
                // publish different position values and table lengths.
                for (int repeat = 0; repeat < 2; repeat++)
                {
                    var actual = dit.GetRopeForForward(positions[index]);
                    Assert.NotNull(actual.Cos);
                    Assert.NotNull(actual.Sin);
                    Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(actual.Cos.AsSpan())
                        .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected[index].Cos.AsSpan())),
                        "A concurrent cache hit returned cosine values for another position snapshot.");
                    Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(actual.Sin.AsSpan())
                        .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected[index].Sin.AsSpan())),
                        "A concurrent cache hit returned sine values for another position snapshot.");
                }
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
        await Task.WhenAll(workers);
    }

    private static void WriteModel(string path)
    {
        const int hidden = 8, head = 8, inner = 16, ffn = 16, time = 4;
        var tensors = new List<(string Name, long[] Shape, float[] Data)>();
        var random = new Random(317);
        void Add(string name, long[] shape, bool gain = false)
        {
            int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            float[] data = Enumerable.Range(0, count)
                .Select(_ => gain ? 1f : (float)(random.NextDouble() - .5) * .3f).ToArray();
            tensors.Add((name, shape, data));
        }
        void Linear(string name, int input, int output, bool bias = false)
        {
            Add(name + ".weight", [input, output]);
            if (bias) Add(name + ".bias", [output]);
        }
        void Block(string prefix, bool modulation)
        {
            foreach (string norm in new[] { "norm1", "norm2" }) Add(prefix + norm + ".weight", [hidden], true);
            foreach (string norm in new[] { "q_norm", "k_norm" }) Add(prefix + "attn." + norm + ".weight", [head], true);
            Linear(prefix + "attn.qkv_proj", hidden, 3 * inner);
            Linear(prefix + "attn.out_proj", inner, hidden);
            Linear(prefix + "mlp.fc1", hidden, 2 * ffn);
            Linear(prefix + "mlp.fc2", ffn, hidden);
            if (modulation) Linear(prefix + "adaln_proj.linear", time, 18 * hidden, true);
        }
        Linear("video_patch_proj", 96, hidden, true);
        Linear("audio_patch_proj", 32, hidden, true);
        Linear("condition_proj", TextDim, hidden, true);
        Block("token_refiner.blocks.0.", false);
        Block("blocks.0.", true);
        Add("token_refiner.final_norm.weight", [hidden], true);
        Add("final_layer.norm.weight", [hidden], true);
        Linear("final_layer.adaln_proj.linear", time, 2 * hidden, true);
        Linear("final_layer.video_out", hidden, 96, true);
        Linear("final_layer.audio_out", hidden, 32, true);
        Add("rope.inv_freq", [1]);
        Add("adaln_t_table", [time, 8]);

        static long Align(long value) => (value + 31) / 32 * 32;
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)tensors.Count); writer.Write(0UL);
        long offset = 0;
        foreach (var (name, shape, data) in tensors)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(name);
            writer.Write((ulong)utf8.Length); writer.Write(utf8); writer.Write((uint)shape.Length);
            foreach (long size in shape) writer.Write((ulong)size);
            writer.Write(0u); writer.Write((ulong)offset); offset += Align(data.LongLength * sizeof(float));
        }
        writer.Write(new byte[Align(stream.Position) - stream.Position]);
        foreach (var (_, _, data) in tensors)
        {
            foreach (float value in data) writer.Write(value);
            writer.Write(new byte[Align(data.LongLength * sizeof(float)) - data.LongLength * sizeof(float)]);
        }
    }
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class MiniMaxH3RefinerDeviceFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models,Device";
    public MiniMaxH3RefinerDeviceFactAttribute()
    {
        string backend = Environment.GetEnvironmentVariable("TS_TEST_H3_BACKEND")?.ToLowerInvariant();
        Skip = TestGates.ModelSkip("TS_MINIMAX_H3_DIR", "minimax_h3_fl2va") ?? (backend switch
        {
            "mlx" => TestGates.MlxSkip,
            "cuda" => TestGates.CudaSkip,
            _ => "Select the actual direct device with TS_TEST_H3_BACKEND=mlx or cuda.",
        });
        string root = Environment.GetEnvironmentVariable("TS_MINIMAX_H3_DIR") ?? ".";
        if (!File.Exists(Path.Combine(root, "fixtures", "h3_dit_text_hidden.bin")))
            Skip ??= "Requires the independent H3 text-conditioning fixture.";
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_H3_DIT_TRACE_DIR")))
            Skip ??= "Refiner boundary tracing intentionally disables the cache.";
    }
}

public sealed class MiniMaxH3RefinerDeviceCacheTests(ITestOutputHelper output)
{
    [MiniMaxH3RefinerDeviceFact]
    public void ActualCheckpointCacheHitsPreserveEveryFloat()
    {
        string backend = Environment.GetEnvironmentVariable("TS_TEST_H3_BACKEND")!.ToLowerInvariant();
        if (backend == "mlx") MlxWorker.Shared.Invoke(() =>
        {
            using var allocator = new MlxAllocator();
            Run(allocator, backend);
        });
        else
        {
            using var allocator = new CudaAllocator();
            Run(allocator, backend);
        }
    }

    private void Run(IAllocator allocator, string backend)
    {
        string root = Environment.GetEnvironmentVariable("TS_MINIMAX_H3_DIR")!;
        byte[] bytes = File.ReadAllBytes(Path.Combine(root, "fixtures", "h3_dit_text_hidden.bin"));
        var text = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, text, 0, bytes.Length);
        using var model = new GgufFile(Path.Combine(root, "minimax_h3_fl2va_pruned-Q4_K.gguf"));
        var config = MiniMaxH3Config.Detect(model.Tensors);
        using var dit = new MiniMaxH3DirectDiT(model, config, allocator);
        int rows = text.Length / config.TextDim;
        var records = new List<object>();
        int elements = rows * config.HiddenSize;
        // First call includes resident weight upload and is excluded from the
        // subsequent cache-miss/cache-hit comparison.
        var timer = Stopwatch.StartNew();
        using (var cold = dit.RefineText(text, rows)) cold.GetElementsAsFloat(elements);
        timer.Stop();
        double coldSeconds = timer.Elapsed.TotalSeconds;
        for (int iteration = 0; iteration < 5; iteration++)
        {
            text[0] += .0625f;
            timer.Restart();
            float[] expected;
            using (var miss = dit.RefineText(text, rows)) expected = miss.GetElementsAsFloat(elements);
            timer.Stop();
            double missSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            float[] actual;
            using (var hit = dit.RefineText((float[])text.Clone(), rows)) actual = hit.GetElementsAsFloat(elements);
            timer.Stop();
            Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            Assert.Equal(expected, actual);
            records.Add(new { iteration, missSeconds, hitSeconds = timer.Elapsed.TotalSeconds, exact = true });
        }
        output.WriteLine(JsonSerializer.Serialize(new
        {
            backend, rows, refinerLayers = config.TokenRefinerLayers, coldSeconds, records,
            scope = "Text projection and refiner with synchronized F32 download; miss/hit both use resident weights",
            limitation = "Refiner only; excludes trunk, sampler, VAE and media export; five small-input runs are not end-to-end speedup evidence",
        }));
    }
}
