using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.MLX;

if (args.Length != 1 || !Path.IsPathFullyQualified(args[0])
    || !OperatingSystem.IsMacOS() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
{
    Console.Error.WriteLine("Usage on macOS ARM64: mlx-runtime-lifetime <absolute-libmlxc.dylib-path>");
    return 2;
}

string entry = Path.GetFullPath(args[0]);
var failures = new List<Exception>();
try
{
    Require(File.Exists(entry), "The selected prebuilt MLX bridge exists.");
    Require(Environment.GetEnvironmentVariable("TS_MLX_DISABLE_COMPILE") != "1",
        "The compiled activation path must be enabled.");
    Environment.SetEnvironmentVariable("TENSORSHARP_MLX_LIBRARY", entry);
    // Retain the loaded library for the process lifetime, including callback retirement.
    _ = NativeLibrary.Load(entry);
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        entry,
        sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(entry))),
        coreMvid = typeof(Tensor).Module.ModuleVersionId,
        mlxMvid = typeof(MlxBackend).Module.ModuleVersionId
    }));
    Require(NativeRuntimeQuarantine.Observe().State == NativeRuntimeQuarantineState.NoRecordedFailure,
        "The fresh process has no recorded cleanup failure.");

    MlxBackend.EnsureAvailable();
    MlxBackend.ClearCache();
    MlxMemorySnapshot before = MlxBackend.GetMemorySnapshot();
    WeakReference[] storage = RunTensorScope();
    MlxMemorySnapshot after = MlxBackend.GetMemorySnapshot();
    Require(after.ActiveBytes == before.ActiveBytes,
        $"Explicit tensor and cache disposal restores active bytes: before={before.ActiveBytes}, after={after.ActiveBytes}.");
    Require(after.CacheBytes == 0, "Explicit allocator disposal clears the native allocation cache.");
    MlxBackend.Shutdown();
    MlxBackend.Shutdown();
    Require(NativeRuntimeQuarantine.Observe().State == NativeRuntimeQuarantineState.NoRecordedFailure,
        "Checked worker retirement records no unsafe cleanup.");
    try
    {
        MlxWorker.Shared.Invoke(() => 0);
        throw new InvalidOperationException("A retired worker accepted new work.");
    }
    catch (InvalidOperationException error) when (error.Message == "Native ownership was already safely released.")
    {
    }
    for (int pass = 0; pass < 10 && storage.Any(reference => reference.IsAlive); pass++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
    Require(storage.All(reference => !reference.IsAlive),
        "Safely disposed real storage owners are not retained after finalizer drainage.");
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        result = "passed",
        before,
        after,
        storagesCollected = storage.Length,
        compiledActivation = "GELU",
        terminalWorkerRetirement = "completed"
    }));
    return 0;
}
catch (Exception error)
{
    failures.Add(error);
}

try { MlxBackend.Shutdown(); }
catch (Exception error) { failures.Add(error); }
foreach (Exception error in failures) Console.Error.WriteLine(error);
Console.Error.WriteLine($"Quarantine state: {NativeRuntimeQuarantine.Observe().State}");
return 1;

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] RunTensorScope()
{
    using var allocator = new MlxAllocator();
    float[] values = [-2f, -0.5f, 0f, 0.75f, 1.25f, 2.5f];
    using var input = Tensor.FromArray(allocator, values);
    using var first = new Tensor(allocator, DType.Float32, values.Length);
    using var reused = new Tensor(allocator, DType.Float32, values.Length);
    Ops.GELU(first, input);
    Ops.GELU(reused, input);
    float[] expected = values.Select(value =>
        0.5f * value * (1f + MathF.Tanh(0.7978845608f * (value + 0.044715f * value * value * value)))).ToArray();
    foreach (Tensor result in new[] { first, reused })
    {
        float[] actual = result.GetElementsAsFloat(values.Length);
        Require(actual.Length == expected.Length && actual.Zip(expected).All(pair =>
            float.IsFinite(pair.First) && MathF.Abs(pair.First - pair.Second) <= 1e-5f),
            "A real compiled MLX activation and its cached reuse match the CPU formula.");
    }
    return [new(input.Storage), new(first.Storage), new(reused.Storage)];
}

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}
