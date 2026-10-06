#nullable enable

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.MLX;

if (args is ["--collectible", var collectibleEntry])
    return ForeignMlxGeneration.Run(collectibleEntry);
if (args is ["--collectible-dump", var dumpEntry])
    return ForeignMlxGeneration.Run(dumpEntry, waitForDump: true);

if (args.Length != 1 || !Path.IsPathFullyQualified(args[0])
    || !OperatingSystem.IsMacOS() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
{
    Console.Error.WriteLine("Usage on macOS ARM64: mlx-runtime-lifetime [--collectible] <absolute-libmlxc.dylib-path>");
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
        mlxMvid = typeof(MlxBackend).Module.ModuleVersionId,
        collectible = AssemblyLoadContext.GetLoadContext(typeof(MlxBackend).Assembly)!.IsCollectible
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

internal static class ForeignMlxGeneration
{
    private sealed record Evidence(WeakReference[] Roots, string[] Names);

    internal static int Run(string entry, bool waitForDump = false)
    {
        try
        {
            Evidence evidence = LoadAndRetire(entry);
            // Match the existing GGML collectible fixture's bounded root observation.
            for (int pass = 0; pass < 30 && evidence.Roots.Any(root => root.IsAlive); pass++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Thread.Sleep(10);
            }
            string[] retained = evidence.Names.Where((_, index) => evidence.Roots[index].IsAlive).ToArray();
            if (waitForDump)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { diagnosticProcessId = Environment.ProcessId, retained }));
                Console.ReadLine();
            }
            if (retained.Length != 0)
                throw new InvalidOperationException("Safely retired real MLX generation remains rooted: " + string.Join(",", retained));
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                result = "passed",
                mode = "collectible",
                collectedRoots = evidence.Roots.Length,
                assemblies = evidence.Names,
                nativeFailureQualification = "not-run",
                actualModelQualification = "not-run"
            }));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            Console.WriteLine(JsonSerializer.Serialize(new { result = "failed", mode = "collectible" }));
            return 1;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Evidence LoadAndRetire(string entry)
    {
        string assemblyPath = typeof(ForeignMlxGeneration).Assembly.Location;
        var generation = new Generation(Path.GetDirectoryName(assemblyPath)!);
        try
        {
            Assembly fixture = generation.LoadFromAssemblyPath(assemblyPath);
            int exit = (int)fixture.EntryPoint!.Invoke(null, [new[] { entry }])!;
            if (exit != 0)
                throw new InvalidOperationException("The foreign native MLX lifetime checks did not complete successfully.");
            Assembly[] assemblies = generation.Assemblies.ToArray();
            foreach (string name in new[] { "mlx-runtime-lifetime", "TensorSharp.Core", "TensorSharp.Backends.MLX" })
                if (!assemblies.Any(assembly => assembly.GetName().Name == name &&
                    AssemblyLoadContext.GetLoadContext(assembly) == generation))
                    throw new InvalidOperationException("The native MLX probe requires its actual private dependency: " + name);
            return new([new(generation), .. assemblies.Select(assembly => new WeakReference(assembly))],
                ["collectible ALC", .. assemblies.Select(assembly => assembly.GetName().Name!)]);
        }
        finally
        {
            generation.Unload();
        }
    }

    private sealed class Generation(string directory) : AssemblyLoadContext("native-mlx-lifetime-" + Guid.NewGuid(), true)
    {
        protected override Assembly? Load(AssemblyName name)
        {
            if (name.Name != "mlx-runtime-lifetime" && !name.Name!.StartsWith("TensorSharp.", StringComparison.Ordinal))
                return null;
            string path = Path.Combine(directory, name.Name + ".dll");
            if (!File.Exists(path))
                throw new FileNotFoundException("The foreign MLX generation requires its actual private dependency.", path);
            return LoadFromAssemblyPath(path);
        }
    }
}
