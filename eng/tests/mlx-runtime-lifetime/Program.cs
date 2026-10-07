#nullable enable

using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args is ["--collectible", var collectibleEntry])
    return ForeignMlxGeneration.Run(collectibleEntry);
if (args is ["--collectible-dump", var dumpEntry])
    return ForeignMlxGeneration.Run(dumpEntry, waitForDump: true);

if (args.Length != 1 || !Path.IsPathFullyQualified(args[0])
    || !OperatingSystem.IsMacOS() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
{
    Console.Error.WriteLine("Usage on macOS ARM64: mlx-runtime-lifetime [--collectible | --collectible-dump] <absolute-libmlxc.dylib-path>");
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
    Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
    {
        ["entry"] = entry,
        ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(entry))),
        ["coreMvid"] = typeof(Tensor).Module.ModuleVersionId,
        ["mlxMvid"] = typeof(MlxBackend).Module.ModuleVersionId,
        ["modelsMvid"] = typeof(Qwen35VisionEncoder).Module.ModuleVersionId,
        ["collectible"] = AssemblyLoadContext.GetLoadContext(typeof(MlxBackend).Assembly)!.IsCollectible
    }));
    Require(NativeRuntimeQuarantine.Observe().State == NativeRuntimeQuarantineState.NoRecordedFailure,
        "The fresh process has no recorded cleanup failure.");

    MlxBackend.EnsureAvailable();
    MlxBackend.ClearCache();
    MlxMemorySnapshot before = MlxBackend.GetMemorySnapshot();
    WeakReference[] owners = [.. TraceFailureProbe.Run(), .. RunTensorScope(), .. RunQuantizedScope(),
        .. RunVisionRollbackScope(), .. RunVisionScope()];
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
    for (int pass = 0; pass < 10 && owners.Any(reference => reference.IsAlive); pass++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
    Require(owners.All(reference => !reference.IsAlive),
        "Safely disposed real storage, closure and callback owners are not retained after finalizer drainage.");
    Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
    {
        ["result"] = "passed",
        ["before"] = SnapshotValues(before),
        ["after"] = SnapshotValues(after),
        ["ownerRootsCollected"] = owners.Length,
        ["compiledActivation"] = "GELU",
        ["quantizedWeight"] = "Q8_0 cached matmul creation/reuse/release",
        ["traceFailure"] = "exact managed cause through native callback; checked closure release",
        ["callbackBusyRefusals"] = "three reentrant and two external-thread owner-specific cleanup refusals; healthy authority",
        ["visionEncoder"] = "tiny Qwen-VL construction, repeated forward/CPU parity and checked disposal",
        ["visionConstructionFailure"] = "real truncated final-weight read; 21 actual MLX storages released before GC",
        ["terminalWorkerRetirement"] = "completed"
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

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] RunQuantizedScope()
{
    const int inDim = 32, outDim = 4, rows = 2, blockBytes = 34;
    const float scale = 0.125f;
    byte[] weights = new byte[outDim * blockBytes];
    float[,] values = new float[rows, inDim];
    float[] expected = new float[rows * outDim];
    for (int row = 0; row < rows; row++)
        for (int index = 0; index < inDim; index++)
            values[row, index] = (index - 12 + row * 3) / 32f;
    for (int output = 0; output < outDim; output++)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(output * blockBytes),
            BitConverter.HalfToUInt16Bits((System.Half)scale));
        for (int index = 0; index < inDim; index++)
        {
            int quantized = (index + output * 7) % 32 - 16;
            weights[output * blockBytes + 2 + index] = unchecked((byte)(sbyte)quantized);
            for (int row = 0; row < rows; row++)
                expected[row * outDim + output] += values[row, index] * quantized * scale;
        }
    }

    using var allocator = new MlxAllocator();
    IntPtr host = Marshal.AllocHGlobal(weights.Length);
    try
    {
        Marshal.Copy(weights, 0, host, weights.Length);
        MlxQuantizedOps.PreloadQuantizedWeight(allocator, host, host,
            (int)GgmlTensorType.Q8_0, inDim, outDim, weights.Length);
        using var input = Tensor.FromArray(allocator, values);
        using var first = new Tensor(allocator, DType.Float32, rows, outDim);
        using var reused = new Tensor(allocator, DType.Float32, rows, outDim);
        foreach (Tensor result in new[] { first, reused })
        {
            Require(MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, input, host, IntPtr.Zero,
                (int)GgmlTensorType.Q8_0, inDim, outDim, weights.Length),
                "A real Q8 matmul reuses the preloaded weight without host data.");
            float[] actual = result.GetElementsAsFloat(expected.Length);
            Require(actual.Length == expected.Length && actual.Zip(expected).All(pair =>
                float.IsFinite(pair.First) && MathF.Abs(pair.First - pair.Second) <= 1e-5f),
                "A real cached Q8 matmul and its reuse match the CPU dot products.");
        }
        return [new(input.Storage), new(first.Storage), new(reused.Storage)];
    }
    finally
    {
        MlxQuantizedOps.ReleaseQuantizedWeight(allocator, host);
        Marshal.FreeHGlobal(host);
    }
}

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] RunVisionRollbackScope()
{
    string directory = Directory.CreateTempSubdirectory("mlx-vision-rollback-").FullName;
    try
    {
        string path = QwenVLSyntheticMmprojBuilder.Write(Path.Combine(directory, "truncated.gguf"));
        using (var truncated = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            truncated.SetLength(truncated.Length - sizeof(float));
        using var allocator = new MlxAllocator();
        var observed = new ObservedMlxAllocator(allocator);
        Qwen35VisionEncoder? partial = null;
        Action<Qwen35VisionEncoder> retain = encoder => partial = encoder;
        ConstructorInfo constructor = typeof(Qwen35VisionEncoder).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic, null,
            [typeof(string), typeof(IAllocator), typeof(bool), typeof(Action<Qwen35VisionEncoder>),
                typeof(Action<Qwen35VisionEncoder, Exception>)], null)!;
        MlxBackend.ClearCache();
        MlxMemorySnapshot before = MlxBackend.GetMemorySnapshot();
        try
        {
            constructor.Invoke([path, observed, false, retain, null]);
            throw new InvalidOperationException("The actual truncated weight read did not fail.");
        }
        catch (TargetInvocationException error) when (error.InnerException is EndOfStreamException)
        {
        }
        Require(partial != null && observed.Roots.Count == 21,
            "The real failed constructor retains its child after 21 actual MLX weight allocations.");
        var cleanup = (NativeConstructionCleanupHandle)typeof(Qwen35VisionEncoder)
            .GetProperty("ConstructionCleanup", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(partial)!;
        Require(cleanup.IsReleased, "The actual failed child's construction cleanup completes.");
        MethodInfo referenceCount = typeof(RefCounted).GetMethod("ReadReferenceCount", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Require(observed.Roots.All(root => root.Target is MlxStorage storage
                && (int)referenceCount.Invoke(storage, null)! == 0),
            "Every observed actual MLX weight storage releases before garbage collection.");
        using (var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
        MlxBackend.ClearCache();
        MlxMemorySnapshot after = MlxBackend.GetMemorySnapshot();
        Require(after.ActiveBytes == before.ActiveBytes && after.CacheBytes == 0,
            "Actual partial-weight rollback restores native active/cache bytes without finalizer cleanup.");
        using var reused = Tensor.FromArray(allocator, new[] { 3f });
        Require(reused.GetElementsAsFloat(1).SequenceEqual(new[] { 3f }),
            "The borrowed actual MLX allocator remains reusable after failed construction.");
        return [new(partial), .. observed.Roots, new(reused.Storage)];
    }
    finally { Directory.Delete(directory, recursive: true); }
}

[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference[] RunVisionScope()
{
    string directory = Directory.CreateTempSubdirectory("mlx-vision-").FullName;
    try
    {
        string path = QwenVLSyntheticMmprojBuilder.Write(Path.Combine(directory, "projector.gguf"));
        float[] pixels = Enumerable.Range(0, 3 * QwenVLSyntheticMmprojBuilder.ImageSize * QwenVLSyntheticMmprojBuilder.ImageSize)
            .Select(index => (index % 17 - 8) * 0.02f).ToArray();
        float[] expected;
        using (var reference = new Qwen35VisionEncoder(path, new CpuAllocator(BlasEnum.DotNet)))
        using (Tensor output = reference.Encode(pixels, QwenVLSyntheticMmprojBuilder.ImageSize, QwenVLSyntheticMmprojBuilder.ImageSize))
            expected = output.GetElementsAsFloat(checked((int)output.ElementCount()));
        Require(expected.Length == QwenVLSyntheticMmprojBuilder.ProjectionDim,
            "The CPU reference produces one real merged token with the declared projection width.");
        using var allocator = new MlxAllocator();
        var roots = new List<WeakReference>();
        using (var encoder = new Qwen35VisionEncoder(path, allocator))
        {
            roots.Add(new(encoder));
            using (var reopened = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
            for (int pass = 0; pass < 2; pass++)
            {
                using Tensor output = encoder.Encode(pixels, QwenVLSyntheticMmprojBuilder.ImageSize, QwenVLSyntheticMmprojBuilder.ImageSize);
                Require(output.Sizes.SequenceEqual(new long[] { 1, QwenVLSyntheticMmprojBuilder.ProjectionDim }),
                    "The MLX encoder produces the actual one-token projection shape.");
                float[] actual = output.GetElementsAsFloat(checked((int)output.ElementCount()));
                Require(actual.Length == expected.Length && actual.Zip(expected).All(pair =>
                    float.IsFinite(pair.First) && MathF.Abs(pair.First - pair.Second) <= 1e-4f),
                    "A real tiny MLX vision encode and its cached reuse match the managed CPU encoder.");
                roots.Add(new(output.Storage));
            }
            var owned = new List<Tensor>();
            typeof(Qwen35VisionEncoder).GetMethod("CollectDisposalOwnership", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(encoder, [owned]);
            Require(owned.Count > 0 && owned.All(tensor => tensor.Storage is MlxStorage),
                "The actual vision encoder owns real MLX weight/cache storages.");
            roots.AddRange(owned.Select(tensor => new WeakReference(tensor.Storage)));
        }
        return roots.ToArray();
    }
    finally { Directory.Delete(directory, recursive: true); }
}

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

// Reflection-based serialization of private types roots a collectible generation.
static Dictionary<string, ulong> SnapshotValues(MlxMemorySnapshot snapshot) => new()
{
    ["ActiveBytes"] = snapshot.ActiveBytes,
    ["CacheBytes"] = snapshot.CacheBytes,
    ["PeakBytes"] = snapshot.PeakBytes
};

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
                Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
                {
                    ["diagnosticProcessId"] = Environment.ProcessId,
                    ["retained"] = retained
                }));
                Console.ReadLine();
            }
            if (retained.Length != 0)
                throw new InvalidOperationException("Safely retired real MLX generation remains rooted: " + string.Join(",", retained));
            Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["result"] = "passed",
                ["mode"] = "collectible",
                ["collectedRoots"] = evidence.Roots.Length,
                ["assemblies"] = evidence.Names,
                ["nativeFailureQualification"] = "not-run",
                ["actualModelQualification"] = "tiny Qwen-VL encoder/CPU parity and actual truncated-read rollback; full models not-run"
            }));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["result"] = "failed",
                ["mode"] = "collectible"
            }));
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
            foreach (string name in new[] { "mlx-runtime-lifetime", "TensorSharp.Core", "TensorSharp.Backends.MLX", "TensorSharp.Models", "TensorSharp.Runtime" })
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

// Observe real supplier allocations without replacing storage or injecting a failure.
internal sealed class ObservedMlxAllocator(MlxAllocator inner) : IAllocator
{
    internal List<WeakReference> Roots { get; } = new();
    BlasEnum IAllocator.BlasEnum => inner.BlasEnum;
    int IAllocator.DeviceId => inner.DeviceId;
    float IAllocator.GetAllocatedMemoryRatio() => inner.GetAllocatedMemoryRatio();
    Storage IAllocator.Allocate(DType type, long count)
    {
        Storage storage = inner.Allocate(type, count);
        Roots.Add(new(storage));
        return storage;
    }
}

internal sealed class TraceFailureProbe
{
    private readonly Exception original = new InvalidOperationException("Actual native trace callback refuses its work.");
    private int calls;
    private Action? beforeFailure;

    private T[] RefuseTrace<T>(T[] inputs)
    {
        calls++;
        beforeFailure?.Invoke();
        throw original;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static WeakReference[] Run()
    {
        const BindingFlags methods = BindingFlags.NonPublic | BindingFlags.Static;
        Type native = typeof(MlxBackend).Assembly.GetType("TensorSharp.MLX.MlxNative", throwOnError: true)!;
        Type arrayType = native.GetNestedType("MlxArray", BindingFlags.NonPublic | BindingFlags.Public)!;
        Type traceType = native.GetNestedType("TraceFunc", BindingFlags.NonPublic)!;
        var target = new TraceFailureProbe();
        Delegate trace = typeof(TraceFailureProbe).GetMethod(nameof(RefuseTrace), BindingFlags.Instance | BindingFlags.NonPublic)!
            .MakeGenericMethod(arrayType).CreateDelegate(traceType, target);
        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, new[] { 1f, 2f });
        using var entered = new ManualResetEventSlim();
        using var resume = new ManualResetEventSlim();
        object closure = native.GetMethod("NewClosure", methods)!.Invoke(null, [trace, true])!;
        target.beforeFailure = () =>
        {
            RequireBusy(() => native.GetMethod("FreeCompiledClosure", methods)!.Invoke(null, [closure]), closure);
            RequireBusy(input.Dispose, input.Storage);
            RequireBusy(MlxBackend.ClearCache, MlxWorker.Shared);
            entered.Set();
            if (!resume.Wait(TimeSpan.FromSeconds(5)))
                throw new TimeoutException("The held actual MLX trace callback did not resume within the fixture bound.");
        };
        object? view = null;
        Task? application = null;
        TargetInvocationException? invocationFailure = null;
        try
        {
            view = input.Storage.GetType().GetMethod("CreateArrayView", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(input.Storage, [input])!;
            application = Task.Run(() => native.GetMethod("ApplyClosure1", methods)!.Invoke(null, [closure, view]));
            if (!entered.Wait(TimeSpan.FromSeconds(5)))
                throw new TimeoutException("The actual MLX callback did not enter its held phase within the fixture bound.");
            RequireBusy(input.Dispose, input.Storage);
            RequireBusy(MlxBackend.ClearCache, MlxWorker.Shared);
            if (!input.Storage.IsOwnerExclusive()
                || NativeRuntimeQuarantine.Observe().State != NativeRuntimeQuarantineState.NoRecordedFailure)
                throw new InvalidOperationException("Healthy callback cleanup refusal changed storage ownership or poisoned native authority.");
        }
        finally
        {
            resume.Set();
            try { application?.GetAwaiter().GetResult(); }
            catch (TargetInvocationException error) { invocationFailure = error; }
            if (view != null) native.GetMethod("FreeArrayReference", methods)!.Invoke(null, [view]);
            native.GetMethod("FreeCompiledClosure", methods)!.Invoke(null, [closure]);
        }
        Exception failure = invocationFailure?.InnerException
            ?? throw new InvalidOperationException("The actual native trace callback did not refuse its work.");
        Exception[] causes = failure is AggregateException aggregate
            ? aggregate.Flatten().InnerExceptions.ToArray() : [failure];
        if (target.calls != 1 || causes.Count(cause => ReferenceEquals(cause, target.original)) != 1)
            throw new InvalidOperationException("The native callback did not preserve its exact managed cause once.", failure);
        return [new(input.Storage), new(closure), new(target)];
    }

    private static void RequireBusy(Action operation, object owner)
    {
        Exception? refusal = null;
        try { operation(); }
        catch (TargetInvocationException error) when (error.InnerException != null) { refusal = error.InnerException; }
        catch (InvalidOperationException error) { refusal = error; }
        if (refusal?.GetType().FullName != "TensorSharp.NativeMlxCallbackBusyException"
            || refusal.GetType().GetMethod("IsFor", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(refusal, [owner]) is not true)
            throw new InvalidOperationException("Callback-time cleanup must refuse for its actual owner before native release.", refusal);
    }
}
