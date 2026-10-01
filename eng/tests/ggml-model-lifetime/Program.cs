#nullable enable

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length != 3 || args[0] is not ("normal" or "observe-refusal") || args[1] is not ("cpu" or "metal"))
    throw new ArgumentException("Expected normal|observe-refusal cpu|metal <absolute-bridge-directory>.");

Retirement.Run(args[0], args[1], Path.GetFullPath(args[2]));

internal static class Retirement
{
    private sealed record Evidence(WeakReference[] Roots, string[] Names, string Report);

    internal static void Run(string mode, string backend, string directory)
    {
        Evidence evidence = LoadAndUnload(mode, backend, directory);
        for (int attempt = 0; attempt < 30 && evidence.Roots.Any(root => root.IsAlive); attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Thread.Sleep(10);
        }
        string[] retained = evidence.Names.Where((_, index) => evidence.Roots[index].IsAlive).ToArray();
        if (mode == "normal" && retained.Length != 0)
            throw new InvalidOperationException("Disposed real model generation remains rooted: " + string.Join(",", retained));
        Console.WriteLine(JsonSerializer.Serialize(new { mode, backend, evidence.Report, retained,
            actualModel = "generated tiny F32 HunyuanDense", pretrainedModelQualification = "not-run",
            wholeNativeExecutor = "not-run", multiDeviceWorkers = "not-run" }));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Evidence LoadAndUnload(string mode, string backend, string directory)
    {
        string assemblyPath = typeof(ForeignModelLifetime).Assembly.Location;
        var context = new Generation(Path.GetDirectoryName(assemblyPath)!);
        Assembly fixture = context.LoadFromAssemblyPath(assemblyPath);
        string report;
        try
        {
            report = (string)fixture.GetType(nameof(ForeignModelLifetime), true)!
                .GetMethod(nameof(ForeignModelLifetime.Run))!.Invoke(null, [mode, backend, directory])!;
        }
        catch (TargetInvocationException error)
        {
            throw new InvalidOperationException("Foreign real-model fixture failed: " + error.InnerException);
        }
        Assembly[] assemblies = context.Assemblies.ToArray();
        var roots = new List<WeakReference> { new(context) };
        var names = new List<string> { "collectible ALC" };
        foreach (Assembly assembly in assemblies)
        {
            roots.Add(new(assembly));
            names.Add(assembly.GetName().Name!);
        }
        context.Unload();
        return new(roots.ToArray(), names.ToArray(), report);
    }

    private sealed class Generation(string directory) : AssemblyLoadContext("real-model-lifetime-" + Guid.NewGuid(), true)
    {
        protected override Assembly? Load(AssemblyName name)
        {
            if (name.Name != "ggml-model-lifetime" && !name.Name!.StartsWith("TensorSharp.", StringComparison.Ordinal)) return null;
            string path = Path.Combine(directory, name.Name + ".dll");
            if (!File.Exists(path)) throw new FileNotFoundException("The foreign model requires its actual private dependency.", path);
            return LoadFromAssemblyPath(path);
        }
    }
}

public static class ForeignModelLifetime
{
    public static string Run(string mode, string backendName, string directory)
    {
        AssemblyLoadContext generation = AssemblyLoadContext.GetLoadContext(typeof(ForeignModelLifetime).Assembly)!;
        foreach (Type type in new[] { typeof(ForeignModelLifetime), typeof(ModelBase), typeof(HunyuanDenseModel),
            typeof(GgufFile), typeof(TensorSharp.Tensor), typeof(GgmlNativeLoader) })
            Require(generation.IsCollectible && AssemblyLoadContext.GetLoadContext(type.Assembly) == generation,
                "Fixture, Models, Runtime, Core and GGML all belong to the foreign generation.");
        GgmlBackendType nativeBackend = backendName == "cpu" ? GgmlBackendType.Cpu : GgmlBackendType.Metal;
        BackendType backend = backendName == "cpu" ? BackendType.GgmlCpu : BackendType.GgmlMetal;
        string entry = Path.Combine(directory, GgmlNativeLoader.EntryLibraryName);
        byte[] bytes = File.ReadAllBytes(entry);
        string hash = Convert.ToHexStringLower(SHA256.HashData(bytes));
        var candidate = new GgmlNativeCandidate(directory, GgmlNativeLoader.TensorSharpBuild,
            GgmlNativeLoader.RuntimeIdentifier, "metal", nativeBackend,
            [new(GgmlNativeLoader.EntryLibraryName, bytes.LongLength, hash)]);
        GgmlNativeLoader.Configure(new(GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.NativeAbi,
            GgmlNativeLoader.RuntimeIdentifier, [candidate], RequestedBackend: nativeBackend));
        GgmlInitializationResult initialized = GgmlNativeLoader.InitializeAsync().GetAwaiter().GetResult();
        Require(initialized.State == GgmlInitializationState.Ready && initialized.ActualBackend == nativeBackend &&
            initialized.ActualNativeAbi == GgmlNativeLoader.NativeAbi && initialized.Selection.LibraryPath == entry,
            "The real pinned bridge initializes the exact requested backend/ABI.");

        string scratch = Path.Combine(Environment.GetEnvironmentVariable("TMPDIR")
            ?? throw new InvalidOperationException("TMPDIR must name repository scratch."), "model-lifetime-" + Guid.NewGuid());
        Directory.CreateDirectory(scratch);
        string modelPath = Path.Combine(scratch, "tiny-hunyuan-dense.gguf");
        try
        {
            new DenseDecoderSyntheticModelBuilder { Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense",
                IncludeMistralControlTokens = false, IncludeQkNorms = true, RopeBase = 10000f,
                ValueHeadDim = mode == "observe-refusal" ? 8 : DenseDecoderSyntheticModelBuilder.HeadDim }.Write(modelPath);
            string modelHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath)));
            string work = mode == "normal" ? UseModel(modelPath, backend) : RefuseModel(modelPath, backend);
            // Finalization is diagnostic here. A failed construction must not require GC to retire its model lease.
            for (int attempt = 0; attempt < 10; attempt++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            int modelLeases = ModelLeaseCount();
            GgmlNativeShutdownResult shutdown = GgmlNativeLoader.Shutdown();
            if (mode == "normal") Require(shutdown.Released && modelLeases == 0, "Normal real model disposal drains guarded shutdown: " + shutdown.Diagnostic);
            else Require(!shutdown.Released && modelLeases == 1, "The unfixed production refusal leaves exactly one abandoned model lease: " + shutdown.Diagnostic);
            return $"bridgeSha256={hash};nativeAbi={GgmlNativeLoader.NativeAbi};modelSha256={modelHash};" +
                $"work={work};modelLeases={modelLeases};shutdownReleased={shutdown.Released};diagnostic={shutdown.Diagnostic}";
        }
        finally
        {
            File.Delete(modelPath);
            Directory.Delete(scratch);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string UseModel(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        Require(model is HunyuanDenseModel && model.Config.NumLayers == 2, "The actual complete two-layer architecture loads.");
        Require(!GgmlNativeLoader.Shutdown().Released, "Guarded shutdown refuses a real live model.");
        float[] prefill = model.ForwardRefill([65, 66, 67]);
        Require(prefill.Length == 256 && prefill.All(float.IsFinite), "The complete real native-backed prefill produces finite logits.");
        float[] decode = model.Forward([68]);
        Require(decode.Length == 256 && decode.All(float.IsFinite), "The complete real native-backed decode produces finite logits.");
        return "full-two-layer-prefill-and-decode";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseModel(string path, BackendType backend)
    {
        try { _ = new HunyuanDenseModel(path, backend); }
        catch (NotSupportedException error) when (error.Message.Contains("equal key/value head dims", StringComparison.Ordinal))
        {
            return error.GetType().Name + ":" + error.Message;
        }
        throw new InvalidOperationException("The controlled production constructor must refuse unequal head dimensions.");
    }

    private static int ModelLeaseCount()
    {
        var resources = (IDictionary)typeof(GgmlNativeLoader).GetField("s_resources", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        return resources.Values.Cast<GgmlRuntimeResourceKind>().Count(kind => kind == GgmlRuntimeResourceKind.Model);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
