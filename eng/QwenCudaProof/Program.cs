using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;

internal static class Program
{
    private const string Prompt = "Write a detailed explanation of how a hash table works, including collisions, resizing, and time complexity. Use concrete examples.";

    private static async Task<int> Main(string[] args)
    {
        if (!OperatingSystem.IsWindows() || args.Length != 3)
        {
            Console.Error.WriteLine("Usage (Windows): QwenCudaProof <application-directory> <gguf-path> <evidence-json>");
            return 2;
        }
        string applicationDirectory = Path.GetFullPath(args[0]);
        AssemblyLoadContext.Default.Resolving += (_, name) =>
        {
            string path = Path.Combine(applicationDirectory, name.Name + ".dll");
            return File.Exists(path) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(path) : null;
        };
        Environment.SetEnvironmentVariable("TS_CUDA_PROFILE", "1");
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "16384");
        return await RunAsync(args[1], args[2]);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static async Task<int> RunAsync(string modelPath, string evidencePath)
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        using var service = new ModelService();
        using var session = new ChatSession();
        service.LoadModel(modelPath, null!, "Cuda");
        Type profile = typeof(TensorSharp.Cuda.CudaAllocator).Assembly.GetType("TensorSharp.Cuda.CudaProfileCounters", throwOnError: true)!;
        Dictionary<string, long> before = ReadCounts(profile, "KernelLaunches", directValues: true);
        Dictionary<string, long> fallbacksBefore = ReadCounts(profile, "Fallbacks", directValues: false);
        string output = "";
        ChatStreamUpdate terminal = default;
        await foreach (var update in service.ChatStreamWithMetricsAsync(session,
            [new ChatMessage { Role = "user", Content = Prompt }], 16, CancellationToken.None,
            new SamplingConfig { Temperature = 0 }, enableThinking: false))
        {
            output += update.Piece;
            if (update.Done)
                terminal = update;
        }
        Dictionary<string, long> launches = Difference(ReadCounts(profile, "KernelLaunches", true), before);
        Dictionary<string, long> fallbacks = Difference(ReadCounts(profile, "Fallbacks", false), fallbacksBefore);
        var evidence = new
        {
            ModelPath = modelPath, Prompt, Backend = "Cuda", MaxContext = 16384, KvCache = "F16",
            ManagedCudaAssembly = typeof(TensorSharp.Cuda.CudaAllocator).Assembly.Location,
            terminal.PromptTokens, terminal.EvalTokens, terminal.FinishReason, Output = output,
            GenerationKernelLaunches = launches, GenerationCpuFallbacks = fallbacks,
            TotalGenerationKernelLaunches = launches.Values.Sum(),
            TotalGenerationCpuFallbacks = fallbacks.Values.Sum()
        };
        string json = JsonSerializer.Serialize(evidence, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(evidencePath, json);
        Console.WriteLine(json);
        return terminal.EvalTokens > 0 && launches.Values.Sum() > 0 ? 0 : 1;
    }

    private static Dictionary<string, long> ReadCounts(Type profile, string field, bool directValues)
    {
        var result = new Dictionary<string, long>();
        object counters = profile.GetField(field, BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
        foreach (object entry in (System.Collections.IEnumerable)counters)
        {
            Type type = entry.GetType();
            string key = (string)type.GetProperty("Key")!.GetValue(entry)!;
            object value = type.GetProperty("Value")!.GetValue(entry)!;
            result[key] = directValues ? (long)value : (long)value.GetType().GetField("Count")!.GetValue(value)!;
        }
        return result;
    }

    private static Dictionary<string, long> Difference(Dictionary<string, long> after, Dictionary<string, long> before)
        => after.Select(kv => new KeyValuePair<string, long>(kv.Key, kv.Value - before.GetValueOrDefault(kv.Key)))
            .Where(kv => kv.Value != 0).ToDictionary(kv => kv.Key, kv => kv.Value);
}
