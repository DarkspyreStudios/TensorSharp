using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (!OperatingSystem.IsWindows() || args.Length is < 1 or > 2)
        {
            Console.Error.WriteLine("Usage (Windows): GgmlCudaCompatibilitySmoke <application-directory> [model.gguf]");
            return 2;
        }
        string directory = Path.GetFullPath(args[0]);
        AssemblyLoadContext.Default.Resolving += (_, name) =>
        {
            string path = Path.Combine(directory, name.Name + ".dll");
            return File.Exists(path) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(path) : null;
        };
        CheckNativeCuda();
        CheckManagedOperations();
        if (args.Length == 2)
            await CheckModelAsync(args[1]);
        Console.WriteLine("PASS GGML CUDA compatibility smoke");
        return 0;
    }

    private static void CheckNativeCuda()
    {
        nint backend = Native.ggml_backend_cuda_init(0);
        if (backend == 0)
            throw new InvalidOperationException("Native CUDA backend initialization failed.");
        try
        {
            if (!Native.ggml_backend_is_cuda(backend))
                throw new InvalidOperationException("Native backend is not CUDA; CPU fallback is not accepted.");
            Console.WriteLine("PASS native CUDA backend: " + Marshal.PtrToStringUTF8(Native.ggml_backend_name(backend)));
        }
        finally { Native.ggml_backend_free(backend); }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void CheckManagedOperations()
    {
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        using var values = new Tensor(allocator, DType.Float32, 1024);
        Ops.Fill(values, 3.25f);
        var actual = new float[1024];
        values.CopyToArray(actual);
        Check(actual, Enumerable.Repeat(3.25f, 1024).ToArray(), "GGML CUDA fill");
        using var sum = Ops.Add(null!, values, values);
        sum.CopyToArray(actual);
        Check(actual, Enumerable.Repeat(6.5f, 1024).ToArray(), "GGML CUDA add");
        using var left = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var right = Tensor.FromArray(allocator, new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        using var destination = new Tensor(allocator, DType.Float32, 2, 2);
        Ops.Fill(destination, 0);
        using var product = Ops.Addmm(null!, 0, destination, 1, left, right);
        var matrix = new float[4];
        product.CopyToArray(matrix);
        Check(matrix, [58, 64, 139, 154], "GGML CUDA matrix multiplication");
    }

    private static void Check(float[] actual, float[] expected, string operation)
    {
        if (actual.Length != expected.Length || actual.Where((value, index) =>
                !float.IsFinite(value) || Math.Abs(value - expected[index]) > 0.001f).Any())
            throw new InvalidOperationException("Incorrect result: " + operation);
        Console.WriteLine("PASS " + operation);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static async Task CheckModelAsync(string path)
    {
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "16384");
        KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
        using var service = new ModelService();
        using var session = new ChatSession();
        service.LoadModel(path, null!, "ggml_cuda");
        ChatStreamUpdate terminal = default;
        await foreach (var update in service.ChatStreamWithMetricsAsync(session,
            [new ChatMessage { Role = "user", Content = "Write a detailed explanation of how a hash table works, including collisions, resizing, and time complexity. Use concrete examples." }],
            16, CancellationToken.None, new SamplingConfig { Temperature = 0 }, enableThinking: false))
        {
            Console.Write(update.Piece);
            if (update.Done)
                terminal = update;
        }
        if (terminal.EvalTokens != 16)
            throw new InvalidOperationException($"Model generated {terminal.EvalTokens} tokens; expected 16. Finish: {terminal.FinishReason}");
        Console.WriteLine($"\nPASS GGML CUDA model: prompt={terminal.PromptTokens}, output={terminal.EvalTokens}, finish={terminal.FinishReason}");
    }
}

internal static class Native
{
    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nint ggml_backend_cuda_init(int device);

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool ggml_backend_is_cuda(nint backend);

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nint ggml_backend_name(nint backend);

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void ggml_backend_free(nint backend);
}
