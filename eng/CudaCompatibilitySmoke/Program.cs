using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

// This process uses tiny tensors and never loads model weights or changes services.
Environment.SetEnvironmentVariable("TS_CUDA_PROFILE", "1");
if (!OperatingSystem.IsWindows())
    throw new PlatformNotSupportedException("This smoke checks the Windows CUDA deployment.");
if (!CudaBackend.IsAvailable())
    throw new InvalidOperationException("CUDA is unavailable.");

using var allocator = new CudaAllocator();
var kernels = typeof(CudaAllocator).GetProperty("Kernels", BindingFlags.Instance | BindingFlags.NonPublic)
    ?? throw new MissingMemberException("CudaAllocator.Kernels");
if (kernels.GetValue(allocator) is null)
    throw new InvalidOperationException("Main PTX module failed to load; CPU fallback is not accepted.");

// Load both native modules explicitly so an unused incompatible module cannot pass.
foreach (var name in new[] { "tensorsharp_kernels.ptx", "tensorsharp_dsv4_kernels.ptx" })
{
    var path = Path.Combine(AppContext.BaseDirectory, "cuda_kernels", name);
    var source = File.ReadAllBytes(path);
    var image = new byte[source.Length + 1];
    source.CopyTo(image, 0);
    var result = Native.cuModuleLoadData(out var module, image);
    if (result != 0)
        throw new InvalidOperationException($"{name}: CUDA module load error {result}.");
    if (name == "tensorsharp_dsv4_kernels.ptx")
        CheckDsv4(module);
    result = Native.cuModuleUnload(module);
    if (result != 0)
        throw new InvalidOperationException($"{name}: CUDA module unload error {result}.");
    Console.WriteLine($"PASS module load: {name}");
}

using var values = new Tensor(allocator, DType.Float32, 1024);
Ops.Fill(values, 3.25f);
var actual = new float[1024];
values.CopyToArray(actual);
Check(actual, Enumerable.Repeat(3.25f, 1024).ToArray(), "CUDA fill");
using var sum = Ops.Add(null!, values, values);
sum.CopyToArray(actual);
Check(actual, Enumerable.Repeat(6.5f, 1024).ToArray(), "CUDA elementwise add");

using var left = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
using var right = Tensor.FromArray(allocator, new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
using var destination = new Tensor(allocator, DType.Float32, 2, 2);
Ops.Fill(destination, 0);
using var product = Ops.Addmm(null!, 0, destination, 1, left, right);
var matrix = new float[4];
product.CopyToArray(matrix);
Check(matrix, new float[] { 58, 64, 139, 154 }, "cuBLAS matrix multiplication");

// Results alone cannot distinguish GPU execution from transparent CPU fallback.
var counters = typeof(CudaBackend).Assembly.GetType("TensorSharp.Cuda.CudaProfileCounters", throwOnError: true)!;
var field = counters.GetField("Fallbacks", BindingFlags.Static | BindingFlags.NonPublic)
    ?? throw new MissingMemberException("CudaProfileCounters.Fallbacks");
if (field.GetValue(null) is not IDictionary fallbacks || fallbacks.Count != 0)
    throw new InvalidOperationException("CUDA smoke recorded CPU fallback operations.");
Console.WriteLine("PASS zero CPU fallback operations");
foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
{
    if (module.ModuleName.StartsWith("cublas", StringComparison.OrdinalIgnoreCase))
        Console.WriteLine($"Loaded cuBLAS: {module.FileName}");
}
Console.WriteLine("PASS Windows CUDA compatibility smoke");

static void Check(float[] actual, float[] expected, string operation)
{
    if (actual.Length != expected.Length || actual.Where((value, index) =>
            !float.IsFinite(value) || Math.Abs(value - expected[index]) > 0.001f).Any())
        throw new InvalidOperationException($"Incorrect result: {operation}.");
    Console.WriteLine($"PASS {operation}");
}

static unsafe void CheckDsv4(nint module)
{
    static void Require(int result)
    {
        if (result != 0)
            throw new InvalidOperationException($"DSV4 kernel CUDA error {result}.");
    }

    Require(Native.cuModuleGetFunction(out var function, module, "ts_dsv4_dspark_argmax_f32"));
    Require(Native.cuMemAlloc(out var logits, 16));
    try
    {
        Require(Native.cuMemAlloc(out var output, 4));
        try
        {
            nint* arguments = stackalloc nint[5];
            // Exercise the changed infinity sentinel and deterministic tie behavior.
            foreach (var (input, expected) in new[]
            {
                (new float[] { float.NegativeInfinity, -9, -3, -3 }, 2),
                (Enumerable.Repeat(float.NegativeInfinity, 4).ToArray(), 0),
                (new float[] { -1, float.PositiveInfinity, float.PositiveInfinity, -2 }, 1)
            })
            {
                fixed (float* host = input)
                    Require(Native.cuMemcpyHtoD(logits, (nint)host, 16));
                nint bias = 0;
                int slot = 0, count = 4, actual = -1;
                arguments[0] = (nint)(&logits);
                arguments[1] = (nint)(&bias);
                arguments[2] = (nint)(&output);
                arguments[3] = (nint)(&slot);
                arguments[4] = (nint)(&count);
                Require(Native.cuLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0, 0, (nint)arguments, 0));
                Require(Native.cuCtxSynchronize());
                Require(Native.cuMemcpyDtoH((nint)(&actual), output, 4));
                if (actual != expected)
                    throw new InvalidOperationException($"DSV4 argmax returned {actual}; expected {expected}.");
            }
            Console.WriteLine("PASS DSV4 argmax infinity and tie cases");
        }
        finally { Require(Native.cuMemFree(output)); }
    }
    finally { Require(Native.cuMemFree(logits)); }
}

internal static class Native
{
    [DllImport("nvcuda.dll")]
    internal static extern int cuModuleLoadData(out nint module, byte[] image);

    [DllImport("nvcuda.dll")]
    internal static extern int cuModuleUnload(nint module);

    [DllImport("nvcuda.dll", CharSet = CharSet.Ansi)]
    internal static extern int cuModuleGetFunction(out nint function, nint module, string name);

    [DllImport("nvcuda.dll", EntryPoint = "cuMemAlloc_v2")]
    internal static extern int cuMemAlloc(out nint pointer, nuint bytes);

    [DllImport("nvcuda.dll", EntryPoint = "cuMemFree_v2")]
    internal static extern int cuMemFree(nint pointer);

    [DllImport("nvcuda.dll", EntryPoint = "cuMemcpyHtoD_v2")]
    internal static extern int cuMemcpyHtoD(nint device, nint host, nuint bytes);

    [DllImport("nvcuda.dll", EntryPoint = "cuMemcpyDtoH_v2")]
    internal static extern int cuMemcpyDtoH(nint host, nint device, nuint bytes);

    [DllImport("nvcuda.dll")]
    internal static extern int cuCtxSynchronize();

    [DllImport("nvcuda.dll")]
    internal static extern int cuLaunchKernel(nint function, uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ, uint sharedBytes, nint stream, nint arguments, nint extra);
}
