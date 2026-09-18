using System.Reflection;
using System.Runtime.InteropServices;

// Runs inside the extracted app, using its runtime and dependencies, not a test SDK.
public static class StartupHook
{
    public static void Initialize()
    {
        Assembly cuda = Assembly.Load("TensorSharp.Backends.Cuda");
        Type resolver = cuda.GetType("TensorSharp.Cuda.Interop.CudaLibraryResolver", throwOnError: true)!;
        MethodInfo resolve = resolver.GetMethod("Resolve", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(resolver.FullName, "Resolve");
        if (resolve.Invoke(null, ["cublas", cuda, null]) is not IntPtr handle || handle == IntPtr.Zero)
            throw new DllNotFoundException("The published CUDA resolver could not load bundled cuBLAS.");

        Type cv = Assembly.Load("OpenCvSharp").GetType("OpenCvSharp.Cv2", throwOnError: true)!;
        MethodInfo buildInfo = cv.GetMethod("GetBuildInformation", Type.EmptyTypes)
            ?? throw new MissingMethodException(cv.FullName, "GetBuildInformation");
        if (buildInfo.Invoke(null, null) is not string info || !info.Contains("OpenCV", StringComparison.Ordinal))
            throw new InvalidOperationException("The packaged OpenCV runtime did not return build information.");

        Type media = Assembly.Load("TensorSharp.Models")
            .GetType("TensorSharp.Models.Media.Desktop.DesktopMediaProvider", throwOnError: true)!;
        object provider = media.GetProperty("Instance")?.GetValue(null)
            ?? throw new MissingMemberException(media.FullName, "Instance");
        MethodInfo encode = media.GetMethod("EncodePng")
            ?? throw new MissingMethodException(media.FullName, "EncodePng");
        if (encode.Invoke(provider, [new byte[] { 255, 0, 0 }, 1, 1, 3]) is not byte[] { Length: > 8 })
            throw new InvalidOperationException("The packaged ImageMagick runtime could not encode a pixel.");

        if (Environment.GetEnvironmentVariable("GB10_REQUIRE_DRIVER") == "1")
            NativeLibrary.Load(Path.Combine(AppContext.BaseDirectory, "libGgmlOps.so"));

        Console.WriteLine("GB10 native dependency check passed.");
    }
}
