using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

// Standalone diagnostic: no project references, model loading or backend compute.
// Read only the unique probe variable and the documented MoE thread setting.
if (args.Length is < 1 or > 2)
{
    Console.Error.WriteLine("Usage: dsv41-native-environment ENGINE_DIRECTORY [REPORT.json]");
    return 2;
}

string engine = Path.GetFullPath(args[0]);
string unique = "TS_DSV41_ENV_PROBE_" + Environment.ProcessId + "_" + Guid.NewGuid().ToString("N");
const string threadsName = "TS_CPU_MOE_THREADS";
string? originalManagedThreads = Environment.GetEnvironmentVariable(threadsName);
var observations = new List<object>();
var result = new Dictionary<string, object?>
{
    ["framework"] = RuntimeInformation.FrameworkDescription,
    ["os"] = RuntimeInformation.OSDescription,
    ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
    ["probe_variable"] = unique,
    ["scope"] = "Managed/native environment observations only; does not inspect the private native override or an actual model's thread count.",
    ["observations"] = observations,
};
int exitCode = 0;
NativeEnvironment? native = null;
MethodInfo? setThreads = null;

try
{
    native = new NativeEnvironment();
    result["native_environment_library"] = native.LibraryName;
    if (Environment.GetEnvironmentVariable(unique) != null || native.Get(unique) != null)
        throw new InvalidOperationException("Unique probe variable unexpectedly exists");

    ObserveUnique("unique_initial");
    Environment.SetEnvironmentVariable(unique, "managed-17");
    ObserveUnique("unique_after_managed_set");
    native.Set(unique, "native-31");
    ObserveUnique("unique_after_native_set_positive_control");
    if (native.Get(unique) != "native-31")
        throw new InvalidOperationException("Native getenv positive control failed");
    Environment.SetEnvironmentVariable(unique, null);
    native.Set(unique, null);
    ObserveUnique("unique_after_cleanup");

    // Resolve the supplied existing host assemblies without copying or editing them.
    AssemblyLoadContext.Default.Resolving += (_, name) =>
    {
        string file = Path.Combine(engine, name.Name + ".dll");
        return File.Exists(file) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(file) : null;
    };
    string nativeFile = Path.Combine(engine, OperatingSystem.IsWindows() ? "GgmlOps.dll" :
        OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so");
    IntPtr bridge = NativeLibrary.Load(nativeFile);
    if (!NativeLibrary.TryGetExport(bridge, "TSGgml_SetHostMoeThreads", out _))
        throw new EntryPointNotFoundException("Existing native MoE thread setter is missing");
    // The process owns this handle until exit; the production import resolver can
    // reuse the same module without leaving dangling function pointers.
    result["native_bridge_path"] = nativeFile;
    result["native_setter_export_present"] = true;
    string modelsPath = Path.Combine(engine, "TensorSharp.Models.dll");
    result["models_assembly_sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelsPath)));
    var models = AssemblyLoadContext.Default.LoadFromAssemblyPath(modelsPath);
    var config = models.GetType("TensorSharp.Models.MoeCpuOffloadConfig", throwOnError: true)!;
    setThreads = config.GetMethod("SetCpuThreads", BindingFlags.Public | BindingFlags.Static)!;
    var property = config.GetProperty("CpuThreads", BindingFlags.Public | BindingFlags.Static)!;
    ObserveThreads("moe_initial", null);
    foreach (int requested in new[] { 9, 0 })
    {
        var captured = new StringWriter(CultureInfo.InvariantCulture);
        TextWriter originalError = Console.Error;
        try
        {
            Console.SetError(captured);
            setThreads.Invoke(null, new object[] { requested });
        }
        finally { Console.SetError(originalError); }
        observations.Add(new
        {
            stage = "moe_after_SetCpuThreads",
            requested,
            managed_property = property.GetValue(null),
            managed_environment = SafeThreadValue(Environment.GetEnvironmentVariable(threadsName)),
            native_environment = SafeThreadValue(native.Get(threadsName)),
            setter_warning_emitted = captured.GetStringBuilder().Length != 0,
        });
        if (captured.GetStringBuilder().Length != 0)
            throw new InvalidOperationException("Production MoE setter emitted a warning; observation is inconclusive");
    }
    result["completed"] = true;

    void ObserveUnique(string stage) => observations.Add(new
    {
        stage,
        managed = SafeProbeValue(Environment.GetEnvironmentVariable(unique)),
        native = SafeProbeValue(native.Get(unique)),
    });
    void ObserveThreads(string stage, int? requested) => observations.Add(new
    {
        stage, requested,
        managed_environment = SafeThreadValue(Environment.GetEnvironmentVariable(threadsName)),
        native_environment = SafeThreadValue(native.Get(threadsName)),
    });
}
catch (Exception error)
{
    result["completed"] = false;
    result["error_type"] = (error.InnerException ?? error).GetType().Name;
    // Avoid printing arbitrary exception messages or other environment contents.
    exitCode = 1;
}
finally
{
    Environment.SetEnvironmentVariable(unique, null);
    native?.Set(unique, null);
    if (setThreads != null)
    {
        try { setThreads.Invoke(null, new object[] { 0 }); }
        catch { /* Standalone process exits immediately; no model was created. */ }
    }
    Environment.SetEnvironmentVariable(threadsName, originalManagedThreads);
    native?.Dispose();
}

string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
if (args.Length == 2) File.WriteAllText(args[1], json + "\n");
Console.WriteLine(json);
return exitCode;

static string? SafeProbeValue(string? value) => value switch
{
    null => null,
    "managed-17" or "native-31" => value,
    _ => "unexpected-present-value-redacted",
};
static string? SafeThreadValue(string? value) => value == null ? null :
    int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int count) && count is >= 0 and <= 65536
        ? count.ToString(CultureInfo.InvariantCulture) : "present-value-redacted";

sealed class NativeEnvironment : IDisposable
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr GetEnv([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetEnv([MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string value, int overwrite);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int UnsetEnv([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PutEnv([MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string value);

    private readonly IntPtr _library;
    private readonly GetEnv _get;
    private readonly SetEnv? _set;
    private readonly UnsetEnv? _unset;
    private readonly PutEnv? _put;
    public string LibraryName { get; }

    public NativeEnvironment()
    {
        string[] names = OperatingSystem.IsWindows() ? new[] { "ucrtbase.dll", "msvcrt.dll" } :
            OperatingSystem.IsMacOS() ? new[] { "/usr/lib/libSystem.B.dylib" } : new[] { "libc.so.6", "libc" };
        foreach (string name in names)
        {
            if (NativeLibrary.TryLoad(name, out _library)) { LibraryName = name; break; }
        }
        if (_library == IntPtr.Zero) throw new DllNotFoundException();
        LibraryName ??= "unknown";
        _get = Marshal.GetDelegateForFunctionPointer<GetEnv>(NativeLibrary.GetExport(_library, "getenv"));
        if (OperatingSystem.IsWindows())
            _put = Marshal.GetDelegateForFunctionPointer<PutEnv>(NativeLibrary.GetExport(_library, "_putenv_s"));
        else
        {
            _set = Marshal.GetDelegateForFunctionPointer<SetEnv>(NativeLibrary.GetExport(_library, "setenv"));
            _unset = Marshal.GetDelegateForFunctionPointer<UnsetEnv>(NativeLibrary.GetExport(_library, "unsetenv"));
        }
    }
    public string? Get(string name) => Marshal.PtrToStringUTF8(_get(name));
    public void Set(string name, string? value)
    {
        int status = _put != null ? _put(name, value ?? "") : value == null ? _unset!(name) : _set!(name, value, 1);
        if (status != 0) throw new InvalidOperationException("Native environment operation failed");
    }
    public void Dispose() => NativeLibrary.Free(_library);
}
