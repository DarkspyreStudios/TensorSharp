using System.Collections;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

if (args.Length != 3)
    throw new ArgumentException("Usage: ggml-package-preflight MANAGED_DLL DEPLOYED_ROOT RID");

string assemblyPath = Path.GetFullPath(args[0]);
string deployedRoot = Path.GetFullPath(args[1]);
var context = new AssemblyLoadContext("actual-package-preflight", isCollectible: true);
context.Resolving += (_, name) =>
{
    string dependency = Path.Combine(Path.GetDirectoryName(assemblyPath)!, name.Name + ".dll");
    return File.Exists(dependency) ? context.LoadFromAssemblyPath(dependency) : null;
};
Assembly assembly = context.LoadFromAssemblyPath(assemblyPath);
Type loader = assembly.GetType("TensorSharp.GGML.GgmlNativeLoader", throwOnError: true)!;
object? Read(string name) => loader.GetProperty(name)!.GetValue(null);
string before = Read("State")!.ToString()!;
if (before != "Unconfigured" || Read("Current") is not null)
    throw new InvalidOperationException("Preflight requires an inert managed loader.");

int count = 0;
Exception? failure = null;
try
{
    // Exercise the actual package resolver, including catalog and deployed byte validation.
    // Explicit roots avoid an ambient installed package or this tool's own output directory.
    MethodInfo resolve = loader.GetMethod("ResolvePackageCandidatesCoreAsync", BindingFlags.NonPublic | BindingFlags.Static)!;
    Task task = (Task)resolve.Invoke(null, [new[] { deployedRoot }, args[2], CancellationToken.None])!;
    await task;
    IEnumerable candidates = (IEnumerable)task.GetType().GetProperty("Result")!.GetValue(task)!;
    foreach (object candidate in candidates)
    {
        count++;
        object? refusal = loader.GetMethod("Check")!.Invoke(null, [candidate]);
        if (refusal is not null)
            throw new InvalidDataException("A resolved candidate failed managed validation.");
    }
    if (count == 0)
        throw new InvalidDataException("Preflight found no package candidates.");
}
catch (Exception error)
{
    failure = error is TargetInvocationException invocation ? invocation.InnerException ?? error : error;
}

string after = Read("State")!.ToString()!;
bool inert = after == "Unconfigured" && Read("Current") is null;
var report = new
{
    assembly = assemblyPath,
    sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(assemblyPath))),
    mvid = assembly.ManifestModule.ModuleVersionId,
    assemblyVersion = assembly.GetName().Version!.ToString(),
    build = Read("TensorSharpBuild"),
    nativeAbi = Read("NativeAbi"),
    ggmlCommit = Read("GgmlCommit"),
    before,
    after,
    inert,
    candidates = count,
    errorType = failure?.GetType().FullName,
    error = failure?.Message
};
Console.WriteLine(JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
context.Unload();
return failure is null && inert ? 0 : 1;
