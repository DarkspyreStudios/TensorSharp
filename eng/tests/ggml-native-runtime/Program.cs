using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;

if (args.Length != 3 || args[0] is not ("selected" or "reject-variant" or "reject-legacy"))
{
    Console.Error.WriteLine("Usage: ggml-native-runtime <selected|reject-variant|reject-legacy> <absolute-bridge-directory> <variant>");
    return 2;
}

try
{
    string mode = args[0], directory = Path.GetFullPath(args[1]), variant = args[2];
    string entry = Path.Combine(directory, GgmlNativeLoader.EntryLibraryName);
    byte[] bytes = File.ReadAllBytes(entry);
    var candidate = new GgmlNativeCandidate(directory, GgmlNativeLoader.TensorSharpBuild,
        GgmlNativeLoader.RuntimeIdentifier, variant, GgmlBackendType.Cpu,
        [new(GgmlNativeLoader.EntryLibraryName, bytes.LongLength, Convert.ToHexStringLower(SHA256.HashData(bytes)))]);
    Require(GgmlNativeLoader.Current == null, "The process starts without a selected bridge.");
    Require(GgmlNativeLoader.Check(candidate) == null, "The candidate passes filesystem validation before loading.");

    if (mode is "reject-variant" or "reject-legacy")
    {
        GgmlNativeCandidate rejected = mode == "reject-variant" ? candidate with { Variant = "wrong-variant" } : candidate;
        GgmlNativeSelection refusal = GgmlNativeLoader.Select([rejected, candidate]);
        Require(refusal.State == GgmlNativeSelectionState.PartiallyInitialized, "A loaded identity refusal requires a fresh process.");
        Require(refusal.Untried.Count == 1 && refusal.Untried[0] == candidate, "The next candidate is not loaded after a loaded refusal.");
        Require(refusal.Refusals.Count == 1 && refusal.Refusals[0].Code == GgmlNativeRefusalCodes.LoadFailed, "The identity refusal has a structured diagnostic.");
        if (mode == "reject-variant")
            Require(refusal.Identity?.NativeAbi == GgmlNativeLoader.NativeAbi, "The refusal exercises a real matching-ABI bridge with a different declared variant.");
        else
            Require(refusal.Identity == null, "A bridge without an exact ABI identity is not accepted as a legacy match.");
        try
        {
            GgmlNativeLoader.Select([candidate]);
            throw new InvalidOperationException("A second selection cannot recover a loaded refusal in this process.");
        }
        catch (InvalidOperationException error) when (error.Message.Contains("earlier selection", StringComparison.Ordinal))
        {
        }
        Console.WriteLine(JsonSerializer.Serialize(new { mode, state = refusal.State.ToString(), refused = refusal.Refusals[0].Message, untried = refusal.Untried.Count }));
        return 0;
    }

    GgmlNativeSelection unavailable = GgmlNativeLoader.Select([candidate with { Files = [candidate.Files[0] with { Sha256 = new string('0', 64) }] }]);
    Require(unavailable.State == GgmlNativeSelectionState.Unavailable, "A hash refusal loads no bridge and permits retry.");
    Require(unavailable.Refusals.Single().Code == GgmlNativeRefusalCodes.HashMismatch, "The hash refusal retains its reason.");
    GgmlNativeSelection selected = GgmlNativeLoader.Select([candidate with { NativeAbi = new string('0', 64) }, candidate]);
    Require(selected.State == GgmlNativeSelectionState.Loaded, "The real CPU backend initializes after pre-load refusals.");
    Require(selected.Identity?.NativeAbi == GgmlNativeLoader.NativeAbi && selected.Identity.GgmlCommit == GgmlNativeLoader.GgmlCommit,
        "The actual bridge reports the managed ABI and pinned upstream revision.");
    Require(selected.Refusals.Single().Code == GgmlNativeRefusalCodes.NotSelected, "A different ABI plan is refused before loading.");
    Require(GgmlDeepSeek4Native.NPast(IntPtr.Zero) == 0, "Other interop classes bind to the same selected bridge.");

    var context = new GgmlContext([0], GgmlBackendType.Cpu);
    var allocator = new GgmlAllocator(context, 0);
    float[] values;
    using (var input = new Tensor(allocator, DType.Float32, 4))
    using (var output = new Tensor(allocator, DType.Float32, 4))
    {
        input.SetElementsAsFloat([1, 2, 3, 4]);
        GgmlBasicOps.Add(output, input, 5);
        values = output.GetElementsAsFloat(4);
        Require(values.SequenceEqual(new float[] { 6, 7, 8, 9 }), "A real GGML CPU operation returns the expected tensor values.");
    }
    context.ReleasePooledMemory();
    Type native = typeof(GgmlNativeLoader).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!;
    Require((bool)native.GetField("s_earlyTunablesApplied", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!, "Import binding applies tunables after identity selection.");
    GgmlNativeShutdownResult shutdown = GgmlNativeLoader.Shutdown();
    Require(shutdown.Released && GgmlNativeLoader.Shutdown().Released, "Teardown after disposing the test tensors is idempotent.");
    Console.WriteLine(JsonSerializer.Serialize(new { mode, state = selected.State.ToString(), selected.Identity, values, shutdown.Released }));
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(error.ToString());
    return 1;
}

static void Require(bool condition, string message)
{
    if (!condition)
        throw new InvalidOperationException(message);
}
