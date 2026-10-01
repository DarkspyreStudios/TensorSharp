#nullable enable

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    private static string ExerciseVisionLifetime(string mode, string scratch, BackendType backend)
    {
        string text = Path.Combine(scratch, "tiny-deepseek41.gguf");
        string vision = Path.Combine(scratch, "tiny-deepseek41.vision.gguf");
        string[] settings = ["MAX_CONTEXT", "TS_DSV4_UBATCH", "TS_DSV4_THREADS", "TS_DSV41_ALLOW_NON_CUDA_GPU"];
        string?[] previous = settings.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            Environment.SetEnvironmentVariable(settings[0], "32");
            Environment.SetEnvironmentVariable(settings[1], "1");
            Environment.SetEnvironmentVariable(settings[2], "2");
            Environment.SetEnvironmentVariable(settings[3], backend == BackendType.GgmlMetal ? "1" : null);
            ulong fingerprint = DeepSeek41LifetimeFixture.WriteText(text);
            DeepSeek41LifetimeFixture.WriteVision(vision, fingerprint, mode == "vision-mismatch" ? 64 : 32);
            string textHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(text)));
            string visionHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(vision)));
            string work;
            if (mode is "vision-load-cleanup-refusal" or "vision-dispose-cleanup-refusal")
            {
                VisionRefusalEvidence failure = mode == "vision-load-cleanup-refusal"
                    ? RefuseVisionLoadCleanup(text, vision, backend) : RefuseLoadedVisionDispose(text, vision, backend);
                for (int attempt = 0; attempt < 10; attempt++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                }
                VerifyRetainedVisionOwner(failure);
                work = failure.Report + ";actual-model-and-exact-text-vision-handles-retained-after-finalizer-drain;terminal-repeat-guard-preserves-original-cleanup-diagnostic";
            }
            else work = UseVisionLifetime(text, vision, backend, mode == "vision-mismatch");
            return $"textBytes={new FileInfo(text).Length};visionBytes={new FileInfo(vision).Length};textSha256={textHash};visionSha256={visionHash};" +
                $"tokenizerFingerprint={fingerprint:x16};work={work};no-image-encode-or-pretrained-or-mixed-logit-qualification";
        }
        finally
        {
            for (int index = 0; index < settings.Length; index++) Environment.SetEnvironmentVariable(settings[index], previous[index]);
            File.Delete(text);
            File.Delete(vision);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string UseVisionLifetime(string text, string vision, BackendType backend, bool mismatch)
    {
        var model = new DeepSeek41Model(text, backend);
        Require(model.Config.HiddenSize == 32 && model.Config.VocabSize == DeepSeek41LifetimeFixture.Vocabulary &&
            model.Tokenizer.LookupToken(ChatTemplate.DeepSeek41ImagePlaceholder) == DeepSeek41LifetimeFixture.ImageToken,
            "The normally constructed real text class has the required tokenizer and complete native model.");
        Require(NativeHandleCount("deepseek-model") == 1 && NativeHandleCount("deepseek-vision") == 0,
            "Actual native text ownership is tracked before loading the real companion.");
        if (mismatch)
        {
            InvalidDataException? validation = null;
            try { model.LoadVisionEncoder(vision); }
            catch (InvalidDataException error) { validation = error; }
            Require(validation?.Message.Contains("does not match", StringComparison.Ordinal) == true && !model.IsVisionEncoderLoaded &&
                NativeHandleCount("deepseek-vision") == 0 && NativeHandleCount("deepseek-model") == 1,
                "Real mismatching VisionInfo rolls back its returned actual native handle and preserves the original validation error.");
            model.ResetKVCache();
            Require(model.Tokenizer.LookupToken(ChatTemplate.DeepSeek41ImagePlaceholder) == DeepSeek41LifetimeFixture.ImageToken,
                "The actual text model resets through its native executor and keeps its original tokenizer after vision rollback.");
        }
        else
        {
            model.LoadVisionEncoder(vision);
            Require(model.IsVisionEncoderLoaded && NativeHandleCount("deepseek-vision") == 1,
                "The real returned vision handle validates and attaches through the normally constructed actual class.");
        }
        Require(!GgmlNativeLoader.Shutdown().Released, "Actual live text/vision ownership refuses guarded shutdown.");
        model.Dispose();
        Require(!model.IsVisionEncoderLoaded && ModelLeaseCount() == 0 && RuntimeResourceCount() == 0 && NativeHandleCount(null) == 0,
            "Successful explicit class disposal releases real text/vision handles and all managed native ownership before GC.");
        return mismatch
            ? "real-VisionInfo-mismatch-original-error-preserved;returned-handle-explicitly-freed;text-native-reset-usable;explicit-clean-disposal"
            : "real-normally-constructed-DeepSeek41;real-vision-loaded-and-attached;explicit-clean-text-and-vision-disposal";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VisionRefusalEvidence RefuseVisionLoadCleanup(string text, string vision, BackendType backend)
    {
        var model = new DeepSeek41Model(text, backend);
        var original = new InvalidDataException("controlled original vision tokenizer validation error");
        ITokenizer tokenizer = model.Tokenizer;
        PropertyInfo property = typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!;
        var wrapper = new RefusingVisionTokenizer(tokenizer, original);
        Exception? failure = null;
        try
        {
            property.SetValue(model, wrapper);
            model.LoadVisionEncoder(vision);
        }
        catch (Exception error) { failure = error; }
        finally { property.SetValue(model, tokenizer); }
        Require(wrapper.Refused && failure is AggregateException aggregate && aggregate.InnerExceptions.Count == 2 &&
            ReferenceEquals(aggregate.InnerExceptions[0], original) && aggregate.InnerExceptions[1] is InvalidOperationException &&
            original.StackTrace?.Contains(nameof(RefusingVisionTokenizer.LookupToken), StringComparison.Ordinal) == true &&
            model.Tokenizer == tokenizer && VisionHandle(model) != IntPtr.Zero,
            "Refused rollback preserves the original validation exception/stack and cleanup failure, plus the actual reserved vision field.");
        Require(NativeHandleCount("deepseek-vision") == 1 && NativeHandleCount("deepseek-model") == 1 && ModelLeaseCount() == 1,
            "The real returned vision/text handles remain tracked after guarded cleanup refuses; no post-poison native call is attempted.");
        var cleanupError = ((AggregateException)failure!).InnerExceptions[1];
        Require(ReferenceEquals(CleanupFailure(model), cleanupError), "Terminal ownership retains the exact first cleanup diagnostic.");
        var identities = NativeHandleKeys();
        RefuseRepeatedVisionDispose(model, cleanupError, identities);
        return new(new WeakReference(model), identities,
            "real-validation-original-and-cleanup-aggregate-preserved;actual-reserved-vision-handle-and-model-retained;" +
            "original-tokenizer-restored;no-native-call-or-pointer-read-after-poison;not-GPU-fault");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VisionRefusalEvidence RefuseLoadedVisionDispose(string text, string vision, BackendType backend)
    {
        var model = new DeepSeek41Model(text, backend);
        model.LoadVisionEncoder(vision);
        Require(model.IsVisionEncoderLoaded && NativeHandleCount("deepseek-model") == 1 && NativeHandleCount("deepseek-vision") == 1,
            "The normally constructed model owns an actual loaded and attached vision companion before controlled refusal.");
        var identities = NativeHandleKeys();
        PoisonOwner();
        InvalidOperationException? failure = null;
        try { model.Dispose(); }
        catch (InvalidOperationException error) { failure = error; }
        Require(failure != null && ReferenceEquals(CleanupFailure(model), failure),
            "Shared disposal preserves the exact ordinary failure type/stack and first terminal ownership diagnostic.");
        RequireExactVisionHandles(model, identities);
        RefuseRepeatedVisionDispose(model, failure!, identities);
        return new(new WeakReference(model), identities,
            "real-loaded-attached-vision-Dispose-managed-refusal;shared-terminal-boundary-retains-actual-model-and-handles;" +
            "no-native-call-or-pointer-read-after-poison;not-captured-graph-or-GPU-fault-proof");
    }

    private sealed record VisionRefusalEvidence(WeakReference Model, (string Kind, IntPtr Handle)[] NativeKeys, string Report);

    private static IntPtr VisionHandle(DeepSeek41Model model)
        => (IntPtr)typeof(DeepSeek41Model).GetField("_vision", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static Exception? CleanupFailure(DeepSeek41Model model)
        => (Exception?)typeof(ModelBase).GetField("_ownershipCleanupFailure", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);

    private static void RefuseRepeatedVisionDispose(DeepSeek41Model model, Exception failure, (string Kind, IntPtr Handle)[] identities)
    {
        string? originalStack = failure.StackTrace;
        Require(!string.IsNullOrEmpty(originalStack), "The initial cleanup failure retains its actual thrown stack.");
        for (int attempt = 0; attempt < 2; attempt++)
        {
            InvalidOperationException? repeated = null;
            try { model.Dispose(); }
            catch (InvalidOperationException error) { repeated = error; }
            Require(repeated?.Message.Contains("repeated teardown is unsafe", StringComparison.Ordinal) == true &&
                ReferenceEquals(repeated.InnerException, failure) && ReferenceEquals(CleanupFailure(model), failure) &&
                failure.StackTrace == originalStack,
                "Repeated teardown refuses at the shared terminal guard and does not replace the exact first cleanup failure.");
            RequireExactVisionHandles(model, identities);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void VerifyRetainedVisionOwner(VisionRefusalEvidence evidence)
    {
        var owners = (IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(evidence.Model.IsAlive && owners.Count == 1 && ReferenceEquals(owners[0], evidence.Model.Target) && owners[0] is DeepSeek41Model,
            "Existing generation-local retention keeps the exact actual failed vision model strongly alive after finalizer drainage.");
        RequireExactVisionHandles((DeepSeek41Model)owners[0]!, evidence.NativeKeys);
    }

    private static void RequireExactVisionHandles(DeepSeek41Model model, (string Kind, IntPtr Handle)[] identities)
    {
        IntPtr text = (IntPtr)typeof(DeepSeek4Model).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        IntPtr vision = VisionHandle(model);
        Require(text != IntPtr.Zero && vision != IntPtr.Zero && identities.Length == 2 &&
            identities.Contains(("deepseek-model", text)) && identities.Contains(("deepseek-vision", vision)) &&
            identities.ToHashSet().SetEquals(NativeHandleKeys()) && ModelLeaseCount() == 1 && RuntimeResourceCount() == 2,
            "The actual model retains exact unchanged text/vision identities and context/Model ownership, not numeric handles alone.");
    }

    private static (string Kind, IntPtr Handle)[] NativeHandleKeys()
        => ((IDictionary)typeof(GgmlNativeLoader).GetField("s_nativeResources", BindingFlags.Static | BindingFlags.NonPublic)!
            .GetValue(null)!).Keys.Cast<(string Kind, IntPtr Handle)>().ToArray();

    private static int NativeHandleCount(string? kind)
    {
        return NativeHandleKeys().Count(key => kind == null || key.Kind == kind);
    }

    private sealed class RefusingVisionTokenizer(ITokenizer original, Exception failure) : ITokenizer
    {
        internal bool Refused;
        public string[] Vocab => original.Vocab;
        public int BosTokenId => original.BosTokenId;
        public int[] EosTokenIds => original.EosTokenIds;
        public int VocabSize => original.VocabSize;
        public List<int> Encode(string text, bool addSpecial = true) => original.Encode(text, addSpecial);
        public string Decode(List<int> ids) => original.Decode(ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => original.AppendTokenBytes(tokenId, buffer);
        public bool IsEos(int tokenId) => original.IsEos(tokenId);
        public int LookupToken(string tokenStr)
        {
            if (tokenStr != ChatTemplate.DeepSeek41ImagePlaceholder) return original.LookupToken(tokenStr);
            Refused = true;
            PoisonOwner();
            throw failure;
        }
    }
}
