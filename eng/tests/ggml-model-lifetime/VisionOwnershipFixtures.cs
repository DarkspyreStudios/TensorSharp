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
            if (mode == "observe-vision-cleanup-refusal")
            {
                var failure = ObserveVisionCleanupRefusal(text, vision, backend);
                for (int attempt = 0; attempt < 10; attempt++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                }
                Require(!failure.Model.IsAlive && NativeHandleCount("deepseek-vision") == 1 && NativeHandleCount("deepseek-model") == 1,
                    "The actual model collects while both unsafe native handles stay tracked; numeric ownership does not retain the lost instance.");
                work = failure.Report + ";actual-model-weak-reference-collected-after-finalizer-drain;native-ownership-still-tracked-not-model-owned";
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
        Require(ModelLeaseCount() == 0 && RuntimeResourceCount() == 0 && NativeHandleCount(null) == 0,
            "Successful explicit class disposal releases real text/vision handles and all managed native ownership before GC.");
        return mismatch
            ? "real-VisionInfo-mismatch-original-error-preserved;returned-handle-explicitly-freed;text-native-reset-usable;explicit-clean-disposal"
            : "real-normally-constructed-DeepSeek41;real-vision-loaded-and-attached;explicit-clean-text-and-vision-disposal";
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (WeakReference Model, string Report) ObserveVisionCleanupRefusal(string text, string vision, BackendType backend)
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
        Require(wrapper.Refused && failure is InvalidOperationException && !ReferenceEquals(failure, original) &&
            model.Tokenizer == tokenizer && !model.IsVisionEncoderLoaded,
            "Current failed Free masks the original error and never transfers its real returned vision handle into the model field.");
        Require(NativeHandleCount("deepseek-vision") == 1 && NativeHandleCount("deepseek-model") == 1 && ModelLeaseCount() == 1,
            "The real returned vision/text handles remain tracked after guarded cleanup refuses; no post-poison native call is attempted.");
        return (new WeakReference(model), "red-original-validation-error-masked-by-guarded-Free-refusal;real-vision-handle-still-native-tracked-but-instance-field-zero;" +
            "original-tokenizer-restored;no-native-call-or-pointer-read-after-poison;not-GPU-fault");
    }

    private static int NativeHandleCount(string? kind)
    {
        var handles = (IDictionary)typeof(GgmlNativeLoader).GetField("s_nativeResources", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        return handles.Keys.Cast<(string Kind, IntPtr Handle)>().Count(key => kind == null || key.Kind == kind);
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
