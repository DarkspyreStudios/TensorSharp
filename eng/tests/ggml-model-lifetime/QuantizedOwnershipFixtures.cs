#nullable enable

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

public static partial class ForeignModelLifetime
{
    private static void WriteQuantizedOwnershipFixture(string path, bool stacked)
    {
        var builder = new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense",
            PreTokenizer = "hunyuan-dense",
            IncludeMistralControlTokens = false,
            RopeBase = 10000f
        };
        var metadata = (List<(string Key, Action<BinaryWriter> Write)>)typeof(DenseDecoderSyntheticModelBuilder)
            .GetMethod("BuildMetadata", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(builder, null)!;
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(1ul);
        writer.Write((ulong)metadata.Count);
        foreach (var (key, write) in metadata) { WriteString(key); write(writer); }
        WriteString(stacked ? "blk.0.ffn_gate_exps.weight" : "output.weight");
        writer.Write(stacked ? 3u : 2u);
        writer.Write(128ul);
        writer.Write(2ul);
        if (stacked) writer.Write(2ul);
        writer.Write((uint)GgmlTensorType.Q4_0);
        writer.Write(0ul);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(new byte[stacked ? 288 : 144]);

        void WriteString(string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ObserveQuantizedTransfer(string mode, string path, BackendType backend)
    {
        var model = new QuantizedOwnershipProbe(path, backend);
        return model.LoadForcedFallback(mode, path);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ObserveBonsaiUnregister(string path, BackendType backend)
    {
        var model = new QuantizedOwnershipProbe(path, backend);
        var weight = new QuantizedWeight(new byte[36], (int)GgmlTensorType.Q2_0, 128, 1);
        typeof(QuantizedWeight).GetMethod("SetBonsaiTransform", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(weight, [Enumerable.Repeat(1f, 128).ToArray(), 128, false, 0, 0, 1]);
        IntPtr hostIdentity = weight.Data;
        IntPtr handleIdentity = weight.EnsureDeviceCacheKey();
        var keys = (ICollection<IntPtr>)typeof(QuantizedWeight).GetField("_bonsaiRegisteredKeys", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(weight)!;
        Require(hostIdentity != IntPtr.Zero && handleIdentity != IntPtr.Zero && hostIdentity != handleIdentity && keys.Count == 2,
            "Actual guarded native Bonsai registrations own both host and GCHandle identities before controlled refusal.");
        model.AddWeight(weight);

        // Invoke the real terminal guard. Do not reset it or bypass it with a raw native import.
        typeof(GgmlNativeLoader).GetMethod("PoisonAfterBackendFailure", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, null);
        bool refused = false;
        try { weight.Dispose(); }
        catch (InvalidOperationException) { refused = true; }
        Require(refused && keys.Count == 0 && !weight.HasHostData && weight.Data == IntPtr.Zero && weight.CacheKey == IntPtr.Zero,
            "Observed defect: unregister refuses, but failed keys, host ownership and GCHandle identity are discarded.");
        Require(!(bool)typeof(QuantizedWeight).GetField("_ownsCacheKeyHandle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(weight)!,
            "The failed native unregister path still frees its GCHandle.");
        try { model.Dispose(); }
        catch (InvalidOperationException) { }
        Require(ModelLeaseCount() == 1 && RuntimeResourceCount() == 2,
            "Shared model disposal retains context/Model after terminal guard refusal, not the already discarded host bytes.");
        return "observed-defect:real-Bonsai-registration;controlled-managed-poison-refusal;registeredIdentitiesBefore=2;" +
            "trackedIdentitiesAfter=0;hostDataAfter=0;cacheHandleOwnedAfter=false;no-native-call-or-pointer-read-after-poison;not-GPU-fault";
    }

    private sealed class QuantizedOwnershipProbe(string path, BackendType backend) : ModelBase(path, backend)
    {
        internal void AddWeight(QuantizedWeight weight) => _quantWeights.Add("controlled-bonsai.weight", weight);

        internal string LoadForcedFallback(string mode, string sourcePath)
        {
            EnsureQuantBackendAvailable();
            bool failRead = mode is "observe-raw-quantized-read" or "observe-stacked-quantized-read";
            var recording = new RecordingReadStream(sourcePath, failRead);
            var streamField = typeof(GgufFile).GetField("_stream", BindingFlags.Instance | BindingFlags.NonPublic)!;
            ((FileStream)streamField.GetValue(_gguf)!).Dispose();
            streamField.SetValue(_gguf, recording);
            QuantizedWeight? firstView = null;
            if (mode == "observe-stacked-owner-insertion")
                typeof(ModelBase).GetField("_stackedExpertWeights", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .SetValue(this, new Dictionary<string, StackedExpertWeights>(new RefusingComparer(static _ => throw new IOException("controlled stack-owner insertion refusal"))));
            if (mode == "stacked-partial-view-refusal")
                typeof(ModelBase).GetField("_quantWeights", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .SetValue(this, new Dictionary<string, QuantizedWeight>(new RefusingComparer(key =>
                    {
                        if (!key.EndsWith(".1.weight", StringComparison.Ordinal)) return;
                        firstView = _quantWeights.Single().Value;
                        throw new IOException("controlled second expert-view insertion refusal");
                    })));

            var classifier = typeof(ModelBase).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)!;
            object original = classifier.GetValue(this)!;
            Exception? failure = null;
            try
            {
                // Every defined backend chooses mmap. Force only the otherwise unreachable fallback classifier.
                classifier.SetValue(this, (BackendType)int.MaxValue);
                LoadWeights();
            }
            catch (Exception error) { failure = error; }
            finally { classifier.SetValue(this, original); }
            Require(failure is IOException && recording.Destination != IntPtr.Zero,
                "The raw fallback read receives an actual allocated native destination before controlled refusal.");
            Require(classifier.GetValue(this)!.Equals(original) && ExecutionPlan.UsesGgmlBackend,
                "The actual CPU/Metal context and ExecutionPlan stay unchanged; classifier restores before cleanup.");
            int views = _quantWeights.Count;
            int owners = _stackedExpertWeights.Count;
            bool partial = mode == "stacked-partial-view-refusal";
            Require(partial ? owners == 1 && views == 1 && firstView!.HasHostData : owners == 0 && views == 0,
                "Registered partial views differ from untransferred raw allocations before rollback.");
            RollBackFailedConstruction(failure!, static () => { });
            Require(_quantWeights.Count == 0 && _stackedExpertWeights.Count == 0 && RuntimeResourceCount() == 0,
                "Explicit rollback drains actual model/context ownership without GC.");
            if (partial) Require(!firstView!.HasHostData, "The registered expert view releases before its stack backing-owner phase.");
            return $"forced-fallback-unreachable-by-current-defined-backends;readDestinationCaptured=true;requestedBytes={recording.RequestedBytes};" +
                $"completedReadBytes={recording.CompletedBytes};viewsBeforeRollback={views};stackOwnersBeforeRollback={owners};" +
                (partial ? "positive-partial-view-rollback-gate" : "observed-untransferred-raw-allocation;free-absence-source-audited-not-measured") +
                ";raw-buffer-liveness-not-inferred-from-GGML-leases;no-post-rollback-pointer-access";
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private sealed class RefusingComparer(Action<string> inspect) : IEqualityComparer<string>
    {
        public bool Equals(string? left, string? right) => StringComparer.Ordinal.Equals(left, right);
        public int GetHashCode(string key) { inspect(key); return StringComparer.Ordinal.GetHashCode(key); }
    }

    private sealed class RecordingReadStream(string path, bool failRead) : FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
    {
        internal IntPtr Destination { get; private set; }
        internal int RequestedBytes { get; private set; }
        internal int CompletedBytes { get; private set; }
        public override unsafe int Read(Span<byte> buffer)
        {
            fixed (byte* pointer = buffer) Destination = (IntPtr)pointer;
            RequestedBytes = buffer.Length;
            if (failRead) return 0;
            int count = base.Read(buffer);
            CompletedBytes += count;
            return count;
        }
    }
}
