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
    private static string ExerciseQuantizedTransfer(string mode, string path, BackendType backend)
    {
        var model = new QuantizedOwnershipProbe(path, backend);
        return model.LoadForcedFallback(mode, path);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseLocalQuantizedTransfer(string path, BackendType backend, bool refuseCleanup = false)
        => new QuantizedOwnershipProbe(path, backend).TestLocalTransfer(refuseCleanup);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseQuantizedFusion(string path, BackendType backend, bool refuseSource = false)
        => new QuantizedOwnershipProbe(path, backend).TestFusionOwnership(refuseSource);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string ExerciseBonsaiRegistration(string path, BackendType backend)
    {
        var model = new QuantizedOwnershipProbe(path, backend);
        var weight = new QuantizedWeight(new byte[36], (int)GgmlTensorType.Q2_0, 128, 1);
        bool refused = false;
        try
        {
            typeof(QuantizedWeight).GetMethod("SetBonsaiTransform", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(weight, [new float[128], 128, false, 0, 0, 1]);
        }
        catch (TargetInvocationException error) when (error.InnerException is InvalidOperationException) { refused = true; }
        var keys = (ICollection<IntPtr>)typeof(QuantizedWeight).GetField("_bonsaiRegisteredKeys", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(weight)!;
        Require(refused && keys.Count == 1 && keys.Contains(weight.Data) && weight.HasHostData,
            "A real native parameter refusal keeps the reserved identity and host owner until explicit unregister confirms cleanup.");
        weight.Dispose();
        Require(keys.Count == 0 && !weight.HasHostData && weight.CacheKey == IntPtr.Zero,
            "Successful guarded unregister explicitly clears the reserved identity and actual owner.");
        model.Dispose();
        Require(RuntimeResourceCount() == 0, "Registration refusal unwinds without GC or runtime poisoning.");
        return "actual-native-Bonsai-parameter-refusal;reserved-key-kept-until-successful-unregister;explicit-clean-owner-retirement";
    }

    private static void VerifyQuantizedRetention(string mode)
    {
        var owners = (System.Collections.IList)typeof(ModelBase).GetField("FailedGgmlModelOwners", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        Require(owners.Count == 1 && RuntimeResourceCount() == 2, "The actual model/context stay retained after finalizer drainage.");
        object model = owners[0]!;
        var weights = (Dictionary<string, QuantizedWeight>)typeof(ModelBase).GetField("_quantWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        IEnumerable<QuantizedWeight> retained = weights.Values;
        if (mode == "local-bonsai-transfer-refusal")
        {
            var resources = (System.Collections.IList)typeof(ModelBase).GetField("_failedOwnershipResources", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            Require(resources.Count == 1 && weights.Count == 0, "The failed unregistered transfer retains its actual owner outside the model dictionary.");
            retained = resources.Cast<QuantizedWeight>();
        }
        Require(retained.Any() && retained.All(weight => weight.HasHostData && weight.Data != IntPtr.Zero),
            "All actual quantized owners keep their host ownership metadata after finalizer drainage.");
        if (mode != "quantized-fusion-source-refusal")
        {
            var weight = retained.Single();
            var keys = (ICollection<IntPtr>)typeof(QuantizedWeight).GetField("_bonsaiRegisteredKeys", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(weight)!;
            Require(keys.Count == 2 && keys.Contains(weight.Data) && keys.Contains(weight.CacheKey), "Both failed native identities remain tracked after finalizers.");
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static string RefuseBonsaiUnregister(string path, BackendType backend)
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
        Require(refused && keys.Count == 2 && weight.HasHostData && weight.Data == hostIdentity && weight.CacheKey == handleIdentity,
            "Unregister refusal retains both failed keys, actual host owner and GCHandle identity.");
        Require((bool)typeof(QuantizedWeight).GetField("_ownsCacheKeyHandle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(weight)!,
            "The failed native unregister path preserves its GCHandle.");
        try { model.Dispose(); }
        catch (InvalidOperationException) { }
        Require(ModelLeaseCount() == 1 && RuntimeResourceCount() == 2,
            "Shared model disposal retains actual quantized owners and context/Model after terminal guard refusal.");
        return "real-Bonsai-registration;controlled-managed-poison-refusal;registeredIdentitiesBefore=2;" +
            "trackedIdentitiesAfter=2;hostDataOwnedAfter=true;cacheHandleOwnedAfter=true;no-native-call-or-pointer-read-after-poison;not-GPU-fault";
    }

    private sealed class QuantizedOwnershipProbe(string path, BackendType backend) : ModelBase(path, backend)
    {
        internal void AddWeight(QuantizedWeight weight) => _quantWeights.Add("controlled-bonsai.weight", weight);

        internal string LoadForcedFallback(string mode, string sourcePath)
        {
            EnsureQuantBackendAvailable();
            bool failRead = mode is "raw-quantized-read-refusal" or "stacked-quantized-read-refusal";
            var recording = new RecordingReadStream(sourcePath, failRead);
            var streamField = typeof(GgufFile).GetField("_stream", BindingFlags.Instance | BindingFlags.NonPublic)!;
            ((FileStream)streamField.GetValue(_gguf)!).Dispose();
            streamField.SetValue(_gguf, recording);
            QuantizedWeight? firstView = null;
            if (mode == "stacked-owner-insertion-refusal")
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
            bool refusedOwnerTransfer = mode == "stacked-owner-insertion-refusal";
            Require(failure is IOException && (refusedOwnerTransfer
                    ? recording.Destination == IntPtr.Zero && recording.RequestedBytes == 0
                    : recording.Destination != IntPtr.Zero),
                "Stack owner insertion precedes the read; other refusals capture the actual read destination.");
            Require(classifier.GetValue(this)!.Equals(original) && ExecutionPlan.UsesGgmlBackend,
                "The actual CPU/Metal context and ExecutionPlan stay unchanged; classifier restores before cleanup.");
            int views = _quantWeights.Count;
            int owners = _stackedExpertWeights.Count;
            bool partial = mode == "stacked-partial-view-refusal";
            int expectedOwners = partial || mode == "stacked-quantized-read-refusal" ? 1 : 0;
            Require(owners == expectedOwners && (partial ? views == 1 && firstView!.HasHostData : views == 0),
                "The raw stack transfers before its read; partial views remain owned before rollback.");
            RollBackFailedConstruction(failure!, static () => { });
            Require(_quantWeights.Count == 0 && _stackedExpertWeights.Count == 0 && RuntimeResourceCount() == 0,
                "Explicit rollback drains actual model/context ownership without GC.");
            if (partial) Require(!firstView!.HasHostData, "The registered expert view releases before its stack backing-owner phase.");
            return $"forced-fallback-unreachable-by-current-defined-backends;readDestinationCaptured={!refusedOwnerTransfer};requestedBytes={recording.RequestedBytes};" +
                $"completedReadBytes={recording.CompletedBytes};viewsBeforeRollback={views};stackOwnersBeforeRollback={owners};" +
                (partial ? "positive-partial-view-rollback-gate" : "guarded-raw-ownership-unwind;raw-free-source-path-not-allocator-liveness-measurement") +
                ";raw-buffer-liveness-not-inferred-from-GGML-leases;no-post-rollback-pointer-access";
        }

        internal string TestLocalTransfer(bool refuseCleanup)
        {
            var weight = new QuantizedWeight(new byte[36], (int)GgmlTensorType.Q2_0, 128, 1);
            if (refuseCleanup)
            {
                RegisterTransform(weight);
                weight.EnsureDeviceCacheKey();
            }
            var original = new IOException("controlled unregistered owner transfer refusal");
            var transfer = typeof(ModelBase).GetMethod("TransferOwnedResource", BindingFlags.Instance | BindingFlags.NonPublic)!
                .MakeGenericMethod(typeof(QuantizedWeight));
            try
            {
                transfer.Invoke(this, [weight, (Action<QuantizedWeight>)(_ =>
                {
                    if (refuseCleanup) PoisonOwner();
                    throw original;
                })]);
            }
            catch (TargetInvocationException error)
            {
                if (refuseCleanup)
                {
                    Require(error.InnerException is AggregateException aggregate && ReferenceEquals(aggregate.InnerExceptions[0], original) &&
                        aggregate.InnerExceptions[1] is InvalidOperationException && weight.HasHostData,
                        "Failed local ownership rollback preserves both original transfer and native guard cleanup errors without releasing host data.");
                    return "real-unregistered-Bonsai-owner-retained;original-and-cleanup-errors-preserved;controlled-managed-refusal-not-GPU-fault";
                }
                Require(ReferenceEquals(error.InnerException, original), "Clean local rollback preserves the original exception object and stack.");
                Require(weight.Data == IntPtr.Zero && weight.CacheKey == IntPtr.Zero && !weight.HasHostData,
                    "The real QuantizedWeight owner explicitly retires its buffer and identity on failed local transfer.");
                Dispose();
                Require(RuntimeResourceCount() == 0, "Actual context/Model cleanup remains explicit and independent of raw ownership assertions.");
                return "actual-unregistered-QuantizedWeight-disposal-state-cleared;original-error-preserved;no-GC-release-substitute";
            }
            throw new InvalidOperationException("Local transfer must report its controlled refusal.");
        }

        internal string TestFusionOwnership(bool refuseSource)
        {
            var info = _gguf.Tensors.Single().Value;
            Require(_gguf.TryGetTensorDataPointer(info, out IntPtr data), "The actual Q4_0 GGUF maps its source bytes.");
            var gate = QuantizedWeight.CreateExternalView(data, 72, (int)GgmlTensorType.Q4_0, 128, 1, _gguf);
            var up = QuantizedWeight.CreateExternalView(data + 72, 72, (int)GgmlTensorType.Q4_0, 128, 1, _gguf);
            if (refuseSource)
            {
                RegisterTransform(gate);
                RegisterTransform(up);
                typeof(ModelBase).GetField("_quantWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(this,
                    new Dictionary<string, QuantizedWeight>(new RefusingComparer(key =>
                    {
                        if (key == "blk.0.ffn_gate_up.weight") PoisonOwner();
                    })));
            }
            _quantWeights.Add("blk.0.ffn_gate.weight", gate);
            _quantWeights.Add("blk.0.ffn_up.weight", up);
            try { FuseGateUpWeights(1); }
            catch (InvalidOperationException) when (refuseSource)
            {
                Require(_quantWeights.Count == 3 && _quantWeights.Values.Contains(gate) && _quantWeights.Values.Contains(up) &&
                    _quantWeights.Values.All(weight => weight.HasHostData),
                    "A failed source unregister keeps both sources and the successfully transferred fused view reachable.");
                return "real-mmap-fusion-source-unregister-refusal;original-sources-not-removed;fused-common-owner-retained;not-GPU-fault";
            }
            QuantizedWeight fused = _quantWeights["blk.0.ffn_gate_up.weight"];
            Require(_quantWeights.Count == 1 && !gate.HasHostData && !up.HasHostData && fused.Data == data && fused.RawBytes == 144,
                "Real mmap fusion transfers a surviving view before retiring the two old views.");
            Require(ReferenceEquals(typeof(QuantizedWeight).GetField("_ownerToken", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(fused), _gguf),
                "The fused view keeps its actual GGUF mapping owner, not an already retired source wrapper.");
            byte[] read = new byte[144];
            System.Runtime.InteropServices.Marshal.Copy(fused.Data, read, 0, read.Length);
            Require(read.All(value => value == 0), "The surviving fused view reads the actual mapped fixture bytes before disposal.");
            fused.Dispose();
            _quantWeights.Remove("blk.0.ffn_gate_up.weight");
            var ownedGate = new QuantizedWeight(Enumerable.Repeat((byte)4, 72).ToArray(), (int)GgmlTensorType.Q4_0, 128, 1);
            var ownedUp = new QuantizedWeight(Enumerable.Repeat((byte)7, 72).ToArray(), (int)GgmlTensorType.Q4_0, 128, 1);
            _quantWeights.Add("blk.0.ffn_gate.weight", ownedGate);
            _quantWeights.Add("blk.0.ffn_up.weight", ownedUp);
            FuseGateUpWeights(1);
            QuantizedWeight copied = _quantWeights["blk.0.ffn_gate_up.weight"];
            Require(!ownedGate.HasHostData && !ownedUp.HasHostData && copied.RawBytes == 144,
                "An owned fusion copy transfers before explicitly retiring both original owned sources.");
            System.Runtime.InteropServices.Marshal.Copy(copied.Data, read, 0, read.Length);
            Require(read.AsSpan(0, 72).ToArray().All(value => value == 4) && read.AsSpan(72).ToArray().All(value => value == 7),
                "The actual fusion copy remains readable after source buffers are retired.");
            Dispose();
            Require(!fused.HasHostData && !copied.HasHostData && RuntimeResourceCount() == 0, "The model explicitly retires fused views/copies and backing mapping.");
            return "real-mmap-fusion-common-owner-preserved;owned-fusion-copy-survives-source-retirement;fused-bytes-readable-before-dispose";
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private static void RegisterTransform(QuantizedWeight weight)
        => typeof(QuantizedWeight).GetMethod("SetBonsaiTransform", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(weight, [Enumerable.Repeat(1f, 128).ToArray(), 128, false, 0, 0, 1]);

    private static void PoisonOwner()
        => typeof(GgmlNativeLoader).GetMethod("PoisonAfterBackendFailure", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, null);

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
