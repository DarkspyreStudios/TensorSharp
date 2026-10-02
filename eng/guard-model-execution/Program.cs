using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

if (args.Length == 1 && args[0] == "--self-test")
{
    string[] bodies = ["ThrowIfOwnershipCleanupFailed(); Run();", " /* comment */ ThrowIfOwnershipCleanupFailed();", "Run(); ThrowIfOwnershipCleanupFailed();",
        "try { ThrowIfOwnershipCleanupFailed(); } catch { }", "if (ready) ThrowIfOwnershipCleanupFailed();", "return;", "_model.ThrowIfOwnershipCleanupFailed(); Run();",
        "_hostModel?.ThrowIfOwnershipCleanupFailed(); Run();"];
    bool[] expected = [true, true, false, false, false, false, true, true];
    for (int index = 0; index < bodies.Length; index++)
    {
        var method = CSharpSyntaxTree.ParseText("class Fixture { void RunCase() { " + bodies[index] + " } }").GetRoot()
            .DescendantNodes().OfType<MethodDeclarationSyntax>().Single();
        string guard = index == 6 ? "_model.ThrowIfOwnershipCleanupFailed();" : index == 7 ? "_hostModel?.ThrowIfOwnershipCleanupFailed();" : "ThrowIfOwnershipCleanupFailed();";
        if (HasEntryGuard(method, guard) != expected[index]) throw new InvalidOperationException("Entry-guard parser case failed: " + index);
    }
    Console.WriteLine("8 positive/negative structured admission parser cases passed; no native calls.");
    return;
}
if (args.Length != 2 || args[0] is not ("--inventory" or "--verify"))
    throw new ArgumentException("Expected --inventory|--verify and the repository root, or --self-test.");
string root = Path.GetFullPath(args[1]);
string models = Path.Combine(root, "TensorSharp.Models");
var files = Directory.EnumerateFiles(models, "*.cs", SearchOption.AllDirectories)
    .Where(path => !path.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar) &&
        !path.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar))
    .Order(StringComparer.Ordinal).Select(path => (Path: path, Text: File.ReadAllText(path)))
    .Select(file => (file.Path, file.Text, Root: CSharpSyntaxTree.ParseText(file.Text).GetRoot())).ToArray();
var classes = files.SelectMany(file => file.Root.DescendantNodes().OfType<ClassDeclarationSyntax>()).ToArray();
var modelTypes = new HashSet<string>(StringComparer.Ordinal) { "ModelBase" };
bool changed;
do
{
    changed = false;
    foreach (var type in classes)
        if (type.BaseList?.Types.Any(parent => modelTypes.Contains(parent.Type.ToString())) == true)
            changed |= modelTypes.Add(type.Identifier.ValueText);
} while (changed);

var operations = new HashSet<string>(StringComparer.Ordinal)
{
    "PrepareForPrefill", "TrimIdleMemory", "ReleaseGgmlDeviceResidency", "BeginDistributedDriver", "Forward", "ForwardRefill", "ResetKVCache",
    "RunDistributedWorkerLoop", "SubmitGreedyDecodeStep", "ResetPipelinedGreedyState", "TruncateKVCache", "TryTruncateKVCache", "TryExtractKVBlock", "TryInjectKVBlock",
    "WarmUpKernels", "WarmUpMultimodalKernels", "ForwardBatch", "TryMigrateLinearKVToPaged", "BindSequenceCache", "AdoptPrimaryCacheToFused", "RestorePrimaryCache",
    "OnSequenceReleased", "TryForwardBatchedFusedDecode", "TryForwardBatchedFusedDecodeSampled", "RetainSequenceCache", "RetainSequenceCacheAs", "TryRebindRetainedCache",
    "DiscardRetainedCache", "TryCheckpointActiveCache", "TryCloneRetainedCache", "TryExportRetainedCache", "TryImportRetainedCache", "AttachPrefixCache", "DetachPrefixCache",
    "TryConvertPrimary", "SettleForCopy", "DiscardRetainedCaches", "TryCaptureCopy", "TryCaptureDonate", "TryMaterialize", "TryReturnDonation", "ReleasePayloads",
    "TryCopyPagedToHolder", "TryExport", "TryImport", "TryBeginImport", "RunImportRead", "TryCommitImport", "AbortImport",
    "LoadMtpDraftWeights", "DraftStep", "DraftBlock", "DraftCatchUp", "DraftCatchUpAndStep", "SpecForward", "SpecEnsureCapacity", "SpecSnapshotRecurrentState",
    "SpecRestoreRecurrentState", "SpecRewindCache", "SpecOnVerifyAccepted", "SpecForwardBatched", "SpecSnapshotRecurrentStateSlots", "SpecRestoreRecurrentStateSlots",
    "DebugTimeQuantMatmul", "LoadVisionEncoder", "LoadAudioEncoder", "SetVisionEmbeddings", "SetAudioEmbeddings", "SetMRoPEPositions", "EncodeImage", "ExpandMultimodalPrompt",
    "SetSequenceVisionEmbeddings", "QueueSequenceVisionEmbeddings", "ClearVisionEmbeddings", "ReadStructured", "ReadStructuredReference", "ClearStructuredCache",
    "ForwardCanvas", "PrefillPrompt", "DecodeCanvas", "DecodeCanvasSampled", "DecodeCanvasSampledSeq", "CreateSequenceState", "DisposeSequenceState",
    "PrefillSequence", "DecodeCanvasSeq", "DecodeCanvasBatched", "EditImage", "GenerateImage", "GenerateVideo", "Predict", "EncodeHidden", "EmbedTokensForMultimodal", "CreateTextEncoder",
    "CreateSeqState", "DisposeSeqState", "PrefillSeq", "UseSequenceVision", "LoadDFlashDraftWeights", "RefreshActiveFusedHolderAfterCacheGrowth",
    "FlushArenaSlotForActiveHolder", "InvalidateFullDecodeState", "InvalidateVerifyCache", "EnterSpecSession", "ExitSpecSession", "TryFullModelDecode",
    "TryFullModelDecodeToken", "DrainDeviceRecurrentState", "TryFullModelVerify", "TryFusedMtpBlock", "TraceLayerResidual", "CreateDiT", "CreateVideoVae", "CreateAudioVae"
};
var injectorOperations = new HashSet<string>(StringComparer.Ordinal)
{
    "LoadProjectors", "ProcessPromptTokens", "TrimPreparedPrompt", "QueuePromptEmbeddings", "QueuePromptEmbeddingsForSlice", "ClearPreparedPromptState", "SetMRoPEPositions", "Dispose"
};
var encoders = new Dictionary<string, string>(StringComparer.Ordinal)
{
    ["Gemma4VisionEncoder"] = "_hostModel",
    ["Gemma4AudioEncoder"] = "_hostModel",
    ["NemotronVisionEncoder"] = "_hostModel",
    ["NemotronAudioEncoder"] = "_host",
    ["Mistral3VisionEncoder"] = "_hostModel",
    ["MuseGlimmerVisionEncoder"] = "_hostModel",
    ["Qwen35VisionEncoder"] = "_hostModel",
    ["GlmNextVisionEncoder"] = "_hostModel"
};
var inventory = new List<object>();
var failures = new List<string>();
var exclusions = new HashSet<string>(StringComparer.Ordinal)
{
    "LogVramSnapshot", "ResetForwardTiming", "CanTruncateKVCache", "YieldGpuComputeLock", "CanPrefillMediaAfterReusedPrefix", "ComputeKVBlockByteSize",
    "HasVisionEncoder", "PrintTimingStats", "SampleGreedy", "Sample", "Dispose", "ThrowIfOwnershipCleanupFailed", "HasFusedSequenceCache", "GetPrefixCacheCapabilities",
    "QuerySpareBytes", "EstimateCloneBytes", "PrintForwardTiming", "CanMaterialize", "MeasureEndState", "IsRetainedHostDirty", "DescribeSpecProfile",
    "ComputePrefillChunkSize", "CanBatchDecode", "VideoVaeResidentBytesEstimate", "TraceLogits", "TraceLayerSetRows", "TraceLayerDone", "RetainedIdleHolderBytes",
    "PrivateIdleHolderBytes", "IsRetainedDeviceAuthoritative", "ResetSpecLayerTimings", "CanReuseLivePrefix", "CanReuseRetainedPrefix", "IsRetainedCheckpoint",
    "RetainedCacheBytes", "DeviceForLayer"
};
int admissions = 0;
foreach (var file in files)
{
    foreach (var method in file.Root.DescendantNodes().OfType<MethodDeclarationSyntax>())
    {
        if (method.Modifiers.Any(SyntaxKind.StaticKeyword) || method.Body == null && method.ExpressionBody == null) continue;
        var type = method.Parent as TypeDeclarationSyntax;
        if (type == null) continue;
        string owner = type.Identifier.ValueText;
        string name = method.Identifier.ValueText;
        bool exposed = method.Modifiers.Any(SyntaxKind.PublicKeyword) || method.Modifiers.Any(SyntaxKind.InternalKeyword) || method.ExplicitInterfaceSpecifier != null;
        string? guard = null;
        if (modelTypes.Contains(owner) && exposed && operations.Contains(name)) guard = "ThrowIfOwnershipCleanupFailed();";
        if (owner == "DeepSeek4Model" && exposed && name is "CanReuseLivePrefix" or "CanReuseRetainedPrefix" or "CanMaterialize" or "MeasureEndState")
            guard = "ThrowIfOwnershipCleanupFailed();";
        if (owner == "ModelMultimodalInjector" && exposed && (injectorOperations.Contains(name) || name.StartsWith("Process", StringComparison.Ordinal) && name.EndsWith("History", StringComparison.Ordinal)))
            guard = "_model.ThrowIfOwnershipCleanupFailed();";
        if (encoders.TryGetValue(owner, out string? host) && exposed && name is "Encode" or "Dispose") guard = host + "?.ThrowIfOwnershipCleanupFailed();";
        if (owner is "QwenImage21Pipeline" or "WanVideoPipeline" or "MiniMaxH3Pipeline" && exposed && name is "Run" or "Generate" or "Dispose")
            guard = "_model.ThrowIfOwnershipCleanupFailed();";
        if (owner == "DiffusionGemmaSampler" && exposed && name is "Generate" or "DenoiseBlock" or "RunBlockBatched")
            guard = "_model.ThrowIfOwnershipCleanupFailed();";
        if (owner == "VisionScope" && name == "Dispose") guard = "_model?.ThrowIfOwnershipCleanupFailed();";
        string relative = Path.GetRelativePath(root, file.Path).Replace('\\', '/');
        int line = method.GetLocation().GetLineSpan().StartLinePosition.Line + 1;
        if (guard == null)
        {
            if (modelTypes.Contains(owner) && exposed)
            {
                bool classified = exclusions.Contains(name);
                inventory.Add(new { path = relative, line, owner, name, classification = classified ? "metadata-diagnostic-or-existing-shared-teardown" : "unclassified", guard });
                if (!classified) failures.Add($"Unclassified model boundary: {relative}:{line} {owner}.{name}");
            }
            continue;
        }
        bool guarded = HasEntryGuard(method, guard);
        admissions++;
        inventory.Add(new { path = relative, line, owner, name, parameters = method.ParameterList.ToString(), classification = "admission", guard, guarded });
        if (!guarded) failures.Add($"Missing first-statement admission guard: {relative}:{line} {owner}.{name}");
        if (owner == "ModelBase" && name == "RunDistributedWorkerLoop")
        {
            var loop = method.Body?.Statements.OfType<WhileStatementSyntax>().SingleOrDefault();
            var block = loop?.Statement as BlockSyntax;
            int outsideGuards = block?.Statements.Count(statement => statement.ToString() == guard) ?? 0;
            var receive = block?.DescendantNodes().OfType<InvocationExpressionSyntax>().SingleOrDefault(call => call.Expression.ToString() == "_tpGroup.ReceiveControl");
            var dispatch = block?.DescendantNodes().OfType<SwitchStatementSyntax>().SingleOrDefault();
            var guards = block?.Statements.Where(statement => statement.ToString() == guard).ToArray();
            if (outsideGuards != 2 || block?.Statements.FirstOrDefault()?.ToString() != guard || receive == null || dispatch == null ||
                guards![1].SpanStart <= receive.SpanStart || guards[1].SpanStart >= dispatch.SpanStart)
                failures.Add("Worker receive and direct dispatch require unconditional terminal guards outside provider catches.");
        }
    }
    foreach (var constructor in file.Root.DescendantNodes().OfType<ConstructorDeclarationSyntax>().Where(constructor => constructor.Identifier.ValueText == "QwenImage21Vae"))
    {
        string guard = "model.ThrowIfOwnershipCleanupFailed();";
        bool guarded = HasEntryGuard(constructor, guard);
        admissions++;
        inventory.Add(new { path = Path.GetRelativePath(root, file.Path).Replace('\\', '/'), owner = "QwenImage21Vae", name = ".ctor", classification = "borrowed-model-admission", guard, guarded });
        if (!guarded) failures.Add("QwenImage21Vae must refuse a failed supplied model before borrowing its source.");
    }
}
if (admissions == 0) failures.Add("The source inventory contains no model admission boundaries.");
if (args[0] == "--inventory") Console.WriteLine(JsonSerializer.Serialize(inventory, new JsonSerializerOptions { WriteIndented = true }));
else
{
    foreach (string failure in failures) Console.Error.WriteLine(failure);
    Console.WriteLine($"Checked {admissions} mapped admission methods; {failures.Count} coverage failures. Source coverage is not model/device execution qualification.");
}
Environment.ExitCode = failures.Count == 0 ? 0 : 1;

static bool HasEntryGuard(BaseMethodDeclarationSyntax method, string guard)
    => method.Body?.Statements.FirstOrDefault()?.WithoutTrivia().NormalizeWhitespace().ToString() == guard;
