// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Threading.Channels;

namespace TensorSharp.Studio;

internal sealed class StudioJobRecord
{
    public StudioJob Job { get; set; } = new();
    public StudioProject? Snapshot { get; set; }
    public StudioGenerationRequest? Generation { get; set; }
    public string Fingerprint { get; set; } = "";
}

public sealed partial class StudioService
{
    readonly object jobSync = new();
    readonly Dictionary<string, StudioJobRecord> jobs = [];
    readonly Dictionary<string, CancellationTokenSource> jobCancellations = [];
    readonly CancellationTokenSource shutdown = new();
    Channel<string> queue = null!;
    Task worker = null!;
    bool stopping;
    string JobDirectory(string jobId) => StudioPaths.SafeChild(RootPath, "jobs", StudioPaths.Id(jobId));
    void SaveJob(StudioJobRecord record) { record.Job.UpdatedAt = DateTimeOffset.UtcNow; StudioStorage.Save(StudioPaths.SafeChild(JobDirectory(record.Job.Id), "job.json"), record); }

    void InitializeJobs()
    {
        queue = Channel.CreateBounded<string>(new BoundedChannelOptions(Options.MaxQueuedJobs) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        int cleanupBudget = StartupCleanupBudget;
        CleanupPortableScratch(ref cleanupBudget);
        foreach (var directory in Directory.EnumerateDirectories(StudioPaths.SafeChild(RootPath, "jobs")).Take(10000))
        {
            var id = Path.GetFileName(directory);
            if (!IsManagedId(id)) continue; // Preserve unrelated directories in the configured root.
            var path = StudioPaths.SafeChild(JobDirectory(id), "job.json");
            CleanupAtomicTemporaries(directory, "job.json", ref cleanupBudget);
            if (!File.Exists(path))
            {
                CleanupScratchTree(StudioPaths.SafeChild(directory, "work"), ref cleanupBudget);
                continue;
            }
            var record = StudioStorage.Read<StudioJobRecord>(path);
            if (record.Job.Id != id) throw new StudioException("invalid_job", "Job identity does not match its directory.");
            if (record.Job.State is "queued" or "running" or "cancelling")
            {
                var adopted = record.Job.Kind == "generate" ? LoadDocument(record.Job.ProjectId).Project.Assets
                    .FirstOrDefault(a => a.Generation?.Request.RequestId == record.Job.RequestId && File.Exists(StudioPaths.SafeChild(ProjectDirectory(record.Job.ProjectId), "assets", a.FileName))) : null;
                if (adopted != null)
                {
                    record.Job.State = "completed"; record.Job.Stage = "completed"; record.Job.Progress = 1;
                    record.Job.OutputAssetId = adopted.Id; record.Job.Error = null;
                }
                else
                {
                    record.Job.State = "interrupted"; record.Job.Stage = "interrupted";
                    record.Job.Error = "The host stopped before this job finished. Retry with a new request identifier. Inference cannot resume mid-denoising.";
                }
                SaveJob(record);
            }
            jobs.Add(id, record);
            // Partial work is never presented as a completed artifact. A restart can safely discard only job-owned scratch.
            var scratch = StudioPaths.SafeChild(directory, "work");
            if (Directory.Exists(scratch) && record.Job.State != "running") CleanupScratchTree(scratch, ref cleanupBudget);
        }
        worker = Task.Run(ProcessJobsAsync);
    }
    public Task<IReadOnlyList<StudioJob>> ListJobsAsync(string projectId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); LoadDocument(projectId);
        lock (jobSync) return Task.FromResult<IReadOnlyList<StudioJob>>(jobs.Values.Where(r => r.Job.ProjectId == projectId).OrderByDescending(r => r.Job.CreatedAt).Select(r => StudioJson.Clone(r.Job)).ToArray());
    }
    public Task<StudioJob> CancelJobAsync(string jobId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); StudioPaths.Id(jobId);
        lock (jobSync)
        {
            if (!jobs.TryGetValue(jobId, out var record)) throw new StudioException("job_not_found", "Job does not exist.");
            if (record.Job.State == "queued") { record.Job.State = "cancelled"; record.Job.Stage = "cancelled"; SaveJob(record); }
            else if (record.Job.State == "running")
            {
                record.Job.State = "cancelling"; record.Job.Stage = "cancelling"; SaveJob(record);
                if (jobCancellations.TryGetValue(jobId, out var source)) source.Cancel();
            }
            return Task.FromResult(StudioJson.Clone(record.Job));
        }
    }
    public string GetJobOutputPath(string projectId, string jobId)
    {
        LoadDocument(projectId); StudioPaths.Id(jobId);
        lock (jobSync)
        {
            if (!jobs.TryGetValue(jobId, out var record) || record.Job.ProjectId != projectId) throw new StudioException("job_not_found", "Job does not belong to this project.");
            if (record.Job.State != "completed" || record.Job.OutputFileName == null) throw new StudioException("output_not_ready", "The job does not have a completed render output.");
            var path = StudioPaths.SafeChild(JobDirectory(jobId), record.Job.OutputFileName);
            if (!File.Exists(path)) throw new StudioException("missing_output", "Rendered output is missing; render the project again.");
            return path;
        }
    }
    public async Task<StudioJob> StartRenderAsync(string projectId, long expectedRevision, string requestId, CancellationToken cancellationToken = default)
    {
        var project = await GetProjectAsync(projectId, cancellationToken); CheckRevision(project, expectedRevision);
        if (project.DurationFrames <= 0) throw new StudioException("empty_timeline", "Add clips to the timeline before rendering.");
        foreach (var c in project.Captions) StudioCaptionBitmap.Validate(c.Text);
        var capability = await Media.GetCapabilitiesAsync(cancellationToken);
        if (!capability.Available) throw new StudioException("render_unavailable", capability.Reason!);
        foreach (var clip in project.Clips) GetSnapshotAssetPath(project, clip.AssetId);
        return Enqueue(new() { Job = new() { ProjectId = projectId, RequestId = requestId, Kind = "render", ProjectRevision = project.Revision }, Snapshot = project,
            Fingerprint = Hash(new { kind = "render", projectId, expectedRevision }) });
    }
    public async Task<StudioJob> StartGenerationAsync(StudioGenerationRequest request, CancellationToken cancellationToken = default)
    {
        if (generation == null) throw new StudioException("generation_unavailable", "No native generation adapter is connected to this host.");
        request = StudioJson.Clone(request);
        if (request.TemporalConditions is null || request.ReferenceAssetIds is null || request.TemporalConditions.Any(c => c is null) || request.ReferenceAssetIds.Any(id => id is null))
            throw new StudioException("invalid_conditioning", "Conditioning collections must not be null.");
        var project = await GetProjectAsync(request.ProjectId, cancellationToken);
        var generationMediaReason = GenerationMediaUnavailableReason(await Media.GetCapabilitiesAsync(cancellationToken));
        if (generationMediaReason != null) throw new StudioException("generation_output_unavailable", generationMediaReason);
        var capabilities = await generation.GetCapabilitiesAsync(cancellationToken);
        var capability = capabilities.FirstOrDefault(c => c.ModelId == request.ModelId) ?? throw new StudioException("model_not_found", "Selected video model is not available from this host registry.");
        if (!capability.Available) throw new StudioException("model_unavailable", capability.UnavailableReason ?? "Selected video model is not installed.");
        if (request.Width == 0) request.Width = capability.DefaultWidth;
        if (request.Height == 0) request.Height = capability.DefaultHeight;
        if (request.Fps == 0) request.Fps = capability.DefaultFps;
        request.GenerateAudio ??= capability.DefaultGenerateAudio;
        if (request.FrameCount == 0) request.FrameCount = capability.DefaultFrameCount;
        if (request.Steps == 0) request.Steps = capability.DefaultSteps;
        if (request.CfgScale == 0) request.CfgScale = capability.DefaultCfgScale;
        if (string.IsNullOrWhiteSpace(request.Prompt) || request.Prompt.Length > 16000 || request.NegativePrompt?.Length > 16000)
            throw new StudioException("invalid_prompt", "Generation prompt must contain 1–16000 characters; negative prompt is limited to 16000.");
        if (request.Width < 16 || request.Height < 16 || request.Width > capability.MaxDimension || request.Height > capability.MaxDimension ||
            request.Width % capability.SpatialMultiple != 0 || request.Height % capability.SpatialMultiple != 0)
            throw new StudioException("invalid_resolution", $"This model requires dimensions divisible by {capability.SpatialMultiple}, up to {capability.MaxDimension} pixels.");
        if (request.FrameCount < capability.TemporalOffset || request.FrameCount > capability.MaxFrames || (request.FrameCount - capability.TemporalOffset) % capability.TemporalMultiple != 0)
            throw new StudioException("invalid_frame_count", $"This model requires {capability.TemporalMultiple}k+{capability.TemporalOffset} frames up to {capability.MaxFrames}.");
        if (request.Fps is < 1 or > 120 || capability.FixedFps is { } fixedFps && request.Fps != fixedFps)
            throw new StudioException("invalid_fps", capability.FixedFps is { } fixedRate ? $"This model requires {fixedRate} fps." : "Frame rate must be between 1 and 120.");
        if (request.Steps is < 1 or > 200 || capability.FixedSteps is { } steps && request.Steps != steps || !float.IsFinite(request.CfgScale) || request.CfgScale is < 1 or > 30 || capability.FixedCfgScale is { } cfg && Math.Abs(request.CfgScale - cfg) > 0.0001)
            throw new StudioException("invalid_sampling", "Steps or CFG do not match the selected model's supported sampling settings.");
        if (!capability.TextToVideo && request.ConditioningAssetId == null) throw new StudioException("conditioning_required", "This checkpoint requires first-frame image conditioning.");
        if (request.ConditioningAssetId != null && !capability.ImageConditioning || request.EndConditioningAssetId != null && !capability.LastFrameConditioning || request.GenerateAudio == true && !capability.AudioOutput)
            throw new StudioException("unsupported_conditioning", "The selected model does not support a requested image/audio conditioning mode.");
        if (request.TemporalConditions.Count > 0 && !capability.TemporalGuidance || request.ReferenceAssetIds.Count > 0 && !capability.IdentityReferences || request.ReferenceAssetIds.Count > capability.MaxReferences || request.TemporalConditions.Count > 32)
            throw new StudioException("unsupported_conditioning", "Temporal guidance or identity-reference images exceed this model's capabilities.");
        foreach (var guide in request.TemporalConditions)
            if (guide.FrameIndex < -1 || guide.FrameIndex >= request.FrameCount || !float.IsFinite(guide.Strength) || guide.Strength is <= 0 or > 1)
                throw new StudioException("invalid_conditioning", "Temporal guide indices and strengths must fit the output frame range and (0, 1].");
        var fingerprint = Hash(new { kind = "generate", request });
        lock (jobSync)
        {
            // An accepted request owns its captured inputs, even after a library deletion.
            // Reconnecting clients may retrieve that job without re-admitting removed assets.
            var existing = FindExistingJob(request.ProjectId, request.RequestId, fingerprint);
            if (existing != null) return existing;
        }
        var paths = ResolveConditioning(request, project);
        await generation.ValidateAsync(request, paths, cancellationToken);
        return Enqueue(new() { Job = new() { ProjectId = project.Id, RequestId = request.RequestId, Kind = "generate", ProjectRevision = project.Revision }, Generation = request, Snapshot = project,
            Fingerprint = fingerprint });
    }
    Dictionary<string, string> ResolveConditioning(StudioGenerationRequest request, StudioProject project)
    {
        var ids = request.ReferenceAssetIds.Concat(request.TemporalConditions.Select(c => c.AssetId)).Concat(new[] { request.ConditioningAssetId, request.EndConditioningAssetId }.OfType<string>()).Distinct(StringComparer.Ordinal);
        var paths = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var id in ids)
        {
            var asset = project.Assets.FirstOrDefault(a => a.Id == id) ?? throw new StudioException("asset_not_found", "A conditioning asset does not belong to the project.");
            if (asset.Kind != "image") throw new StudioException("invalid_conditioning", "Conditioning assets must be still images.");
            paths.Add(id, GetSnapshotAssetPath(project, id));
        }
        return paths;
    }
    StudioJob Enqueue(StudioJobRecord record)
    {
        StudioPaths.Id(record.Job.RequestId); CheckDisk();
        lock (jobSync)
        {
            if (stopping) throw new StudioException("host_stopping", "Studio is stopping; restart the host before submitting work.");
            var existing = FindExistingJob(record.Job.ProjectId, record.Job.RequestId, record.Fingerprint);
            if (existing != null) return existing;
            if (jobs.Values.Count(r => r.Job.State is "queued" or "running" or "cancelling") >= Options.MaxQueuedJobs)
                throw new StudioException("queue_full", "The local media job queue is full. Wait or cancel queued work before retrying.");
            jobs.Add(record.Job.Id, record); SaveJob(record);
            if (!queue.Writer.TryWrite(record.Job.Id)) { jobs.Remove(record.Job.Id); Directory.Delete(JobDirectory(record.Job.Id), true); throw new StudioException("queue_full", "The local job queue is full."); }
            return StudioJson.Clone(record.Job);
        }
    }
    // Call under jobSync.
    StudioJob? FindExistingJob(string projectId, string requestId, string fingerprint)
    {
        var existing = jobs.Values.FirstOrDefault(r => r.Job.ProjectId == projectId && r.Job.RequestId == requestId);
        if (existing == null) return null;
        if (existing.Fingerprint != fingerprint) throw new StudioException("idempotency_conflict", "This job request identifier was already used for different parameters.");
        return StudioJson.Clone(existing.Job);
    }
    sealed class InlineProgress(Action<StudioProgress> callback) : IProgress<StudioProgress> { public void Report(StudioProgress value) => callback(value); }
    async Task ProcessJobsAsync()
    {
        try
        {
            await foreach (var id in queue.Reader.ReadAllAsync(shutdown.Token))
            {
                StudioJobRecord record; CancellationTokenSource cancellation;
                lock (jobSync)
                {
                    record = jobs[id]; if (record.Job.State != "queued") continue;
                    cancellation = CancellationTokenSource.CreateLinkedTokenSource(shutdown.Token); jobCancellations[id] = cancellation;
                    record.Job.State = "running"; record.Job.Stage = "starting"; SaveJob(record);
                }
                var workDirectory = StudioPaths.SafeChild(JobDirectory(id), "work"); Directory.CreateDirectory(workDirectory);
                DateTimeOffset lastSave = DateTimeOffset.MinValue;
                var progress = new InlineProgress(p =>
                {
                    lock (jobSync)
                    {
                        if (record.Job.State != "running") return;
                        record.Job.Stage = p.Stage; record.Job.Progress = Math.Clamp(p.Fraction, 0, .99);
                        if (DateTimeOffset.UtcNow - lastSave > TimeSpan.FromMilliseconds(300)) { CheckDisk(); SaveJob(record); lastSave = DateTimeOffset.UtcNow; }
                    }
                });
                try
                {
                    CheckDisk();
                    if (record.Job.Kind == "render")
                    {
                        var output = await Media.RenderAsync(record.Snapshot!, a => GetSnapshotAssetPath(record.Snapshot!, a), workDirectory, progress, cancellation.Token);
                        cancellation.Token.ThrowIfCancellationRequested();
                        var target = StudioPaths.SafeChild(JobDirectory(id), "output.mp4"); File.Move(output, target, true);
                        record.Job.OutputFileName = "output.mp4";
                    }
                    else
                    {
                        var request = record.Generation!;
                        var snapshot = record.Snapshot ?? await GetProjectAsync(request.ProjectId, cancellation.Token);
                        var output = await generation!.GenerateAsync(request, workDirectory, ResolveConditioning(request, snapshot), progress, cancellation.Token);
                        cancellation.Token.ThrowIfCancellationRequested();
                        var actualPath = Path.GetFullPath(output.FilePath);
                        if (!actualPath.StartsWith(workDirectory + Path.DirectorySeparatorChar, StringComparison.Ordinal) || File.GetAttributes(actualPath).HasFlag(FileAttributes.ReparsePoint))
                            throw new StudioException("unsafe_generation_output", "Generation adapter returned a file outside its managed job directory.");
                        await using var stream = File.OpenRead(actualPath);
                        var updated = await ImportAssetInternalAsync(request.ProjectId, stream, Path.GetFileName(actualPath), new()
                        { Request = StudioJson.Clone(request), ModelVersion = output.ModelVersion, EffectiveParameters = output.EffectiveParameters }, cancellation.Token);
                        record.Job.OutputAssetId = updated.Assets.Last(a => a.Generation?.Request.RequestId == request.RequestId).Id;
                    }
                    lock (jobSync) { record.Job.State = "completed"; record.Job.Stage = "completed"; record.Job.Progress = 1; SaveJob(record); }
                }
                catch (OperationCanceledException)
                {
                    lock (jobSync)
                    {
                        record.Job.State = shutdown.IsCancellationRequested ? "interrupted" : "cancelled"; record.Job.Stage = record.Job.State;
                        record.Job.Error = shutdown.IsCancellationRequested ? "Host stopped. Retry with a new request identifier; no mid-inference resume is available." : null; SaveJob(record);
                    }
                }
                catch (Exception e)
                {
                    lock (jobSync) { record.Job.State = "failed"; record.Job.Stage = "failed"; record.Job.Error = e is OutOfMemoryException ? "Insufficient memory. Release other loaded models or explicitly choose lower settings, then retry." : e.Message; SaveJob(record); }
                }
                finally
                {
                    lock (jobSync) jobCancellations.Remove(id);
                    cancellation.Dispose();
                    try { if (Directory.Exists(workDirectory)) Directory.Delete(workDirectory, true); } catch (IOException) { }
                }
            }
        }
        catch (OperationCanceledException) when (shutdown.IsCancellationRequested) { }
    }
    async Task StopJobsAsync()
    {
        lock (jobSync) { if (stopping) return; stopping = true; queue.Writer.TryComplete(); shutdown.Cancel(); }
        await worker;
        lock (jobSync)
        {
            foreach (var record in jobs.Values.Where(r => r.Job.State == "queued")) { record.Job.State = "interrupted"; record.Job.Stage = "interrupted"; record.Job.Error = "Host stopped before this queued job started. Retry with a new request identifier."; SaveJob(record); }
        }
        shutdown.Dispose();
    }
}
