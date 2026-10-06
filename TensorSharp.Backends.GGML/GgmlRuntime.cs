#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.GGML;

/// <summary>The process owner's lifecycle. Reading it invokes no native code.</summary>
public enum GgmlRuntimeState { Unconfigured, Configured, Initializing, Ready, Unavailable, Poisoned, Stopping, Stopped }

/// <summary>The result of an explicit initialization attempt.</summary>
public enum GgmlInitializationState { Ready, Unavailable, Unsupported, RequiresProcessRestart }

/// <summary>An immutable initialization request. Null candidates require explicit default package probing.</summary>
public sealed record GgmlRuntimePlan(
    string TensorSharpBuild,
    string NativeAbi,
    string Rid,
    IReadOnlyList<GgmlNativeCandidate>? Candidates,
    bool DefaultNuGetProbing = false,
    GgmlBackendType RequestedBackend = GgmlBackendType.Cpu);

/// <summary>The requested plan, actual selection and structured outcome.</summary>
public sealed record GgmlInitializationResult(
    GgmlRuntimePlan Plan,
    GgmlInitializationState State,
    GgmlNativeSelection Selection,
    string? Diagnostic)
{
    public GgmlBackendType? ActualBackend => Selection.Backend;
    public string? ActualVariant => Selection.Identity?.Variant;
    public string? ActualBuild => Selection.Identity?.TensorSharpBuild;
    public string? ActualRid => Selection.Identity?.Rid;
    public string? ActualNativeAbi => Selection.Identity?.NativeAbi;
}

/// <summary>A managed resource that retains the process runtime.</summary>
public enum GgmlRuntimeResourceKind { Context, Tensor, Model }

public static partial class GgmlNativeLoader
{
    private const string ProcessOwnerKey = "Darkspyre.TensorSharp.GGML.ProcessOwner";
    private static readonly Dictionary<long, ResourceRecord> s_resources = new();
    private static readonly Dictionary<(string Kind, IntPtr Handle), NativeResource> s_nativeResources = new();
    private static readonly Dictionary<long, NativeCallLease> s_calls = new();
    private static readonly GgmlProcessOwner s_processOwnerToken = new();
    private static long s_nextLease;
    private static GgmlRuntimeState s_runtimeState;
    private static GgmlRuntimePlan? s_plan;
    private static Task<GgmlInitializationResult>? s_initialization;
    private static int s_initializationThread;
    private static int s_teardownThread;
    private static bool s_implicitPlan;
    private static bool s_nativeUseStarted;
    private static bool s_otherProcessOwner;
    private static GgmlBackendType? s_actualBackend;
    private static GgmlNativeBuildIdentity? s_defaultIdentity;
    private static string? s_defaultLibraryPath;
    private static GgmlNativeShutdownResult? s_shutdownResult;

    /// <summary>The lifecycle state. This accessor never loads native code.</summary>
    public static GgmlRuntimeState State { get { lock (s_gate) return s_runtimeState; } }

    /// <summary>Snapshots a plan without loading native code. A used runtime cannot be reconfigured.</summary>
    public static void Configure(GgmlRuntimePlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (string.IsNullOrWhiteSpace(plan.TensorSharpBuild) || string.IsNullOrWhiteSpace(plan.Rid) ||
            plan.NativeAbi == null || plan.NativeAbi.Length != 64 ||
            plan.NativeAbi.Any(c => !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f')) ||
            !Enum.IsDefined(plan.RequestedBackend))
            throw new ArgumentException("A plan requires an exact build/ABI identity, RID and supported backend value.", nameof(plan));
        if (plan.DefaultNuGetProbing != (plan.Candidates == null))
            throw new ArgumentException("A plan specifies ordered candidates or explicit default package probing, never both.", nameof(plan));
        IReadOnlyList<GgmlNativeCandidate>? candidates = plan.Candidates == null ? null : Array.AsReadOnly(plan.Candidates.Select(candidate =>
        {
            ArgumentNullException.ThrowIfNull(candidate);
            if (!System.IO.Path.IsPathFullyQualified(candidate.Directory))
                throw new ArgumentException("Every candidate directory is absolute.", nameof(plan));
            return candidate with { Files = Array.AsReadOnly((candidate.Files ?? Array.Empty<GgmlNativeFile>()).ToArray()) };
        }).ToArray());
        lock (s_gate)
        {
            CheckProcessDependency();
            if (s_runtimeState is GgmlRuntimeState.Stopping or GgmlRuntimeState.Stopped)
                throw new InvalidOperationException("A terminal runtime cannot be configured again.");
            GgmlRuntimePlan snapshot = plan with { Candidates = candidates };
            if (SamePlan(s_plan, snapshot) && s_runtimeState != GgmlRuntimeState.Unavailable)
                return;
            if (s_runtimeState is GgmlRuntimeState.Initializing or GgmlRuntimeState.Stopping or GgmlRuntimeState.Stopped or GgmlRuntimeState.Poisoned ||
                s_nativeUseStarted || s_resources.Count != 0 || s_calls.Count != 0)
                throw new InvalidOperationException("A configured runtime cannot change after native use, while owned resources exist, or after terminal shutdown.");
            s_plan = snapshot;
            s_implicitPlan = false;
            s_initialization = null;
            s_runtimeState = GgmlRuntimeState.Configured;
        }
    }

    private static bool SamePlan(GgmlRuntimePlan? left, GgmlRuntimePlan right)
    {
        if (left == null || left.TensorSharpBuild != right.TensorSharpBuild || left.NativeAbi != right.NativeAbi ||
            left.Rid != right.Rid || left.DefaultNuGetProbing != right.DefaultNuGetProbing || left.RequestedBackend != right.RequestedBackend)
            return false;
        if (left.Candidates == null || right.Candidates == null) return left.Candidates == null && right.Candidates == null;
        return left.Candidates.Count == right.Candidates.Count && left.Candidates.Zip(right.Candidates).All(pair =>
            pair.First.Directory == pair.Second.Directory && pair.First.TensorSharpBuild == pair.Second.TensorSharpBuild &&
            pair.First.NativeAbi == pair.Second.NativeAbi && pair.First.Rid == pair.Second.Rid && pair.First.Variant == pair.Second.Variant &&
            pair.First.Backend == pair.Second.Backend && pair.First.Files.SequenceEqual(pair.Second.Files));
    }

    /// <summary>Shares one initialization of the configured plan. Cancellation stops only this caller's wait.</summary>
    public static Task<GgmlInitializationResult> InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Task<GgmlInitializationResult> initialization;
        lock (s_gate)
        {
            CheckProcessDependency();
            if (s_runtimeState is GgmlRuntimeState.Stopping or GgmlRuntimeState.Stopped)
                throw new InvalidOperationException("Initialization cannot follow terminal shutdown.");
            if (s_plan == null)
                throw new InvalidOperationException("Configure the runtime before explicit initialization.");
            if (s_runtimeState == GgmlRuntimeState.Poisoned && s_initialization is not { IsCompletedSuccessfully: true })
                return Task.FromResult(new GgmlInitializationResult(s_plan, GgmlInitializationState.RequiresProcessRestart,
                    s_current ?? EmptySelection(), "The native runtime is poisoned; a fresh process is required."));
            if (s_runtimeState == GgmlRuntimeState.Poisoned && s_initialization!.Result.State != GgmlInitializationState.RequiresProcessRestart)
                s_initialization = Task.FromResult(new GgmlInitializationResult(s_plan, GgmlInitializationState.RequiresProcessRestart,
                    s_current ?? EmptySelection(), "The native runtime is poisoned; a fresh process is required."));
            if (s_initialization == null)
            {
                if (s_runtimeState == GgmlRuntimeState.Ready && s_current != null)
                    return s_initialization = Task.FromResult(new GgmlInitializationResult(s_plan, GgmlInitializationState.Ready, s_current, null));
                if (s_runtimeState == GgmlRuntimeState.Poisoned)
                    throw new InvalidOperationException("Native calls already used this runtime outside explicit initialization.");
                s_runtimeState = GgmlRuntimeState.Initializing;
                GgmlRuntimePlan plan = s_plan;
                s_initialization = Task.Run(() => InitializeOwned(plan));
            }
            initialization = s_initialization;
        }
        return PublishInitializationAsync(initialization, cancellationToken);
    }

    private static async Task<GgmlInitializationResult> PublishInitializationAsync(Task<GgmlInitializationResult> initialization, CancellationToken token)
    {
        var result = await initialization.WaitAsync(token).ConfigureAwait(false);
        token.ThrowIfCancellationRequested();
        CheckProcessDependency();
        return result;
    }

    /// <summary>Retains the runtime for a managed resource. Native imports also acquire mandatory call leases.</summary>
    public static IDisposable AcquireLease(GgmlRuntimeResourceKind kind)
    {
        if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        lock (s_gate)
        {
            EnsureOperational();
            long id = ++s_nextLease;
            CheckCleanupAdmission();
            s_resources.Add(id, new ResourceRecord(kind));
            return new ResourceLease(id);
        }
    }

    public static GgmlNativeSelection Select(IReadOnlyList<GgmlNativeCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        lock (s_gate)
            if (s_candidateInProcess || s_defaultProbingBound)
                throw new InvalidOperationException("A GgmlOps library from an earlier selection or default binding is already in this process.");
        Configure(new(TensorSharpBuild, NativeAbi, RuntimeIdentifier, candidates, RequestedBackend: candidates.FirstOrDefault()?.Backend ?? GgmlBackendType.Cpu));
        return InitializeAsync().GetAwaiter().GetResult().Selection;
    }

    private static GgmlInitializationResult InitializeOwned(GgmlRuntimePlan plan)
    {
        lock (s_gate) s_initializationThread = Environment.CurrentManagedThreadId;
        try
        {
            if (plan.Rid != RuntimeIdentifier || plan.TensorSharpBuild != TensorSharpBuild || plan.NativeAbi != NativeAbi)
                return CompleteInitialization(plan, EmptySelection(), GgmlInitializationState.Unsupported, "The plan does not match this process RID or managed build/ABI.");
            if (plan.RequestedBackend == GgmlBackendType.Cuda) EstablishCudaDependency();
            GgmlNativeSelection selection = plan.DefaultNuGetProbing ? InitializeDefault(plan) : SelectCore(plan.Candidates!);
            return CompleteInitialization(plan, selection, selection.State switch
            {
                GgmlNativeSelectionState.Loaded => GgmlInitializationState.Ready,
                GgmlNativeSelectionState.PartiallyInitialized => GgmlInitializationState.RequiresProcessRestart,
                _ => GgmlInitializationState.Unavailable,
            }, null);
        }
        catch (Exception error) when (error is not NativeRuntimeQuarantinedException && (error is InvalidOperationException or PlatformNotSupportedException or System.IO.IOException or
            System.UnauthorizedAccessException or System.Runtime.InteropServices.ExternalException or TypeInitializationException or
            DllNotFoundException or EntryPointNotFoundException or BadImageFormatException))
        {
            bool retained = s_nativeUseStarted || s_otherProcessOwner;
            GgmlNativeSelection selection = s_current ?? EmptySelection();
            return CompleteInitialization(plan, selection, retained ? GgmlInitializationState.RequiresProcessRestart :
                error is PlatformNotSupportedException ? GgmlInitializationState.Unsupported : GgmlInitializationState.Unavailable, error.Message);
        }
        finally
        {
            lock (s_gate)
            {
                s_initializationThread = 0;
                ReleaseUnusedProcessClaim();
            }
        }
    }

    private static GgmlInitializationResult CompleteInitialization(GgmlRuntimePlan plan, GgmlNativeSelection selection, GgmlInitializationState state, string? diagnostic)
    {
        lock (s_gate)
        {
            s_runtimeState = state switch
            {
                GgmlInitializationState.Ready => GgmlRuntimeState.Ready,
                GgmlInitializationState.RequiresProcessRestart => GgmlRuntimeState.Poisoned,
                _ => GgmlRuntimeState.Unavailable,
            };
            s_actualBackend = selection.Backend;
            CheckProcessDependency();
            return new(plan, state, selection, diagnostic);
        }
    }

    private static GgmlNativeSelection EmptySelection() => new(GgmlNativeSelectionState.Unavailable,
        null, null, null, null, Array.Empty<GgmlNativeRefusal>(), Array.Empty<GgmlNativeCandidate>());

    private static void EnsureOperational()
    {
        CheckProcessDependency();
        if (s_runtimeState is GgmlRuntimeState.Poisoned or GgmlRuntimeState.Stopping or GgmlRuntimeState.Stopped)
            throw new InvalidOperationException("The native runtime is poisoned, stopping or stopped; a fresh process is required.");
        if (s_runtimeState == GgmlRuntimeState.Initializing && s_initializationThread != Environment.CurrentManagedThreadId)
            throw new InvalidOperationException("The native runtime is still initializing.");
        if (s_plan == null)
        {
            s_plan = new(TensorSharpBuild, NativeAbi, RuntimeIdentifier, null, DefaultNuGetProbing: true);
            s_implicitPlan = true;
            s_runtimeState = GgmlRuntimeState.Configured;
        }
        if (!s_implicitPlan && s_runtimeState is GgmlRuntimeState.Configured or GgmlRuntimeState.Unavailable)
            throw new InvalidOperationException("Initialize the configured candidate plan before native calls or resource creation.");
    }

    private sealed class ResourceLease(long id) : IDisposable
    {
        public void Dispose()
        {
            lock (s_gate)
            {
                if (s_resources.TryGetValue(id, out ResourceRecord? record) && record.CleanupFailure == null)
                    s_resources.Remove(id);
                ReleaseUnusedProcessClaim();
            }
        }
    }

    internal static NativeReleaseLease BeginNativeHandleRelease(string kind, IntPtr handle)
    {
        lock (s_gate)
        {
            if (handle != IntPtr.Zero)
            {
                if (!s_nativeResources.TryGetValue((kind, handle), out NativeResource? resource))
                    throw new InvalidOperationException("A native handle is no longer owned.");
                if (resource.CleanupFailure != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(resource.CleanupFailure).Throw();
                if (resource.Releasing || resource.Calls != 0)
                    throw new InvalidOperationException("A native handle cannot be released while in use, already releasing or no longer owned.");
                resource.Releasing = true;
            }
            return new(kind, handle);
        }
    }

    internal readonly struct NativeReleaseLease(string kind, IntPtr handle) : IDisposable
    {
        internal void MarkAttempt()
        {
            lock (s_gate)
                if (s_nativeResources.TryGetValue((kind, handle), out var resource)) resource.State = AcquisitionState.ReleaseAttempted;
        }
        internal void RetainFailure(Exception error)
        {
            lock (s_gate)
                if (s_nativeResources.TryGetValue((kind, handle), out var resource) && resource.State == AcquisitionState.ReleaseAttempted)
                    resource.CleanupFailure ??= error;
        }
        public void Complete()
        {
            NativeResource? resource;
            lock (s_gate) s_nativeResources.TryGetValue((kind, handle), out resource);
            resource?.CompleteRelease();
            lock (s_gate) s_nativeResources.Remove((kind, handle));
        }
        public void MarkReturned()
        {
            lock (s_gate)
                if (s_nativeResources.TryGetValue((kind, handle), out var resource)) resource.ReleaseReturned = true;
        }
        public void Dispose()
        {
            lock (s_gate)
                if (s_nativeResources.TryGetValue((kind, handle), out NativeResource? resource) && resource.CleanupFailure == null)
                    resource.Releasing = false;
        }
    }

    internal static void ReleaseNativeHandle(string kind, IntPtr handle, Action release)
    {
        using var resource = BeginNativeHandleRelease(kind, handle);
        try
        {
            using var call = EnterNativeCall();
            resource.MarkAttempt();
            release();
            resource.MarkReturned();
            resource.Complete();
        }
        catch (Exception error) { resource.RetainFailure(error); throw; }
    }

    internal static void ValidateRequestedBackend(GgmlBackendType backend)
    {
        lock (s_gate)
        {
            EnsureOperational();
            if (s_actualBackend.HasValue && s_actualBackend != backend)
                throw new InvalidOperationException("The process cannot change its initialized GGML backend.");
            if (!s_implicitPlan && s_plan!.RequestedBackend != backend)
                throw new InvalidOperationException("The backend does not match the configured runtime plan.");
            if (s_implicitPlan && s_actualBackend == null)
                s_plan = s_plan! with { RequestedBackend = backend };
        }
    }

    internal static string PrepareModelBackend(string backendName)
    {
        GgmlBackendType backend;
        lock (s_gate)
        {
            EnsureOperational();
            backend = backendName?.ToLowerInvariant() switch
            {
                null or "" => s_plan!.RequestedBackend,
                "cpu" => GgmlBackendType.Cpu,
                "metal" => GgmlBackendType.Metal,
                "cuda" => GgmlBackendType.Cuda,
                "vulkan" => GgmlBackendType.Vulkan,
                _ => throw new ArgumentException("A whole-model native backend must be cpu, metal, cuda or vulkan.", nameof(backendName)),
            };
        }
        GgmlNative.EnsureAvailable(backend);
        return backend.ToString().ToLowerInvariant();
    }

    private static GgmlNativeSelection InitializeDefault(GgmlRuntimePlan plan)
    {
        GgmlNative.EnsureAvailable(plan.RequestedBackend);
        return Finish(new(GgmlNativeSelectionState.Loaded, null, s_defaultLibraryPath, plan.RequestedBackend, s_defaultIdentity,
            Array.Empty<GgmlNativeRefusal>(), Array.Empty<GgmlNativeCandidate>()));
    }

    internal static IntPtr ResolveDefault(System.Reflection.Assembly assembly)
    {
        lock (s_gate)
        {
            if (!(s_runtimeState == GgmlRuntimeState.Stopping && s_teardownThread == Environment.CurrentManagedThreadId))
                EnsureOperational();
            if (TryGetSelectedHandle(out IntPtr selected)) return selected;
            if (!s_plan!.DefaultNuGetProbing) throw new DllNotFoundException("The configured candidate plan has not selected a library; ambient probing is disabled.");
            string[] paths = GgmlNative.GetCandidatePaths(assembly).Where(System.IO.File.Exists)
                .Select(System.IO.Path.GetFullPath).Distinct(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal).ToArray();
            if (paths.Length > 1) throw new InvalidOperationException("Default package probing found competing native bridges; configure one explicit candidate plan.");
            ClaimProcess();
            IntPtr handle;
            if (OperatingSystem.IsIOS() || OperatingSystem.IsTvOS())
                handle = System.Runtime.InteropServices.NativeLibrary.GetMainProgramHandle();
            else if (paths.Length == 1) handle = GgmlLibraryLoader.Load(paths[0]);
            else throw new DllNotFoundException("No native bridge exists in the package's absolute runtime paths.");
            s_nativeUseStarted = true;
            s_candidateInProcess = true;
            s_processOwnerToken.Library = handle;
            GgmlNativeBuildIdentity? identity = ReadIdentity(handle, admitted: true);
            s_processOwnerToken.Identity = identity;
            if (identity == null || identity.NativeAbi != NativeAbi || identity.GgmlCommit != GgmlCommit ||
                identity.TensorSharpBuild != TensorSharpBuild || identity.Rid != RuntimeIdentifier)
            {
                s_runtimeState = GgmlRuntimeState.Poisoned;
                throw new InvalidOperationException("The loaded default bridge does not match the exact managed ABI, source, version and RID; a fresh process is required.");
            }
            s_defaultIdentity = identity;
            s_defaultLibraryPath = paths.FirstOrDefault();
            s_defaultProbingBound = true;
            s_selectedHandle = handle;
            RegisterProcessExitHook();
            return handle;
        }
    }

    internal static void PoisonAfterBackendFailure()
    {
        lock (s_gate)
            if (s_nativeUseStarted) s_runtimeState = GgmlRuntimeState.Poisoned;
    }

    internal static bool RecreateOwnedBackend(Func<bool> recreate)
    {
        lock (s_gate)
        {
            EnsureOperational();
            if (s_resources.Count != 0 || s_nativeResources.Count != 0 || s_calls.Count != 0)
                throw new InvalidOperationException("Backend recreation cannot invalidate active contexts, tensors, models, native handles or calls.");
        }
        using var reservation = ReserveResourceCleanup(s_processOwnerToken);
        try
        {
            bool result = recreate();
            CheckProcessDependency();
            reservation.Complete();
            return result;
        }
        catch (NativeRuntimeQuarantinedException) { throw; }
        catch { PoisonAfterBackendFailure(); throw; }
    }

    internal static T WithResourceCleanup<T>(Func<T> cleanup)
    {
        using var reservation = ReserveResourceCleanup(s_processOwnerToken);
        T result = cleanup();
        reservation.Complete();
        return result;
    }

    internal static void RecordBackendInitialized(GgmlBackendType backend)
    {
        lock (s_gate)
        {
            s_actualBackend = backend;
            CheckProcessDependency();
            if (s_runtimeState != GgmlRuntimeState.Initializing)
            {
                s_runtimeState = GgmlRuntimeState.Ready;
                s_current = new(GgmlNativeSelectionState.Loaded, null, s_defaultLibraryPath, backend, s_defaultIdentity,
                    Array.Empty<GgmlNativeRefusal>(), Array.Empty<GgmlNativeCandidate>());
            }
        }
    }

    private static void ClaimProcess()
    {
        lock (AppDomain.CurrentDomain)
        {
            object? owner = AppDomain.CurrentDomain.GetData(ProcessOwnerKey);
            if (owner == null) AppDomain.CurrentDomain.SetData(ProcessOwnerKey, s_processOwnerToken);
            else if (!ReferenceEquals(owner, s_processOwnerToken))
            {
                s_otherProcessOwner = true;
                s_runtimeState = GgmlRuntimeState.Poisoned;
                throw new InvalidOperationException("Another managed GGML runtime owns this process; a fresh process is required.");
            }
        }
    }

    private static void ReleaseUnusedProcessClaim()
    {
        if (s_nativeUseStarted || s_resources.Count != 0 || s_calls.Count != 0 || s_initializationThread != 0) return;
        lock (AppDomain.CurrentDomain)
            if (ReferenceEquals(AppDomain.CurrentDomain.GetData(ProcessOwnerKey), s_processOwnerToken))
                AppDomain.CurrentDomain.SetData(ProcessOwnerKey, null);
    }

    private static void RegisterProcessExitHook()
    {
        lock (s_gate)
        {
            if (s_processExitHooked) return;
            AppDomain.CurrentDomain.ProcessExit += OnProcessExit;
            s_processExitHooked = true;
        }
    }

    private static void OnProcessExit(object? sender, EventArgs args) => Shutdown();

    private static void RemoveProcessExitHook()
    {
        if (!s_processExitHooked) return;
        AppDomain.CurrentDomain.ProcessExit -= OnProcessExit;
        s_processExitHooked = false;
    }

    private static GgmlNativeShutdownResult ShutdownOwned()
    {
        lock (s_gate)
        {
            if (s_shutdownResult != null) return s_shutdownResult;
            if (s_runtimeState == GgmlRuntimeState.Initializing || s_initialization is { IsCompleted: false } ||
                s_calls.Count != 0 || s_resources.Count != 0 || s_nativeResources.Count != 0)
                return new(false, $"The runtime has active initialization, {s_calls.Count} native calls, {s_resources.Count} managed leases and {s_nativeResources.Count} native handles.");
            if (s_runtimeState == GgmlRuntimeState.Poisoned || s_otherProcessOwner)
                return new(false, "The runtime is poisoned or belongs to another owner; a fresh process is required.");
        }
        ResourceCleanupReservation? reservation = null;
        bool teardownEntered = false;
        try
        {
            reservation = ReserveResourceCleanup(s_processOwnerToken);
            lock (s_gate)
            {
                if (s_resources.Count != 0 || s_nativeResources.Count != 0)
                    return new(false, "The runtime acquired resources before shutdown admission.");
                s_runtimeState = GgmlRuntimeState.Stopping;
                s_teardownThread = Environment.CurrentManagedThreadId;
                teardownEntered = true;
            }
            if (s_selectedHandle != IntPtr.Zero) GgmlNative.ShutdownCore();
            if (s_processOwnerToken.CudaDependency is { } dependency)
            {
                using var effect = dependency.EnterEffect();
                effect.CompleteSafeRelease(s_processOwnerToken);
                s_processOwnerToken.CudaDependency = null;
            }
            reservation.Complete();
            lock (s_gate)
            {
                RemoveProcessExitHook();
                // Retain the terminal process claim without rooting the retired assembly.
                lock (AppDomain.CurrentDomain)
                    if (ReferenceEquals(AppDomain.CurrentDomain.GetData(ProcessOwnerKey), s_processOwnerToken))
                        AppDomain.CurrentDomain.SetData(ProcessOwnerKey, new object());
                s_shutDown = true;
                s_runtimeState = GgmlRuntimeState.Stopped;
                return s_shutdownResult = new(true, null);
            }
        }
        catch (Exception error) when (error is InvalidOperationException or System.Runtime.InteropServices.ExternalException or TypeInitializationException)
        {
            lock (s_gate)
            {
                if (error is NativeRuntimeQuarantinedException)
                {
                    s_processOwnerToken.ShutdownFailure ??= System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error);
                    return s_shutdownResult = new(false, error.Message);
                }
                if (reservation == null) return new(false, error.Message);
                s_processOwnerToken.ShutdownFailure ??= System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error);
                s_runtimeState = GgmlRuntimeState.Poisoned;
                return s_shutdownResult = new(false, error.Message);
            }
        }
        finally
        {
            if (teardownEntered) { lock (s_gate) s_teardownThread = 0; }
            reservation?.Dispose();
        }
    }
}
