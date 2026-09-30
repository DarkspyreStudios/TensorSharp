// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;

namespace TensorSharp.GGML
{
    /// <summary>One file of a native candidate directory, as its artifact manifest lists it.</summary>
    /// <param name="Path">Relative path inside the candidate directory, with '/' separators.</param>
    /// <param name="Size">Size in bytes.</param>
    /// <param name="Sha256">SHA-256 of the file, 64 lowercase hexadecimal characters.</param>
    public sealed record GgmlNativeFile(string Path, long Size, string Sha256);

    /// <summary>
    /// A directory that holds one GgmlOps build. The caller orders candidates; the loader tries
    /// them in that order.
    /// </summary>
    /// <param name="Directory">Absolute path of the directory that holds the GgmlOps library.</param>
    /// <param name="TensorSharpBuild">The TensorSharp build the directory was built for. It must equal <see cref="GgmlNativeLoader.TensorSharpBuild"/>.</param>
    /// <param name="Rid">The runtime identifier the directory was built for. It must equal <see cref="GgmlNativeLoader.RuntimeIdentifier"/>.</param>
    /// <param name="Variant">The artifact variant, for example cpu, metal, vulkan or cuda13.</param>
    /// <param name="Backend">The GGML backend the loader initializes to accept the candidate.</param>
    /// <param name="Files">
    /// The directory's file list. When it is not empty, every listed file must exist with the listed
    /// size and SHA-256 before the library loads. When it is empty, only the library must exist.
    /// </param>
    public sealed record GgmlNativeCandidate(
        string Directory,
        string TensorSharpBuild,
        string Rid,
        string Variant,
        GgmlBackendType Backend,
        IReadOnlyList<GgmlNativeFile> Files);

    /// <summary>Refusal codes for candidate validation and loading.</summary>
    public static class GgmlNativeRefusalCodes
    {
        /// <summary>A library the candidate depends on is absent, for example a GPU driver or loader.</summary>
        public const string MissingPrerequisite = "missing-prerequisite";

        /// <summary>The candidate targets another RID or architecture, or its backend found no usable device.</summary>
        public const string IncompatibleHardware = "incompatible-hardware";

        /// <summary>The library is absent, failed to load, or failed its identity check after loading.</summary>
        public const string LoadFailed = "load-failed";

        /// <summary>The directory does not match its file list: a file is missing, a size or hash differs, or a path is unsafe.</summary>
        public const string HashMismatch = "hash-mismatch";

        /// <summary>The candidate was built for another TensorSharp build, or an earlier candidate was loaded.</summary>
        public const string NotSelected = "not-selected";
    }

    /// <summary>Why the loader skipped or rejected one candidate.</summary>
    public sealed record GgmlNativeRefusal(GgmlNativeCandidate Candidate, string Code, string Message);

    /// <summary>The build identity compiled into a GgmlOps binary by TSGgml_GetBuildIdentity.</summary>
    public sealed record GgmlNativeBuildIdentity(
        string TensorSharpBuild,
        string SourceCommit,
        string GgmlCommit,
        string Variant,
        string Rid,
        string CpuProfile,
        string Raw)
    {
        /// <summary>Parses the "key=value;..." string the native export returns.</summary>
        public static GgmlNativeBuildIdentity Parse(string raw)
        {
            ArgumentNullException.ThrowIfNull(raw);
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string pair in raw.Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                int eq = pair.IndexOf('=');
                if (eq > 0)
                    values[pair[..eq]] = pair[(eq + 1)..];
            }

            string Get(string key) => values.TryGetValue(key, out string? v) ? v : string.Empty;
            return new GgmlNativeBuildIdentity(Get("tensorsharp"), Get("source"), Get("ggml"), Get("variant"),
                Get("rid"), Get("cpu"), raw);
        }
    }

    /// <summary>The outcome of <see cref="GgmlNativeLoader.Select"/>.</summary>
    public enum GgmlNativeSelectionState
    {
        /// <summary>A candidate loaded and its backend initialized. Every GgmlOps import binds to it.</summary>
        Loaded = 1,

        /// <summary>
        /// No candidate loaded, and no candidate library is in the process. The caller may call
        /// <see cref="GgmlNativeLoader.Select"/> again in this process with other candidates.
        /// </summary>
        Unavailable = 2,

        /// <summary>
        /// A candidate library loaded into the process, then failed its identity check or backend
        /// initialization. The library stays in the process, and the loader tries no further
        /// candidate. The remaining candidates must be tried in a fresh process.
        /// </summary>
        PartiallyInitialized = 3,
    }

    /// <summary>What a selection loaded, and why it skipped each earlier candidate.</summary>
    /// <param name="State">The outcome.</param>
    /// <param name="Candidate">The loaded candidate, or the partially initialized one; null when unavailable.</param>
    /// <param name="LibraryPath">The absolute path of the library in the process; null when unavailable.</param>
    /// <param name="Backend">The initialized backend when loaded; otherwise null.</param>
    /// <param name="Identity">The build identity the loaded library reports; null when it was not read.</param>
    /// <param name="Refusals">One refusal for every candidate tried or skipped before the outcome, in candidate order.</param>
    /// <param name="Untried">Candidates after a partially initialized one, in order. Empty in every other state.</param>
    public sealed record GgmlNativeSelection(
        GgmlNativeSelectionState State,
        GgmlNativeCandidate? Candidate,
        string? LibraryPath,
        GgmlBackendType? Backend,
        GgmlNativeBuildIdentity? Identity,
        IReadOnlyList<GgmlNativeRefusal> Refusals,
        IReadOnlyList<GgmlNativeCandidate> Untried);

    /// <summary>The outcome of <see cref="GgmlNativeLoader.Shutdown"/>.</summary>
    /// <param name="Released">True when the native teardown ran, or when no GgmlOps library is in the process.</param>
    /// <param name="Diagnostic">Why the teardown did not run, or null.</param>
    public sealed record GgmlNativeShutdownResult(bool Released, string? Diagnostic);

    /// <summary>
    /// Selects the GgmlOps library a process uses from an ordered list of candidate directories.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without a call to <see cref="Select"/>, GgmlOps loads through the existing probing: the
    /// application directory, a repository build tree, and the runtime's default probing of
    /// <c>runtimes/&lt;rid&gt;/native</c> from the NuGet package.
    /// </para>
    /// <para>
    /// <see cref="Select"/> must run before the first GgmlOps call in the process. It refuses once
    /// a GgmlOps import has bound through default probing. A process loads at most one GgmlOps
    /// library. A library that loaded is never unloaded; process exit releases it.
    /// </para>
    /// <para>
    /// <see cref="TensorSharpBuild"/>, <see cref="RuntimeIdentifier"/>, <see cref="EntryLibraryName"/>
    /// and <see cref="Check"/> load no native code. Settings and discovery code can use them.
    /// </para>
    /// </remarks>
    public static class GgmlNativeLoader
    {
        private static readonly object s_gate = new();
        private static int s_resolverRegistered;
        private static IntPtr s_selectedHandle;
        private static bool s_defaultProbingBound;
        private static bool s_candidateInProcess;
        private static bool s_processExitHooked;
        private static bool s_shutDown;
        private static GgmlNativeSelection? s_current;

        /// <summary>The TensorSharp build of this assembly. Candidates must match it exactly.</summary>
        public static string TensorSharpBuild { get; } =
            typeof(GgmlNativeLoader).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
                .FirstOrDefault(a => a.Key == "TensorSharpBuild")?.Value ?? string.Empty;

        /// <summary>The runtime identifier of this process, for example osx-arm64. Candidates must match it exactly.</summary>
        public static string RuntimeIdentifier => RuntimeInformation.RuntimeIdentifier;

        /// <summary>The GgmlOps library file name on this operating system.</summary>
        public static string EntryLibraryName =>
            OperatingSystem.IsWindows() ? "GgmlOps.dll" :
            OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst() ? "libGgmlOps.dylib" :
            "libGgmlOps.so";

        /// <summary>The last selection in this process, or null when <see cref="Select"/> has not run.</summary>
        public static GgmlNativeSelection? Current
        {
            get { lock (s_gate) return s_current; }
        }

        /// <summary>
        /// Checks a candidate's identity and files without loading native code.
        /// </summary>
        /// <returns>The refusal, or null when the candidate may be loaded.</returns>
        public static GgmlNativeRefusal? Check(GgmlNativeCandidate candidate)
        {
            ArgumentNullException.ThrowIfNull(candidate);

            try
            {
                return CheckCandidate(candidate);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                return Refuse(candidate, GgmlNativeRefusalCodes.LoadFailed,
                    $"The candidate files could not be inspected: {ex.Message}");
            }
        }

        private static GgmlNativeRefusal? CheckCandidate(GgmlNativeCandidate candidate)
        {
            if (!Enum.IsDefined(candidate.Backend))
                return Refuse(candidate, GgmlNativeRefusalCodes.IncompatibleHardware, "The candidate names an unknown backend.");
            if (string.IsNullOrWhiteSpace(candidate.Variant))
                return Refuse(candidate, GgmlNativeRefusalCodes.NotSelected, "The candidate has no variant identity.");

            if (!string.Equals(candidate.Rid, RuntimeIdentifier, StringComparison.Ordinal))
            {
                return Refuse(candidate, GgmlNativeRefusalCodes.IncompatibleHardware,
                    $"The candidate targets {candidate.Rid}; this process runs {RuntimeIdentifier}.");
            }

            if (!string.Equals(candidate.TensorSharpBuild, TensorSharpBuild, StringComparison.Ordinal))
            {
                return Refuse(candidate, GgmlNativeRefusalCodes.NotSelected,
                    $"The candidate was built for TensorSharp {candidate.TensorSharpBuild}; this process runs TensorSharp {TensorSharpBuild}.");
            }

            if (string.IsNullOrEmpty(candidate.Directory) || !Path.IsPathFullyQualified(candidate.Directory))
            {
                return Refuse(candidate, GgmlNativeRefusalCodes.LoadFailed,
                    $"The candidate directory '{candidate.Directory}' is not an absolute path.");
            }

            string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(candidate.Directory));
            for (DirectoryInfo? directory = new(root); directory != null; directory = directory.Parent)
            {
                if (directory.LinkTarget != null)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"The candidate directory traverses a link at '{directory.FullName}'.");
            }

            string entryPath = Path.Combine(root, EntryLibraryName);
            IReadOnlyList<GgmlNativeFile> files = candidate.Files ?? Array.Empty<GgmlNativeFile>();
            if (files.Count == 0)
            {
                if (new FileInfo(entryPath).LinkTarget != null)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"The library '{entryPath}' is a link.");
                return File.Exists(entryPath)
                    ? null
                    : Refuse(candidate, GgmlNativeRefusalCodes.LoadFailed, $"{entryPath} does not exist.");
            }

            if (!files.Any(f => f != null && string.Equals(f.Path, EntryLibraryName, StringComparison.Ordinal)))
            {
                return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch,
                    $"The file list does not name the library {EntryLibraryName}.");
            }

            string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar) ? root : root + Path.DirectorySeparatorChar;
            var paths = new HashSet<string>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
            foreach (GgmlNativeFile file in files)
            {
                if (file == null)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, "The file list contains a null entry.");
                string? pathProblem = CheckRelativePath(file.Path);
                if (pathProblem != null)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, pathProblem);
                if (!paths.Add(file.Path))
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"The file list repeats '{file.Path}'.");
                if (file.Size < 0 || file.Sha256 == null || file.Sha256.Length != 64 ||
                    file.Sha256.Any(c => !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f')))
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"'{file.Path}' has an invalid size or SHA-256.");

                string fullPath = Path.GetFullPath(Path.Combine(root, file.Path.Replace('/', Path.DirectorySeparatorChar)));
                if (!fullPath.StartsWith(rootWithSeparator, StringComparison.Ordinal))
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"'{file.Path}' resolves outside the candidate directory.");

                for (DirectoryInfo? directory = new FileInfo(fullPath).Directory;
                     directory != null && !string.Equals(directory.FullName, root, StringComparison.Ordinal);
                     directory = directory.Parent)
                {
                    if (directory.LinkTarget != null)
                        return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"'{file.Path}' traverses a linked directory.");
                }

                var info = new FileInfo(fullPath);
                if (!info.Exists)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"'{file.Path}' is missing.");
                if (info.LinkTarget != null)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch, $"'{file.Path}' is a link.");
                if (info.Length != file.Size)
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch,
                        $"'{file.Path}' is {info.Length} bytes; the file list says {file.Size}.");

                string actual;
                using (FileStream stream = File.OpenRead(fullPath))
                    actual = Convert.ToHexStringLower(SHA256.HashData(stream));
                if (!string.Equals(actual, file.Sha256, StringComparison.Ordinal))
                    return Refuse(candidate, GgmlNativeRefusalCodes.HashMismatch,
                        $"'{file.Path}' has SHA-256 {actual}; the file list says {file.Sha256}.");
            }

            return null;
        }

        /// <summary>
        /// Loads the first candidate that passes <see cref="Check"/>, loads, reports a matching build
        /// identity and initializes its backend.
        /// </summary>
        /// <remarks>
        /// A candidate that fails before its library loads is refused, and the next one is tried. A
        /// candidate whose library loads and then fails ends the selection in
        /// <see cref="GgmlNativeSelectionState.PartiallyInitialized"/>: the library stays in the
        /// process, and <see cref="GgmlNativeSelection.Untried"/> lists the candidates that a fresh
        /// process must try next. A loaded selection registers <see cref="Shutdown"/> on process exit.
        /// </remarks>
        /// <exception cref="InvalidOperationException">
        /// A GgmlOps library is already in the process, through default probing or an earlier selection.
        /// </exception>
        public static GgmlNativeSelection Select(IReadOnlyList<GgmlNativeCandidate> candidates)
        {
            ArgumentNullException.ThrowIfNull(candidates);

            lock (s_gate)
            {
                if (s_shutDown)
                    throw new InvalidOperationException("GgmlOps was shut down in this process.");
                if (s_candidateInProcess)
                    throw new InvalidOperationException("A GgmlOps library from an earlier selection is already in this process.");
                if (s_defaultProbingBound)
                    throw new InvalidOperationException("GgmlOps already bound through default probing in this process.");

                EnsureImportResolverRegistered();

                var refusals = new List<GgmlNativeRefusal>();
                for (int i = 0; i < candidates.Count; i++)
                {
                    GgmlNativeCandidate candidate = candidates[i];
                    GgmlNativeRefusal? refusal = Check(candidate);
                    if (refusal != null)
                    {
                        refusals.Add(refusal);
                        continue;
                    }

                    string libraryPath = Path.Combine(Path.GetFullPath(candidate.Directory), EntryLibraryName);
                    if (!TryLoad(candidate, libraryPath, out IntPtr handle, out refusal))
                    {
                        refusals.Add(refusal!);
                        continue;
                    }

                    s_candidateInProcess = true;
                    IReadOnlyList<GgmlNativeCandidate> untried = candidates.Skip(i + 1).ToArray();

                    GgmlNativeBuildIdentity? identity = ReadIdentity(handle);
                    string? identityProblem = CompareIdentity(candidate, identity);
                    if (identityProblem != null)
                    {
                        refusals.Add(Refuse(candidate, GgmlNativeRefusalCodes.LoadFailed, identityProblem));
                        return Finish(new GgmlNativeSelection(GgmlNativeSelectionState.PartiallyInitialized,
                            candidate, libraryPath, null, identity, refusals, untried));
                    }

                    s_selectedHandle = handle;
                    string? backendProblem = InitializeBackend(candidate.Backend);
                    if (backendProblem != null)
                    {
                        refusals.Add(Refuse(candidate, GgmlNativeRefusalCodes.IncompatibleHardware, backendProblem));
                        return Finish(new GgmlNativeSelection(GgmlNativeSelectionState.PartiallyInitialized,
                            candidate, libraryPath, null, identity, refusals, untried));
                    }

                    foreach (GgmlNativeCandidate skipped in untried)
                        refusals.Add(Refuse(skipped, GgmlNativeRefusalCodes.NotSelected,
                            $"An earlier candidate ({candidate.Variant}) was loaded."));

                    if (!s_processExitHooked)
                    {
                        s_processExitHooked = true;
                        AppDomain.CurrentDomain.ProcessExit += (_, _) => Shutdown();
                    }

                    return Finish(new GgmlNativeSelection(GgmlNativeSelectionState.Loaded,
                        candidate, libraryPath, candidate.Backend, identity, refusals, Array.Empty<GgmlNativeCandidate>()));
                }

                return Finish(new GgmlNativeSelection(GgmlNativeSelectionState.Unavailable,
                    null, null, null, null, refusals, Array.Empty<GgmlNativeCandidate>()));
            }
        }

        /// <summary>
        /// Reads the build identity of the GgmlOps library this process uses, loading it through
        /// the selection or default probing when it is not loaded yet.
        /// </summary>
        /// <returns>The identity, or null when no library loads or it predates the identity export.</returns>
        public static GgmlNativeBuildIdentity? ReadLoadedIdentity()
        {
            try
            {
                return GgmlNative.ReadBuildIdentity() is string raw ? GgmlNativeBuildIdentity.Parse(raw) : null;
            }
            catch (DllNotFoundException)
            {
                return null;
            }
            catch (EntryPointNotFoundException)
            {
                return null;
            }
        }

        /// <summary>
        /// Runs the native teardown (TSGgml_Shutdown): it frees every backend, cached buffer and
        /// graph the library holds. Idempotent. No GgmlOps call may follow it in this process.
        /// Only process exit releases the library itself.
        /// </summary>
        public static GgmlNativeShutdownResult Shutdown()
        {
            lock (s_gate)
            {
                if (s_shutDown)
                    return new GgmlNativeShutdownResult(true, null);

                if (!s_defaultProbingBound && s_selectedHandle == IntPtr.Zero)
                {
                    if (s_candidateInProcess)
                    {
                        // A library that failed its identity check is not bound to the imports,
                        // so its teardown cannot run. Process exit releases it.
                        return new GgmlNativeShutdownResult(false,
                            "The partially initialized library is not bound; only process exit releases it.");
                    }

                    s_shutDown = true;
                    return new GgmlNativeShutdownResult(true, null);
                }

                s_shutDown = true;
            }

            try
            {
                GgmlNative.Shutdown();
                return new GgmlNativeShutdownResult(true, null);
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return new GgmlNativeShutdownResult(false, ex.Message);
            }
        }

        internal static void EnsureImportResolverRegistered()
        {
            if (Interlocked.Exchange(ref s_resolverRegistered, 1) == 0)
                NativeLibrary.SetDllImportResolver(typeof(GgmlNativeLoader).Assembly, GgmlNative.ImportResolver);
        }

        /// <summary>
        /// Called by the import resolver. Returns the selected library, or records that the imports
        /// bind through default probing.
        /// </summary>
        /// <exception cref="DllNotFoundException">
        /// A candidate library is in the process but not bound, because it failed its identity check.
        /// Binding another library would put two GgmlOps builds in one process.
        /// </exception>
        internal static bool TryGetSelectedHandle(out IntPtr handle)
        {
            lock (s_gate)
            {
                handle = s_selectedHandle;
                if (handle != IntPtr.Zero)
                    return true;
                if (s_candidateInProcess)
                    throw new DllNotFoundException(
                        "The GgmlOps selection in this process is partially initialized; a fresh process must load the next candidate.");
                s_defaultProbingBound = true;
                return false;
            }
        }

        private static GgmlNativeSelection Finish(GgmlNativeSelection selection)
        {
            s_current = selection;
            return selection;
        }

        private static GgmlNativeRefusal Refuse(GgmlNativeCandidate candidate, string code, string message) =>
            new(candidate, code, message);

        private static string? CheckRelativePath(string path)
        {
            if (string.IsNullOrEmpty(path))
                return "A file list entry has an empty path.";
            if (path.StartsWith('/') || path.Contains('\\') || path.Contains(':') || path.Any(char.IsControl) || Path.IsPathRooted(path))
                return $"'{path}' is not a relative path with '/' separators.";
            foreach (string segment in path.Split('/'))
            {
                if (segment.Length == 0 || segment == "." || segment == "..")
                    return $"'{path}' has an empty, '.' or '..' segment.";
            }

            return null;
        }

        private static bool TryLoad(GgmlNativeCandidate candidate, string libraryPath, out IntPtr handle, out GgmlNativeRefusal? refusal)
        {
            refusal = null;
            GgmlNative.EnsureWindowsNativeDependencySearchPaths();
            try
            {
                handle = NativeLibrary.Load(libraryPath);
                return true;
            }
            catch (BadImageFormatException ex)
            {
                handle = IntPtr.Zero;
                refusal = Refuse(candidate, GgmlNativeRefusalCodes.IncompatibleHardware, ex.Message);
                return false;
            }
            catch (DllNotFoundException ex)
            {
                handle = IntPtr.Zero;
                refusal = Refuse(candidate, ClassifyLoadFailure(ex.Message), ex.Message);
                return false;
            }
        }

        /// <summary>
        /// Classifies a loader error for a library that exists. A missing dependency means an
        /// absent prerequisite; a foreign architecture means incompatible hardware.
        /// </summary>
        internal static string ClassifyLoadFailure(string message)
        {
            string m = message ?? string.Empty;
            if (m.Contains("wrong ELF class", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("incompatible architecture", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("wrong architecture", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("0x800700C1", StringComparison.OrdinalIgnoreCase))
                return GgmlNativeRefusalCodes.IncompatibleHardware;
            if (m.Contains("cannot open shared object file", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("Library not loaded", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("0x8007007E", StringComparison.OrdinalIgnoreCase))
                return GgmlNativeRefusalCodes.MissingPrerequisite;
            return GgmlNativeRefusalCodes.LoadFailed;
        }

        private static unsafe GgmlNativeBuildIdentity? ReadIdentity(IntPtr handle)
        {
            if (!NativeLibrary.TryGetExport(handle, "TSGgml_GetBuildIdentity", out IntPtr export))
                return null;
            IntPtr text = ((delegate* unmanaged<IntPtr>)export)();
            string? raw = text == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(text);
            return raw == null ? null : GgmlNativeBuildIdentity.Parse(raw);
        }

        private static string? CompareIdentity(GgmlNativeCandidate candidate, GgmlNativeBuildIdentity? identity)
        {
            if (identity == null)
                return "The library has no TSGgml_GetBuildIdentity export; it predates versioned selection.";
            if (!string.Equals(identity.TensorSharpBuild, candidate.TensorSharpBuild, StringComparison.Ordinal))
                return $"The library reports TensorSharp {identity.TensorSharpBuild}; the candidate says {candidate.TensorSharpBuild}.";
            if (!string.Equals(identity.Rid, candidate.Rid, StringComparison.Ordinal))
                return $"The library reports RID '{identity.Rid}'; the candidate says {candidate.Rid}.";
            if (!string.Equals(identity.Variant, candidate.Variant, StringComparison.Ordinal))
                return $"The library reports variant '{identity.Variant}'; the candidate says {candidate.Variant}.";
            return null;
        }

        private static string? InitializeBackend(GgmlBackendType backend)
        {
            // The first call into GgmlNative runs its static constructor, which applies the
            // tunables that must precede backend creation, through the selected library.
            GgmlNative.EnsureImportResolverRegistered();
            try
            {
                GgmlNative.EnsureAvailable(backend);
                return null;
            }
            catch (Exception ex) when (ex is InvalidOperationException or PlatformNotSupportedException)
            {
                return ex.Message;
            }
        }
    }
}
