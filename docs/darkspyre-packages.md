# Darkspyre package identity

The `darkspyre` branch publishes fork-owned packages with a `Darkspyre.` prefix.
Assemblies, namespaces, and project names retain their upstream TensorSharp
identity so upstream changes can be reapplied without source-level renaming.

The fork is based on upstream `main` (`v2026.09.01-41-gcff7ea39`). It adds:

- public direct inference primitives;
- the restricted PyTorch ZIP state-dictionary reader, for source checkpoints that have no GGUF or
  safetensors publication;
- bounded metadata inspection for GGUF, safetensors and restricted PyTorch files;
- GGUF shard validation: shard numbering, counts and duplicate tensor names;
- the documented four-dimensional channels-last output shape from direct image group
  normalization.

Model files load from file paths through the upstream path APIs.

`Darkspyre.TensorSharp.Backends.GGML` contains managed code only. A consumer installs a separate
`Darkspyre.TensorSharp.Backends.GGML.Native.<rid>` baseline package for default native probing.
The native packaging tool selects these baseline variants from its staged input:

| Runtime | File | Backends | Toolchain |
|---|---|---|---|
| `osx-arm64` | `libGgmlOps.dylib` | CPU, Metal | `build-macos.sh` |
| `linux-x64` | `libGgmlOps.so` | CPU | CMake, Ubuntu 24.04, gcc 13 |
| `linux-arm64` | `libGgmlOps.so` | CPU | CMake, Ubuntu 24.04, gcc 13 |
| `win-x64` | `GgmlOps.dll` | CPU | CMake, Ninja, MSVC 14.51 |
| `win-arm64` | `GgmlOps.dll` | CPU | CMake, Ninja, clang-cl 23.1.2; OpenMP off |

Each baseline package places its bridge at `runtimes/<rid>/native/`. Optional variant packages
place their files under `ggml/<variant>/` and copy them into that separate output directory.
The artifact manifest records each binary's actual identity, dependencies and notices. A package
layout or successful cross-build does not establish runtime qualification on the target device.

The managed GGML project excludes native binaries from its package. `eng/pack-ggml-natives.py`
packages staged bridges into separate per-RID native packages and variant archives. Ordinary source
builds still build and copy the platform bridge unless `TensorSharpSkipGgmlNative=true` is set.

The packer validates all staged inputs before writing release output. It accepts the five baseline
RIDs above, Vulkan and CUDA13 on Linux and Windows x64, and Vulkan on Windows ARM64. It rejects
unknown pairs, links and special files, nonportable paths, missing licenses, unresolved or uninspected
dependencies, wrong binary architecture, and a bridge without the exact `TSGgml_GetBuildIdentity`
export. Binary identity must match the source, ggml, RID, variant and version in the build record.
Inspection-tool failures do not count as empty dependency lists. A supplied managed package must
have the release's exact identity/version and no native payload.

`--validate-only` runs those checks without creating an output directory, package, archive or
manifest. Validation does not require a fetched ggml checkout. Staged inputs remain caller-owned
and must stay immutable through packaging. Existing output is not removed on validation failure.

```sh
python3 -B eng/pack-ggml-natives.py --stage artifacts/ggml-natives/2.8.6.8 --validate-only
python3 -B -m unittest discover -s eng/tests -p test_pack_ggml_natives.py
```

The packaging tests inspect actual native headers and exports from a tiny compiled identity
fixture, ordinary files and symbolic links. The fixture contains no GGML backend. These tests do
not qualify model loading, native lifecycle or accelerator execution.

## Native candidate validation

`GgmlNativeLoader.Check` inspects a candidate without loading native code. It checks the managed
package build, exact bridge source/interop identity, current process RID, declared backend, variant,
absolute directory and bridge file.
A supplied file list must name the bridge. Every entry has a unique relative path, a nonnegative
size and a lowercase SHA-256 digest that matches the file. Paths cannot contain traversal segments,
control characters or alternate separators. Candidate directories, their ancestors, listed files
and intermediate directories cannot be symbolic links. Filesystem inspection failures return a
structured refusal. The caller owns keeping the validated directory immutable through loading;
validation does not lock the filesystem against another writer.

The managed assembly's `GgmlNativeAbi` metadata and the bridge's `TSGgml_GetBuildIdentity` export
carry the same SHA-256 identity. CMake and MSBuild compute it from sorted bridge implementation
files, every top-level managed GGML backend source file and the pinned ggml revision. The broader
managed input set includes interop declarations and their type/owner dependencies regardless of
filename. CRLF line endings normalize
to LF. The identity does not use the package display version or Git commit. Explicit selection
rejects a missing or different ABI identity, malformed or repeated identity fields, and unknown
native/upstream source commits. The reported ggml commit must match `GgmlUpstreamCommit` in the
managed assembly. A loaded refusal retains its library handle and requires a new
process; it never falls through to another library. Resolver registration and static construction
load no native code. First import binding applies the native environment tunables after selection.

## Native runtime ownership

The existing `GgmlNativeLoader` owns configuration, loading, initialization and terminal teardown.
`Configure(GgmlRuntimePlan)` snapshots its ordered candidate and file lists without loading native
code. `InitializeAsync(CancellationToken)` shares one initialization task for the same plan;
cancellation cancels that caller's wait, not the process initialization. Results distinguish
`Ready`, `Unavailable`, `Unsupported` and `RequiresProcessRestart`, and retain actual selection,
identity and refusal details. Repeating an identical configuration is allowed; changing a used
runtime, or configuring after shutdown, is not. No-load unavailability permits a retry.

Null candidates require `DefaultNuGetProbing: true`; an explicit empty list never probes ambient
libraries. Default probing examines absolute application, assembly, package runtime and nearest
repository build paths, refuses competing bridges, and checks the loaded exact identity. Windows
uses `LoadLibraryExW` with the chosen library directory and System32 only. It does not mutate PATH,
search CUDA toolkits, or use the working directory. Windows dependency closure still requires
real target qualification; static loading flags do not prove a driver package works.

Every bridge import acquires a mandatory native-call lease. Model/vision/embedding handles,
paged-KV pools, MTP handles, state snapshots and aligned allocations are retained until their
guarded release; owned handle calls cannot race their release. GGML contexts and tensor storages
retain managed resource leases. Models retain their own lease and release contexts they create;
externally supplied contexts remain borrowed. `GgmlContext` is disposable and refuses disposal
while tensor storages remain; disposal drains pending compute and invalidates pooled host buffers
before freeing memory. An abandoned context retains its lease if safe synchronization fails.

`Shutdown()` refuses active initialization, calls, contexts, tensors, models or native handles.
Every public shutdown/recreation path uses that guard. Success is terminal and idempotent; cached
native imports cannot execute afterwards. The loaded library is never unloaded. A process-wide
BCL-only ownership token prevents another managed load context from loading a second GGML build
without pinning a foreign collectible assembly. `AcquireLease(GgmlRuntimeResourceKind)` lets
adapters retain additional resources; it does not replace the mandatory supplier leases.

The focused loader tests use ordinary files, symbolic links and isolated collectible managed load
contexts. They do not load a bridge or qualify a CPU or accelerator backend:

```sh
dotnet test eng/tests/ggml-native-loader/ggml-native-loader.csproj -p:TensorSharpSkipGgmlNative=true
```

`eng/tests/ggml-native-runtime` runs each selection scenario in a separate process against a real
bridge directory. `selected` verifies pre-load hash/ABI refusals, retry, exact loaded identity,
cross-class import binding, shared initialization, cancellation, immutable plans, CPU tensor
arithmetic, active-resource/call shutdown refusal, double-free refusal, foreign-owner refusal,
collectible-context release and terminal teardown. `default` verifies absolute package-path probing;
`ambiguous-default` verifies no-load ambiguity refusal followed by a corrected default plan.
`reject-variant` verifies that a loaded identity mismatch leaves later candidates untried and
rejects a second selection. `reject-legacy` verifies the same refusal for a bridge without an
exact ABI identity. The probes do not qualify model generation, real whole-model handles, GPU
execution, Windows dependency loading or other RIDs, nor prove a published package's layout.

```sh
dotnet build eng/tests/ggml-native-runtime/ggml-native-runtime.csproj -c Release -p:TensorSharpSkipGgmlNative=true
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll selected /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll default /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll ambiguous-default /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll reject-variant /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll reject-legacy /absolute/legacy/bridge/directory metal
```

`dotnet run --project eng/guard-ggml-interop/guard-ggml-interop.csproj -- --verify .` checks every
interop declaration for private raw binding, mandatory call guards and owned handle-family
acquisition/use/release coverage. New unclassified native handles fail verification.

`eng/build-ggml-natives.sh` refuses uncommitted native, managed ABI and build inputs. Its Linux
CUDA release profile includes SASS 75/80/86/89/120 and PTX 120 on both x64 and ARM64. Local GPU
detection and extra CMake arguments cannot reduce that profile. The builder checks the actual
CMake cache before staging. A toolkit that cannot compile the profile fails the build.
`test_build_ggml_natives.py` verifies these command/provenance gates with mock build commands;
it does not qualify CUDA binaries or devices.

Stable fork releases use a fourth numeric version component. Each published package version is
immutable; a later compatible fork release increments the fourth component.

The model package and its direct dependency closure are:

| Package | Assembly |
|---|---|
| `Darkspyre.TensorSharp.Models` | `TensorSharp.Models` |
| `Darkspyre.TensorSharp.Runtime` | `TensorSharp.Runtime` |
| `Darkspyre.TensorSharp.Tensors` | `TensorSharp.Core` |
| `Darkspyre.TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` |
| `Darkspyre.TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` |
| `Darkspyre.TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` |

Other packable projects follow the same `Darkspyre.` plus project-name rule.
Packages are published to the private Darkspyre GitHub Packages feed. The
`darkspyre` branch is the long-lived integration branch; upstream synchronization
is performed onto that branch while preserving the focused fork commits.

## Bounded metadata inspection

`GgufFile.InspectMetadata`, `SafetensorsFile.InspectMetadata` and
`TorchStateDictionaryFile.InspectMetadata` reuse the corresponding loading parsers over a
caller-owned seekable stream. They return metadata only. They never open sibling files, map
weights or allocate native tensors. The stream reports the full artifact length, so a caller can
serve bounded remote ranges without downloading the artifact. Inspection leaves the stream open and
defaults to a 16 MiB total-read budget, including ZIP seeks. Restricted pickle interpretation keeps
its whitelist; no repository code is executed.

GGUF strings, arrays and counts, and safetensors header allocations, are checked against the
remaining bytes before allocation. Duplicate tensor names, invalid GGUF rank or alignment, negative
safetensors dimensions and overflowing safetensors shape sizes are rejected. Ordinary GGUF and
safetensors file parsing uses the same bounded parser with a 64 MiB header budget. The API exposes
metadata, not a remotely backed model instance.

## Chat stop sequences

The autoregressive chat pipeline honors `SamplingConfig.StopSequences` before publishing decoded
text. It withholds the longest suffix that can become a stop string in the next chunk. A completed
match ends generation and excludes the stop string and the rest of its chunk. The first match in
the decoded text wins, independently of the configured list order. An unmatched prefix is flushed
on natural completion or cancellation. The pending buffer is shorter than the longest stop string;
it does not accumulate the whole response. Empty or null stop strings fail before execution.

Stop matching snapshots the configured list. Mutating that list during a request does not change
its stop policy. The full-text `TokenSampler.CheckStopSequences` uses the same first-match rule.
The transcript retains raw generated tokens separately from the emitted text, including tokens
hidden by a stop or forwarded before an abort settles. Continuation uses that raw-token boundary;
it does not add hidden stop text to the public assistant content.

The deterministic stop tests drive the managed scheduling engine with a scripted model/tokenizer.
They do not load weights or qualify native backends:

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter FullyQualifiedName~StreamingStopSequenceTests
```
