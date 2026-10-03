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

`GgmlNativeLoader.ResolvePackageCandidatesAsync` explicitly reads deployed package catalogs without
loading native code. It inspects only `AppContext.BaseDirectory` and the managed supplier assembly's
directory. Baselines use `ggml/baseline.artifact.json` with the fixed
`runtimes/<rid>/native/` payload. Optional packages use `ggml/<variant>.artifact.json` with
`ggml/<variant>/`. The resolver does not scan CWD, PATH, parent repositories or download locations.
Missing catalogs return no candidates. Loose libraries without a catalog do not establish candidates.

The packer derives each catalog from the validated artifact record. Catalogs retain package/native
source identities, exact ABI and ggml revision, backend declarations, file sizes/hashes and component
evidence. Runtime catalog file entries contain exactly `path`, `size` and `sha256`. The full artifact
inventory separately retains `executable` facts. Baselines retain flat package notices and also carry the complete verified file closure,
including licenses, inside the native directory. Their RID-conditional `buildTransitive` targets copy
that directory and its catalog for output and publish. An explicit target RID takes precedence over
the SDK host RID. Optional targets apply the same RID condition and copy their payload and sibling
catalog for output and publish. Other RID packages remain inert and cannot overwrite these bindings.

Resolution rejects malformed/duplicate JSON, unknown fields, conflicting bindings, wrong build/ABI/
upstream/RID/variant, incomplete component mappings, unsafe or linked paths and missing/stale file
bytes. It throws `InvalidDataException` with a fixed safe message and no raw filesystem diagnostic.
Cancellation never returns a partial candidate list. Results and nested file lists are read-only.
Catalog checks do not establish native initialization, device availability or redistribution permission.
Package/native source commits pass exact hash syntax and core-component coherence checks. This resolver
does not compare them to a managed assembly source commit or read a native bridge identity. Verified
supplier release provenance and actual loaded bridge identity remain separate downstream gates.

Candidates put accelerator primaries first (CUDA then Vulkan; Metal on macOS), followed by baseline
CPU and CPU alternatives advertised by optional artifacts. Each artifact/backend pair uses the same
verified files; it does not represent a second independent library. The configured runtime still
refuses fallback after a potentially loaded failure. A consumer can derive a stable pair identity from
the catalog RID/variant and candidate backend. Actual bridge identity and backend remain initialization
observations, not catalog claims. Existing `Check` and initialization validate candidates again.

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

`TensorSharpSkipCudaNative=true` disables CUDA compiler discovery, architecture resolution,
PTX compilation, intermediate PTX copying and committed-PTX updates. The committed PTX content
still copies through the existing output/publish items. The flag also applies to direct target
invocation and takes precedence over `TensorSharpUpdateCommittedPtx=true`. An unset or false
flag preserves normal CUDA build behavior. This flag permits managed-source verification; it
does not establish CUDA execution, artifact identity, native availability or package closure.

The packer validates all staged inputs before writing release output. It accepts the five baseline
RIDs above, Vulkan and CUDA13 on Linux and Windows x64, and Vulkan on Windows ARM64. It rejects
unknown pairs, links and special files, nonportable paths, missing licenses, unresolved or uninspected
dependencies, wrong binary architecture, and a bridge without the exact `TSGgml_GetBuildIdentity`
export. Binary identity must match the source, ggml, RID, variant and version in the build record.
Inspection-tool failures do not count as empty dependency lists. A supplied managed package must
have the release's exact identity/version and no native payload.

`eng/ggml-required-exports.json` records the literal entrypoints used by the guarded managed
interop declarations. The existing Roslyn guard tool generates it with `--inventory` without
rewriting sources. The inventory binds to the exact native ABI and pinned ggml commit. Artifact
validation requires every listed symbol and refuses missing or uninspected exports. Additional
raw backend exports are permitted. CPU artifacts do not require CUDA/Vulkan-only upstream APIs.

Release output requires all 12 RID/variant pairs: macOS ARM64 Metal; Linux x64 and ARM64 CPU,
Vulkan and CUDA13; Windows x64 CPU, Vulkan and CUDA13; and Windows ARM64 CPU and Vulkan.
Missing, unsupported or duplicate pairs fail before output is written. Bundled dependencies must
be inspected native siblings with the matching target format and architecture. A text file with a
library name does not satisfy closure. Linux bundled dependencies require `$ORIGIN` linkage.
Dependency names cannot refer to absolute build-machine paths or escape the selected directory.
macOS accepts system libraries and direct `@loader_path` siblings; unverified `@rpath` scopes fail.

`--validate-only` runs those checks without creating an output directory, package, archive or
manifest. This mode permits a partial stage; add `--complete-release` to require the entire matrix.
Validation does not require a fetched ggml checkout. Staged inputs remain caller-owned
and must stay immutable through packaging. Existing output is not removed on validation failure.

```sh
python3 -B eng/pack-ggml-natives.py --stage artifacts/ggml-natives/2.8.6.8 --validate-only
python3 -B eng/pack-ggml-natives.py --stage artifacts/ggml-natives/2.8.6.8 --validate-only --complete-release
python3 -B -m unittest discover -s eng/tests -p test_pack_ggml_natives.py
dotnet run --project eng/guard-ggml-interop/guard-ggml-interop.csproj -- --inventory .
```

The packaging tests inspect actual native headers and exports from a tiny compiled ABI-stub
fixture, ordinary files and symbolic links. The fixture contains no GGML backend. These tests do
not qualify model loading, native lifecycle or accelerator execution.

`eng/build-ggml-natives.sh` records provenance through `eng/record-ggml-native-build.py`. The record
checks the built bridge's identity, native header, architecture and identity export against the
committed source and actual CMake cache. It rejects modified source or ggml inputs, a host-specific
CPU profile, a mismatched accelerator, and a Linux runpath other than `$ORIGIN`. The record includes
the bridge hash, CMake cache hash and settings, actual compiler output, exact CPU floor and macOS
deployment target. CUDA13 records include observed toolkit compiler output and every compiled
SASS/PTX architecture. Target and GPU execution remain explicitly unrecorded by this inspection.

The recorder and packer use one release-profile policy. Collection requires the recorded portable
CPU floor, target/backend settings, Linux `$ORIGIN` configuration and an explicit macOS deployment
target. CUDA13 requires the full recorded SASS/PTX profile and consistent observed CUDA13 compiler
and toolkit evidence. The recorded bridge SHA-256 must match the actual staged bridge.

Each build record includes `cmake-settings.txt` and `cmake-cache.snapshot.txt`. Both are ordinary
files. The snapshot preserves the original observed cache bytes, including line endings and
comments. Its SHA-256 must match `cmakeCacheSha256`. A strict parser rejects empty, malformed or
duplicate cache entries. Parsed snapshot settings, normalized settings and `cmakeConfiguration`
must agree exactly. Records without the snapshot fail validation, including partial-matrix stages.
The snapshot is evidence only. Build scripts do not use it as a CMake input. These checks validate
recorded observations; they do not authenticate the recorder, reproduce the build or qualify
target/compiler/GPU execution. Native source commit evidence remains a separate release check.

### Mac Identity Relink

`eng/relink-ggml-native-identity.py` compiles only the Mac bridge identity object and relinks the
bridge from verified retained inputs. It accepts the osx-arm64 Metal/CPU baseline with deployment
target 14.0. It compares the original and current native source trees. All 59 other object/archive
inputs retain their exact bytes. A fresh configure-only CMake observation supplies the current
build profile and cache snapshot. The historical cache remains separate evidence and is never a
build input. Original and fresh compiler flags and link topology must match.
The replacement identity uses the committed current TensorSharp build version. The original
bridge identity and build record retain their actual original version. Version replacement does
not change the pinned upstream, native source tree, target, backend or CPU floor.

The additive `identityRelink` record contains both original observations and actual new compiler
and linker arguments. The packer verifies ordinary evidence files, hashes, source-tree equality,
the original bridge/profile/cache, exact reused input membership, replacement identity source and
compiler/linker arguments. The ordinary current-ABI, bridge-hash, profile and raw-cache gates also
remain required. Every execution invariant is checked before invoking the compiler or linker.
Recorded absolute command paths bind to explicit immutable execution roots. The evidence files
use portable relative paths. Copying a stage or pruning its producer worktree does not change the
recorded arguments or prevent validation in another checkout. This operation does not compile
upstream ggml or accelerator kernels. It does
not qualify native lifecycle, target execution or GPU safety.

The packer's explicit `--mac-prerelease` mode requires exactly one osx-arm64 Metal/CPU baseline.
All artifact, ABI, profile, cache, dependency, component and relink checks remain required. The
default output mode requires the complete twelve-artifact release matrix. `--complete-release`
and `--mac-prerelease` cannot be combined. Partial staging validation alone does not authorize
package output.

### Package Resolver Preflight

`eng/tests/ggml-package-preflight` checks an actual packaged managed loader against an unpacked
native package. Its three arguments are the absolute managed GGML DLL path, deployed native
package root and target RID. It invokes the supplier's existing filesystem resolver with that
explicit root, validates every returned candidate and reports executed DLL hash/MVID/build/ABI.
Both entry and exit require `Unconfigured` state and no selected native handle. The tool does not
configure, initialize or execute native code. Invalid catalogs or absent candidates return exit 1.
This validates actual package/catalog interoperability without claiming native runtime qualification.

### Component evidence

The build record and schema-1 artifact manifest include `components`. Each component records an
ID, name, kind, supplier, exact version and source ID. `binaryFiles` and `evidenceFiles` contain
relative paths, positive byte sizes and lowercase SHA-256 hashes. The two core records associate
the bridge with TensorSharp and statically linked ggml. The provenance helper copies their actual
source `LICENSE` bytes after verifying clean TensorSharp and pinned ggml sources. The ggml record
uses its exact pinned commit as its version identity.

`--redist-dir` requires `--redist-manifest`. The UTF-8 mapping has this shape:

```json
{
  "schema": "tensorsharp-native-redistribution/1",
  "components": [
    {
      "id": "supplier-runtime",
      "name": "Supplier runtime",
      "kind": "redistributed",
      "supplier": "Supplier name",
      "version": "exact-supplier-version",
      "sourceId": "exact-supplier-release-or-package-identity",
      "binaryFiles": [{ "path": "runtime-library-name", "size": 123, "sha256": "actual-64-character-lowercase-digest" }],
      "evidenceFiles": [{ "path": "licenses/supplier-notice.txt", "size": 456, "sha256": "actual-64-character-lowercase-digest" }]
    }
  ]
}
```

The example describes fields, not usable redistribution inputs. The provenance helper copies only
declared files. It rejects extra input files, stale hashes, absent or empty evidence, non-text notices,
links, unsafe or case-conflicting paths, conflicting ownership and core-file overwrites. Binary files
must sit at the redistribution directory's top level. Evidence must sit under `licenses/`. The
mapping may sit outside that directory or inside it; the helper does not copy the mapping itself.
Multiple components may reference the same supplied evidence. Each redistributed native sibling
has one component owner. The packer independently verifies exhaustive staged-file coverage and
actual inspected native formats. Partial validation applies the same evidence checks without
requiring every RID/variant or a fetched ggml checkout.

Flat manifest `notices` and staged package/archive license bytes remain unchanged. Generated
package `NOTICE.md` identifies components, exact supplier identities and supplied evidence paths.
A filename does not establish legal permission. Core license records do not certify complete
inline/static third-party attribution coverage, optional compiled dependencies or toolkit
redistribution permission. Actual source/toolkit notice review remains a separate release gate.

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

Direct CUDA groups compare every owned allocator's live storage references before communicator
or rank teardown. The Distributed wrapper performs that preflight before releasing pins, TCP or
its local CUDA group. Healthy Busy keeps allocation/reference and group execution admission
fenced. Callers release escaped storage and explicitly retry disposal. Both group implementations
carry one validated owning plan and one same-thread context restoration token through cleanup;
child cleanup does not start a competing parent census or restore an intermediate context.
The internal composition applies to the actual CUDA and Distributed implementations, not an
arbitrary custom group. The managed checks do not qualify physical multi-device execution,
communicator construction, TCP operation or model-family cleanup.

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

Concrete model constructors roll back base and family-owned resources without virtual disposal
of a partially constructed subclass. The guarded host-read barrier drains deferred work before
any family graph reset (some native reset entrypoints have no barrier). Family graphs retire
before generic GGML scratch/cache
bindings, then derived buffers, base weights/mappings, late owned raw buffers and final context/
model ownership retire. Failed construction does not dispose a caller-supplied tensor-parallel
group, its allocator or its context. A tensor allocated during F32 weight reading transfers to
the model only after a successful read; failure explicitly disposes the unregistered allocation.
Clean rollback preserves the original exception and stack. Failed cleanup reports both errors
and stops further dependent teardown. For GGML, a generation-local failed-owner collection
retains the actual unsafe model and unregistered storage, supported by the existing live runtime
owner’s process-exit root and guarded shutdown refusal. It adds no global root or finalizer and
does not establish equivalent terminal-failure retention for CUDA or MLX owners. This retention
applies to construction rollback and failure inside the shared normal `Dispose` pipeline. Cleanup
outside that pipeline does not automatically enter this retention path.
The shared pipeline preserves the original cleanup exception and stack; repeated teardown after failure refuses before any phase
and retains the first diagnostic. This is terminal disposal retention, not forward/reset operation
fencing or proof of failed-GPU synchronization safety.

The DeepSeek V4.1 vision loader reserves its returned native handle before validating companion
metadata and attaching it to the text model. Successful rollback explicitly frees and clears
the handle, then rethrows the original validation error unchanged. Refused rollback retains
the actual model and reserved handle through the existing GGML failed-owner collection and
reports the original validation and cleanup errors together. Normal V4.1 disposal releases its
vision handle and text executor through one shared graph-release callback after the host-read
barrier. A failure retains the still-owned fields and native identities; repeated teardown
refuses at the shared terminal guard. These paths do not fence forward/reset operations or
establish equivalent non-GGML retention, captured GPU teardown or image-encoding qualification.

Weight loading keeps a local disposable owner until ordinary quantized, Bonsai or F32 storage
transfers to a model dictionary. Owned stacked expert buffers transfer before reading or creating
expert views. Failure before that transfer explicitly frees the raw allocation. Fusion transfers
the fused owner before disposing source weights, and removes each source entry only after
successful disposal. Contiguous mapped fusion views retain their common backing owner.
Bonsai registration reserves an identity before native registration. Unregister removes only
successfully retired identities. Failed unregister keeps host storage and GCHandle identity
alive; the model cleanup boundary retains unsafe local owners and preserves both work and
cleanup errors. These paths do not establish tensor-parallel backing-owner or CUDA/MLX shared
allocator teardown safety.

Tensor-parallel shard arrays reserve model ownership before allocation/copy. Quantized raw copies
and requantization use owned wrappers before writing. Removed column-parallel sources remain
explicitly model-owned until their views retire. Temporary source views unwind through the local
rollback helper; original work and failed cleanup errors remain visible. GGML TP views retire
before backing owners, bulk buffers and the GGUF mapping. Source disposal precedes dictionary
removal. GGML group synchronization and its host barrier precede graph/cache/storage teardown.
A borrowed group's allocator-wide CUDA arena remains with that group. Direct CUDA/MLX retains
its existing late TP-view/cache order and does not newly release column backing owners. These
source rules do not qualify its synchronization, device cache closure, terminal strong retention
or physical multi-device execution.

GGML broadcast borrows its input and allocates an independent tensor for every logical rank.
Partial allocation/copy failure explicitly rolls back the whole destination array. A refused
rollback retains its distinct borrowed source dependency without disposing that input. Owning
model callers retire their original embedding/router/Mamba source only after copying succeeds.
Temporary-output callers retire every GGML copy, including rank zero, and their original source
through one ownership boundary. A refused cleanup retains the actual source and remaining
copies through existing failed-model ownership; work and cleanup errors stay visible. Direct
CUDA/MLX broadcast cleanup preserves its existing rank-one-onward/source disposal sequence.

`Shutdown()` refuses active initialization, calls, contexts, tensors, models or native handles.
Every public shutdown/recreation path uses that guard. Success is terminal and idempotent; cached
native imports cannot execute afterwards. The loaded library is never unloaded. A process-wide
BCL-only ownership token prevents another managed load context from loading a second GGML build
without pinning a foreign collectible assembly. `AcquireLease(GgmlRuntimeResourceKind)` lets
adapters retain additional resources; it does not replace the mandatory supplier leases.
Both explicit selection and default probing register the same owned, idempotent `ProcessExit`
handler. Successful guarded teardown removes that external delegate root before reporting
`Stopped` and `Released`. Busy or poisoned teardown retains cleanup ownership and does not
report release. Removing the handler does not reset terminal process ownership or unload the
native library.

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
`selected-metal` repeats the explicit selection and lifecycle checks on the Metal backend. It
requires an actual detected GPU, reports its description, and checks tensor arithmetic after GPU
dispatch and host synchronization. A CPU fallback fails this mode.
`reject-variant` verifies that a loaded identity mismatch leaves later candidates untried and
rejects a second selection. `reject-legacy` verifies the same refusal for a bridge without an
exact ABI identity. The probes do not qualify model generation, real whole-model handles, CUDA or
Vulkan execution, Windows dependency loading or other RIDs, nor prove a published package's layout.

`retire-selected-cpu`, `retire-default-cpu`, `retire-selected-metal` and `retire-default-metal`
load the real fixture, GGML backend and Core assemblies privately in a collectible generation.
They assert the provenance of the fixture, loader, context, allocator, group, tensor and operation
registry. They perform real tensor arithmetic, drain host reads, refuse shutdown while contexts,
tensors, aligned allocations or resource/call leases remain, and drain a pending rank-one callback
before disposal. A Model-kind lease does not establish actual model loading. Rank-one callbacks
do not establish multi-device persistent worker shutdown. The foreign invocation frame returns
only native identity strings and weak references. The outer frame requests GC and verifies
collection of the generation and its three real assemblies before process exit. The tests do not
serialize foreign objects through a permanently rooted reflection serializer.

```sh
dotnet build eng/tests/ggml-native-runtime/ggml-native-runtime.csproj -c Release -p:TensorSharpSkipGgmlNative=true
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll selected /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll selected-metal /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll default /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll ambiguous-default /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll reject-variant /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll reject-legacy /absolute/legacy/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll retire-selected-cpu /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll retire-default-cpu /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll retire-selected-metal /absolute/bridge/directory metal
dotnet eng/tests/ggml-native-runtime/bin/Release/net10.0/ggml-native-runtime.dll retire-default-metal /absolute/bridge/directory metal
```

`eng/tests/ggml-model-lifetime` privately loads its fixture and all real TensorSharp dependencies
in a collectible generation. `normal` generates a deterministic two-layer F32 HunyuanDense GGUF,
loads it, refuses shutdown while the model is live, executes complete prefill and decode, and
explicitly disposes it before guarded shutdown and outer-frame collection of all eight roots.
`refusal` exercises the actual unequal-head-dimension constructor error. `constructor-matrix`
checks malformed-input unwind of all 18 concrete model implementations, including internal image
components, with zero resources immediately after each failure. `partial-weight` injects a real
empty-file read after the first registered F32 weight and verifies local and registered tensor
cleanup without virtual disposal or GC. `borrowed-tp` checks disposable allocator ownership and
a real caller-owned nested GGML context after base and derived construction failures; using a
CUDA constructor label with a supplied allocator/context does not qualify CUDA execution.
`phase-order` queues real asynchronous native flash attention and verifies the pipeline barrier
drains it before the first family graph callback, with real KV device-copy cache bindings retired
before the first derived buffer free. It does not build a captured CUDA graph.

`derived-cleanup-failure`, `base-cleanup-failure`, `local-cleanup-failure` and `dispose-cleanup-failure` use labeled managed
fault injection with real native resources. They verify both errors, retained derived/base or
unregistered native buffers through finalizer drainage, shutdown refusal and all foreign roots
remaining alive. The base mode holds a controlled managed call lease, not a blocked native call.
The disposal mode constructs the probe successfully, then refuses at its graph phase; repeated
disposal preserves the first exception without rerunning teardown, and actual model/storage
owners and all four leases survive finalizer drainage. `observe-dispose-refusal` is the historical
red reproduction of actual model/storage collection despite an abandoned numeric Model lease.
These tests do not qualify pretrained, quantized, whole-native-executor or multi-device models,
every nested allocation helper, actual captured-graph teardown, other RIDs or CUDA/Vulkan devices.
`observe-refusal` is a historical red-reproduction mode for the pre-fix checkpoint and is not a
current success gate. Run each mode/backend in a separate process against a matching real bridge:

```sh
env TMPDIR="$PWD/tmp" TENSORSHARP_GGML_NATIVE_SKIP=true TENSORSHARP_MLX_NATIVE_SKIP=true dotnet build eng/tests/ggml-model-lifetime/ggml-model-lifetime.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:TensorSharpSkipCudaNative=true -p:GeneratePackageOnBuild=false -p:PublishAot=false
env TMPDIR="$PWD/tmp" dotnet eng/tests/ggml-model-lifetime/bin/Release/net10.0/ggml-model-lifetime.dll normal cpu /absolute/bridge/directory
env TMPDIR="$PWD/tmp" dotnet eng/tests/ggml-model-lifetime/bin/Release/net10.0/ggml-model-lifetime.dll normal metal /absolute/bridge/directory
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
