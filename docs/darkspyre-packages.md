# Darkspyre package identity

The `darkspyre` branch publishes fork-owned packages with a `Darkspyre.` prefix.
Assemblies, namespaces, and project names retain their upstream TensorSharp
identity so upstream changes can be reapplied without source-level renaming.

The fork is based on upstream tag `v2026.09.01`. The historical prereleases
`2.8.6-darkspyre.1` and `2.8.6-darkspyre.2` introduced persistence-backed model
loading, public direct inference primitives, and the restricted PyTorch ZIP
state-dictionary reader needed for source checkpoints that have no GGUF or
safetensors publication. Stable fork releases use a fourth numeric component;
`2.8.6.3` is the stable release of that feature set. It additionally preserves the documented
four-dimensional channels-last output shape from direct image group normalization. Each published
package version is immutable; later compatible fork releases increment the fourth component.

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

## 2.8.6.4: complete persistence-backed GGUF sets

`PersistenceFileSet` names the primary artifact and an immutable map of portable relative paths
to `PersistenceFileReference` objects. Pass the entire set to `GgufFile.OpenAsync` or
`ModelService.LoadModelAsync` for split GGUF models. The single-reference overload remains for
single-file artifacts; persistence loading never discovers unlisted files beside a supplied path.
The raw-path GGUF API still supports its existing sibling discovery.

The one persistence lease implementation retains all source handles when file-backed artifacts
already have the declared layout. Otherwise it copies the complete set into a private temporary
directory, retaining names/subdirectories and removing the projection on reader disposal/model
unload. Mixed stores and opaque asset keys are supported. Failure or cancellation closes opened
streams and removes partial projections. The reader rejects missing/misnamed shards, inconsistent
numbering/counts and duplicate tensors before native model allocation.

Verification uses synthetic split GGUFs, including actual tensor reads from each shard, in-place
files, memory/mixed stores, malformed sets, cancellation, service lifetime and replacement rollback.
This is managed persistence work; no upstream ggml or native kernels change and no full-weight or
GPU inference parity is claimed. The earlier local-only `2.8.6.4-local.gguf.<commit>` integration
builds are superseded by the 2.8.6.4 release package set.

Focused verification: 22 persistence tests pass in Debug and Release; both builds report zero
warnings. Changed-file formatting verification and the Chat transitive dependency vulnerability
audit pass. The existing single-file safetensors and replacement-rollback tests remain included.

## 2.8.6.4: bounded metadata inspection

`GgufFile.InspectMetadata`, `SafetensorsFile.InspectMetadata` and
`TorchStateDictionaryFile.InspectMetadata` reuse the corresponding loading parsers over a caller-owned
seekable stream. They return metadata only, never open sibling files, create filesystem projections,
map weights or allocate native tensors. The stream reports the full artifact length, so a caller can
serve bounded remote ranges without downloading the artifact. Inspection leaves the stream open and
defaults to a 16 MiB total-read budget, including ZIP seeks. Restricted pickle interpretation retains
the existing whitelist; no repository code is executed.

GGUF strings/arrays/counts and safetensors header allocations are checked against remaining bytes
before allocation. Duplicate tensor names, invalid GGUF rank/alignment, negative safetensors dimensions
and overflowing safetensors shape sizes are rejected. Ordinary GGUF/safetensors file parsing uses
the same bounded parser with a 64 MiB header budget. The API exposes metadata, not a remotely backed
model instance. Model loading still uses the existing persistence/file APIs.

Synthetic inspection tests prove header-only access and caller ownership, bounds before allocation,
duplicate-name rejection, declared safetensors storage validation and shared pickle whitelist behavior.
These tests and the persistence regressions require no weights, GPU or native backend changes.
Focused verification passes 36 tests in Debug and Release with zero build warnings, including all
22 persistence regressions, the existing synthetic safetensors reads and restricted Torch tests.

## 2.8.6.4 release verification (2026-09-23)

The Inference four-package release requires this managed dependency closure: Tensors,
Runtime.Logging, Runtime, AgentHost, Backends.Cuda, Backends.GGML, Backends.MLX, Models and Chat.
All nine use the one 2.8.6.4 version and retain declared NuGet dependencies. No native build or
full-weight/device test is claimed. The managed-only pack switches are the existing GGML/MLX
packaging policy, not removal of previously embedded runtime assets.

The 38 focused persistence, metadata, restricted Torch and safetensors cases pass in Debug and
Release. The Chat transitive vulnerability audit reports no known vulnerable dependency. The
source changes are the previously verified persistence/inspection commits; this release changes
only versioning and release documentation. Package inspection and fresh downstream Inference
restore are required before publication to the private DarkspyreStudios feed.
