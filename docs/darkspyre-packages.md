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
package build, current process RID, declared backend, variant, absolute directory and bridge file.
A supplied file list must name the bridge. Every entry has a unique relative path, a nonnegative
size and a lowercase SHA-256 digest that matches the file. Paths cannot contain traversal segments,
control characters or alternate separators. Candidate directories, their ancestors, listed files
and intermediate directories cannot be symbolic links. Filesystem inspection failures return a
structured refusal. The caller owns keeping the validated directory immutable through loading;
validation does not lock the filesystem against another writer.

The focused loader tests use ordinary files and symbolic links. They do not load a bridge or qualify
a CPU or accelerator backend:

```sh
dotnet test eng/tests/ggml-native-loader/ggml-native-loader.csproj -p:TensorSharpSkipGgmlNative=true
```

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
