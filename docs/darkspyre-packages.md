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

`Darkspyre.TensorSharp.Backends.GGML` ships the macOS arm64 native bridge as
`runtimes/osx-arm64/native/libGgmlOps.dylib`, built from the same commit with
`TensorSharp.GGML.Native/build-macos.sh`. A consuming app therefore loads the GGML CPU and Metal
backends on Apple silicon without building natives. The release packs it by passing
`-p:DarkspyreGgmlNativeOsxArm64=<path>`; ordinary builds are unchanged. Other platforms still build
their bridge with their own `build-*` script.

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
