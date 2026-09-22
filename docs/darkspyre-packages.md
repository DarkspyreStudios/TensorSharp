# Darkspyre package identity

The `darkspyre` branch publishes fork-owned packages with a `Darkspyre.` prefix.
Assemblies, namespaces, and project names retain their upstream TensorSharp
identity so upstream changes can be reapplied without source-level renaming.

The initial fork version is `2.8.6-darkspyre.1`, based on upstream tag
`v2026.09.01`. Each published package version is immutable; later fork releases
increment the final numeric prerelease component.

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
