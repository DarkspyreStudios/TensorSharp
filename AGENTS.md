# External dependencies

- Keep ggml upstream sources unchanged. Never add or apply ggml patches, rewrite
  fetched ggml files, or make TensorSharp builds depend on a modified ggml tree.
- Implement behavior that ggml does not provide in TensorSharp-owned code,
  including native kernels and backend integration when necessary.
- Validate native changes against an unchanged upstream checkout. Record the
  dependency revision, actual test coverage, and benchmark limitations; do not
  count skipped or unavailable model/device scenarios as passing validation.

# Validation artifacts

- Keep generated validation logs, reports, and snapshots in ignored `docs/validation/`
  or `artifacts/`; do not force-add them to Git.
- Store reusable validation tools in `eng/` and required test fixtures in the
  relevant test project, outside the generated evidence directories.

# Darkspyre

- Never build the native GGML, MLX or CUDA libraries unless the user asks for it in that task. This overrides the native build steps in DEVELOPMENT.md. Managed builds leave native compilation off; it only runs when `TensorSharpBuildGgmlNative`, `TensorSharpBuildMlxNative` or `TensorSharpBuildCudaNative` is set.

## Operating-system boundaries

All OS-specific code in every Darkspyre project must live behind an OS-agnostic capability contract.

Portable consumers call that contract. Platform implementations own OS detection, native bindings and ABI layouts, native-library names and loading, system commands, platform paths, filesystem semantics, and platform-specific process and UI behaviour. Only platform composition selects the implementation. SDK contracts, plugins, extensions, core modules, and portable host code must not require consumers to know the operating system or call a platform implementation directly.

An interface that exposes OS-specific commands, types, paths, or layouts does not satisfy this rule. Reuse the existing capability seam. A platform without an implementation reports the capability unavailable through the same contract.

This boundary lets another operating system supply an implementation without changing portable consumers.

**Check.** Audit source and package configuration for OS branches, native imports, platform paths and commands, filesystem assumptions, and platform-only dependencies. Trace each hit through its callers and registration. Every OS-specific operation must remain inside a platform implementation selected through an OS-agnostic contract. A filename, namespace, annotation, or `IsMacOS()` guard alone does not establish that boundary.
