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

## Branches

This repository is a fork of `zhongkaifu/TensorSharp`. `origin/main` is the upstream baseline that the Darkspyre
version sits on, and it must match upstream's `main`.

- Never push to `origin/main`. No commit, merge, fast-forward or force-push goes there.
- `origin/darkspyre` is this repository's main branch. Work merges into `origin/dev`, and `origin/dev` merges into
  `origin/darkspyre`.
- When the user says "push to main", "push to master" or "merge dev into main" for the Darkspyre repositories, in
  this repository that means `origin/dev` into `origin/darkspyre`.
- Upstream changes come from `upstream/main` (`https://github.com/zhongkaifu/TensorSharp.git`) and are merged into
  `dev` only when the user asks.

**Check.** `git ls-remote origin refs/heads/main` names the same commit as upstream's `main`.

## Native builds

- Never build the native GGML, MLX or CUDA libraries unless the user asks for it in that task. This overrides the native build steps in DEVELOPMENT.md. Managed builds leave native compilation off; it only runs when `TensorSharpBuildGgmlNative`, `TensorSharpBuildMlxNative` or `TensorSharpBuildCudaNative` is set.
