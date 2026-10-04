## Workflow and builds

Two shared documents hold the rules every Darkspyre repository follows. Read both before the first command:

- `~/.agent-ds/workflow.md`: cutting a worktree off `origin/dev`, committing and pushing your own branch, merging into `dev`, and the Jira and ticket rules.
- `~/.agent-ds/build-instructions.md`: interim builds from `origin/dev` source, release-candidate builds from NuGet, and test copies with agent-ds `scripts/test-copy.sh`.

They apply to agent-ds, AgentDSPlugins, Inference, TensorSharp and RightSize. Rules in this repository's own documents add to them.

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

## Local development and releases

Normal builds and tests use local source from the seven sibling workspace worktrees: agent-ds,
AgentDSPlugins, Inference, TensorSharp, RightSize, SecureStore and PersistenceStore. No source-root
flags or per-session local props files are required. Missing source must fail without a package fallback.
Do not build in primary checkouts or create interim NuGet packages. Read `~/.agent-ds/build-instructions.md`.

`DarkspyreDependencyMode=Packages` is explicit release publishing/deployment mode. Debug/Release,
signing and interim installation do not select it. Test copies retain compatible prebuilt GGML driver
hardlinks. Signed bundles may copy prebuilt drivers before signing. Native compilation always needs
an explicit request.
