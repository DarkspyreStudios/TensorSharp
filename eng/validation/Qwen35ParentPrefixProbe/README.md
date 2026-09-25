# Qwen35 parent-to-child public prefix reuse

This opt-in probe compares the same parent and two reviewer prompts with automatic
ancestor checkpoints enabled versus the previous full-system-prefix behavior.
**Both arms enable radix caching and use the default public checkpoint budget of
two.** Each arm gets a fresh engine and distinct private request scopes. The model
loads once; arm order alternates over three pairs by default.
No persistent checkpoint store is attached, so disk save/import costs are excluded;
the HTTP host validation exercises persistence separately.

```powershell
dotnet build eng/validation/Qwen35ParentPrefixProbe -c Release
dotnet eng/validation/Qwen35ParentPrefixProbe/bin/Release/net10.0/Qwen35ParentPrefixProbe.dll --describe-only --model C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ3_XXS.gguf --out artifacts/qwen35-parent-prefix-reuse/prompt-description.json
dotnet eng/validation/Qwen35ParentPrefixProbe/bin/Release/net10.0/Qwen35ParentPrefixProbe.dll --model C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ3_XXS.gguf --steps 32 --pairs 3 --out artifacts/qwen35-parent-prefix-reuse/direct-probe.json
```

`--describe-only` parses GGUF metadata and vocabulary and renders prompts without
loading tensor payloads or initializing the GPU. The report records the actual
parent/reviewer common token prefix, the production-nominated checkpoint,
full public prefix lengths, tool declarations, messages and token IDs.

The fixture uses production `SkillTools`, a shell declaration matching enabled
execution/install/network flags, and `MultiAgentTools`. The parent advertises nine
tools and reviewers seven. The shell is resolved only for its declaration and never
executed. `MultiAgentSession` constructs both child messages and possible child
profiles. The real GGUF tokenizer/template renders these messages, and reflection
calls the production `ChatGenerationPipeline` boundary calculation without adding
a validation API to the product. A concise governing prompt is used instead of the
HTTP host's full skill catalog. The parent includes private assignment text that
must not appear in the child context.

Each timed workflow first runs the parent, then queues both reviewers behind a
closed `ComputeGate` before awaiting either completion. The ancestor arm must reuse
`[parent/child ancestor, complete child public prefix]`; the baseline must reuse
`[0, complete child public prefix]`. The parent is cold in both arms. Time includes
parent prefill and **all checkpoint capture costs**, parent decoding, child
adoption/prefill/decoding, and scheduling between the parent and child phases.
Loading, rendering and engine construction are excluded. Separate parent and
children phase durations and per-request time to first token are also reported.
The headline speedup uses the entire parent-plus-children workflow, so an earlier
checkpoint whose cost exceeds its benefit cannot appear as an overall improvement.

`--baseline full-public` is the default and preserves the original comparison:
the baseline has no intermediate checkpoints in either parent or child requests.
`--baseline child-boundaries` instead nominates the same intermediate checkpoints
for children in both arms; the parent receives the extra checkpoint only in the
ancestor arm. This makes the child's prefill partitions identical: the baseline
computes the common prefix before continuing, while the ancestor arm restores
that same prefix from the parent before the identical continuation.

The second policy is a **mechanistic counterfactual**, isolating the contribution
of parent-to-child reuse. It does not measure speed relative to an older release.
Both policies include all parent and child checkpoint capture costs and retain
the same strict full-token/count/status/finish validation. The JSON records
`BaselinePolicy` and `BenchmarkComparison`; neither policy may qualify performance
after a correctness failure. Save the counterfactual to a separate output path,
for example `--baseline child-boundaries --out
artifacts/qwen35-parent-prefix-reuse/child-boundaries-probe.json`, to preserve an
earlier result. A successful same-partition state diagnostic should precede using
this policy to interpret performance.

After each arm of the first pair, untimed requests verify that the complete parent
and child public checkpoints remain reusable under budget two. Every warm request
uses a new scope; private conversation reuse cannot satisfy these checks.

All completed raw rows are written before correctness validation. Paired parent,
child, and warm continuations must have identical full generated token arrays,
actual counts, completion statuses, and finish reasons. Every request must produce
non-EOS content and end by normal EOS or its token cap. Errors, missing stages,
unexpected reuse, and mismatched outputs disqualify correctness and performance
and produce a nonzero exit code. These short greedy continuations test state
restoration; they do not by themselves establish complete reviewer reasoning or
HTTP orchestration quality.

Qwen35 still copies model-owned checkpoint state into separate writable request
caches. This measures avoided prefill, not zero-copy attention-page sharing or
VRAM savings. Post-stage cached-payload memory samples exclude active peak memory,
weights, allocator reserves, and primary/private cache bytes. Only the recorded
model, backend, cache dtype and device are covered. Optional
`TS_VALIDATION_GGML_REVISION` and `TS_VALIDATION_DEVICE` label the unchanged upstream
dependency and actual device; loaded native and managed binaries are hashed.

The design was compared read-only with SGLang commit
`8ca82118e0e1a0a1b49f85f675843b19ebb388ba`, particularly
`python/sglang/srt/mem_cache/unified_cache/components/mamba.py` (valid exact
recurrent snapshots, branch snapshots, and private writable state),
`unified_cache/unified_tree_core.py` (all-component match validation), and
`python/sglang/srt/managers/schedule_policy.py` (cold shared-prefix scheduling).
SGLang is Apache-2.0; no SGLang source is copied by this probe or required to run it.

## State diagnostics

`--diagnose-state --out artifacts/qwen35-parent-prefix-reuse/state-diagnostic.json`
runs a separate direct-holder diagnostic instead of the timed workflows. It does
not weaken their exact-output criteria or qualify a performance result.

The six variants prefill the child public prefix in one call, split it at the
production ancestor with or without an intermediate checkpoint capture, restore
that ancestor from a parent that has continued and decoded, or repeat the
split/restore comparison at the nearest lower multiple of 64 tokens. Capturing
split controls match state synchronization side effects, then keep computing in
their own holder without adopting that checkpoint. The no-capture split control
distinguishes pure prefill partition effects from capture/flush effects. Every
variant captures the complete child public prefix and clones it into proposal B
before the same private suffix and greedy generation.

The report includes complete first-token logit arrays, top ten tokens and
probabilities, first-token margins, generated token IDs and text, exact-match flags,
maximum absolute error, normalized and centered RMSE, cosine, KL divergence, and
total variation. It records errors per scenario. Same-partition clone comparisons
must be bitwise exact in first logits and match the entire greedy continuation to
set `SamePartitionCloneExact`; no tolerance turns differences into a pass.
`IntermediateCaptureExact` independently checks first logits and generated tokens
with and without the intermediate checkpoint in the otherwise identical split.
Either control failure produces a nonzero diagnostic exit code.
`ValidationPassed` and `PerformanceQualified` remain false in diagnostic mode.
Direct diagnostic prefill call shapes are explicitly recorded; they may differ
from the scheduler's chunk sizes. Later divergent greedy logits are not compared
without teacher forcing.
