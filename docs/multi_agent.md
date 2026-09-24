# Multiple agents

TensorSharp can let a model delegate independent parts of a request to bounded
subagents, collect their results, and synthesize one answer. Delegation runs in
`TensorSharp.AgentHost/Agents/`, alongside the existing
[skill and code-execution loop](agent_skills.md), and uses the host's loaded
model and generation backend.

Automatic delegation is enabled by default on supported server chat paths.
The model decides whether to delegate and which task to assign. Enabling the
feature does not force a fixed number of agents or guarantee a faster or more
accurate answer.

## Choosing work and models

The coordination prompt asks the model to identify a substantial, independent
task with a concrete expected result before spawning. Suitable examples include
analyzing separate documents, investigating distinct components, and reviewing
a complex result independently. Short requests, sequential dependencies, and
tasks competing over the same mutable resource should stay with the parent.
The parent should continue useful independent work, avoid repeating delegated
work, and verify important claims before combining the results.

Each child uses the same model as the parent. `agent_type` chooses instructions
and tool access, not another set of model weights:

| Role | Intended work | Tool access |
|---|---|---|
| `explorer` | Focused investigation; default role | Advertised skill listing and reading; `read_file` when the parent offers it |
| `reviewer` | Independent checks and evidence | Advertised skill listing and reading; `read_file` when the parent offers it |
| `worker` | Bounded implementation work | Read-only by default; mutable host tools require explicit operator opt-in |

There is no automatic selection among local or remote models in this version.
Sharing the loaded model avoids loading a separate copy of its weights for each
child. It does not imply shared mutable conversation state or a fork of the
parent's KV cache.

## Context, tools, and lifecycle

Each child starts with governing system/developer instructions and a
self-contained task supplied by the parent. It does not receive the parent's
conversation transcript or attachments. Task text
must therefore include the facts, scope, ownership boundaries, and expected
evidence needed for the assignment. Follow-up turns reuse the child's own
conversation. The parent receives bounded result reports rather than the
child's full tool transcript.

Agents belong to a request-scoped tree with parent identity and depth. Limits
apply across that tree, including children created by other children. The
runtime manages their background tasks, status, cancellation, and completion;
the host supplies generation through its existing inference path.

The model sees these native tools:

| Tool | Behavior |
|---|---|
| `spawn_agent(task_name, task, agent_type?)` | Starts an independent child and immediately returns its ID and status. The task name must use letters, digits, underscores, or hyphens. |
| `wait_agent(agent_id?, timeout_ms?)` | Waits for the named direct child, or all direct children when the ID is omitted. Default timeout is 10,000 ms; maximum is 60,000 ms. A timeout does not mean the child completed or was cancelled. |
| `send_input(agent_id, message)` | Queues a message for a running child at its next generation boundary, or starts a follow-up turn on a completed child. |
| `list_agents()` | Reports direct children and their state. Waiting uses `wait_agent`, rather than repeated listing. |
| `close_agent(agent_id)` | Cancels a child and its descendants. Cancellation is not successful completion. |

Children inherit only permitted host-owned tools. Client-owned function tools
are not delegated: their implementations belong to the caller, and an internal
child cannot ask that caller to service them. Read-only roles cannot execute
shell commands, run skill scripts, or modify files. They can analyze evidence
included in their task and use the parent's `read_file` tool when offered,
within its existing filesystem restrictions.
Enabling worker tools does
not enable an execution surface that the operator has otherwise disabled.

Worker tools share the parent's sandbox and workspace. This version does not
create Git worktrees or merge independent patches. Assign disjoint file scopes
when opting into workers; read-only delegation is the default. Agent IDs do not
grant access to another request's agents or workspace.

Failures, timeouts, cancellation, and exhausted budgets are reported distinctly
from completed work. Required child results must be collected before the parent
claims completion. Request cancellation also stops the request's descendants.
Agent state is not a durable cross-request session API.

## Host controls

The server startup flags below configure the entire request tree. The same
options are available through `ServerHostingOptions.MultiAgent` and
`MultiAgentOptions` in C#. Existing server JSON configuration expands to these
flags. A request can set the top-level boolean `"multi_agent": false` to use a
single agent. `true` or an omitted field follows the host policy and cannot
enable delegation when the host has disabled it. Requests cannot raise host
limits or enable worker tools.

| Flag | Default | Allowed range or meaning |
|---|---|---|
| `--no-multi-agent` | Absent | Disables coordination tools and automatic delegation. `TS_NO_MULTI_AGENT` set to a nonempty value other than `0` also disables it. |
| `--agents-max-concurrent` | `3` | `1`–`32` active descendants; excludes the root |
| `--agents-max-count` | `8` | `1`–`128` total children across the tree |
| `--agents-max-depth` | `2` | `1`–`8` child levels below the root |
| `--agents-max-rounds` | `8` | `1`–`64` tool-loop rounds per child turn; a capped loop permits one final answer generation |
| `--agents-max-generations` | `48` | `1`–`1024` child generations across the request |
| `--agents-timeout` | `180` | `1`–`3600` seconds per child run, reset for a follow-up turn |
| `--agents-max-result-chars` | `8000` | `256`–`64000` characters per result report |
| `--agents-allow-worker-tools` | Absent | Allows the `worker` role to use the parent's enabled mutable host tools |

`MultiAgentOptions.MaxTaskCharacters` bounds task and message text, defaults to
16,000, and permits values from 256 through 64,000. It is a C# option, without a
dedicated startup flag. These limits bound orchestration; existing context,
generation, skill, and code-execution limits still apply. Timeouts signal
cancellation; generation callbacks must honor their cancellation token, and
host tool execution retains the existing runner's time limits.

The integrated server paths are OpenAI-compatible chat completions and
Responses, Ollama chat, and the Web UI/TensorAgent chat path. TensorAgent uses
the host defaults; it does not add a separate settings toggle. Structured-output
requests that suppress tools, and model families whose templates cannot render
tools, do not offer coordination tools.

Direct C# callers enable `SkillAgentLoopOptions.MultiAgent` and provide
`SubagentGeneratorFactory`. That factory must allocate independent generation
state for each child ID; passing callbacks that capture the root's mutable
session or KV state is not safe. The server integration supplies a separate
`ChatSession` and generation context per child.

`SkillsChatClient` local delivery supplies independent HTTP conversations
automatically. Configure `SkillsChatClientOptions.MultiAgent` to set local
limits, or set `SkillsChatRequest.MultiAgent = false` for a single-agent request.
Against a detected TensorSharp server, local delivery suppresses server-side
orchestration so only one host owns the tools. Token usage includes the children;
`SkillToolInvocation.AgentId` identifies each callback's owner. Callbacks may run
concurrently and must be thread-safe.
The returned `SkillsChatResponse.Messages` can be reused for the next turn;
it preserves the caller's preamble without accumulating injected host policies.

For example, the server can decide how to divide a substantial document review:

```json
{
  "model": "your-loaded-model",
  "messages": [{
    "role": "user",
    "content": "Review the supplied service specifications for migration risks. Investigate independent components as useful, verify conflicting findings, and produce one prioritized report with evidence."
  }],
  "multi_agent": true,
  "stream": true
}
```

Use `/v1/chat/completions` and supply the actual documents or authorized skills
alongside that request. Delegation does not itself expose any new documents.
The standalone CLI's direct decode loop is not integrated with this coordinator;
use the server or the C# agent host for subagents. No mobile-device performance
claim follows from the shared TensorAgent integration.

## Design sources

The user-provided [TensorSharp design discussion](https://chatgpt.com/share/6ab43386-9464-83e8-828a-a1e97583bee4)
proposed independent agent sessions, a runtime manager, tree-wide bounds,
lifecycle tools, and a read-only first stage integrated with existing skills.
Its later-stage proposals include isolated workspaces, heterogeneous models,
task DAGs, and KV-cache-aware context forks. Those later features are not
implemented by this change.

Implementation and prompt design were compared with the unchanged public
[OpenAI Codex checkout](https://github.com/openai/codex/tree/12fd929f724371104a1dea7c14930b078d6b4a18),
pinned at `12fd929f724371104a1dea7c14930b078d6b4a18` on 2026-09-24.
TensorSharp adapts these ideas to its existing C# host and local-model tools:

| Codex reference at the pinned revision | Design applied here |
|---|---|
| [Delegation guidance](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/core/src/tools/handlers/multi_agents_spec.rs#L711) | Bounded independent tasks, useful parent work during delegation, distinct ownership, and evidence-based synthesis |
| [Proactive mode instructions](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/prompts/src/model_messages/multi_agent.rs) | Explicit prompting for the model's delegation decision |
| [Shared agent registry](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/core/src/agent/registry.rs) | Tree identity, depth, and shared capacity accounting |
| [Child configuration](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/core/src/agent/child_config.rs) | Inheriting effective runtime permissions and model settings |
| [Completion routing](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/core/src/agent/control/completion.rs) | Separate child context and parent result delivery |
| [Wait handling](https://github.com/openai/codex/blob/12fd929f724371104a1dea7c14930b078d6b4a18/codex-rs/core/src/tools/handlers/multi_agents_v2/wait.rs) | Bounded asynchronous waiting and explicit timeout state |

This is not a claim of API compatibility with Codex. In particular, this version
uses fresh child contexts and the five tools listed above, rather than exposing
all of Codex's history-fork, messaging, and resume options.

## Validation and performance

Evaluate single-agent and automatic multi-agent modes on the same tasks,
model, backend, hardware, sampling settings, and completion criteria. Include
both substantial independent work and small or dependent tasks, so unnecessary
delegation appears as overhead. Warm up each path, run repeated trials, and
record latency distribution, completed-task score, input/output token usage,
failed or truncated work, and observed concurrency.

Use separate evidence for two questions:

1. Deterministic generators and controlled tools establish lifecycle behavior,
   concurrency, isolation, cancellation, and overlap. A synthetic delay
   benchmark measures harness overlap and overhead; it cannot establish model
   quality or GPU speed.
2. Real-model end-to-end runs establish whether the model selects useful tasks,
   follows the tool protocol, finds evidence, and integrates results correctly.
   Grade against independent expected facts or executable acceptance checks.
   Compare repeated trials and report the exact model revision and device.

The reusable [MultiAgentBench harness](../eng/validation/MultiAgentBench/README.md)
supports both modes. For example:

```sh
dotnet run --project eng/validation/MultiAgentBench -c Release -- --iterations 8 --warmup 1 --work-ms 80 --out artifacts/multi-agent/scripted.json
dotnet run --project eng/validation/MultiAgentBench -c Release -- --endpoint http://localhost:8000/v1/chat/completions --model MODEL --tensorsharp --iterations 5 --warmup 1 --out artifacts/multi-agent/real-model.json
```

The scripted mode performs real skill reads with a controlled analysis delay.
The endpoint mode lets the actual model choose whether to delegate while the
benchmark owns orchestration and fixture tools; the endpoint supplies generation.
Follow the harness README when configuring a TensorSharp endpoint so that the
server does not independently orchestrate those calls. To evaluate the server's
own orchestration, compare API requests with `multi_agent: false` and
`multi_agent: true` separately. The harness's nine-fact recall and unsupported-fact
counts are narrow, reproducible quality proxies; they do not establish
superiority on general coding or reasoning tasks.

Multiple agents can improve coverage and reduce wall time when useful work
overlaps. They also consume additional generation tokens and context memory.
On a single saturated inference device, scheduling more conversations may
increase latency. Improved quality and performance are workload-dependent
outcomes to measure, not guarantees of enabling the feature. Unavailable models
or devices, skipped scenarios, and synthetic-only measurements must be recorded
as such, never counted as successful real-model validation.

Generated validation reports and logs belong in ignored `docs/validation/` or
`artifacts/`; reusable validation programs belong in `eng/`, with fixtures in
their test project. This implementation is TensorSharp-owned managed host
code. It does not modify ggml upstream sources or require a patched native
dependency, and makes no claim to a new native KV-sharing optimization.
