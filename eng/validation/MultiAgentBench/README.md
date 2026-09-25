# MultiAgentBench

Runs the same independent service-review fixture through `SkillAgentLoop` with a
single parent and with bounded child agents. Both paths must return all nine
current facts, and neither may introduce obsolete facts. Reports per-sample wall
time, p50/p95, exact fact coverage, unsupported facts, failed tool invocations,
generation counts, prompt/completion token sums (when reported by the endpoint),
child count, and exhausted round budgets. Run order alternates;
warmups are excluded from summaries. Reports belong in ignored `artifacts/` or
`docs/validation/`.

```
dotnet run --project eng/validation/MultiAgentBench -c Release -- --iterations 8 --warmup 1 --work-ms 80 --out artifacts/multi-agent/scripted.json
```

The default **scripted orchestration** mode uses real skill-file tools and actual
orchestration, with deterministic generators and an injected delay representing
independent analysis work. It verifies concurrency, evidence preservation and
synthesis. It does **not** measure model inference or establish better reasoning
quality. A speedup in this mode is only a scheduling result. The same fact coverage
in both arms establishes fixture parity, not a quality improvement.

To measure a real model, supply the full OpenAI-compatible chat-completions URL and
model name. The model chooses whether to delegate; child count records what it
actually did. `MULTI_AGENT_BENCH_API_KEY` optionally supplies a bearer token. No
credentials are written to the report.

```
dotnet run --project eng/validation/MultiAgentBench -c Release -- --endpoint http://localhost:8000/v1/chat/completions --model your-model --tensorsharp --iterations 5 --warmup 1 --out artifacts/multi-agent/real-model.json
```

The endpoint supplies generation only; the benchmark host owns fixture reads and
child orchestration. `--tensorsharp` sends `multi_agent:false` to disable delegation
owned by the endpoint in both arms, ensuring the single-agent baseline stays
single. Omit this flag for other OpenAI-compatible endpoints, and configure them
to leave supplied tool calls to this client and disable any autonomous agent
orchestration. Remote calls send these fixture documents and orchestration instructions.
Use the same model, hardware, inference settings and load for both arms. Record
model revision, device and server flags alongside the report; this executable
cannot discover them reliably. Native dependencies are not modified by this
benchmark.

Add `--distractor-lines 60` for longer service documents containing superseded
audit proposals before the current evidence. Generated sources are saved beside
the report under `fixtures/`. Both arms use identical sources. This adds context
volume and distractors; the quality score remains narrow exact-fact extraction.

This is a small extraction task with objective but narrow quality metrics. For
larger coding/research quality claims, add representative tasks and independent
correctness checks, compare equal budgets, and inspect errors. A saturated single
device can make concurrent inference slower. Do not count an unavailable endpoint
as a passing test, and do not generalize scripted results to model performance.
The executable returns a nonzero exit code for missing/unsupported facts, tool
errors or exhausted round budgets. It deliberately does not assert a speedup.
