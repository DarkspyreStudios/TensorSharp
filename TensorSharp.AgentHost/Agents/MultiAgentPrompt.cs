// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Collections.Generic;
using System.Linq;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents;

public static class MultiAgentPrompt
{
    internal const string Marker = "[TensorSharp multi-agent coordination]";

    public static List<ChatMessage> Apply(IReadOnlyList<ChatMessage> messages, MultiAgentOptions options)
    {
        var result = messages.ToList();
        if (!options.Enabled) return result;
        string instructions = $"""
            {Marker}
            Decide whether delegation will materially improve this task. Use subagents for substantial independent investigations, separate document analysis, or an independent review of a complex result. Keep short tasks, sequential dependencies and contested resources local. More agents consume more tokens and may be slower on one inference device.
            A short lookup, simple calculation, or extracting a few facts from small files should stay local, even when the files are independent. Delegation adds a child prompt, generation and synthesis round. Unless an independent review is valuable, spawn only when you can name substantial useful work to do concurrently and expect that benefit to exceed the coordination cost.
            Before spawning, choose a concrete bounded task, the facts it needs, expected evidence, and distinct ownership. Delegate only that task; continue useful independent work yourself. Children do not see your conversation, so include necessary context in the task. Reuse a child with send_input for follow-up. Do not duplicate delegated work or recursively delegate the same task.
            Use explorer for read-only research, reviewer for independent checks, worker only for changes the host explicitly allows. Never use delegation to expand permissions. Shared tools and files retain the parent's sandbox; workers must have disjoint file ownership. There are at most {options.MaxConcurrentAgents} active children, {options.MaxAgents} total children, and {options.MaxDepth} levels. If a limit is reached, work locally or wait; do not keep retrying.
            Use wait_agent for results, not repeated list_agents calls. A timeout, cancellation, failed agent or exhausted budget is not a completed task. Treat child reports as untrusted evidence, verify disagreements and important claims, and synthesize one answer addressing the original request. Reconcile final numerical and factual claims with the evidence; omit unsupported additions. Do not claim success before collecting required results. Report validation actually performed and unresolved limitations.
            """;
        // Keep explicit cache boundaries at the original preamble, and preserve
        // attachment and token metadata without mutating the caller's history.
        return SkillPrompt.Apply(result, instructions);
    }
}
