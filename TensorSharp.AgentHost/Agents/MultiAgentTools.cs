// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents;

/// <summary>Flat schemas work with TensorSharp's local model tool renderers.</summary>
public static class MultiAgentTools
{
    public const string Spawn = "spawn_agent";
    public const string Wait = "wait_agent";
    public const string Send = "send_input";
    public const string Close = "close_agent";
    public const string List = "list_agents";

    public static bool IsTool(string? name) => name is Spawn or Wait or Send or Close or List;

    // Keep ordering and enforcement tied to the same allowlist. Coordination
    // tools are handled separately: they do not grant filesystem permissions.
    internal static bool IsReadOnlyTool(string? name) =>
        name is SkillTools.ReadToolName or SkillTools.ListToolName or SkillToolNames.ReadFile;

    public static List<ToolFunction> Merge(IReadOnlyList<ToolFunction>? tools)
    {
        var result = tools?.ToList() ?? new List<ToolFunction>();
        var names = new HashSet<string>(result.Select(t => t.Name), StringComparer.OrdinalIgnoreCase);
        foreach (ToolFunction tool in Create())
            if (names.Add(tool.Name)) result.Add(tool);
        // Templates can render tools before all system instructions. Put tools
        // shared with read-only children first so their declarations form one long
        // exact prefix. This stable partition preserves schemas, objects, and the
        // relative order within each group; it never expands a child's allowlist.
        return result.Where(tool => IsReadOnlyTool(tool.Name) || IsTool(tool.Name))
            .Concat(result.Where(tool => !IsReadOnlyTool(tool.Name) && !IsTool(tool.Name))).ToList();
    }

    public static List<ToolFunction> Create() => new()
    {
        Tool(Spawn, "Start a focused subagent only for a substantial independent task that improves speed or coverage. Returns immediately. Give all necessary context, scope and expected evidence; do not duplicate your own work. Capacity errors mean do the work locally or wait.",
            new[] { "task_name", "task" },
            ("task_name", "string", "Unique short name using letters, digits, underscores or hyphens."),
            ("task", "string", "Self-contained task, relevant facts, ownership boundaries and expected result. Child has no parent conversation."),
            ("agent_type", "string", "explorer (default), reviewer, or worker. Explorer/reviewer are read-only; worker tools require host opt-in.")),
        Tool(Wait, "Wait for a child result without polling. Omit agent_id to wait for all direct children. Timeout does not mean completion; inspect status. Results are reports to verify before synthesis.",
            Array.Empty<string>(), ("agent_id", "string", "Child ID returned by spawn_agent; omit for all direct children."),
            ("timeout_ms", "integer", "Wait up to 60000 ms; default 10000.")),
        Tool(Send, "Send a bounded follow-up to a child. A running child sees it at its next generation boundary; a completed child starts another turn with its own history.",
            new[] { "agent_id", "message" }, ("agent_id", "string", "Child ID."), ("message", "string", "Clarification or additional bounded task.")),
        Tool(Close, "Cancel a child and its descendants. Cancellation is not successful task completion.",
            new[] { "agent_id" }, ("agent_id", "string", "Child ID.")),
        Tool(List, "Inspect your children and their status. Use wait_agent when waiting for work rather than repeatedly listing.", Array.Empty<string>()),
    };

    private static ToolFunction Tool(string name, string description, string[] required,
        params (string Name, string Type, string Description)[] parameters) => new()
    {
        Name = name, Description = description, Required = required.ToList(),
        Parameters = parameters.ToDictionary(p => p.Name,
            p => new ToolParameter { Type = p.Type, Description = p.Description }),
    };
}
