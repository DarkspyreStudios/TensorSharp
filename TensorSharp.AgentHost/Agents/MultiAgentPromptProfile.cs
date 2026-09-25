// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents;

/// <summary>A possible agent's governing messages and offered tools, without an
/// identity, assigned task, or conversation. Hosts can render these profiles to
/// identify exact public prefix checkpoints before any child is spawned.</summary>
public sealed class MultiAgentPromptProfile
{
    public MultiAgentPromptProfile(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolFunction> tools)
    {
        Messages = messages.ToArray();
        Tools = tools.ToArray();
    }

    public IReadOnlyList<ChatMessage> Messages { get; }
    public IReadOnlyList<ToolFunction> Tools { get; }
}
