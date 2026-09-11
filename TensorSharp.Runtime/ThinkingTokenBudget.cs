// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Immutable sampling policy for a prompt that already opened its reasoning
    /// channel. The closing token counts toward the original output limit.
    /// Tracking belongs to each sampler, so configurations may be cloned safely.
    /// </summary>
    public sealed class ThinkingTokenBudget
    {
        public ThinkingTokenBudget(int tokenLimit, int endTokenId, bool closeOnRepetition = false)
        {
            if (tokenLimit <= 0) throw new ArgumentOutOfRangeException(nameof(tokenLimit));
            if (endTokenId < 0) throw new ArgumentOutOfRangeException(nameof(endTokenId));
            TokenLimit = tokenLimit;
            EndTokenId = endTokenId;
            CloseOnRepetition = closeOnRepetition;
        }

        public int TokenLimit { get; }
        public int EndTokenId { get; }
        /// <summary>Allow the repetition guard to close an open reasoning channel
        /// through ordinary sampling instead of stopping the entire answer.</summary>
        public bool CloseOnRepetition { get; }
    }
}
