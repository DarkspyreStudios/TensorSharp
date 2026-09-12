// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the trailing-token rewind in the executor's live-cache continuation.
///
/// <para>
/// The rule this guards: a turn can end on a control token that the chat template
/// never re-renders as history. Gemma 4 answers a tool call by emitting
/// <c>&lt;|tool_response&gt;</c>; the engine forwards it into the KV cache, and the
/// next render frames that boundary as <c>&lt;turn|&gt;\n&lt;|turn&gt;tool</c>
/// instead. The live cache is then an exact prefix of the new prompt for every token
/// but its last, and requiring an EXACT match threw the whole conversation away — an
/// Agent Skills lookup re-prefilled from token 0, and (the symptom the executor's own
/// comment used to record) an EOS-terminated turn reported 0% reuse where a
/// max_tokens-terminated turn of the same conversation reported ~95%.
/// </para>
/// <para>
/// These tests exercise the decision arithmetic directly rather than through a model:
/// the executor's <c>ComputeLiveContinuationLcp</c> needs a loaded GGUF and a live
/// KV cache, so what is pinned here is the contract every caller of that decision
/// depends on — how far a rewind may go, and that it stays bounded. The end-to-end
/// behaviour is verified against a real model; see docs/agent_skills.md for the
/// measured numbers.
/// </para>
/// </summary>
public class LiveCacheRewindTests
{
    /// <summary>
    /// Mirrors the executor's decision so the boundary conditions are testable without
    /// a model. Kept deliberately small and in step with
    /// <c>BatchExecutor.ComputeLiveContinuationLcp</c>: if that method's rules change,
    /// this must change with it or the tests below stop meaning anything.
    /// </summary>
    private static int Decide(
        IReadOnlyList<int> prompt,
        IReadOnlyList<int> liveCache,
        int pooledCap,
        bool canTruncate,
        int maxRewind = 16,
        int granularity = 1)
    {
        int liveLen = liveCache.Count;
        if (liveLen <= pooledCap)
            return 0;

        int lcp = 0;
        int limit = Math.Min(liveLen, prompt.Count);
        while (lcp < limit && prompt[lcp] == liveCache[lcp])
            lcp++;

        if (prompt.Count <= lcp)
        {
            // The cache holds the whole prompt: keep all but its last token and
            // rewind the rest, which is only possible for a model that can rewind.
            if (!canTruncate)
                return 0;
            lcp = prompt.Count - 1;
            if (lcp <= 0)
                return 0;
        }
        if (lcp == liveLen)
            return liveLen;

        // A rewind has to land where the model can put its head: DeepSeek V4.1 stops only
        // on a compression-block boundary. Aligned DOWN before the rewind is measured, so
        // the limit below and the truncate at execution time mean the same target.
        if (granularity > 1 && lcp % granularity != 0)
        {
            lcp -= lcp % granularity;
            if (lcp <= 0)
                return 0;
        }

        int rewind = liveLen - lcp;
        if (rewind > maxRewind)
            return 0;
        if (!canTruncate)
            return 0;
        if (lcp <= pooledCap)
            return 0;
        return lcp;
    }

    private static List<int> Tokens(int count, int seed = 1)
        => Enumerable.Range(0, count).Select(i => seed * 1000 + i).ToList();

    // ---- a model whose head only stops on a block boundary -----------------

    /// <summary>
    /// DeepSeek V4.1 can only rewind to a multiple of its widest compression ratio, so the
    /// matched prefix is rounded DOWN. A plan that promised an odd position would be
    /// declined at execution time, after admission had already announced the reuse.
    /// </summary>
    [Fact]
    public void GranularModel_RoundsTheMatchedPrefixDownToABoundary()
    {
        var cache = Tokens(2000);
        // Diverges at 1995 (odd), 5 tokens of trailing rewind.
        var prompt = Tokens(1995).Concat(Tokens(40, seed: 2)).ToList();

        Assert.Equal(1995, Decide(prompt, cache, pooledCap: 0, canTruncate: true));
        Assert.Equal(1994, Decide(prompt, cache, pooledCap: 0, canTruncate: true, granularity: 2));
    }

    /// <summary>
    /// Aligning down never costs more than granularity-1 tokens, and never turns a
    /// continuation into a re-prefill while a usable prefix remains - unless it pushes the
    /// rewind past the depth limit, which is the honest outcome at the boundary.
    /// </summary>
    [Fact]
    public void GranularModel_AligningIsVisibleAtTheRewindLimitRatherThanSilent()
    {
        // An aligned target already inside the limit is unaffected.
        var evenCache = Tokens(2000);
        var evenPrompt = Tokens(1984).Concat(Tokens(40, seed: 2)).ToList();
        Assert.Equal(1984, Decide(evenPrompt, evenCache, pooledCap: 0, canTruncate: true, granularity: 2));

        // The one case where alignment changes the verdict: a 16-token rewind that
        // alignment widens to 17, one past the limit. Declining is the honest outcome -
        // the alternative is a plan that announces a reuse the model will refuse.
        var oddCache = Tokens(2001);
        var oddPrompt = Tokens(1985).Concat(Tokens(40, seed: 2)).ToList();
        Assert.Equal(1985, Decide(oddPrompt, oddCache, pooledCap: 0, canTruncate: true));
        Assert.Equal(0, Decide(oddPrompt, oddCache, pooledCap: 0, canTruncate: true, granularity: 2));
    }

    [Fact]
    public void GranularModel_ExactExtensionIsNeverAligned()
    {
        // The prompt extends the cache exactly: nothing is truncated, so the odd cache
        // length must be kept whole rather than rounded into a real rewind.
        var cache = Tokens(1999);
        var prompt = Tokens(1999).Concat(Tokens(40, seed: 2)).ToList();
        Assert.Equal(1999, Decide(prompt, cache, pooledCap: 0, canTruncate: true, granularity: 2));
    }

    // ---- the case the fix exists for ---------------------------------------

    [Fact]
    public void OneTrailingControlToken_IsRewoundInsteadOfDiscardingTheWholePrefix()
    {
        // Round 1 left 1984 tokens resident, the last being a control token the
        // template does not reproduce. Round 2's prompt reproduces the first 1983 and
        // then diverges. Before the fix this returned 0 and re-prefilled everything.
        List<int> live = Tokens(1984);
        List<int> prompt = Tokens(1983);
        prompt.Add(999_999);                 // the template's framing, not the model's
        prompt.AddRange(Tokens(2200, seed: 7));   // the freshly fetched skill body

        Assert.Equal(1983, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AnExactPrefix_StillNeedsNoRewindAtAll()
    {
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000);
        prompt.AddRange(Tokens(50, seed: 7));

        Assert.Equal(1000, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    // ---- the bounds that keep it safe --------------------------------------

    [Fact]
    public void ARewindLongerThanTheLimit_IsDeclined()
    {
        // A long divergence is not a stale control token — it is an edited turn or a
        // changed system prompt, where a clean re-prefill is the correct answer. On a
        // sliding-window model the cache is also circular, so rewinding far is not
        // faithful even when it is cheap.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(950);
        prompt.AddRange(Tokens(200, seed: 7));   // diverges 50 tokens from the end

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void ARewindAtExactlyTheLimit_IsStillAllowed()
    {
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(984);
        prompt.AddRange(Tokens(200, seed: 7));   // 16-token rewind

        Assert.Equal(984, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AModelThatCannotRewind_IsNeverAskedTo()
    {
        // Recurrent / SSM state has no reverse: Qwen 3.5's GatedDeltaNet cannot be
        // rolled back to an earlier position, so for those the only valid reuse is an
        // exact prefix. SupportsKVCacheTruncation is what says so.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(999);
        prompt.Add(999_999);
        prompt.AddRange(Tokens(200, seed: 7));

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: false));
        Assert.Equal(999, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AMatchedPrefixInsideThePooledCap_IsLeftToThePooledPath()
    {
        // Below the cap the pooled snapshot already covers it, and adopting the live
        // cache would buy nothing while pinning the model to one sequence.
        List<int> live = Tokens(600);
        List<int> prompt = Tokens(500);
        prompt.AddRange(Tokens(200, seed: 7));

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void APromptWithNoNewSuffix_RewindsOneTokenSoThereIsSomethingToForward()
    {
        // Nothing left to forward as-is: the cache holds exactly this prompt. Adopting
        // it whole would leave the sequence with no work and the caller expecting a
        // token, so the last prompt token is given back to the model instead.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000);

        Assert.Equal(999, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void ADivergenceAtTheVeryStart_IsDeclined()
    {
        // A different system prompt (a changed skill selection, for instance) shares
        // nothing, and must re-prefill rather than rewind almost the whole cache.
        List<int> live = Tokens(1000);
        List<int> prompt = Tokens(1000, seed: 7);

        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void APromptTheCacheAlreadyHoldsEntirely_RewindsToItsLastToken()
    {
        // The same question asked again in a new chat, or a regenerated turn: the
        // cache holds the prompt AND the answer it produced. Everything up to the
        // prompt's last token is kept; that token is re-forwarded for fresh logits.
        var prompt = Tokens(1000);
        var live = new List<int>(prompt) { 9001, 9002, 9003 };   // the answer
        Assert.Equal(999, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void APromptTheCacheAlreadyHoldsEntirely_IsStillBoundedByTheRewindLimit()
    {
        var prompt = Tokens(1000);
        var live = new List<int>(prompt);
        live.AddRange(Tokens(40, seed: 7));   // a long answer: 41 tokens back is too far
        Assert.Equal(0, Decide(prompt, live, pooledCap: 512, canTruncate: true));
    }

    [Fact]
    public void AModelThatCannotRewind_DeclinesAPromptTheCacheAlreadyHoldsEntirely()
    {
        var prompt = Tokens(1000);
        var live = new List<int>(prompt) { 9001 };
        Assert.Equal(0, Decide(prompt, live, pooledCap: 0, canTruncate: false));
    }
}
