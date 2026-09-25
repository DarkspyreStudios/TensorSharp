using System.Collections;
using System.Diagnostics;
using System.Reflection;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

// These diagnostics deliberately bypass scheduling. They isolate prefix chunk
// shape from checkpoint copying, using only the production holder APIs for state.
static class StateDiagnostics
{
    public static StateDiagnosticResult Run(Qwen35Model model, FixtureSet fixtures, int steps)
    {
        var sequences = (IBatchedPagedModel)model;
        if (!sequences.SupportsPerSequenceFusedForward || !sequences.SupportsPrefixCheckpoints)
            throw new InvalidOperationException("Diagnostics require per-sequence fused caches and exact checkpoints.");
        PromptFixture child = fixtures.Children[1];
        int ancestor = fixtures.Parent.PublicCheckpointBoundaries.Single();
        int aligned = ancestor / 64 * 64;
        if (aligned <= 0 || aligned >= child.SharedPrefixTokens)
            throw new InvalidOperationException("The fixture cannot exercise the aligned control.");
        var variants = new List<StateVariant>();
        foreach (var specification in new[]
        {
            (Name: "full-public-cold", Split: 0, Restore: false, Capture: false),
            (Name: "split-no-capture", Split: ancestor, Restore: false, Capture: false),
            (Name: "split-cold", Split: ancestor, Restore: false, Capture: true),
            (Name: "restored-parent-ancestor", Split: ancestor, Restore: true, Capture: true),
            (Name: "aligned-split-cold", Split: aligned, Restore: false, Capture: true),
            (Name: "aligned-restored-parent", Split: aligned, Restore: true, Capture: true),
        })
        {
            StateVariant variant = RunVariant(model, sequences, fixtures.Parent, child,
                specification.Name, specification.Split, specification.Restore, specification.Capture, steps);
            variants.Add(variant);
            Console.WriteLine($"[state-diagnostic] {variant.Name} split={variant.SplitTokens} generated={variant.GeneratedTokens.Length} first={variant.FirstTopTokens.FirstOrDefault()?.TokenId} margin={variant.FirstLogitMargin:G8} error={variant.Error != null}");
        }
        var comparisons = new List<LogitComparison>();
        Compare("pure-prefix-partition", "full-public-cold", "split-no-capture");
        Compare("intermediate-capture-flush", "split-no-capture", "split-cold");
        Compare("partition-plus-capture", "full-public-cold", "split-cold");
        Compare("same-partition-parent-clone", "split-cold", "restored-parent-ancestor");
        Compare("aligned-prefix-partition", "full-public-cold", "aligned-split-cold");
        Compare("same-aligned-partition-parent-clone", "aligned-split-cold", "aligned-restored-parent");
        bool complete = variants.All(variant => variant.Error == null);
        bool samePartitionExact = complete && comparisons.Where(comparison => comparison.Name.StartsWith("same-", StringComparison.Ordinal))
            .All(comparison => comparison.ExactFirstLogits && comparison.ExactGreedyTokens);
        bool captureExact = complete && comparisons.Where(comparison => comparison.Name == "intermediate-capture-flush")
            .All(comparison => comparison.ExactFirstLogits && comparison.ExactGreedyTokens);
        return new StateDiagnosticResult(complete, samePartitionExact, captureExact, variants, comparisons,
            "Diagnostic only; regular workflow validation/performance remains unqualified. All variants clone their full child public checkpoint before the same proposal-B private suffix. The split-no-capture control performs no intermediate capture or adoption, separating pure prefill partition effects from checkpoint synchronization. Other split controls capture/synchronize intermediate state but do not adopt it; restored variants clone the parent's exact intermediate snapshot after the parent has finished its prompt and up to the output cap. Same-partition comparisons isolate adoption/state copying. Each greedy continuation ends on EOS or the configured cap. FirstLogits stores every first-token vocabulary logit, with top10/margin and pair metrics. Later greedy paths may diverge, so later logits are not compared without teacher forcing. The aligned control rounds down the common prefix to 64 tokens; it is a diagnostic boundary, not the production cache nomination. No tolerance silently converts an inexact comparison into a pass.");

        void Compare(string name, string expected, string observed)
        {
            StateVariant a = variants.Single(variant => variant.Name == expected);
            StateVariant b = variants.Single(variant => variant.Name == observed);
            if (a.Error != null || b.Error != null) return;
            LogitComparison comparison = CompareLogits(name, a, b);
            comparisons.Add(comparison);
            Console.WriteLine($"[state-diagnostic] {name} exact_logits={comparison.ExactFirstLogits} exact_tokens={comparison.ExactGreedyTokens} max_abs={comparison.MaxAbsoluteError:G8} centered_nrmse={comparison.CenteredNormalizedRmse:G8} kl={comparison.SoftmaxKl:G8} tv={comparison.TotalVariation:G8}");
        }
    }

    private static StateVariant RunVariant(Qwen35Model model, IBatchedPagedModel sequences, PromptFixture parent,
        PromptFixture child, string name, int split, bool restoreParent, bool captureIntermediate, int steps)
    {
        var result = new StateVariant(name, split, restoreParent, captureIntermediate, child.SharedPrefixTokens, child.Tokens.Count);
        string tag = "parent-state-" + Guid.NewGuid().ToString("N");
        string producerId = tag + "-producer", parentId = tag + "-parent", childId = tag + "-child";
        string ancestorKey = tag + "-ancestor", parentFullKey = tag + "-parent-public", fullKey = tag + "-public";
        var clock = Stopwatch.StartNew();
        try
        {
            sequences.RestorePrimaryCache();
            model.ResetKVCache();
            if (restoreParent)
            {
                Require(split > 0 && parent.Tokens.Take(split).SequenceEqual(child.Tokens.Take(split)), "Parent prefix does not match child.");
                Require(sequences.BindSequenceCache(parentId), "Parent diagnostic holder was not fresh.");
                model.PrepareForPrefill(parent.Tokens.Count + steps + 2);
                result.InitialProducerCapacity = Capacity(model, parentId);
                Forward(parent.Tokens, 0, split, "parent-ancestor");
                Capture(ancestorKey);
                Forward(parent.Tokens, split, parent.SharedPrefixTokens - split, "parent-public-tail");
                Capture(parentFullKey);
                float[] parentLogits = Forward(parent.Tokens, parent.SharedPrefixTokens,
                    parent.Tokens.Count - parent.SharedPrefixTokens, "parent-private-tail");
                result.ParentGeneratedAfterCapture = Generate(parentLogits).ToArray();
                sequences.RestorePrimaryCache();
                sequences.OnSequenceReleased(parentId);
                Require(sequences.TryCloneRetainedCache(ancestorKey, producerId), "Parent ancestor clone declined.");
                Require(!sequences.BindSequenceCache(producerId), "Restored ancestor holder appeared fresh.");
                model.PrepareForPrefill(child.Tokens.Count + steps + 2);
                Forward(child.Tokens, split, child.SharedPrefixTokens - split, "child-public-tail");
            }
            else
            {
                Require(sequences.BindSequenceCache(producerId), "Cold public producer was not fresh.");
                model.PrepareForPrefill(child.Tokens.Count + steps + 2);
                result.InitialProducerCapacity = Capacity(model, producerId);
                if (split > 0)
                {
                    Forward(child.Tokens, 0, split, "cold-ancestor");
                    // Match the synchronization and capture side effects, without
                    // adopting the resulting checkpoint into this live holder.
                    if (captureIntermediate) Capture(ancestorKey);
                    Forward(child.Tokens, split, child.SharedPrefixTokens - split, "cold-public-tail");
                }
                else Forward(child.Tokens, 0, child.SharedPrefixTokens, "cold-full-public");
            }
            result.CompletedPublicProducerCapacity = Capacity(model, producerId);
            Capture(fullKey);
            sequences.RestorePrimaryCache();
            sequences.OnSequenceReleased(producerId);
            Require(sequences.TryCloneRetainedCache(fullKey, childId), "Full child checkpoint clone declined.");
            Require(!sequences.BindSequenceCache(childId), "Restored child public holder appeared fresh.");
            model.PrepareForPrefill(child.Tokens.Count + steps + 2);
            result.FinalChildCapacity = Capacity(model, childId);
            float[] logits = Forward(child.Tokens, child.SharedPrefixTokens,
                child.Tokens.Count - child.SharedPrefixTokens, "child-private-tail");
            Require(logits.Length > 1 && logits.All(float.IsFinite), "First logits are empty or nonfinite.");
            result.FirstLogits = (float[])logits.Clone();
            int[] top = Enumerable.Range(0, logits.Length).OrderByDescending(index => logits[index]).Take(10).ToArray();
            double denominator = logits.Sum(logit => Math.Exp((double)logit - logits[top[0]]));
            result.FirstTopTokens = top.Select(index => new TopLogit(index, model.Tokenizer.Decode(new List<int> { index }),
                logits[index], Math.Exp((double)logits[index] - logits[top[0]]) / denominator)).ToArray();
            result.FirstLogitMargin = (double)logits[top[0]] - logits[top[1]];
            result.GeneratedTokens = Generate(logits).ToArray();
            result.OutputText = model.Tokenizer.Decode(result.GeneratedTokens.ToList());
            result.FinishReason = model.Tokenizer.IsEos(result.GeneratedTokens[^1]) ? "eos" : "length";
            Require(result.GeneratedTokens.Any(token => !model.Tokenizer.IsEos(token)), "Diagnostic generated only EOS.");
        }
        catch (Exception error) { result.Error = error.ToString(); }
        finally
        {
            sequences.RestorePrimaryCache();
            foreach (string id in new[] { parentId, producerId, childId }) sequences.OnSequenceReleased(id);
            foreach (string key in new[] { ancestorKey, parentFullKey, fullKey }) sequences.DiscardRetainedCache(key);
            result.Milliseconds = clock.Elapsed.TotalMilliseconds;
        }
        return result;

        float[] Forward(IReadOnlyList<int> tokens, int start, int count, string phase)
        {
            Require(count > 0, $"Empty forward phase {phase}.");
            result.PrefillPhases.Add(new ForwardPhase(phase, start, count));
            return (float[])model.Forward(tokens.Skip(start).Take(count).ToArray()).Clone();
        }
        void Capture(string key) => Require(sequences.TryCheckpointActiveCache(key), $"Checkpoint {key} declined.");
        List<int> Generate(float[] logits)
        {
            var tokens = new List<int>();
            for (int step = 0; step < steps; step++)
            {
                Require(logits.All(float.IsFinite), $"Nonfinite greedy logits at step {step}.");
                int token = ArgMax(logits);
                tokens.Add(token);
                if (model.Tokenizer.IsEos(token) || step + 1 == steps) break;
                logits = (float[])model.Forward(new[] { token }).Clone();
            }
            return tokens;
        }
    }

    private static LogitComparison CompareLogits(string name, StateVariant expected, StateVariant observed)
    {
        float[] a = expected.FirstLogits, b = observed.FirstLogits;
        Require(a.Length == b.Length && a.Length > 1, "Diagnostic logit shapes differ.");
        double dot = 0, normA = 0, normB = 0, squareError = 0, errorSum = 0, sum = 0, maxError = 0;
        bool exact = true;
        for (int i = 0; i < a.Length; i++)
        {
            double error = (double)b[i] - a[i];
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i]; normB += (double)b[i] * b[i];
            sum += a[i]; squareError += error * error; errorSum += error;
            maxError = Math.Max(maxError, Math.Abs(error));
            exact &= BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]);
        }
        double variance = Math.Max(1e-20, normA / a.Length - Math.Pow(sum / a.Length, 2));
        double mse = squareError / a.Length, meanError = errorSum / a.Length;
        int topA = ArgMax(a), topB = ArgMax(b);
        double logSumA = Math.Log(a.Sum(logit => Math.Exp((double)logit - a[topA])));
        double logSumB = Math.Log(b.Sum(logit => Math.Exp((double)logit - b[topB])));
        double kl = 0, tv = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double logP = (double)a[i] - a[topA] - logSumA, logQ = (double)b[i] - b[topB] - logSumB;
            double p = Math.Exp(logP), q = Math.Exp(logQ);
            kl += p * (logP - logQ); tv += .5 * Math.Abs(p - q);
        }
        return new LogitComparison(name, expected.Name, observed.Name, exact,
            expected.GeneratedTokens.SequenceEqual(observed.GeneratedTokens) && expected.FinishReason == observed.FinishReason,
            maxError, Math.Sqrt(mse / variance), Math.Sqrt(Math.Max(0, mse - meanError * meanError) / variance),
            meanError, dot / Math.Sqrt(Math.Max(1e-20, normA * normB)), Math.Max(0, kl), tv, topA, topB,
            expected.FirstLogitMargin, observed.FirstLogitMargin);
    }
    private static int Capacity(Qwen35Model model, string id)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        // Binding loads the holder into active fields; growth is written back to
        // the dictionary only when another holder (or primary) is restored.
        if ((string?)typeof(Qwen35Model).GetField("_activeFusedKey", flags)!.GetValue(model) == id)
            return (int)typeof(Qwen35Model).GetField("_kvCacheCapacity", flags)!.GetValue(model)!;
        var holders = (IDictionary)typeof(Qwen35Model).GetField("_fusedHolders", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        object holder = holders[id]!;
        return (int)holder.GetType().GetField("KvCapacity")!.GetValue(holder)!;
    }
    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
}

sealed record StateDiagnosticResult(bool Completed, bool SamePartitionCloneExact, bool IntermediateCaptureExact, List<StateVariant> Variants,
    List<LogitComparison> Comparisons, string Scope);
sealed class StateVariant(string name, int splitTokens, bool parentRestored, bool intermediateCaptured, int publicTokens, int promptTokens)
{
    public string Name { get; } = name;
    public int SplitTokens { get; } = splitTokens;
    public bool ParentRestored { get; } = parentRestored;
    public bool IntermediateCheckpointCaptured { get; } = intermediateCaptured;
    public int PublicTokens { get; } = publicTokens;
    public int PromptTokens { get; } = promptTokens;
    public int InitialProducerCapacity { get; set; }
    public int CompletedPublicProducerCapacity { get; set; }
    public int FinalChildCapacity { get; set; }
    public List<ForwardPhase> PrefillPhases { get; } = [];
    public float[] FirstLogits { get; set; } = [];
    public TopLogit[] FirstTopTokens { get; set; } = [];
    public double? FirstLogitMargin { get; set; }
    public int[] ParentGeneratedAfterCapture { get; set; } = [];
    public int[] GeneratedTokens { get; set; } = [];
    public string? OutputText { get; set; }
    public string? FinishReason { get; set; }
    public double Milliseconds { get; set; }
    public string? Error { get; set; }
}
sealed record ForwardPhase(string Name, int Position, int Tokens);
sealed record TopLogit(int TokenId, string TokenText, float Logit, double Probability);
sealed record LogitComparison(string Name, string Expected, string Observed, bool ExactFirstLogits, bool ExactGreedyTokens,
    double MaxAbsoluteError, double NormalizedRmse, double CenteredNormalizedRmse, double MeanError, double Cosine,
    double SoftmaxKl, double TotalVariation, int ExpectedTopToken, int ObservedTopToken,
    double? ExpectedMargin, double? ObservedMargin);
