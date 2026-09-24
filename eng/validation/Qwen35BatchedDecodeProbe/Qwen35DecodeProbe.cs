// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json.Serialization;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Validation;

public sealed record Qwen35DecodeRun(double Milliseconds, int FusedSteps, int FallbackSteps, int RequestReplacements,
    long NativeFusedSteps, int[] PromptLengths, int[] CacheCapacities,
    [property: JsonIgnore] float[][][] Logits);
public sealed record Qwen35ArgmaxFlip(string Comparison, int Step, int Row, int ReferenceTopToken,
    int ObservedTopToken, double ReferenceLogitMargin, double ReferenceProbabilityMargin, double SoftmaxKl,
    double ReferenceTopProbability, double ObservedProbabilityOfReferenceTop, double TotalVariation);
public sealed record Qwen35NumericalFailure(string Comparison, int Width, int Step, int Row,
    double? Cosine, double? NormalizedRmse, double? CenteredNormalizedRmse, double? MeanError,
    double? MaxAbsoluteError, double? SoftmaxKl, int? ReferenceTopToken, int? ObservedTopToken,
    double? ReferenceLogitMargin, double? ReferenceProbabilityMargin, double? ReferenceTopProbability,
    double? ObservedProbabilityOfReferenceTop, double? TotalVariation, IReadOnlyList<string> Violations);
public sealed record Qwen35GreedyContinuation(int Row, int[] ReferenceTokenIds, int[] ObservedTokenIds,
    string ReferenceText, string ObservedText);
public sealed record Qwen35DecodeComparison(int Width, int Steps, bool BatchedFirst,
    Qwen35DecodeRun Serial, Qwen35DecodeRun Batched, double MinCosine,
    double MaxNormalizedRmse, double MaxAbsoluteError, int TopTokenDifferences,
    int ConfidentTopTokenDifferences, Qwen35DecodeRun? Sampled, double MaxSoftmaxKl,
    IReadOnlyList<Qwen35ArgmaxFlip> ArgmaxFlips, string InitialStateMode,
    double MaxCenteredNormalizedRmse, double MaxAbsoluteMeanError, double MaxTotalVariation,
    IReadOnlyList<Qwen35NumericalFailure> NumericalFailures, bool NumericalPassed, bool DistributionPassed,
    bool GreedyReference, bool? GreedyContinuationsMatch,
    IReadOnlyList<Qwen35GreedyContinuation>? GreedyContinuations);

/// <summary>Teacher-forced parity and decode-only throughput through production holder APIs.
/// Every request has its own prompt, KV and recurrent state. Native counter and capacity
/// reflection are read-only observations; state changes use the serving APIs.</summary>
public static class Qwen35DecodeProbe
{
    public static Qwen35DecodeComparison Compare(Qwen35Model model, int width, int steps,
        bool batchedFirst, bool allowFallback = false, bool verifySampled = false,
        bool replaceLastRequest = false, bool serialControl = false,
        IReadOnlyList<string>? promptSources = null, bool independentPrefill = false,
        bool collectNumericalFailures = false, bool greedyReference = false)
    {
        if (width is < 2 or > 4 || steps is < 2 or > 512)
            throw new ArgumentOutOfRangeException(nameof(width), "Use 2..4 sequences and 2..512 steps.");
        if (greedyReference && independentPrefill)
            throw new ArgumentException("Greedy reference diagnostics require matched checkpoint state.");
        if (greedyReference) batchedFirst = false;
        var sequences = (IBatchedPagedModel)model;
        if (!sequences.SupportsPerSequenceFusedForward)
            throw new InvalidOperationException("This probe requires per-sequence fused decode.");
        string? snapshotRoot = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_SNAPSHOT_DIR");
        string? snapshotDirectory = string.IsNullOrWhiteSpace(snapshotRoot) ? null
            : Path.Combine(Path.GetFullPath(snapshotRoot), $"width{width}-" + Guid.NewGuid().ToString("N"));
        int snapshotRun = 0;

        // Deliberately different lengths, including a 128-row capacity boundary when
        // TS_KV_INITIAL_TOKENS=128. Subsequent decode remains inside each allocation.
        int[] lengths = new[] { 32, 73, 137, 273 }.Take(width).ToArray();
        string? lengthsOverride = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS");
        if (!string.IsNullOrWhiteSpace(lengthsOverride))
        {
            string[] entries = lengthsOverride.Split(',');
            if (entries.Length < width)
                throw new ArgumentException($"TS_QWEN35_PROBE_LENGTHS requires at least {width} comma-separated positive token lengths for width {width}.");
            var configured = new int[entries.Length];
            for (int i = 0; i < entries.Length; i++)
                if (!int.TryParse(entries[i].Trim(), out configured[i]) || configured[i] <= 0)
                    throw new ArgumentException($"TS_QWEN35_PROBE_LENGTHS entry {i + 1} ('{entries[i]}') must be a positive integer.");
            lengths = configured.Take(width).ToArray();
        }
        for (int row = 0; row < width; row++)
            if ((long)lengths[row] + steps + 2 > model.MaxContextLength)
                throw new ArgumentException($"Prompt row {row} requires {lengths[row]} + {steps} + 2 = {(long)lengths[row] + steps + 2} context tokens, exceeding model.MaxContextLength={model.MaxContextLength}. Increase MAX_CONTEXT or reduce TS_QWEN35_PROBE_LENGTHS/TS_QWEN35_PROBE_STEPS.");
        promptSources ??= new[]
        {
            "Revenue is price multiplied by quantity. Profit is revenue minus variable and fixed costs. ",
            "The river flows through a green valley. Trees grow beside the water and birds fly overhead. ",
            "A computer stores numbers in memory. A program performs arithmetic and produces a result. ",
            "Paris is the capital of France. Tokyo is the capital of Japan. Rome is the capital of Italy. "
        };
        if (promptSources.Count < width || promptSources.Take(width).Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Provide a non-empty prompt source for every requested row.", nameof(promptSources));
        int[][] sources = Enumerable.Range(0, width).Select(row =>
        {
            int[] pool = model.Tokenizer.Encode(promptSources[row], addSpecial: false).ToArray();
            if (pool.Length == 0) throw new InvalidOperationException($"Prompt row {row} tokenized to an empty sequence.");
            return Enumerable.Range(0, lengths[row] + steps + 2).Select(i => pool[i % pool.Length]).ToArray();
        }).ToArray();

        float[][][]? sampledExpected = null;
        string checkpointTag = "qwen35-initial-" + Guid.NewGuid().ToString("N");
        string[] checkpointKeys = Enumerable.Range(0, width).Select(row => checkpointTag + "-" + row).ToArray();
        try
        {
            if (!independentPrefill)
            {
                if (!sequences.SupportsPrefixCheckpoints)
                    throw new InvalidOperationException("Matched-state validation requires production prefix checkpoints.");
                sequences.RestorePrimaryCache();
                model.ResetKVCache();
                for (int row = 0; row < width; row++)
                {
                    string sourceId = checkpointTag + "-source-" + row;
                    try
                    {
                        sequences.BindSequenceCache(sourceId);
                        model.PrepareForPrefill(lengths[row] + steps + 2);
                        model.Forward(sources[row].Take(lengths[row]).ToArray());
                        // Build descriptors once and freeze the exact KV/GDN state
                        // shared by every arm, including the sampled-token check.
                        float[] seedLogits = model.Forward(new[] { sources[row][lengths[row]] });
                        if (greedyReference) sources[row][lengths[row] + 1] = ArgMax(seedLogits);
                        if (!sequences.TryCheckpointActiveCache(checkpointKeys[row]))
                            throw new InvalidOperationException($"Initial checkpoint creation declined for row {row}.");
                    }
                    finally
                    {
                        sequences.RestorePrimaryCache();
                        sequences.OnSequenceReleased(sourceId);
                    }
                }
            }
            return CompareRuns();
        }
        finally
        {
            sequences.RestorePrimaryCache();
            if (!independentPrefill)
                foreach (string key in checkpointKeys) sequences.DiscardRetainedCache(key);
        }

        Qwen35DecodeComparison CompareRuns()
        {
            Qwen35DecodeRun serial, batched;
            if (batchedFirst) { batched = Run(!serialControl); serial = Run(false); }
            else { serial = Run(false, buildGreedyReference: greedyReference); batched = Run(!serialControl); }
            Qwen35DecodeRun? sampled = null;
            if (verifySampled)
            {
                sampledExpected = batched.Logits;
                sampled = Run(true, sample: true);
            }
            if (!independentPrefill && (!serial.CacheCapacities.SequenceEqual(batched.CacheCapacities)
                || (sampled != null && !serial.CacheCapacities.SequenceEqual(sampled.CacheCapacities))))
                throw new InvalidOperationException("Matched checkpoint arms allocated different cache capacities; the comparison is not qualified.");
            double minCosine = 1, maxNrmse = 0, maxAbsolute = 0, maxSoftmaxKl = 0;
            double maxCenteredNrmse = 0, maxAbsoluteMeanError = 0, maxTotalVariation = 0;
            int topDifferences = 0, confidentDifferences = 0;
            var flips = new List<Qwen35ArgmaxFlip>();
            var failures = new List<Qwen35NumericalFailure>();
            bool distributionPassed = true;
            var comparisons = new List<(Qwen35DecodeRun Expected, Qwen35DecodeRun Actual, int FirstStep, string Label)>
            {
                (serial, batched, 0, serialControl ? "serial-repeat" : "serial-batched")
            };
            // Sampled calls check every token against the host-logit batch below;
            // only their final serial continuation produces logits for comparison.
            if (sampled != null) comparisons.Add((batched, sampled, steps, "batched-sampled-continuation"));
            foreach (var comparison in comparisons)
            for (int step = comparison.FirstStep; step <= steps; step++)
            for (int row = 0; row < width; row++)
            {
                float[] expected = comparison.Expected.Logits[step][row], actual = comparison.Actual.Logits[step][row];
                if (expected.Length != actual.Length || expected.Length < 2)
                    throw new InvalidOperationException("Logit shapes differ or are empty.");
                double dot = 0, normA = 0, normB = 0, squaredError = 0, sum = 0, errorSum = 0, absolute = 0;
                int topA = 0, topB = 0, nonFiniteIndex = -1;
                for (int i = 0; i < expected.Length; i++)
                {
                    if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
                    {
                        nonFiniteIndex = i;
                        break;
                    }
                    double error = (double)actual[i] - expected[i];
                    dot += (double)expected[i] * actual[i];
                    normA += (double)expected[i] * expected[i];
                    normB += (double)actual[i] * actual[i];
                    sum += expected[i];
                    squaredError += error * error;
                    errorSum += error;
                    absolute = Math.Max(absolute, Math.Abs(error));
                    if (expected[i] > expected[topA]) topA = i;
                    if (actual[i] > actual[topB]) topB = i;
                }
                if (nonFiniteIndex >= 0)
                {
                    distributionPassed = false;
                    failures.Add(new(comparison.Label, width, step, row,
                        null, null, null, null, null, null, null, null, null, null, null, null, null,
                        new[] { $"Non-finite logits at index {nonFiniteIndex}" }));
                    if (!collectNumericalFailures)
                    {
                        var error = new InvalidOperationException($"Non-finite logits: width={width} step={step} row={row} index={nonFiniteIndex}.");
                        error.Data["NumericalFailures"] = failures.ToArray();
                        error.Data["ArgmaxFlips"] = flips.ToArray();
                        throw error;
                    }
                    continue;
                }
                double variance = Math.Max(1e-20, normA / expected.Length - Math.Pow(sum / expected.Length, 2));
                double rmse = Math.Sqrt(squaredError / expected.Length);
                double nrmse = rmse / Math.Sqrt(variance);
                double meanError = errorSum / expected.Length;
                double centeredNrmse = Math.Sqrt(Math.Max(0, rmse * rmse - meanError * meanError) / variance);
                double cosine = dot / Math.Sqrt(Math.Max(1e-20, normA * normB));
                double softmaxSumA = 0, softmaxSumB = 0;
                for (int i = 0; i < expected.Length; i++)
                {
                    softmaxSumA += Math.Exp((double)expected[i] - expected[topA]);
                    softmaxSumB += Math.Exp((double)actual[i] - actual[topB]);
                }
                double logSumA = Math.Log(softmaxSumA), logSumB = Math.Log(softmaxSumB), kl = 0, totalVariation = 0;
                for (int i = 0; i < expected.Length; i++)
                {
                    double logP = (double)expected[i] - expected[topA] - logSumA;
                    double logQ = (double)actual[i] - actual[topB] - logSumB;
                    double p = Math.Exp(logP), q = Math.Exp(logQ);
                    kl += p * (logP - logQ);
                    totalVariation += 0.5 * Math.Abs(p - q);
                }
                // Roundoff can make the mathematically nonnegative KL slightly negative.
                kl = Math.Max(0, kl);
                float runnerUp = float.NegativeInfinity;
                for (int i = 0; i < expected.Length; i++)
                    if (i != topA) runnerUp = Math.Max(runnerUp, expected[i]);
                double margin = (double)expected[topA] - runnerUp;
                double probabilityMargin = (1 - Math.Exp(-margin)) / softmaxSumA;
                double referenceTopProbability = 1 / softmaxSumA;
                double observedProbabilityOfReferenceTop = Math.Exp((double)actual[topA] - actual[topB] - logSumB);
                minCosine = Math.Min(minCosine, cosine);
                maxNrmse = Math.Max(maxNrmse, nrmse);
                maxAbsolute = Math.Max(maxAbsolute, absolute);
                maxSoftmaxKl = Math.Max(maxSoftmaxKl, kl);
                maxCenteredNrmse = Math.Max(maxCenteredNrmse, centeredNrmse);
                maxAbsoluteMeanError = Math.Max(maxAbsoluteMeanError, Math.Abs(meanError));
                maxTotalVariation = Math.Max(maxTotalVariation, totalVariation);
                if (Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_TRACE") == "1")
                    Console.WriteLine($"[qwen35-parity] comparison={comparison.Label} width={width} step={step} row={row} cosine={cosine:F8} nrmse={nrmse:F6} centered_nrmse={centeredNrmse:F6} mean_error={meanError:F6} maxabs={absolute:F6} softmax_kl={kl:F8} reference_top={topA} observed_top={topB} reference_margin={margin:F6} probability_margin={probabilityMargin:F8} reference_top_probability={referenceTopProbability:F8} observed_reference_top_probability={observedProbabilityOfReferenceTop:F8} total_variation={totalVariation:F8}");
                if (topA != topB)
                {
                    topDifferences++;
                    flips.Add(new(comparison.Label, step, row, topA, topB, margin, probabilityMargin, kl,
                        referenceTopProbability, observedProbabilityOfReferenceTop, totalVariation));
                    // A fixed reference margin keeps the decision independent of how
                    // much error the observed path introduces.
                    if (margin > 0.25) confidentDifferences++;
                }
                // KL is an explicit empirical distribution-error budget, not a proof
                // of model quality. Raw NRMSE and max error remain visible diagnostics.
                var violations = new List<string>();
                if (!double.IsFinite(cosine)) violations.Add("Non-finite cosine");
                else if (cosine < 0.995) violations.Add("Raw cosine below 0.995");
                if (!double.IsFinite(kl)) violations.Add("Non-finite softmax KL");
                else if (kl > 0.02) violations.Add("Softmax KL exceeds 0.02");
                if (topA != topB && margin > 0.25) violations.Add("Argmax changed above fixed reference logit margin 0.25");
                if (!double.IsFinite(kl) || kl > 0.02 || (topA != topB && margin > 0.25))
                    distributionPassed = false;
                if (violations.Count != 0)
                {
                    failures.Add(new(comparison.Label, width, step, row,
                        double.IsFinite(cosine) ? cosine : null, nrmse, centeredNrmse, meanError,
                        absolute, double.IsFinite(kl) ? kl : null, topA, topB, margin, probabilityMargin,
                        referenceTopProbability, observedProbabilityOfReferenceTop, totalVariation, violations));
                    if (!collectNumericalFailures)
                    {
                        var error = new InvalidOperationException($"Decode distribution comparison failed: comparison={comparison.Label}, width={width} step={step} row={row}, cosine={cosine:F8}, normalized RMSE={nrmse:F6}, centered normalized RMSE={centeredNrmse:F6}, mean error={meanError:F6}, max absolute={absolute:F6}, softmax KL={kl:F8}, reference top={topA}, observed top={topB}, reference logit margin={margin:F6}, reference probability margin={probabilityMargin:F8}, reference top probability={referenceTopProbability:F8}, observed probability of reference top={observedProbabilityOfReferenceTop:F8}, total variation={totalVariation:F8}. Budgets: cosine >= 0.995, KL <= 0.02, no argmax flip above reference margin 0.25.");
                        error.Data["ArgmaxFlips"] = flips.ToArray();
                        error.Data["NumericalFailures"] = failures.ToArray();
                        throw error;
                    }
                }
            }
            Qwen35GreedyContinuation[]? continuations = greedyReference
                ? Enumerable.Range(0, width).Select(row =>
                {
                    int[] reference = new[] { sources[row][lengths[row] + 1] }
                        .Concat(serial.Logits.Select(logits => ArgMax(logits[row]))).ToArray();
                    int[] observed = new[] { sources[row][lengths[row] + 1] }
                        .Concat(batched.Logits.Select(logits => ArgMax(logits[row]))).ToArray();
                    return new Qwen35GreedyContinuation(row, reference, observed,
                        model.Tokenizer.Decode(reference.ToList()), model.Tokenizer.Decode(observed.ToList()));
                }).ToArray()
                : null;
            return new(width, steps, batchedFirst,
                serial with { Logits = Array.Empty<float[][]>() }, batched with { Logits = Array.Empty<float[][]>() },
                minCosine, maxNrmse, maxAbsolute, topDifferences, confidentDifferences,
                sampled is null ? null : sampled with { Logits = Array.Empty<float[][]>() },
                maxSoftmaxKl, flips, independentPrefill ? "independent-prefill" : "shared-checkpoint-clones",
                maxCenteredNrmse, maxAbsoluteMeanError, maxTotalVariation, failures, failures.Count == 0,
                distributionPassed, greedyReference,
                continuations == null ? null : continuations.All(row => row.ReferenceTokenIds.SequenceEqual(row.ObservedTokenIds)),
                continuations);
        }

        Qwen35DecodeRun Run(bool useBatch, bool sample = false, bool buildGreedyReference = false)
        {
            int runNumber = ++snapshotRun;
            string tag = "qwen35-probe-" + Guid.NewGuid().ToString("N");
            string[] ids = Enumerable.Range(0, width).Select(i => tag + "-" + i).ToArray();
            var positions = lengths.Select(length => length + 1).ToArray();
            var rows = new float[steps + 1][][];
            int accepted = 0, declined = 0, replacements = 0;
            try
            {
                sequences.RestorePrimaryCache();
                model.ResetKVCache();
                for (int row = 0; row < width; row++)
                {
                    if (!independentPrefill && !sequences.TryCloneRetainedCache(checkpointKeys[row], ids[row]))
                        throw new InvalidOperationException($"Initial checkpoint cloning declined for row {row}.");
                    sequences.BindSequenceCache(ids[row]);
                    model.PrepareForPrefill(lengths[row] + steps + 2);
                    if (independentPrefill)
                    {
                        model.Forward(sources[row].Take(lengths[row]).ToArray());
                        model.Forward(new[] { sources[row][lengths[row]] });
                    }
                }
                sequences.RestorePrimaryCache();
                int[] capacities = CacheCapacities(model, ids);
                for (int row = 0; row < width; row++)
                    if (capacities[row] < lengths[row] + steps + 2)
                        throw new InvalidOperationException($"Row {row} lacks reserved decode headroom: capacity={capacities[row]}, required={lengths[row] + steps + 2}.");
                long initialCounter = NativeSteps(model);
                var watch = Stopwatch.StartNew();
                for (int step = 0; step < steps; step++)
                {
                    if (replaceLastRequest && step == steps / 2)
                    {
                        // Replace a request by cloning its exact current state,
                        // avoiding a fresh prefill that could introduce unrelated
                        // numerical variation. Copying is outside decode timing.
                        watch.Stop();
                        sequences.RestorePrimaryCache();
                        int row = width - 1;
                        string replacementKey = tag + "-replacement-checkpoint";
                        try
                        {
                            sequences.BindSequenceCache(ids[row]);
                            if (!sequences.TryCheckpointActiveCache(replacementKey))
                                throw new InvalidOperationException("Replacement checkpoint creation declined.");
                            sequences.RestorePrimaryCache();
                            sequences.OnSequenceReleased(ids[row]);
                            ids[row] += "-replacement";
                            if (!sequences.TryCloneRetainedCache(replacementKey, ids[row]))
                                throw new InvalidOperationException("Replacement checkpoint cloning declined.");
                            sequences.BindSequenceCache(ids[row]);
                            model.PrepareForPrefill(lengths[row] + steps + 2);
                            sequences.RestorePrimaryCache();
                        }
                        finally
                        {
                            sequences.RestorePrimaryCache();
                            sequences.DiscardRetainedCache(replacementKey);
                        }
                        replacements++;
                        watch.Start();
                    }
                    // Exercise canonical slot order with changing caller order.
                    int[] order = Enumerable.Range(0, width).Select(i => (i + step) % width).ToArray();
                    if (step % 2 != 0) Array.Reverse(order);
                    int[] tokens = order.Select(row => sources[row][positions[row]]).ToArray();
                    var result = new float[width][];
                    var sampledTokens = new int[width];
                    bool fused = useBatch && (sample
                        ? sequences.TryForwardBatchedFusedDecodeSampled(
                            order.Select(row => ids[row]).ToArray(), tokens,
                            order.Select(row => positions[row]).ToArray(), sampledTokens)
                        : sequences.TryForwardBatchedFusedDecode(
                            order.Select(row => ids[row]).ToArray(), tokens,
                            order.Select(row => positions[row]).ToArray(), result));
                    if (fused) accepted++;
                    else
                    {
                        if (useBatch)
                        {
                            declined++;
                            if (!allowFallback || sample)
                                throw new InvalidOperationException($"Batched fused decode declined: width={width} step={step}, reason={sequences.BatchedFusedDecodeDeclineReason ?? "not supplied"}.");
                        }
                        for (int slot = 0; slot < width; slot++)
                        {
                            sequences.BindSequenceCache(ids[order[slot]]);
                            // Forward reuses the model-wide logits buffer across
                            // holders; retain this row before the next Forward.
                            result[slot] = (float[])model.Forward(new[] { tokens[slot] }).Clone();
                        }
                    }
                    rows[step] = new float[width][];
                    for (int slot = 0; slot < width; slot++)
                    {
                        int row = order[slot];
                        if (sample)
                        {
                            float[] expected = sampledExpected![step][row];
                            int expectedToken = 0;
                            for (int i = 1; i < expected.Length; i++)
                                if (expected[i] > expected[expectedToken]) expectedToken = i;
                            if (sampledTokens[slot] != expectedToken)
                                throw new InvalidOperationException($"Sampled/host-logit batch argmax mismatch: width={width} step={step} row={row}, expected={expectedToken}, actual={sampledTokens[slot]}.");
                            rows[step][row] = Array.Empty<float>();
                        }
                        else
                        {
                            rows[step][row] = fused ? (float[])result[slot].Clone() : result[slot];
                        }
                        if (buildGreedyReference)
                            sources[row][positions[row] + 1] = ArgMax(rows[step][row]);
                        positions[row]++;
                    }
                    if (snapshotDirectory != null && width >= 3 && step == 0)
                    {
                        // Diagnostic only: checkpointing flushes the arena. It is
                        // excluded from timing and disqualifies benchmark results.
                        watch.Stop();
                        const int row = 2;
                        string key = tag + "-diagnostic-checkpoint";
                        string arm = sample ? "sampled" : useBatch ? "batched" : "serial";
                        Directory.CreateDirectory(snapshotDirectory);
                        string snapshotPath = Path.Combine(snapshotDirectory, $"{runNumber}-{arm}-row2-step0.q5kc");
                        try
                        {
                            sequences.BindSequenceCache(ids[row]);
                            if (!sequences.TryCheckpointActiveCache(key))
                                throw new InvalidOperationException("Diagnostic checkpoint creation declined.");
                            using (var stream = File.Create(snapshotPath))
                                if (!sequences.TryExportRetainedCache(key, stream))
                                    throw new InvalidOperationException("Diagnostic checkpoint export declined.");
                            File.WriteAllText(snapshotPath + ".json", JsonSerializer.Serialize(new
                            {
                                Arm = arm, Run = runNumber, Width = width, Step = step, LogicalRow = row,
                                CachedTokens = positions[row], KvDtype = model.KvCacheDtype.ToString(),
                                DiagnosticOnly = true,
                                Note = "Checkpoint/export flushes arena ownership and changes later rejoin behavior; performance is unqualified."
                            }, new JsonSerializerOptions { WriteIndented = true }));
                            Console.WriteLine($"[qwen35-probe] diagnostic snapshot {snapshotPath}");
                        }
                        finally
                        {
                            sequences.DiscardRetainedCache(key);
                            sequences.RestorePrimaryCache();
                        }
                        watch.Start();
                    }
                }
                watch.Stop();
                long nativeSteps = NativeSteps(model) - initialCounter;
                if (nativeSteps != accepted)
                    throw new InvalidOperationException($"Successful native steps ({nativeSteps}) differ from accepted calls ({accepted}).");
                // A serial continuation after the batch validates arena-to-holder
                // coherence; it is checked numerically but excluded from timing.
                rows[steps] = new float[width][];
                for (int row = width - 1; row >= 0; row--)
                {
                    sequences.BindSequenceCache(ids[row]);
                    rows[steps][row] = (float[])model.Forward(new[] { sources[row][positions[row]] }).Clone();
                }
                return new(watch.Elapsed.TotalMilliseconds, accepted, declined, replacements, nativeSteps, lengths, capacities, rows);
            }
            finally
            {
                sequences.RestorePrimaryCache();
                foreach (string id in ids) sequences.OnSequenceReleased(id);
            }
        }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int index = 1; index < logits.Length; index++)
            if (logits[index] > logits[best]) best = index;
        return best;
    }

    private static long NativeSteps(Qwen35Model model) => (long)(typeof(Qwen35Model)
        .GetProperty("ArenaBatchedDecodeSteps", BindingFlags.Instance | BindingFlags.NonPublic)!
        .GetValue(model) ?? throw new InvalidOperationException("Missing native engagement counter."));

    private static int[] CacheCapacities(Qwen35Model model, string[] ids)
    {
        var holders = (IDictionary)typeof(Qwen35Model).GetField("_fusedHolders", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        return ids.Select(id => (int)holders[id]!.GetType().GetField("KvCapacity")!.GetValue(holders[id])!).ToArray();
    }
}
