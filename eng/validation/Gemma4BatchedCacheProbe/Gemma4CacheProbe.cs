using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text.Json.Serialization;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Validation;

public sealed record CacheProbeRun(double Milliseconds, int FusedSteps, int FallbackSteps, int GrowthRows, long NativeBatchedTokens,
    int[] InitialCapacities, int[] FinalCapacities, [property: JsonIgnore] float[][][] Logits);
public sealed record CacheProbeComparison(int Width, bool BatchedFirst, CacheProbeRun Serial,
    CacheProbeRun Batched, double MinCosine, double MaxRmse, double MaxAbsoluteError, int TopTokenDifferences,
    LogitComparison? UniformBatchedReference);
public sealed record LogitComparison(double MinCosine, double MaxRmse, double MaxAbsoluteError, int TopTokenDifferences);

/// <summary>Actual retained-prefix cloning, independent holder growth and token-batched decode.
/// Reflection observes capacities only; all cache changes use production model APIs.</summary>
public static class Gemma4CacheProbe
{
    public static CacheProbeComparison Compare(Gemma4Model model, int width, int steps, bool batchedFirst, bool requireFused,
        bool compactClones = true, bool requireNumericalParity = true)
    {
        if (width is < 2 or > 4 || steps < 4) throw new ArgumentOutOfRangeException(nameof(width));
        var sequences = (IBatchedPagedModel)model;
        if (!sequences.SupportsPerSequenceFusedForward || !sequences.SupportsPrefixCheckpoints)
            throw new InvalidOperationException("This regression requires fused decode and checkpoint support.");
        int[] pool = model.Tokenizer.Encode("The capital of France is Paris. The river flows through the valley. Describe the history of computing clearly. ", addSpecial: false).ToArray();
        int[] source = Enumerable.Range(0, 1200 + steps).Select(i => pool[i % pool.Length]).ToArray();
        int[] lengths = new[] { 128, 192, 510, 1020 }.Take(width).ToArray();
        CacheProbeRun serial, batched;
        if (batchedFirst)
        {
            batched = Run(true);
            serial = Run(false);
        }
        else
        {
            serial = Run(false);
            batched = Run(true);
        }
        if (!serial.FinalCapacities.SequenceEqual(batched.FinalCapacities))
            throw new InvalidOperationException("Batching changed unrelated holders' cache capacity.");

        // The original quantized batched GEMM already differs from GEMV. Record
        // that difference separately; do not mislabel it cache-copy error or hide
        // it by loosening tolerances. Equal batch shapes isolate this change.
        LogitComparison serialDifference = Measure(serial.Logits, batched.Logits, strict: false);
        LogitComparison? referenceComparison = null;
        if (requireNumericalParity && requireFused)
        {
            CacheProbeRun reference = Run(true, uniformReference: true);
            referenceComparison = Measure(reference.Logits, batched.Logits, strict: true);
        }
        return new(width, batchedFirst, serial with { Logits = Array.Empty<float[][]>() },
            batched with { Logits = Array.Empty<float[][]>() }, serialDifference.MinCosine,
            serialDifference.MaxRmse, serialDifference.MaxAbsoluteError, serialDifference.TopTokenDifferences, referenceComparison);

        CacheProbeRun Run(bool useBatch, bool uniformReference = false)
        {
            string tag = "probe-" + Guid.NewGuid().ToString("N");
            string checkpoint = tag + "-checkpoint";
            string[] ids = Enumerable.Range(0, width).Select(i => tag + "-" + i).ToArray();
            int accepted = 0, declined = 0, growthRows = 0;
            var results = new float[steps + 1][][];
            try
            {
                model.ResetKVCache();
                model.Forward(source.Take(lengths[0]).ToArray());
                if (!sequences.TryCheckpointActiveCache(checkpoint)) throw new InvalidOperationException("Checkpoint creation declined.");
                sequences.AdoptPrimaryCacheToFused(ids[0]);
                for (int i = 1; i < width; i++)
                {
                    if (compactClones && !uniformReference && !sequences.TryCloneRetainedCache(checkpoint, ids[i])) throw new InvalidOperationException("Prefix cloning declined.");
                    sequences.BindSequenceCache(ids[i]);
                    if (!compactClones || uniformReference) model.Forward(source.Take(lengths[0]).ToArray());
                    model.Forward(source.Skip(lengths[0]).Take(lengths[i] - lengths[0]).ToArray());
                }
                sequences.RestorePrimaryCache();
                int[] initial = Capacities(model, ids);
                if (initial[0] != 16384 || (compactClones && !uniformReference && initial.Distinct().Count() < 2))
                    throw new InvalidOperationException($"Regression failed to construct a 16K root and compact cloned cache: {string.Join(',', initial)}.");
                var positions = (int[])lengths.Clone();
                int[] simulatedCapacity = new[] { 16384, 512, 512, 1024 }.Take(width).ToArray();
                long beforeFusedSteps = model.BatchedFusedDecodeSteps, beforeFusedTokens = model.BatchedFusedDecodeTokens;
                var watch = Stopwatch.StartNew();
                for (int step = 0; step < steps; step++)
                {
                    // Reverse and rotate membership order while the native graph stays live.
                    int[] order = Enumerable.Range(0, width).Select(i => (i + step) % width).ToArray();
                    if (step % 2 != 0) Array.Reverse(order);
                    int[] tokens = order.Select(i => source[lengths[i] + step]).ToArray();
                    var logits = new float[width][];
                    int[] ready = useBatch ? Enumerable.Range(0, width).Where(slot =>
                        uniformReference && compactClones ? positions[order[slot]] < simulatedCapacity[order[slot]]
                        : model.CanBatchDecode(ids[order[slot]], positions[order[slot]])).ToArray() : Array.Empty<int>();
                    if (useBatch) growthRows += width - ready.Length;
                    var batchLogits = new float[ready.Length][];
                    bool fused = ready.Length >= 2 && sequences.TryForwardBatchedFusedDecode(ready.Select(slot => ids[order[slot]]).ToArray(), ready.Select(slot => tokens[slot]).ToArray(), ready.Select(slot => positions[order[slot]]).ToArray(), batchLogits);
                    if (fused)
                    {
                        accepted++;
                        for (int slot = 0; slot < ready.Length; slot++) logits[ready[slot]] = batchLogits[slot];
                    }
                    else if (ready.Length >= 2)
                    {
                        declined++;
                        if (requireFused) throw new InvalidOperationException($"Batched fused decode declined: width={width} step={step}, capacities={string.Join(',', Capacities(model, ids))}, positions={string.Join(',', positions)}, reason={model.BatchedFusedDecodeDeclineReason}.");
                    }
                    // The production scheduler lets only a holder that needs growth take
                    // a serial step; its unrelated ready peers continue in a batch.
                    for (int slot = 0; slot < width; slot++)
                    {
                        if (logits[slot] != null) continue;
                        int row = order[slot];
                        sequences.BindSequenceCache(ids[row]);
                        logits[slot] = (float[])model.Forward(new[] { tokens[slot] }).Clone();
                        while (simulatedCapacity[row] <= positions[row]) simulatedCapacity[row] *= 2;
                    }
                    results[step] = new float[width][];
                    for (int slot = 0; slot < width; slot++)
                    {
                        int row = order[slot];
                        results[step][row] = logits[slot];
                        positions[row]++;
                    }
                }
                watch.Stop();
                if (model.BatchedFusedDecodeSteps - beforeFusedSteps != accepted)
                    throw new InvalidOperationException("Native successful-execution counter does not match accepted calls.");
                sequences.RestorePrimaryCache();
                int[] final = Capacities(model, ids);
                results[steps] = new float[width][];
                for (int row = width - 1; row >= 0; row--)
                {
                    sequences.BindSequenceCache(ids[row]);
                    results[steps][row] = (float[])model.Forward(new[] { source[lengths[row] + steps] }).Clone();
                }
                return new(watch.Elapsed.TotalMilliseconds, accepted, declined, growthRows,
                    model.BatchedFusedDecodeTokens - beforeFusedTokens, initial, final, results);
            }
            finally
            {
                sequences.RestorePrimaryCache();
                foreach (string id in ids) sequences.OnSequenceReleased(id);
                sequences.DiscardRetainedCache(checkpoint);
            }
        }

        LogitComparison Measure(float[][][] reference, float[][][] observed, bool strict)
        {
            double minCosine = 1, maxRmse = 0, maxAbsolute = 0;
            int topDifferences = 0;
            // Includes the final solo continuation after the batched episode.
            for (int step = 0; step <= steps; step++)
            for (int row = 0; row < width; row++)
            {
                float[] expected = reference[step][row], actual = observed[step][row];
                if (expected.Length != actual.Length) throw new InvalidOperationException("Logit lengths differ.");
                double dot = 0, normA = 0, normB = 0, errors = 0, maxError = 0;
                int topA = 0, topB = 0;
                for (int i = 0; i < expected.Length; i++)
                {
                    if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i])) throw new InvalidOperationException($"Non-finite logit row={row} step={step} index={i}.");
                    double error = (double)expected[i] - actual[i];
                    if (strict && Math.Abs(error) > 1e-4 + 1e-4 * Math.Abs(expected[i]))
                        throw new InvalidOperationException($"Uniform-batched cache reference mismatch: width={width} row={row} step={step} logit={i} expected={expected[i]} actual={actual[i]} error={error}.");
                    errors += error * error;
                    maxError = Math.Max(maxError, Math.Abs(error));
                    dot += (double)expected[i] * actual[i];
                    normA += (double)expected[i] * expected[i]; normB += (double)actual[i] * actual[i];
                    if (expected[i] > expected[topA]) topA = i;
                    if (actual[i] > actual[topB]) topB = i;
                }
                minCosine = Math.Min(minCosine, dot / Math.Sqrt(normA * normB));
                maxRmse = Math.Max(maxRmse, Math.Sqrt(errors / expected.Length)); maxAbsolute = Math.Max(maxAbsolute, maxError);
                if (topA != topB) topDifferences++;
                if (strict && topA != topB) throw new InvalidOperationException("Uniform-batched reference top-token mismatch.");
            }
            return new(minCosine, maxRmse, maxAbsolute, topDifferences);
        }
    }

    private static int[] Capacities(Gemma4Model model, string[] ids)
    {
        var holders = (IDictionary)typeof(Gemma4Model).GetField("_fusedHolders", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
        return ids.Select(id => (int)holders[id]!.GetType().GetField("GlobalCapacity")!.GetValue(holders[id])!).ToArray();
    }
}
