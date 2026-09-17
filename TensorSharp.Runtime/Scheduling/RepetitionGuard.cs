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
using System.Diagnostics.CodeAnalysis;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Recognises a generation that has locked into a loop, so the engine can end it
    /// instead of streaming the same few tokens until somebody presses Stop.
    ///
    /// <para>
    /// Observed on a phone: a 4-bit 9B model, asked for a slide deck, was writing a
    /// long block of XML inside a Python string when it fell into <c>","+","+","+"</c>
    /// and stayed there — 230 repetitions, five and a half minutes, ended only by the
    /// user tapping Stop. Nothing in the pipeline could end it: the penalties are
    /// deliberately off for code (they corrupt legitimately repetitive structure), the
    /// reply limit was hundreds of thousands of tokens, and a loop never reaches EOS.
    /// The output was not wrong because of the cache (99.8% reuse) or the prompt; it
    /// was a sampling degeneration in a long, low-entropy stretch, which is the one
    /// failure that sampling settings cannot rule out for every model and every text.
    /// </para>
    /// <para>
    /// The test is purely structural — no decoding, no language model — and errs on
    /// the side of letting output through: a loop is declared only when the LAST
    /// <c>max(MinSpan, MinRepeats × period)</c> generated tokens are exactly periodic
    /// with a period of at most <see cref="MaxPeriod"/> tokens. That is 128 identical
    /// tokens in a row, or a 3-token phrase 43 times over, or a 32-token paragraph 8
    /// times word for word. Real writing repeats shorter than that (a table separator
    /// row, a rule of dashes, a list of identical values) or with variation; and the
    /// cost of the rare false stop is one round in which the model is told why and
    /// writes it differently, while the cost of a miss is minutes of garbage on a
    /// battery.
    /// </para>
    /// <para>
    /// Cost: bounded by <c>MaxPeriod × MinSpan</c> comparisons per token in the
    /// pathological case and a handful in the normal one, because a period is only
    /// examined when the newest token equals the one that period back.
    /// </para>
    /// </summary>
    public static class RepetitionGuard
    {
        /// <summary>The longest repeating unit, in tokens, that counts as a loop.</summary>
        public const int MaxPeriod = 64;

        /// <summary>
        /// A loop must repeat its unit at least this many times over.
        ///
        /// <para>
        /// Eight rather than four, and that is the parameter that decides the false
        /// positives. A degenerate loop's unit is SHORT — the observed one was three
        /// tokens — and a short unit is bounded by <see cref="MinSpan"/> anyway, so
        /// raising this costs the real case nothing. What it buys is the long-unit
        /// corner: four byte-identical copies of a 32-token block is ordinary
        /// structured output (four empty paragraphs in a row of OOXML, four identical
        /// rows of a table), and stopping that would be wrong. Eight in a row, with no
        /// variation at all, is not something a model writes on purpose.
        /// </para>
        /// </summary>
        public const int MinRepeats = 8;

        /// <summary>
        /// ...and the periodic tail must be at least this long in tokens, so that a
        /// short unit has to recur many more times than a long one before it counts:
        /// 128 copies of one token, 43 of a three-token phrase, 8 of a 32-token block.
        /// </summary>
        public const int MinSpan = 128;

        /// <summary>The finish reason the engine reports when this guard ends a sequence.</summary>
        public const string FinishReason = "repetition";

        /// <summary>
        /// Whether the first <paramref name="count"/> entries of
        /// <paramref name="tokens"/> end in a loop. Only the newest token is new since
        /// the previous call, so the check is meant to run once per emitted token.
        /// </summary>
        /// <param name="tokens">The generated tokens (never the prompt).</param>
        /// <param name="count">How many of them count; the rest may be an unpublished speculative tail.</param>
        /// <param name="period">The loop's period, when one is found.</param>
        /// <param name="repeats">How many whole copies of the unit the periodic tail holds.</param>
        public static bool IsLooping(IReadOnlyList<int> tokens, int count, out int period, out int repeats)
        {
            period = 0;
            repeats = 0;
            if (tokens == null)
                return false;
            if (count > tokens.Count)
                count = tokens.Count;
            if (count < MinSpan)
                return false;

            int last = tokens[count - 1];
            for (int p = 1; p <= MaxPeriod; p++)
            {
                int span = Math.Max(MinSpan, MinRepeats * p);
                if (span > count)
                    break;
                // Cheap pre-check: the newest token must equal the one a period back,
                // which for ordinary text prunes almost every period immediately.
                if (tokens[count - 1 - p] != last)
                    continue;

                // The tail is p-periodic when every token in it equals the one p
                // before it; the first p tokens of the tail are the unit itself.
                bool periodic = true;
                for (int i = count - span; i < count - p; i++)
                {
                    if (tokens[i] != tokens[i + p])
                    {
                        periodic = false;
                        break;
                    }
                }
                if (!periodic)
                    continue;

                period = p;
                // Extend backwards past the minimum span so the report says how far
                // the loop really reached, capped so this stays cheap.
                int start = count - span;
                int limit = Math.Max(0, count - 64 * MaxPeriod);
                while (start - 1 >= limit && tokens[start - 1] == tokens[start - 1 + p])
                    start--;
                repeats = (count - start) / p;
                return true;
            }
            return false;
        }

        /// <summary>
        /// The same test on CHARACTERS, for a layer that has text rather than tokens:
        /// true when <paramref name="text"/> ends in a unit of at most 512 characters
        /// repeated at least 4 times over at least 64 characters. Used to quote the loop
        /// back to the model; it is not what decides to stop.
        /// </summary>
        public static bool TryFindTextLoop(string text, [NotNullWhen(true)] out string? unit, out int repeats)
        {
            unit = null;
            repeats = 0;
            if (string.IsNullOrEmpty(text) || text.Length < 64)
                return false;
            // Looser than the token test on purpose, and in the safe direction: this
            // one only decides how the loop is DESCRIBED, and a 64-token unit (the
            // token cap) can be several hundred characters, so a tighter cap here
            // would fall back to the vague sentence for exactly the cases the token
            // guard just stopped. Run once per looping round, never per token.
            const int maxPeriod = 512, minRepeats = 4, minSpan = 64;
            int n = text.Length;
            char last = text[n - 1];
            for (int p = 1; p <= maxPeriod; p++)
            {
                int span = Math.Max(minSpan, minRepeats * p);
                if (span > n)
                    break;
                if (text[n - 1 - p] != last)
                    continue;
                bool periodic = true;
                for (int i = n - span; i < n - p; i++)
                {
                    if (text[i] != text[i + p])
                    {
                        periodic = false;
                        break;
                    }
                }
                if (!periodic)
                    continue;
                int start = n - span;
                while (start - 1 >= 0 && text[start - 1] == text[start - 1 + p])
                    start--;
                unit = text.Substring(n - p, p);
                repeats = (n - start) / p;
                return true;
            }
            return false;
        }

        /// <summary>
        /// One sentence for a log line or a note to the model: what repeated and how
        /// many times, with the unit rendered by <paramref name="decode"/> when a
        /// tokenizer is at hand.
        /// </summary>
        public static string Describe(IReadOnlyList<int> tokens, int count, int period, int repeats, Func<List<int>, string?> decode)
        {
            string unit = string.Empty;
            if (tokens != null && decode != null && period > 0 && count >= period)
            {
                var ids = new List<int>(period);
                for (int i = count - period; i < count; i++)
                    ids.Add(tokens[i]);
                try
                {
                    unit = decode(ids) ?? string.Empty;
                }
                catch (Exception)
                {
                    unit = string.Empty;
                }
                unit = unit.Replace("\r", "\\r").Replace("\n", "\\n");
                if (unit.Length > 48)
                    unit = unit.Substring(0, 48) + "…";
            }
            string what = unit.Length > 0 ? $"`{unit}`" : $"a {period}-token sequence";
            return $"the output repeated {what} {repeats} times in a row";
        }
    }
}
