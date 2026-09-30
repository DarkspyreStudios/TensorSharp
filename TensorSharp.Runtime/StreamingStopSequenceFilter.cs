using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    internal static class StopSequenceMatcher
    {
        internal static string[] Snapshot(IReadOnlyList<string>? sequences)
        {
            if (sequences == null || sequences.Count == 0)
                return Array.Empty<string>();
            var copy = new string[sequences.Count];
            for (int i = 0; i < copy.Length; i++)
                copy[i] = !string.IsNullOrEmpty(sequences[i]) ? sequences[i]
                    : throw new ArgumentException("Stop sequences must contain non-empty strings.", nameof(sequences));
            return copy;
        }

        internal static int FirstMatch(string text, IReadOnlyList<string> sequences)
        {
            int first = -1;
            foreach (string stop in sequences)
            {
                int index = text.IndexOf(stop, StringComparison.Ordinal);
                if (index >= 0 && (first < 0 || index < first))
                    first = index;
            }
            return first;
        }
    }

    /// <summary>Withholds only a suffix that can become a stop sequence in a later chunk.</summary>
    internal sealed class StreamingStopSequenceFilter
    {
        private readonly string[] _sequences;
        private string _pending = string.Empty;
        private bool _completed;

        internal StreamingStopSequenceFilter(IReadOnlyList<string> sequences)
            => _sequences = StopSequenceMatcher.Snapshot(sequences);

        internal bool Stopped { get; private set; }
        internal int PendingLength => _pending.Length;

        internal string Append(string piece)
        {
            if (_completed)
                throw new InvalidOperationException("The stop-sequence filter is complete.");
            if (Stopped)
                return string.Empty;
            string text = _pending + piece;
            int match = StopSequenceMatcher.FirstMatch(text, _sequences);
            if (match >= 0)
            {
                Stopped = true;
                _pending = string.Empty;
                return text[..match];
            }

            int retain = 0;
            foreach (string stop in _sequences)
            {
                for (int length = Math.Min(stop.Length - 1, text.Length); length > retain; length--)
                {
                    if (text.AsSpan(text.Length - length).SequenceEqual(stop.AsSpan(0, length)))
                    {
                        retain = length;
                        break;
                    }
                }
            }
            _pending = text[(text.Length - retain)..];
            return text[..(text.Length - retain)];
        }

        internal string Complete()
        {
            _completed = true;
            string tail = _pending;
            _pending = string.Empty;
            return tail;
        }

        internal string Trim(string rawText)
        {
            int match = StopSequenceMatcher.FirstMatch(rawText, _sequences);
            return match < 0 ? rawText : rawText[..match];
        }
    }
}
