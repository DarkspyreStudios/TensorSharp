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
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server
{
    internal static class ChatHistoryPreparer
    {
        private const string FileBackedCsvPrefix = "[Attached CSV available to tools:";

        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => PrepareHistoryForInference(history, arch, NullLogger.Instance);

        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch, logger);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        /// <summary>
        /// Replace browser-inlined CSV rows with a compact reference when the named
        /// uploads have already been staged into the tool workspace. Every row remains
        /// available without spending the model context on the whole table.
        /// </summary>
        /// <remarks>
        /// This runs on the server rather than only in the page. Saved conversations
        /// created by older builds already contain the large <c>[File: ...]</c>
        /// envelope and would otherwise reproduce the overflow whenever reopened.
        /// Non-CSV documents and files that were not successfully staged retain their
        /// existing inline behavior; data is never silently discarded when no tool can
        /// read the file. The returned inference-only clone no longer marks the file as
        /// inlined text, so ordinary history trimming can resume on later turns. The
        /// original request and its attachment metadata remain untouched for persistence
        /// and audit logging.
        /// </remarks>
        internal static List<ChatMessage> UseFileBackedCsvAttachments(
            List<ChatMessage> history,
            IReadOnlyDictionary<string, string> stagedFiles)
        {
            if (history == null || history.Count == 0 ||
                stagedFiles == null || stagedFiles.Count == 0)
            {
                return history;
            }

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                ChatMessage message = history[i];
                if (message?.TextFilePaths is not { Count: > 0 } ||
                    string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase) == false)
                {
                    continue;
                }

                List<AttachedTextFile> attached = AttachedTextFiles(message);
                List<AttachedTextFile> csvFiles = attached.Where(file => file.IsCsv).ToList();
                if (csvFiles.Count == 0 ||
                    csvFiles.Any(file => !stagedFiles.ContainsKey(file.Path)))
                    continue;

                string content = message.Content ?? string.Empty;
                if (content.StartsWith(FileBackedCsvPrefix, StringComparison.Ordinal))
                    continue;

                List<string> referencedCsvNames = csvFiles
                    .Select(file => stagedFiles[file.Path])
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
                var csvDisplayNames = new HashSet<string>(
                    csvFiles.Select(file => file.Name), StringComparer.OrdinalIgnoreCase);
                var csvPaths = new HashSet<string>(
                    csvFiles.Select(file => file.Path), StringComparer.Ordinal);

                // Both bundled pages put inlined text envelopes before the user's own
                // words. Parse at most the structured attachment count and remove only
                // staged CSV envelopes; a Markdown/PDF attached beside the table stays
                // inline and keeps its document-preservation semantics.
                if (content.StartsWith("[File: ", StringComparison.Ordinal) &&
                    TrySplitLeadingFileEnvelopes(
                        content, attached.Count, out List<FileEnvelope> envelopes, out string trailing))
                {
                    // Old clients did not send TextFileNames. When every structured
                    // text attachment has an envelope, their documented ordering gives
                    // us a safe fallback for a header that uses the friendly name while
                    // the path carries a generated storage name.
                    bool positionalMapping = envelopes.Count == attached.Count;
                    var kept = new List<string>();
                    for (int envelopeIndex = 0; envelopeIndex < envelopes.Count; envelopeIndex++)
                    {
                        FileEnvelope envelope = envelopes[envelopeIndex];
                        bool isCsv = csvDisplayNames.Contains(envelope.Name)
                            || (positionalMapping && attached[envelopeIndex].IsCsv);
                        if (!isCsv)
                            kept.Add(envelope.Content);
                    }
                    if (trailing.Length > 0)
                        kept.Add(trailing);
                    content = string.Join("\n\n", kept);
                }

                string names = string.Join(", ", referencedCsvNames.Select(QuoteFileName));
                string noun = referencedCsvNames.Count == 1 ? "file is" : "files are";
                string reference = FileBackedCsvPrefix + " " + names + "]\n"
                    + "The complete attached " + noun + " in the working directory. Before answering about "
                    + "the data, you must inspect it with read_file, shell, or an applicable table-analysis skill. "
                    + "Compute over all rows when reporting statistics; do not infer from the filename or a row "
                    + "preview. The rows are intentionally kept out of this prompt so a large table does not "
                    + "consume the model context.\n"
                    + "[End of attached CSV reference]";

                ChatMessage copy = CloneShallow(message);
                copy.Content = content.Length == 0 ? reference : reference + "\n\n" + content;
                // Offsets into the old, inlined body no longer name the same bytes.
                copy.ContentCacheBreakpoints = null;
                // These fields mean "the bytes are inline in Content". Remove the CSV
                // entries after externalising them, while retaining any companion prose
                // document that is still inline. Otherwise the overflow guard preserves
                // all prior turns forever. AttachmentPaths remains as provenance; the
                // tool plan already owns the CodeInputFile instances.
                RemoveFileBackedCsvMetadata(message, copy, csvPaths);
                copy.HasFileBackedTextAttachments = false;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = copy;
            }

            return prepared ?? history;
        }

        /// <summary>
        /// Put metadata-only CSV uploads back into the old full-inline shape when this
        /// request cannot offer a readable tool workspace. This is a compatibility and
        /// safety fallback: a small CSV still works with a non-tool model, while a large
        /// one reaches the normal attached-document context check and is rejected
        /// honestly instead of inviting an answer about rows the model never received.
        /// </summary>
        internal static List<ChatMessage> RestoreUnstagedFileBackedCsvAttachments(
            List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                ChatMessage message = history[i];
                if (message?.HasFileBackedTextAttachments != true ||
                    message.TextFilePaths is not { Count: > 0 })
                {
                    continue;
                }

                List<AttachedTextFile> csvFiles = AttachedTextFiles(message)
                    .Where(file => file.IsCsv)
                    .ToList();
                if (csvFiles.Count == 0)
                    continue;

                var envelopes = new List<string>();
                // The display name is not an identity. Two uploads from different
                // directories may both be called responses.csv, and the tool workspace
                // deliberately refuses to collapse those onto one file. Preserve both
                // bodies in this no-tools fallback; only an exact repeated source path
                // is redundant.
                var seenPaths = new HashSet<string>(StringComparer.Ordinal);
                foreach (AttachedTextFile file in csvFiles)
                {
                    if (!seenPaths.Add(file.Path))
                        continue;
                    string fullText = TextUploadHelper.PreserveFullText(File.ReadAllText(file.Path));
                    envelopes.Add("[File: " + file.Name + "]\n" + fullText + "\n[End of file]");
                }

                ChatMessage copy = CloneShallow(message);
                string prefix = string.Join("\n\n", envelopes);
                copy.Content = string.IsNullOrEmpty(message.Content)
                    ? prefix
                    : prefix + "\n\n" + message.Content;
                copy.ContentCacheBreakpoints = null;
                copy.HasFileBackedTextAttachments = false;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = copy;
            }

            return prepared ?? history;
        }

        private readonly record struct AttachedTextFile(string Path, string Name, bool IsCsv);

        private readonly record struct FileEnvelope(string Name, string Content);

        private static List<AttachedTextFile> AttachedTextFiles(ChatMessage message)
        {
            var result = new List<AttachedTextFile>();
            for (int i = 0; i < message.TextFilePaths.Count; i++)
            {
                string path = message.TextFilePaths[i] ?? string.Empty;
                string supplied = message.TextFileNames != null && i < message.TextFileNames.Count
                    ? message.TextFileNames[i]
                    : null;
                string name = Path.GetFileName(string.IsNullOrWhiteSpace(supplied) ? path : supplied);
                if (name.Length == 0)
                    continue;
                bool isCsv = string.Equals(Path.GetExtension(name), ".csv", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(Path.GetExtension(path), ".csv", StringComparison.OrdinalIgnoreCase);
                result.Add(new AttachedTextFile(path, name, isCsv));
            }
            return result;
        }

        private static void RemoveFileBackedCsvMetadata(
            ChatMessage source, ChatMessage copy, IReadOnlySet<string> csvPaths)
        {
            var paths = new List<string>();
            List<string> names = source.TextFileNames != null ? new List<string>() : null;
            for (int i = 0; i < source.TextFilePaths.Count; i++)
            {
                string path = source.TextFilePaths[i] ?? string.Empty;
                string supplied = source.TextFileNames != null && i < source.TextFileNames.Count
                    ? source.TextFileNames[i]
                    : null;
                if (csvPaths.Contains(path))
                    continue;

                paths.Add(path);
                names?.Add(string.IsNullOrWhiteSpace(supplied) ? path : supplied);
            }

            copy.TextFilePaths = paths.Count > 0 ? paths : null;
            copy.TextFileNames = paths.Count > 0 ? names : null;
        }

        private static bool TrySplitLeadingFileEnvelopes(
            string content,
            int maximumEnvelopeCount,
            out List<FileEnvelope> envelopes,
            out string remainder)
        {
            const string startMarker = "[File: ";
            const string endMarker = "\n[End of file]";
            envelopes = new List<FileEnvelope>();
            int cursor = 0;

            while (envelopes.Count < maximumEnvelopeCount)
            {
                while (cursor < content.Length && (content[cursor] == '\r' || content[cursor] == '\n'))
                    cursor++;

                if (!content.AsSpan(cursor).StartsWith(startMarker, StringComparison.Ordinal))
                    break;

                int headerEnd = content.IndexOf('\n', cursor);
                int nameEnd = headerEnd < 0
                    ? -1
                    : content.IndexOf(']', cursor + startMarker.Length);
                int envelopeEnd = headerEnd < 0
                    ? -1
                    : content.IndexOf(endMarker, headerEnd, StringComparison.Ordinal);
                if (nameEnd < 0 || nameEnd > headerEnd || envelopeEnd < 0)
                {
                    remainder = content;
                    envelopes.Clear();
                    return false;
                }

                string name = Path.GetFileName(
                    content.Substring(cursor + startMarker.Length, nameEnd - cursor - startMarker.Length));
                int afterEnvelope = envelopeEnd + endMarker.Length;
                envelopes.Add(new FileEnvelope(name, content.Substring(cursor, afterEnvelope - cursor)));
                cursor = afterEnvelope;
            }

            remainder = content[cursor..].TrimStart('\r', '\n');
            return true;
        }

        private static string QuoteFileName(string name)
        {
            // A file name belongs in prose, not in prompt structure. Strip control
            // characters so a crafted name cannot manufacture another instruction line.
            string safe = new(name.Where(c => !char.IsControl(c)).ToArray());
            return "'" + safe.Replace("'", "’", StringComparison.Ordinal) + "'";
        }

        /// <summary>
        /// Rebuild <paramref name="incoming"/> so its render reproduces the token
        /// sequence the live KV cache already holds.
        ///
        /// <para>
        /// Two mechanisms, walked with independent cursors over the two lists. A plain
        /// assistant turn gets the tracked <see cref="ChatMessage.RawOutputTokens"/>
        /// spliced onto it, exactly as before. A turn that ran the in-process tool loop
        /// is harder: the client sends ONE clean assistant message, but the cache holds
        /// the whole loop transcript — assistant round, tool result, assistant round,
        /// ... — so a positional walk breaks at the first tool result and the next
        /// render diverged thousands of tokens before the cache's end. The engine only
        /// rewinds a few trailing tokens (see BatchExecutor.MaxLiveContinuationRewindTokens),
        /// so that turn re-prefilled the entire conversation: measured on the pdf-skill
        /// incident, the follow-up turn re-prefilled 15.7k tokens (10.7s to first token)
        /// where 99% was reusable. The fix is EXPANSION: when the tracked history holds
        /// a tool transcript where the client sent one assistant message, substitute the
        /// transcript — raw tokens and all — so the rendered prefix stays byte-identical
        /// to the cache.
        /// </para>
        /// </summary>
        public static List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming, IReadOnlyList<ChatMessage> trackedHistory)
        {
            if (incoming == null)
                return null;

            var result = new List<ChatMessage>(incoming.Count);
            int t = 0;
            bool diverged = trackedHistory == null || trackedHistory.Count == 0;

            for (int i = 0; i < incoming.Count; i++)
            {
                ChatMessage src = incoming[i];

                if (diverged || t >= trackedHistory.Count || src.Role != trackedHistory[t].Role)
                {
                    // Past the tracked prefix, or the conversation genuinely diverges
                    // here (an edited or regenerated turn): everything from this point
                    // renders from the client's own content.
                    diverged = true;
                    result.Add(src);
                    continue;
                }

                ChatMessage tracked = trackedHistory[t];

                if (src.Role != "assistant")
                {
                    // Compare on Content for non-assistant roles only. Assistant content
                    // can be legitimately altered by the streaming output parser between
                    // turns.
                    if (!string.Equals(src.Content ?? string.Empty, tracked.Content ?? string.Empty, StringComparison.Ordinal))
                    {
                        diverged = true;
                        result.Add(src);
                        continue;
                    }

                    result.Add(src);
                    t++;
                    continue;
                }

                string nextIncomingRole = i + 1 < incoming.Count ? incoming[i + 1].Role : null;
                if (TryMatchToolTranscript(trackedHistory, t, src, nextIncomingRole, out int runLength))
                {
                    var expanded = new List<ChatMessage>(runLength);
                    for (int k = 0; k < runLength; k++)
                        expanded.Add(CloneShallow(trackedHistory[t + k]));
                    PreserveCollapsedCacheMarkers(src, expanded);
                    result.AddRange(expanded);
                    t += runLength;
                    continue;
                }

                bool useTracked = tracked.RawOutputTokens is { Count: > 0 }
                    && (src.RawOutputTokens == null || src.RawOutputTokens.Count == 0);

                if (useTracked)
                {
                    result.Add(new ChatMessage
                    {
                        Role = src.Role,
                        Content = src.Content,
                        ImagePaths = src.ImagePaths,
                        ImageTimestamps = src.ImageTimestamps,
                        AudioPaths = src.AudioPaths,
                        TextFilePaths = src.TextFilePaths,
                        TextFileNames = src.TextFileNames,
                        HasFileBackedTextAttachments = src.HasFileBackedTextAttachments,
                        IsVideo = src.IsVideo,
                        ToolCalls = src.ToolCalls,
                        ToolCallId = src.ToolCallId,
                        Thinking = src.Thinking,
                        RawOutputTokens = tracked.RawOutputTokens,
                        RawPromptTrailingWhitespace = tracked.RawPromptTrailingWhitespace,
                        RawGenerationSuffix = tracked.RawGenerationSuffix,
                        CacheControl = src.CacheControl,
                        ContentCacheBreakpoints = src.ContentCacheBreakpoints,
                    });
                }
                else
                {
                    result.Add(src);
                }
                t++;
            }
            return result;
        }

        /// <summary>Map cache markers from a client-visible assistant message onto
        /// the tracked assistant/tool transcript that replaces it. A message-level
        /// marker belongs after the whole expansion. Content offsets are mapped over
        /// the intermediate assistant text that TryMatchToolTranscript verified; if
        /// an offset reaches the final raw model round, mark its start conservatively
        /// because parser-stripped thinking text makes the exact offset unknowable.</summary>
        private static void PreserveCollapsedCacheMarkers(
            ChatMessage source, List<ChatMessage> expanded)
        {
            if (expanded.Count == 0) return;

            int lastAssistant = expanded.FindLastIndex(m => m.Role == "assistant");
            if (lastAssistant < 0) return;

            if (source.CacheControl != null)
            {
                expanded[lastAssistant].CacheControl = new CacheControlMarker
                {
                    Type = source.CacheControl.Type,
                };
            }

            if (source.ContentCacheBreakpoints == null
                || source.ContentCacheBreakpoints.Count == 0)
                return;

            foreach (int rawOffset in source.ContentCacheBreakpoints)
            {
                int remaining = Math.Max(0, rawOffset);
                bool mapped = false;
                for (int i = 0; i < lastAssistant; i++)
                {
                    if (expanded[i].Role != "assistant") continue;
                    int length = expanded[i].Content?.Length ?? 0;
                    if (remaining <= length)
                    {
                        AddMappedContentCacheBreakpoint(expanded[i], remaining);
                        mapped = true;
                        break;
                    }
                    remaining -= length;
                }

                if (!mapped)
                    AddMappedContentCacheBreakpoint(expanded[lastAssistant], 0);
            }
        }

        private static void AddMappedContentCacheBreakpoint(ChatMessage message, int offset)
        {
            message.AddContentCacheBreakpoint(offset);
            message.ContentCacheBreakpoints.Sort();
        }

        /// <summary>
        /// Does the tracked history hold, at <paramref name="start"/>, an in-process
        /// tool-loop transcript that <paramref name="src"/> is the clean client-visible
        /// form of? A transcript is a contiguous run of assistant and <c>role: "tool"</c>
        /// messages ending on an assistant, with at least one tool result — the shape
        /// the skills/code loop leaves behind. Verified by content: each intermediate
        /// round's parsed content concatenates, in order, into a prefix of what the
        /// client sent back (the final round's tracked content is the RAW model text and
        /// is not compared, the same tolerance the plain splice has always had).
        ///
        /// <para>
        /// Families that feed tool results back as user turns (Mistral 3) never form
        /// this shape, so they keep today's behavior — a conservative miss, never a
        /// wrong splice.
        /// </para>
        /// </summary>
        private static bool TryMatchToolTranscript(
            IReadOnlyList<ChatMessage> trackedHistory, int start, ChatMessage src,
            string nextIncomingRole, out int runLength)
        {
            runLength = 0;

            // A client whose NEXT message is a tool result is carrying the transcript
            // itself (an OpenAI-style tool flow): every message lines up one-to-one,
            // and expanding here would insert the tracked rounds a second time.
            if (nextIncomingRole == "tool")
                return false;

            // The run: assistant, then tool results and further assistant rounds, up to
            // the last assistant before anything else (the next turn's user message).
            int end = start;
            int lastAssistant = start;
            bool sawTool = false;
            while (end < trackedHistory.Count)
            {
                string role = trackedHistory[end].Role;
                if (role == "assistant")
                    lastAssistant = end;
                else if (role == "tool")
                    sawTool = true;
                else
                    break;
                end++;
            }

            if (!sawTool || lastAssistant == start)
                return false;

            // The client message must not itself be a tool-aware transcript: a caller
            // that sends its own tool messages is matched positionally, not expanded.
            if (src.RawOutputTokens is { Count: > 0 })
                return false;

            // Sanity: the intermediate rounds' parsed content must lead the client's
            // concatenated text. This is what separates "the same turn, re-sent clean"
            // from "a different conversation that happens to align" (an edited turn
            // falls through to the plain splice and diverges there, as before).
            string clientContent = src.Content ?? string.Empty;
            int offset = 0;
            for (int k = start; k < lastAssistant; k++)
            {
                if (trackedHistory[k].Role != "assistant")
                    continue;
                string piece = trackedHistory[k].Content ?? string.Empty;
                if (piece.Length == 0)
                    continue;
                if (offset + piece.Length > clientContent.Length
                    || string.CompareOrdinal(clientContent, offset, piece, 0, piece.Length) != 0)
                    return false;
                offset += piece.Length;
            }

            runLength = lastAssistant - start + 1;
            return true;
        }

        public static void UpdateTrackedHistory(
            List<ChatMessage> trackedHistory,
            List<ChatMessage> incomingHistory,
            string assistantText,
            List<int> generatedTokens,
            string? rawPromptTrailingWhitespace = null,
            string? rawGenerationSuffix = null)
        {
            trackedHistory.Clear();
            if (incomingHistory != null)
            {
                for (int i = 0; i < incomingHistory.Count; i++)
                    trackedHistory.Add(CloneShallow(incomingHistory[i]));
            }

            trackedHistory.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantText,
                RawOutputTokens = generatedTokens,
                RawPromptTrailingWhitespace = rawPromptTrailingWhitespace,
                RawGenerationSuffix = rawGenerationSuffix,
            });
        }

        public static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        public static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        public static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch, ILogger logger)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            // maxVideoFrames <= 0 means "no cap" (pure time-based extraction); leave history untouched.
            // Whether a family expands a video into per-frame images (and so needs the
            // cap) is declared with the rest of its chat protocol, not matched on here.
            bool capsFrames = ChatProtocolRegistry.For(arch)?.CapsVideoFrames ?? false;
            if (!capsFrames || maxVideoFrames <= 0 || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampledIndices = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames);
            var sampled = sampledIndices.Select(i => msg.ImagePaths[i]).ToList();

            // A Warning, not an Information: frames the user sent are being thrown away,
            // and the answer may miss what happened between the kept ones.
            (logger ?? NullLogger.Instance).LogWarning(LogEventIds.VideoFrameDownsample,
                "video.downsample originalFrames={OriginalFrames} sampledFrames={SampledFrames} architecture={Architecture}: " +
                "the video exceeds the per-message frame cap, so only the sampled, evenly spaced frames reach the model " +
                "and detail between them is lost. Raise VIDEO_MAX_FRAMES to keep more.",
                msg.ImagePaths.Count, sampled.Count, arch);

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                ImageTimestamps = msg.ImageTimestamps?.Count == msg.ImagePaths.Count
                    ? sampledIndices.Select(i => msg.ImageTimestamps[i]).ToList() : null,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                TextFilePaths = msg.TextFilePaths != null ? new List<string>(msg.TextFilePaths) : null,
                TextFileNames = msg.TextFileNames != null ? new List<string>(msg.TextFileNames) : null,
                HasFileBackedTextAttachments = msg.HasFileBackedTextAttachments,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                ToolCallId = msg.ToolCallId,
                Thinking = msg.Thinking,
                RawOutputTokens = msg.RawOutputTokens,
                RawPromptTrailingWhitespace = msg.RawPromptTrailingWhitespace,
                RawGenerationSuffix = msg.RawGenerationSuffix,
                CacheControl = msg.CacheControl,
                ContentCacheBreakpoints = msg.ContentCacheBreakpoints != null
                    ? new List<int>(msg.ContentCacheBreakpoints)
                    : null,
            };
        }

        private static ChatMessage CloneShallow(ChatMessage src)
        {
            return new ChatMessage
            {
                Role = src.Role,
                Content = src.Content,
                ImagePaths = src.ImagePaths,
                ImageTimestamps = src.ImageTimestamps,
                AudioPaths = src.AudioPaths,
                TextFilePaths = src.TextFilePaths,
                TextFileNames = src.TextFileNames,
                HasFileBackedTextAttachments = src.HasFileBackedTextAttachments,
                AttachmentPaths = src.AttachmentPaths,
                AttachmentNames = src.AttachmentNames,
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                ToolCallId = src.ToolCallId,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
                RawPromptTrailingWhitespace = src.RawPromptTrailingWhitespace,
                RawGenerationSuffix = src.RawGenerationSuffix,
                CacheControl = src.CacheControl != null
                    ? new CacheControlMarker { Type = src.CacheControl.Type }
                    : null,
                ContentCacheBreakpoints = src.ContentCacheBreakpoints != null
                    ? new List<int>(src.ContentCacheBreakpoints)
                    : null,
            };
        }
    }
}
