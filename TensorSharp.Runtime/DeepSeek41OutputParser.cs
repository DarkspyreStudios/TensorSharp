// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// V4.1 uses spaced DSML tags: &lt;｜DSML｜ calls&gt;,
    /// &lt;｜DSML｜ invoke&gt; and &lt;｜DSML｜ parameter&gt;.
    /// String arguments are lossless, and incomplete calls are never dispatched.
    /// </summary>
    public sealed class DeepSeek41OutputParser : DeepSeek4OutputParser
    {
        public DeepSeek41OutputParser() : base(true) { }

        internal static bool TryParseParameters(string body, out Dictionary<string, object> arguments)
        {
            const string dsmlOpen = "<｜DSML｜ parameter name=\"";
            const string plainOpen = "<parameter name=\"";
            const string dsmlClose = "</｜DSML｜ parameter>";
            const string plainClose = "</parameter>";
            arguments = new Dictionary<string, object>();
            int pos = 0;
            while (true)
            {
                while (pos < body.Length && char.IsWhiteSpace(body[pos]))
                    pos++;
                if (pos == body.Length)
                    return true;

                // The Q2_K checkpoint also emits plain XML parameter tags inside
                // complete DSML invokes, and occasionally a plain closing tag for
                // a DSML opening tag. Both have unambiguous name/value boundaries.
                int prefixLength = body.AsSpan(pos).StartsWith(dsmlOpen, StringComparison.Ordinal) ? dsmlOpen.Length
                    : body.AsSpan(pos).StartsWith(plainOpen, StringComparison.Ordinal) ? plainOpen.Length : 0;
                if (prefixLength == 0)
                    return false; // Never turn unrecognised argument markup into {}.
                int keyEnd = body.IndexOf('"', pos + prefixLength);
                if (keyEnd <= pos + prefixLength)
                    return false;
                string key = body.Substring(pos + prefixLength, keyEnd - pos - prefixLength);
                int tagEnd = body.IndexOf('>', keyEnd + 1);
                if (tagEnd < 0)
                    return false;
                string attributes = body.Substring(keyEnd + 1, tagEnd - keyEnd - 1).Trim();
                bool isString = attributes.Length == 0 || attributes == "string=\"true\"";
                if (!isString && attributes != "string=\"false\"")
                    return false;

                int dsmlEnd = body.IndexOf(dsmlClose, tagEnd + 1, StringComparison.Ordinal);
                int plainEnd = body.IndexOf(plainClose, tagEnd + 1, StringComparison.Ordinal);
                bool usePlain = plainEnd >= 0 && (dsmlEnd < 0 || plainEnd < dsmlEnd);
                int valueEnd = usePlain ? plainEnd : dsmlEnd;
                if (valueEnd < 0)
                    return false;
                string value = body.Substring(tagEnd + 1, valueEnd - tagEnd - 1);
                // A missing close must not consume the next parameter as part of
                // this value. Other literal markup in a string is preserved.
                if (value.Contains("<｜DSML｜ parameter", StringComparison.Ordinal) ||
                    value.Contains("<parameter", StringComparison.Ordinal) ||
                    value.Contains("<｜DSML｜ invoke", StringComparison.Ordinal))
                    return false;

                object parsed = value;
                if (!isString)
                {
                    try
                    {
                        using var document = JsonDocument.Parse(value);
                        parsed = ChatMlOutputParser.JsonElementToObject(document.RootElement);
                    }
                    catch (JsonException)
                    {
                        return false;
                    }
                }
                if (!arguments.TryAdd(key, parsed))
                    return false;
                pos = valueEnd + (usePlain ? plainClose.Length : dsmlClose.Length);
            }
        }
    }
}
