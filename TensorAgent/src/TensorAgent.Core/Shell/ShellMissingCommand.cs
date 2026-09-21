// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;

namespace TensorAgent.Core.Shell;

/// <summary>
/// What to say when a command does not exist here.
///
/// <para>
/// "bc: command not found" is a true sentence and a dead end, and a dead end is where a
/// model stops using the shell. Observed on a phone: asked for the days between two
/// dates, the model correctly reached for the shell, correctly got one date's epoch
/// seconds out of it, then reached for <c>bc</c> to do the subtraction, was told 127,
/// and finished the job in its own head — arriving at 164 days instead of 20,863 and
/// presenting it with a formula. Nothing in the output suggested a next move, so it
/// made one up.
/// </para>
/// <para>
/// The usual answer to a missing program is to install it, and here that is not
/// available to anybody: iOS runs no child processes and will not execute a binary that
/// was not signed into the app bundle, so there is no package manager to reach for and
/// no <c>bc</c> to fetch. What IS available is a full CPython and a JavaScript engine,
/// either of which does everything the missing tool would have. When the user has
/// allowed the network, CPython can also add libraries that ship as pure-Python wheels;
/// JavaScript packages and native dependencies cannot be added. So the message names
/// the thing that works instead of the thing that does not exist, which is the difference
/// between a model that recovers in one round and a model that gives up.
/// </para>
/// </summary>
internal static class ShellMissingCommand
{
    /// <summary>
    /// The advice for a specific missing command. Keyed by the name a model actually
    /// types; the value completes the sentence "… is not available here; ".
    /// </summary>
    private static readonly Dictionary<string, string> Instead = new(StringComparer.Ordinal)
    {
        // The one this was written for, and its relatives.
        ["bc"] = "do integer arithmetic with $(( ... )) and anything with decimals in python3, "
               + "for example: python3 -c 'print((1234567 - 89) / 86400)'",
        ["dc"] = "do integer arithmetic with $(( ... )) and anything with decimals in python3",
        ["expr"] = "use $(( ... )), which this shell supports",

        // Other languages: there is exactly one interpreter here, and it is not theirs.
        ["perl"] = "write it in python3",
        ["ruby"] = "write it in python3",
        ["php"] = "write it in python3",
        ["lua"] = "write it in python3 or node",
        ["jq"] = "parse JSON in python3, for example: "
               + "python3 -c 'import json,sys; print(json.load(sys.stdin)[\"key\"])'",

        // Nothing native can be built or run: the app is a signed bundle and iOS will
        // not execute anything that was not signed into it.
        ["gcc"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["cc"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["clang"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["make"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["cmake"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["java"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["go"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["cargo"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",
        ["rustc"] = "nothing can be compiled or run as a native program on this device; write it in python3 or node",

        // System package managers. Saying "not found" here is the least useful possible
        // answer, because the model is in the middle of trying to fix a missing tool.
        ["apt"] = PackageManagers,
        ["apt-get"] = PackageManagers,
        ["brew"] = PackageManagers,
        ["yum"] = PackageManagers,
        ["dnf"] = PackageManagers,
        ["pacman"] = PackageManagers,
        ["apk"] = PackageManagers,
        ["port"] = PackageManagers,

        // Node itself is embedded, but its package manager is not. This needs explicit
        // advice rather than the generic "node is available": the latter sounds like a
        // transient missing command that can be repaired, and sends the model straight
        // back to another npm spelling.
        ["npm"] = "npm/JavaScript packages cannot be installed on this device; use Node's built-in modules "
                + "or write it in python3, whose pure-Python `none-any` wheels can be installed and which already "
                + "has numpy, Pillow, lxml, python-pptx and python-docx built in",
        ["npx"] = "this embedded JavaScript runtime is not Node.js and cannot execute npm packages or child processes; "
                + "a skill that requires native processes must run on a desktop host",

        ["sudo"] = "there is nothing to escalate to; run the command by itself",
        ["su"] = "there is nothing to escalate to; run the command by itself",
        ["systemctl"] = "there are no services on this device",
        ["docker"] = "there are no containers on this device; run the program directly in python3 or node",
        ["ssh"] = "this device cannot open a shell anywhere else",
        ["open"] = "there is no desktop to open a file on; print what you want the user to see",
        ["man"] = "there are no manual pages; every command here is a builtin and takes the usual POSIX options",
    };

    private const string PackageManagers =
        "there is no system package manager on this device and no native program can be installed. "
        + "Only Python libraries distributed as pure-Python `none-any` wheels can be installed when "
        + "network access is on: use `pip install <name>` or `python3 -m pip install <name>`; "
        + "numpy, Pillow, lxml, python-pptx, python-docx and openpyxl are already built in and need no install. "
        + "Dependencies are not resolved automatically, and npm/JavaScript packages cannot be installed";

    /// <summary>
    /// The whole message for <paramref name="name"/>: that it is missing, what to use
    /// instead when that is known, and — always — what this shell actually has.
    /// </summary>
    /// <param name="name">The command the model typed.</param>
    /// <param name="known">Every command that does exist, for the near-miss suggestion.</param>
    /// <param name="python">True when python3 can run here.</param>
    /// <param name="javaScript">True when node can run here.</param>
    public static string Describe(
        string name, IEnumerable<string> known, bool python, bool javaScript)
    {
        var message = new StringBuilder(name).Append(": command not found");

        // A typo is the cheapest possible fix and the model cannot see the list.
        if (Nearest(name, known) is { } close)
            message.Append(". Did you mean `").Append(close).Append("`?");

        if (Instead.TryGetValue(name, out string? advice))
        {
            message.Append(". `").Append(name).Append("` is not available here; ").Append(advice);
        }
        else if (python || javaScript)
        {
            // No specific advice, but the general fact still turns a dead end into a
            // move: this shell's commands are fixed and cannot be added to, and the
            // interpreters can do whatever the missing one would have.
            message.Append(". This shell's commands are built in and cannot be installed; ")
                .Append(python && javaScript ? "python3 and node are available"
                     : python ? "python3 is available" : "node is available")
                .Append(" and can do what this command would have");
        }

        return message.ToString();
    }

    /// <summary>
    /// The known command within one or two edits of <paramref name="name"/>, or null.
    /// Bounded to short names and a small distance so that a genuinely different word
    /// is not "corrected" into something the model never meant.
    /// </summary>
    private static string? Nearest(string name, IEnumerable<string> known)
    {
        if (name.Length < 2)
            return null;

        string? best = null;
        int bestDistance = int.MaxValue;
        int limit = name.Length <= 4 ? 1 : 2;
        foreach (string candidate in known)
        {
            if (Math.Abs(candidate.Length - name.Length) > limit)
                continue;
            int distance = Distance(name, candidate, limit);
            if (distance <= limit && distance < bestDistance)
            {
                bestDistance = distance;
                best = candidate;
            }
        }
        return best;
    }

    /// <summary>
    /// Damerau-Levenshtein distance (optimal string alignment), abandoned once it
    /// passes <paramref name="limit"/>.
    ///
    /// <para>
    /// Transpositions count as ONE edit, and that is not a detail: the typos a model
    /// makes are overwhelmingly transpositions — <c>ehco</c>, <c>gerp</c>, <c>mkdir</c>
    /// as <c>mkdri</c> — and plain Levenshtein scores every one of them as two, which
    /// is exactly the threshold that would have to be loosened to catch them, letting
    /// in genuinely different words at the same time.
    /// </para>
    /// </summary>
    private static int Distance(string a, string b, int limit)
    {
        int[] beforePrevious = new int[b.Length + 1];
        int[] previous = new int[b.Length + 1];
        int[] current = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++)
            previous[j] = j;

        for (int i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            int rowBest = current[0];
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                int value = Math.Min(
                    Math.Min(current[j - 1] + 1, previous[j] + 1),
                    previous[j - 1] + cost);
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
                    value = Math.Min(value, beforePrevious[j - 2] + 1);
                current[j] = value;
                rowBest = Math.Min(rowBest, value);
            }
            if (rowBest > limit)
                return limit + 1;
            int[] recycled = beforePrevious;
            beforePrevious = previous;
            previous = current;
            current = recycled;
        }
        return previous[b.Length];
    }
}
