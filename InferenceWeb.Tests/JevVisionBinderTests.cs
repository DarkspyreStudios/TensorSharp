// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;
using TensorSharp.Server.Jev;

namespace InferenceWeb.Tests;

/// <summary>
/// The rules that keep a Jev image request pointing at the right pixels: every chunk prompt gets
/// its own expanded spans, a read installs the set belonging to ITS prompt, repeated reads of one
/// chunk do not reinstall (which would invalidate the retained prompt K/V on every sample), and
/// nothing is left installed when the request ends.
/// </summary>
public sealed class JevVisionBinderTests
{
    [Fact]
    public void TextOnlyRequestTouchesNoVisionState()
    {
        var target = new RecordingTarget();
        using var binder = new JevVisionBinder(target, []);

        Assert.False(binder.HasImages);
        Assert.Null(binder.ImagePathsOrNull());
        int[] prompt = [1, 2, 3];
        Assert.Same(prompt, binder.Expand([], prompt));
        binder.Read(prompt, [0], [0], [[1]], default);

        Assert.Empty(target.Injector.Installed);
        Assert.Equal(0, target.Cleared);
    }

    [Fact]
    public void EachChunkReadsItsOwnSpansAndOneChunkInstallsOnce()
    {
        var target = new RecordingTarget();
        var binder = new JevVisionBinder(target, ["/tmp/a.png"]);

        int[] first = binder.Expand([], [1, 2]);
        int[] second = binder.Expand([], [3, 4]);
        Assert.NotEqual(first, second);

        // Two samples of chunk one, then chunk two, then back to chunk one: the switch is what
        // reinstalls, never the repeat.
        binder.Read(first, [0], [0], [[1]], default);
        binder.Read(first, [0], [0], [[1]], default);
        binder.Read(second, [0], [0], [[1]], default);
        binder.Read(first, [0], [0], [[1]], default);

        Assert.Equal(3, target.Injector.Installed.Count);
        Assert.Equal(target.Injector.Installed[0], target.Injector.Installed[2]);
        Assert.NotEqual(target.Injector.Installed[0], target.Injector.Installed[1]);
        // One clear per install: the previous chunk's spans are released before the next land.
        Assert.Equal(3, target.Cleared);
        Assert.Equal(4, target.Reads);

        binder.Dispose();
        Assert.Equal(4, target.Cleared);
        Assert.Empty(target.Injector.Prepared);
    }

    [Fact]
    public void RefusesAPromptWhoseImagesNeverReachedTheModel()
    {
        var target = new RecordingTarget { Injector = { PrepareSpans = false } };
        using var binder = new JevVisionBinder(target, ["/tmp/a.png"]);

        Assert.Throws<JevModelUnavailableException>(() => binder.Expand([], [1, 2]));
        Assert.Equal(0, target.Reads);
    }

    [Fact]
    public void RefusesAReadForAPromptItNeverPrepared()
    {
        var target = new RecordingTarget();
        using var binder = new JevVisionBinder(target, ["/tmp/a.png"]);
        binder.Expand([], [1, 2]);

        Assert.Throws<InvalidOperationException>(() => binder.Read([9, 9], [0], [0], [[1]], default));
    }

    private sealed class RecordingTarget : IJevVisionTarget
    {
        public FakeInjector Injector { get; } = new();
        IMultimodalInjector IJevVisionTarget.Injector => Injector;
        public int Cleared { get; private set; }
        public int Reads { get; private set; }

        public void ClearVisionEmbeddings() => Cleared++;

        public float[][] ReadStructured(int[] prompt, int[] canvas, int[] positions, int[][] tokenIds,
            CancellationToken cancellationToken)
        {
            Reads++;
            return [[1f]];
        }
    }

    private sealed class FakeInjector : IMultimodalInjector
    {
        public bool PrepareSpans { get; set; } = true;
        public HashSet<string> Prepared { get; } = new(StringComparer.Ordinal);
        public List<string> Installed { get; } = new();

        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens, string requestId = null)
        {
            if (PrepareSpans) Prepared.Add(requestId);
            // One image soft row, so an expanded prompt is a distinct array as well as longer.
            return [.. inputTokens, 0];
        }

        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null)
        {
            if (!Prepared.Contains(requestId)) return false;
            Installed.Add(requestId);
            return true;
        }

        public void ClearPreparedPromptState(string requestId) => Prepared.Remove(requestId);
        public bool HasPendingEmbeddings(string requestId) => Prepared.Contains(requestId);
        public void LoadProjectors(string mmProjPath) { }
        public bool QueuePromptEmbeddingsForSlice(int promptStartToken, int tokenCount, string requestId = null) => false;
        public int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null) => reusablePrefixTokenCount;
        public int ClampTrimStart(int trimStartTokenCount, string requestId = null) => trimStartTokenCount;
        public void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null) { }
    }
}
