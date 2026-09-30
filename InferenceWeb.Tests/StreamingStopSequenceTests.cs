using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class StreamingStopSequenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void EveryStopSplit_OmitsTheWholeStopAndTrailingChunk(int split)
    {
        const string stop = "<STOP>";
        var filter = new StreamingStopSequenceFilter(new[] { stop });
        var output = new StringBuilder(filter.Append("answer " + stop[..split]));
        output.Append(filter.Append(stop[split..] + "hidden"));
        output.Append(filter.Append("also hidden"));
        output.Append(filter.Complete());
        Assert.Equal("answer ", output.ToString());
        Assert.True(filter.Stopped);
        Assert.Equal(0, filter.PendingLength);
    }

    [Theory]
    [InlineData("END", "STOP")]
    [InlineData("STOP", "END")]
    public void EarliestMatch_DoesNotDependOnListOrder(string first, string second)
    {
        var filter = new StreamingStopSequenceFilter(new[] { first, second });
        Assert.Equal("answer ", filter.Append("answer STOP hidden END"));
        Assert.Equal(("answer ", true), new TokenSampler(new SamplingConfig
        { StopSequences = new() { first, second } }).CheckStopSequences("answer STOP hidden END"));
    }

    [Theory]
    [InlineData("ab", "abc", "ab")]
    [InlineData("abcd", "bc", "abc")]
    [InlineData("aba", "bab", "abab")]
    public void OverlappingSequences_StopAtFirstCompletedMatch(string first, string second, string text)
    {
        var filter = new StreamingStopSequenceFilter(new[] { first, second });
        string output = string.Concat(text.Select(c => filter.Append(c.ToString())));
        Assert.Equal(first == "abcd" ? "a" : string.Empty, output);
        Assert.True(filter.Stopped);
        Assert.Empty(filter.Complete());
    }

    [Fact]
    public void UnmatchedPrefix_IsFlushedExactlyOnceAtCompletion()
    {
        var filter = new StreamingStopSequenceFilter(new[] { "<STOP>" });
        Assert.Equal("answer ", filter.Append("answer <ST"));
        Assert.Equal(3, filter.PendingLength);
        Assert.Equal("<ST", filter.Complete());
        Assert.Empty(filter.Complete());
        Assert.Throws<InvalidOperationException>(() => filter.Append("later"));
        Assert.False(filter.Stopped);
    }

    [Fact]
    public void MismatchedPrefix_StreamsWithoutWaitingForTheResponseToEnd()
    {
        var filter = new StreamingStopSequenceFilter(new[] { "<STOP>" });
        Assert.Equal("a", filter.Append("a<ST"));
        Assert.Equal("<STx", filter.Append("x<STO"));
        Assert.Equal(4, filter.PendingLength);
        Assert.Equal("<STO", filter.Complete());
    }

    [Fact]
    public void UnicodeStopAndAnswer_AreMatchedOrdinallyAcrossChunks()
    {
        var filter = new StreamingStopSequenceFilter(new[] { "\u7d42\u308f\u308a" });
        Assert.Equal("\u2713 ", filter.Append("\u2713 \u7d42"));
        Assert.Empty(filter.Append("\u308f"));
        Assert.Empty(filter.Append("\u308a trailing"));
        Assert.Empty(filter.Complete());
        Assert.True(filter.Stopped);
    }

    [Fact]
    public void ResponseLength_DoesNotGrowThePendingBuffer()
    {
        var filter = new StreamingStopSequenceFilter(new[] { "aaaaab" });
        int emitted = 0;
        for (int i = 0; i < 10_000; i++)
        {
            emitted += filter.Append("a").Length;
            Assert.InRange(filter.PendingLength, 0, 5);
        }
        Assert.Equal(10_000, emitted + filter.Complete().Length);
    }

    [Fact]
    public void StopLists_AreSnapshottedForBothConsumers()
    {
        var config = new SamplingConfig { StopSequences = new() { "STOP" } };
        var filter = new StreamingStopSequenceFilter(config.StopSequences);
        var sampler = new TokenSampler(config);
        config.StopSequences[0] = "OTHER";
        config.StopSequences.Clear();
        Assert.Equal("answer ", filter.Append("answer STOP hidden"));
        Assert.Equal(("answer ", true), sampler.CheckStopSequences("answer STOP hidden"));
    }

    [Theory]
    [InlineData("")]
    [InlineData(null)]
    public void InvalidStop_FailsBeforeSamplingOrStreaming(string? stop)
    {
        var stops = new List<string> { stop! };
        Assert.Throws<ArgumentException>(() => new StreamingStopSequenceFilter(stops));
        Assert.Throws<ArgumentException>(() => new TokenSampler(new SamplingConfig { StopSequences = stops }));
    }

    [Fact]
    public async Task Pipeline_OmitsStopButRetainsTheRealRawTokensForContinuation()
    {
        using var fixture = new PipelineFixture("answer <ST", "OP> hidden");
        using var session = new ChatSession();
        var history = new List<ChatMessage> { new() { Role = "user", Content = "question" } };
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in fixture.Stream(session, history))
            updates.Add(update);
        Assert.Equal("answer ", string.Concat(updates.Select(u => u.Piece)));
        ChatStreamUpdate done = Assert.Single(updates, u => u.Done);
        Assert.Equal("stop_sequence", done.FinishReason);
        Assert.Equal(new[] { 1, 2 }, done.RawOutputTokens.Take(2));
        Assert.Contains("<STOP>", fixture.Model.Tokenizer.Decode(done.RawOutputTokens.ToList()), StringComparison.Ordinal);

        var next = new List<ChatMessage>(history)
        {
            new() { Role = "assistant", Content = "answer " },
            new() { Role = "user", Content = "continue" },
        };
        var augmented = session.Transcripts.Augment(next).History;
        ChatMessage restored = augmented[1];
        Assert.Equal(done.RawOutputTokens, restored.RawOutputTokens);
        Assert.Equal("answer ", restored.Content);
        Assert.Contains("<STOP>", fixture.Model.Tokenizer.Decode(restored.RawOutputTokens.ToList()), StringComparison.Ordinal);
        Assert.Equal(done.RawGenerationSuffix, restored.RawGenerationSuffix);
    }

    [Fact]
    public async Task Pipeline_NaturalEndFlushesUnmatchedPrefixBeforeTerminal()
    {
        using var fixture = new PipelineFixture("answer <ST");
        using var session = new ChatSession();
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in fixture.Stream(session, new() { new() { Role = "user", Content = "question" } }))
            updates.Add(update);
        Assert.Equal("answer <ST", string.Concat(updates.Select(u => u.Piece)));
        Assert.True(updates[^1].Done);
        Assert.NotEqual("stop_sequence", updates[^1].FinishReason);
        Assert.Equal("<ST", updates[^2].Piece);
    }

    [Fact]
    public async Task Pipeline_NoStopsStreamsTheOriginalChunksUnchanged()
    {
        using var fixture = new PipelineFixture("answer <ST", "OP> visible");
        using var session = new ChatSession();
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in fixture.Stream(session, new() { new() { Role = "user", Content = "question" } },
            sampling: SamplingConfig.Greedy))
            updates.Add(update);
        Assert.Equal(new[] { "answer <ST", "OP> visible" }, updates.Where(u => !u.Done).Select(u => u.Piece));
        Assert.True(updates[^1].Done);
        Assert.NotEqual("stop_sequence", updates[^1].FinishReason);
    }

    [Fact]
    public async Task Pipeline_InvalidStopFailsBeforeModelOrEngineExecution()
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance);
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance);
        using var session = new ChatSession();
        var pipeline = new ChatGenerationPipeline(lifecycle, host, new KVCachePromptRenderer(new PlainRenderer()),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await foreach (var update in pipeline.ChatStreamWithMetricsAsync(session, new(), 16, CancellationToken.None,
                new SamplingConfig { StopSequences = new() { string.Empty } }))
                Assert.Fail("An invalid stop cannot produce an event.");
        });
    }

    [Fact]
    public async Task Pipeline_CallerMutationCannotChangeAnActiveStopPolicy()
    {
        using var fixture = new PipelineFixture("answer <ST", "OP> hidden");
        using var session = new ChatSession();
        var sampling = new SamplingConfig { Temperature = 0, StopSequences = new() { "<STOP>" } };
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in fixture.Stream(session, new() { new() { Role = "user", Content = "question" } }, sampling: sampling))
        {
            updates.Add(update);
            sampling.StopSequences.Clear();
        }
        Assert.Equal("answer ", string.Concat(updates.Select(u => u.Piece)));
        Assert.Equal("stop_sequence", updates[^1].FinishReason);
    }

    [Fact]
    public async Task Pipeline_CancellationFlushesUnmatchedPrefixAndRecordsTheEmittedTurn()
    {
        using var fixture = new PipelineFixture("answer <ST", "unseen");
        using var session = new ChatSession();
        using var cancellation = new CancellationTokenSource();
        var history = new List<ChatMessage> { new() { Role = "user", Content = "question" } };
        var updates = new List<ChatStreamUpdate>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var update in fixture.Stream(session, history, cancellation.Token))
            {
                updates.Add(update);
                cancellation.Cancel();
            }
        });
        Assert.Equal("answer <ST", string.Concat(updates.Select(u => u.Piece)));
        Assert.DoesNotContain(updates, u => u.Done);
        var next = new List<ChatMessage>(history)
        {
            new() { Role = "assistant", Content = "answer <ST" },
            new() { Role = "user", Content = "continue" },
        };
        Assert.NotNull(session.Transcripts.Augment(next).History[1].RawOutputTokens);
    }

    private sealed class PipelineFixture : IDisposable
    {
        private readonly string _directory = Path.Combine(Path.GetTempPath(), "stop-stream-" + Guid.NewGuid().ToString("N"));
        private readonly ModelLifecycleService _lifecycle;
        private readonly InferenceEngineHost _host;
        private readonly ChatGenerationPipeline _pipeline;
        internal ScriptedModel Model { get; }

        internal PipelineFixture(params string[] pieces)
        {
            Directory.CreateDirectory(_directory);
            string path = Path.Combine(_directory, "header.gguf");
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(0UL);
                writer.Write(new byte[8]);
            }
            Model = new ScriptedModel(path, pieces);
            _lifecycle = new ModelLifecycleService(NullLogger.Instance, (_, _, _, _) => Model);
            _lifecycle.LoadModel(path, null, "cpu");
            _host = new InferenceEngineHost(_lifecycle, NullLogger.Instance)
            {
                SchedulerConfigOverride = new()
                {
                    BlockSize = 8,
                    NumBlocks = 32,
                    MaxNumBatchedTokens = 32,
                    MaxPrefillChunkSize = 16,
                    MaxNumRunningSequences = 1,
                    EnablePrefixCaching = false,
                    StopRepetition = false,
                },
            };
            _pipeline = new ChatGenerationPipeline(_lifecycle, _host, new KVCachePromptRenderer(new PlainRenderer()),
                new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        }

        internal IAsyncEnumerable<ChatStreamUpdate> Stream(ChatSession session, List<ChatMessage> history,
            CancellationToken cancellationToken = default, SamplingConfig? sampling = null)
            => _pipeline.ChatStreamWithMetricsAsync(session, history, 16, cancellationToken,
                sampling ?? new SamplingConfig { Temperature = 0, StopSequences = new() { "<STOP>" } });

        public void Dispose()
        {
            _host.Dispose();
            _lifecycle.Dispose();
            Directory.Delete(_directory, recursive: true);
        }
    }

    private sealed class ScriptedModel : ModelBase
    {
        internal ScriptedModel(string path, string[] pieces) : base(path, BackendType.Cpu)
        {
            Tokenizer = new PieceTokenizer(pieces);
            Config = new ModelConfig { Architecture = "stop-fixture", VocabSize = Tokenizer.VocabSize, NumLayers = 1 };
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override long ComputeKVBlockByteSize(int tokenCount) => tokenCount;
        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        { destination.Clear(); return true; }
        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            var logits = new float[Tokenizer.VocabSize];
            logits[Math.Min(tokens[^1] + 1, Tokenizer.VocabSize - 1)] = 10;
            return logits;
        }
        protected override void ResetKVCacheCore() { }
    }

    private sealed class PlainRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false) => "prompt";
    }

    private sealed class PieceTokenizer : ITokenizer
    {
        internal PieceTokenizer(string[] pieces) => Vocab = new[] { "prompt" }.Concat(pieces).Append(string.Empty).ToArray();
        public string[] Vocab { get; }
        public int BosTokenId => 0;
        public int[] EosTokenIds => new[] { Vocab.Length - 1 };
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new() { 0 };
        public string Decode(List<int> ids) => string.Concat(ids.Select(id => Vocab[id]));
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == Vocab.Length - 1;
        public int LookupToken(string tokenStr) => Array.IndexOf(Vocab, tokenStr);
    }
}
