// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Embeddings;
using System.Text;

namespace InferenceWeb.Tests;

public sealed class EmbeddingModelTests
{
    [Fact]
    public void MalformedQkvExtraDimensionsFailBeforeNativeUpload()
    {
        using var fixture = new TinyEncoderFixture(invalidQkv: true);
        var error = Assert.Throws<InvalidOperationException>(() => EmbeddingModel.Load(fixture.Path, new() { Backend = "GGML_CPU" }));
        Assert.Contains("invalid dimensions", error.Message);
    }

    [Fact]
    public void ManagedBackendRejectsMalformedDimensionsAtLoad()
    {
        using var fixture = new TinyEncoderFixture(invalidQkv: true);
        var error = Assert.Throws<InvalidDataException>(() => EmbeddingModel.Load(fixture.Path));
        Assert.Contains("invalid dimensions", error.Message);
    }

    [Fact]
    public void ManagedBackendRejectsNonzeroDevice()
    {
        using var fixture = new TinyEncoderFixture();
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingModel.Load(fixture.Path, new() { Device = 1 }));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task DefaultBackendIsManagedAndMatchesIndependentPoolingCalculation(int pooling)
    {
        using var fixture = new TinyEncoderFixture(pooling: pooling);
        using var model = EmbeddingModel.Load(fixture.Path, new() { Threads = 2 });
        Assert.True(model.IsManaged);
        Assert.Equal("CPU", model.Backend);
        int[] tokens = [1, 0, 2];
        var actual = (await model.EmbedTokensAsync([tokens])).Embeddings[0];
        // Fixture projections are zero and normalization weights are one. Its
        // three residual layer norms reduce to centered, normalized input rows.
        var rows = tokens.Select((token, position) => Enumerable.Range(0, 32)
            .Select(i => (double)(float)Math.Sin(token * 32 + i + 1) + (float)(0.1 * Math.Cos(position * 32 + i + 1))).ToArray()).ToArray();
        foreach (var row in rows)
        for (int layerNorm = 0; layerNorm < 3; ++layerNorm)
        {
            double mean = row.Average();
            double variance = row.Sum(value => (value - mean) * (value - mean)) / row.Length;
            for (int i = 0; i < row.Length; ++i) row[i] = (row[i] - mean) / Math.Sqrt(variance + 1e-5);
        }
        double[] expected = pooling == 1 ? Enumerable.Range(0, 32).Select(i => rows.Average(row => row[i])).ToArray() : rows[pooling == 2 ? 0 : 2];
        double norm = Math.Sqrt(expected.Sum(value => value * value));
        for (int i = 0; i < expected.Length; ++i) Assert.InRange(Math.Abs(actual[i] - expected[i] / norm), 0, 1e-6);
        AssertNormalized(actual, 32);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task ManagedAttentionTilesMatchScalarEncoderAcrossSequenceAndQueryTails(int pooling)
    {
        using var fixture = new TinyEncoderFixture(pooling: pooling, projectionWeights: true, heads: 4);
        using var model = EmbeddingModel.Load(fixture.Path, new() { Threads = 3 });
        int[][][] cases = [[[1], [1, 0, 2], [1, 0, 0, 0, 2], [1, 2, 0, 1, 0, 2, 1]],
            [[1, 0, 2], [1, 0, 0, 0, 2]]]; // Total8, last sequence starts at3: vector-tail regression.
        foreach (var inputs in cases)
        {
            var batch = await model.EmbedTokensAsync(inputs);
            for (int sequence = 0; sequence < inputs.Length; ++sequence)
            {
                double[] expected = ScalarFixtureEncoder(inputs[sequence], pooling);
                for (int dimension = 0; dimension < 32; ++dimension)
                    Assert.InRange(Math.Abs(batch.Embeddings[sequence][dimension] - expected[dimension]), 0, 2e-6);
            }
        }
    }

    internal static double[] ScalarFixtureEncoder(int[] tokens, int pooling, bool nativeHalfGelu = false)
    {
        static double[] Norm(double[] row)
        {
            double mean = row.Average();
            double scale = 1.0 / Math.Sqrt(row.Sum(value => (value - mean) * (value - mean)) / row.Length + 1e-5);
            return row.Select(value => (value - mean) * scale).ToArray();
        }
        double[][] input = tokens.Select((token, position) => Norm(Enumerable.Range(0, 32)
            .Select(i => (double)(float)Math.Sin(token * 32 + i + 1) + (float)(0.1 * Math.Cos(position * 32 + i + 1))).ToArray())).ToArray();
        double[][] encoded = new double[tokens.Length][];
        for (int query = 0; query < tokens.Length; ++query)
        {
            if (pooling != 1 && query != (pooling == 2 ? 0 : tokens.Length - 1)) continue;
            var attention = new double[32];
            for (int head = 0; head < 4; ++head)
            {
                var probabilities = new double[tokens.Length];
                for (int key = 0; key < tokens.Length; ++key)
                {
                    double score = 0;
                    for (int dimension = head * 8; dimension < (head + 1) * 8; ++dimension)
                        score += input[query][dimension] * input[key][dimension];
                    probabilities[key] = score * 0.04 / Math.Sqrt(8);
                }
                double max = probabilities.Max();
                probabilities = probabilities.Select(value => Math.Exp(value - max)).ToArray();
                double sum = probabilities.Sum();
                for (int dimension = head * 8; dimension < (head + 1) * 8; ++dimension)
                for (int key = 0; key < tokens.Length; ++key)
                    attention[dimension] += probabilities[key] / sum * input[key][dimension] * 0.2;
            }
            var residual = Norm(Enumerable.Range(0, 32).Select(i => input[query][i] + attention[i] * 0.2).ToArray());
            encoded[query] = Norm(residual.Select(value =>
            {
                double x = value * 0.2;
                double gelu = 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
                if (nativeHalfGelu)
                {
                    // ggml-cpu/vec.h defines GGML_GELU_FP16: F32 GELU rounds
                    // both its argument and lookup-table result through F16.
                    float rounded = (float)(Half)(float)x;
                    gelu = (float)(Half)(0.5f * rounded * (1.0f + MathF.Tanh(
                        0.7978845608028654f * rounded * (1.0f + 0.044715f * rounded * rounded))));
                }
                return value + 0.2 * gelu;
            }).ToArray());
        }
        var result = pooling == 1 ? Enumerable.Range(0, 32).Select(i => encoded.Average(row => row[i])).ToArray() : encoded[pooling == 2 ? 0 : encoded.Length - 1];
        double length = Math.Sqrt(result.Sum(value => value * value));
        return result.Select(value => value / length).ToArray();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task ManagedLongAttentionSchedulingPreservesScalarResultsAndMixedBatchIsolation(int pooling)
    {
        using var fixture = new TinyEncoderFixture(pooling: pooling, projectionWeights: true, heads: 4, context: 1025);
        using var model = EmbeddingModel.Load(fixture.Path, new() { Threads = 3 });
        int[][] inputs = new[] { 3, 1023, 1024, 1025 }.Select(length =>
            Enumerable.Range(0, length).Select(position => (position * 7 + length) % 3).ToArray()).ToArray();
        var batch = await model.EmbedTokensAsync(inputs);
        for (int sequence = 0; sequence < inputs.Length; ++sequence)
        {
            // The mixed batch uses tiled attention; the short and 1023-token
            // single requests retain whole-row softmax. Both must
            // agree with the independent full encoder and with each other.
            double[] expected = ScalarFixtureEncoder(inputs[sequence], pooling);
            float[] single = (await model.EmbedTokensAsync([inputs[sequence]])).Embeddings[0];
            for (int dimension = 0; dimension < 32; ++dimension)
            {
                Assert.InRange(Math.Abs(batch.Embeddings[sequence][dimension] - single[dimension]), 0, 2e-6);
                Assert.InRange(Math.Abs(batch.Embeddings[sequence][dimension] - expected[dimension]), 0, 2e-6);
            }
        }
    }

    [Fact]
    public void VocabularyAndEmbeddingRowMismatchIsRejectedAtLoad()
    {
        using var fixture = new TinyEncoderFixture(vocabularyMismatch: true);
        var error = Assert.Throws<InvalidDataException>(() => EmbeddingModel.Load(fixture.Path));
        Assert.Contains("tokenizer vocabulary", error.Message);
    }

    [ModelFact("TENSORSHARP_MINILM_EMBED_MODEL")]
    public async Task MeanPoolingMasksPaddingPreservesOrderAndMatchesIndividualInference()
    {
        using var model = Load("TENSORSHARP_MINILM_EMBED_MODEL");
        await VerifyBatch(model, ["Hello world", "A longer sentence about searching a collection of documents.", "Cats", "Hello world"]);
    }

    [ModelFact("TENSORSHARP_SNOWFLAKE_EMBED_MODEL")]
    public async Task ClsPoolingMasksPaddingPreservesOrderAndMatchesIndividualInference()
    {
        using var model = Load("TENSORSHARP_SNOWFLAKE_EMBED_MODEL");
        await VerifyBatch(model, ["Hello world", "检索文档的多语言语义向量。", "A longer sentence about searching a collection of documents.", "Hello world"]);
        int[] shortInput = model.Tokenize("Hello world");
        float[] expected = (await model.EmbedTokensAsync([shortInput])).Embeddings[0];
        // Odd batch counts and tile boundaries previously selected different CPU
        // accumulation kernels and changed the same short input's embedding.
        foreach (int length in new[] { 9, 17, 25 })
        foreach (int count in new[] { 3, 5 })
        {
            var inputs = Enumerable.Range(0, count).Select(i => i == 0 ? shortInput :
                Enumerable.Range(0, length).Select(j => shortInput[j % shortInput.Length]).ToArray()).ToArray();
            var result = await model.EmbedTokensAsync(inputs);
            double cosine = expected.Zip(result.Embeddings[0], (a, b) => (double)a * b).Sum();
            Assert.True(cosine > 0.99999, $"Batch {count}, length {length}: short input cosine {cosine}.");
        }
    }

    [ModelFact("TENSORSHARP_MINILM_EMBED_MODEL")]
    public async Task InputValidationCancellationAndDisposalLeaveModelUsable()
    {
        var model = Load("TENSORSHARP_MINILM_EMBED_MODEL", 8);
        int[] longTokens = model.Tokenize("one two three four five six seven eight nine ten", truncate: true);
        Assert.Equal(8, longTokens.Length);
        Assert.Equal(102, longTokens[^1]);
        Assert.Throws<ArgumentException>(() => model.Tokenize("one two three four five six seven eight nine ten"));
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([Array.Empty<int>()]));
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([new[] { model.VocabularySize }]));
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([Enumerable.Repeat(101, 9).ToArray()]));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => model.EmbedTokensAsync([longTokens], cancellation.Token));
        var result = await model.EmbedTokensAsync([longTokens]);
        Assert.Equal(8, result.PromptTokens);
        AssertNormalized(result.Embeddings[0], model.Dimensions);
        model.Dispose();
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.Tokenize("Hello"));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => model.EmbedTokensAsync([longTokens]));
    }

    [ModelFact("TENSORSHARP_MINILM_EMBED_MODEL")]
    public async Task ManagedMeanEncoderMatchesNativeReferenceAndBatching() =>
        await VerifyManagedReference("TENSORSHARP_MINILM_EMBED_MODEL");

    [ModelFact("TENSORSHARP_SNOWFLAKE_EMBED_MODEL")]
    public async Task ManagedClsEncoderMatchesNativeReferenceAndBatching() =>
        await VerifyManagedReference("TENSORSHARP_SNOWFLAKE_EMBED_MODEL");

    [ModelFact("TENSORSHARP_MINILM_EMBED_MODEL")]
    public async Task ManagedCancellationDuringInferenceLeavesEncoderUsable()
    {
        using var model = EmbeddingModel.Load(Environment.GetEnvironmentVariable("TENSORSHARP_MINILM_EMBED_MODEL")!,
            new() { Backend = "CPU", Threads = 2 });
        int[] tokens = Enumerable.Repeat(100, model.MaxTokens).ToArray();
        using var cancellation = new CancellationTokenSource();
        var inFlight = model.EmbedTokensAsync([tokens, tokens, tokens, tokens], cancellation.Token);
        await Task.Delay(20);
        Assert.False(inFlight.IsCompleted);
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => inFlight);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        var recovered = await model.EmbedAsync(["Inference after cancellation"]);
        AssertNormalized(recovered.Embeddings[0], model.Dimensions);
    }

    private static async Task VerifyManagedReference(string variable)
    {
        string path = Environment.GetEnvironmentVariable(variable)!;
        using var managed = EmbeddingModel.Load(path, new() { Backend = "CPU", Threads = 4 });
        Assert.True(managed.IsManaged);
        string[] texts = ["Hello world", "Searching documents using semantic similarity.", "检索多语言文档。"];
        await VerifyBatch(managed, texts);
        using var native = EmbeddingModel.Load(path, new() { Backend = "GGML_CPU", Threads = 4 });
        Assert.False(native.IsManaged);
        var managedResult = await managed.EmbedAsync(texts);
        var nativeResult = await native.EmbedAsync(texts);
        for (int i = 0; i < texts.Length; ++i)
        {
            double cosine = managedResult.Embeddings[i].Zip(nativeResult.Embeddings[i], (a, b) => (double)a * b).Sum();
            Assert.True(cosine > 0.999, $"Pure C# / ggml CPU reference cosine is {cosine} for sequence {i}.");
        }
    }

    private static EmbeddingModel Load(string variable, int maxTokens = 0) => EmbeddingModel.Load(
        Environment.GetEnvironmentVariable(variable)!, new EmbeddingModelOptions
        {
            Backend = Environment.GetEnvironmentVariable("TS_TEST_EMBEDDING_BACKEND") ?? "GGML_CPU",
            Threads = 4,
            MaxTokens = maxTokens,
        });

    private static async Task VerifyBatch(EmbeddingModel model, string[] texts)
    {
        int[][] tokens = texts.Select(text => model.Tokenize(text)).ToArray();
        var batch = await model.EmbedTokensAsync(tokens);
        Assert.Equal(tokens.Sum(row => row.Length), batch.PromptTokens);
        Assert.Equal(texts.Length, batch.Embeddings.Length);
        for (int i = 0; i < texts.Length; ++i)
        {
            AssertNormalized(batch.Embeddings[i], model.Dimensions);
            var single = await model.EmbedTokensAsync([tokens[i]]);
            double dot = single.Embeddings[0].Zip(batch.Embeddings[i], (a, b) => (double)a * b).Sum();
            Assert.True(dot > 0.99999, $"Sequence {i} batch/individual cosine is {dot}.");
        }
        var concurrent = await Task.WhenAll(model.EmbedTokensAsync([tokens[0]]), model.EmbedTokensAsync([tokens[1]]));
        for (int i = 0; i < concurrent.Length; ++i)
            Assert.True(concurrent[i].Embeddings[0].Zip(batch.Embeddings[i], (a, b) => (double)a * b).Sum() > 0.99999);
        Assert.Empty((await model.EmbedTokensAsync(Array.Empty<int[]>())).Embeddings);
    }

    private static void AssertNormalized(float[] vector, int dimensions)
    {
        Assert.Equal(dimensions, vector.Length);
        Assert.All(vector, value => Assert.True(float.IsFinite(value)));
        Assert.InRange(vector.Sum(value => (double)value * value), 0.99999, 1.00001);
    }

    internal sealed class TinyEncoderFixture : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"embedding-model-{Guid.NewGuid():N}.gguf");

        public TinyEncoderFixture(bool invalidQkv = false, bool vocabularyMismatch = false, int pooling = 2,
            bool projectionWeights = false, int heads = 1, int context = 16)
        {
            const int dim = 32;
            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "bert",
                ["bert.embedding_length"] = (uint)dim,
                ["bert.attention.head_count"] = (uint)heads,
                ["bert.block_count"] = 1U,
                ["bert.feed_forward_length"] = (uint)dim,
                ["bert.context_length"] = (uint)context,
                ["bert.pooling_type"] = (uint)pooling,
                ["bert.attention.layer_norm_epsilon"] = 1e-5f,
                ["tokenizer.ggml.model"] = "bert",
                ["tokenizer.ggml.tokens"] = new[] { "[UNK]", "[CLS]", "[SEP]" },
                ["tokenizer.ggml.token_type"] = new[] { 2, 3, 3 },
                ["tokenizer.ggml.unknown_token_id"] = 0U,
                ["tokenizer.ggml.bos_token_id"] = 1U,
                ["tokenizer.ggml.eos_token_id"] = 2U,
            };
            var tensors = new List<(string Name, ulong[] Shape)>
            {
                ("token_embd.weight", new ulong[] { dim, vocabularyMismatch ? 4UL : 3UL }),
                ("position_embd.weight", new ulong[] { dim, (ulong)context }),
                ("token_embd_norm.weight", new ulong[] { dim }),
                ("token_embd_norm.bias", new ulong[] { dim }),
            };
            foreach (string name in new[] { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down" })
                tensors.Add(($"blk.0.{name}.weight", invalidQkv && (name is "attn_q" or "attn_k" or "attn_v")
                    ? new ulong[] { dim, dim, 2 } : new ulong[] { dim, dim }));
            foreach (string name in new[] { "attn_output_norm", "layer_output_norm" })
            {
                tensors.Add(($"blk.0.{name}.weight", new ulong[] { dim }));
                tensors.Add(($"blk.0.{name}.bias", new ulong[] { dim }));
            }
            using var writer = new BinaryWriter(File.Create(Path));
            void Text(string value) { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
            writer.Write(Encoding.ASCII.GetBytes("GGUF")); writer.Write(3U);
            writer.Write((ulong)tensors.Count); writer.Write((ulong)metadata.Count);
            foreach (var (key, value) in metadata)
            {
                Text(key);
                switch (value)
                {
                    case string text: writer.Write(8U); Text(text); break;
                    case uint integer: writer.Write(4U); writer.Write(integer); break;
                    case float number: writer.Write(6U); writer.Write(number); break;
                    case string[] strings:
                        writer.Write(9U); writer.Write(8U); writer.Write((ulong)strings.Length);
                        foreach (string text in strings) Text(text);
                        break;
                    case int[] integers:
                        writer.Write(9U); writer.Write(5U); writer.Write((ulong)integers.Length);
                        foreach (int integer in integers) writer.Write(integer);
                        break;
                }
            }
            ulong offset = 0;
            foreach (var (name, shape) in tensors)
            {
                Text(name); writer.Write((uint)shape.Length);
                foreach (ulong size in shape) writer.Write(size);
                writer.Write(0U); writer.Write(offset);
                offset += shape.Aggregate(4UL, (size, dimension) => size * dimension);
            }
            while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
            foreach (var (name, shape) in tensors)
            {
                int count = checked((int)shape.Aggregate(1UL, (size, dimension) => size * dimension));
                for (int i = 0; i < count; ++i)
                    writer.Write(name == "token_embd.weight" ? (float)Math.Sin(i + 1) :
                        name == "position_embd.weight" ? (float)(0.1 * Math.Cos(i + 1)) :
                        name.EndsWith("norm.weight", StringComparison.Ordinal) ? 1.0f :
                        projectionWeights && shape.Length == 2 && name.StartsWith("blk.", StringComparison.Ordinal) && i / dim == i % dim ? 0.2f : 0.0f);
            }
        }

        public void Dispose() => File.Delete(Path);
    }
}
