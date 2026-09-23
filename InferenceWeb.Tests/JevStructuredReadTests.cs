// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class JevStructuredReadTests(ITestOutputHelper output)
{
    [Fact]
    public void Validation_RejectsMalformedInputsBeforeExecutingAModel()
    {
        void Check(int[] prompt, int[] canvas, int[] positions, int[][] labels)
            => DiffusionGemmaModel.ValidateStructuredRequest(prompt, canvas, positions, labels, 10, 4, 8);
        Check([1], [2, 3], [1], [[4, 5]]);
        Assert.Throws<ArgumentNullException>(() => Check(null, [2], [0], [[4]]));
        Assert.Throws<ArgumentException>(() => Check([], [2], [0], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [], [0], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [1, 2, 3, 4, 5], [0], [[4]]));
        Assert.Throws<ArgumentException>(() => Check([1, 1, 1, 1, 1, 1], [1, 2, 3], [0], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([10], [2], [0], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [-1], [0], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [2], [-1], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [2], [1], [[4]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Check([1], [2], [0], [[10]]));
        Assert.Throws<ArgumentException>(() => Check([1], [2], [], []));
        Assert.Throws<ArgumentException>(() => Check([1], [2], [0], []));
        Assert.Throws<ArgumentException>(() => Check([1], [2], [0], [null]));
        Assert.Throws<ArgumentException>(() => Check([1], [2], [0], [[]]));
        Assert.Throws<ArgumentException>(() => Check([1], [2], [0], [[4, 4]]));
    }

    [Fact]
    public void ConditionalSoftmax_IsStableAndUsesOnlyRequestedLabelsAtTemperatureOne()
    {
        float[] row = [10000, 10001, 9999];
        DiffusionGemmaModel.NormalizeStructuredLabels(row);
        Assert.InRange(Math.Abs(row.Sum() - 1), 0, 1e-6);
        Assert.InRange(Math.Abs(row[1] / row[0] - Math.E), 0, 1e-6);
        float[] singleton = [-10000];
        DiffusionGemmaModel.NormalizeStructuredLabels(singleton);
        Assert.Equal(1f, singleton[0]);
        Assert.Throws<InvalidOperationException>(() => DiffusionGemmaModel.NormalizeStructuredLabels([float.NaN]));
        Assert.Throws<InvalidOperationException>(() => DiffusionGemmaModel.NormalizeStructuredLabels([float.PositiveInfinity]));
    }

    [ModelFact("TS_TEST_MODEL_DIR", "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion")]
    public void SelectedHead_MatchesFullVocabulary_AndDoesNotMutateTheSeed()
    {
        string modelPath = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"),
            "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion");
        string backendName = Environment.GetEnvironmentVariable("TS_TEST_BACKEND") ?? "ggmlcpu";
        BackendType backend = backendName.ToLowerInvariant() switch
        {
            "cpu" => BackendType.Cpu, "cuda" => BackendType.Cuda, "mlx" => BackendType.Mlx,
            "ggmlcuda" => BackendType.GgmlCuda, "ggmlmetal" => BackendType.GgmlMetal,
            "ggmlcpu" => BackendType.GgmlCpu,
            _ => throw new ArgumentException($"Unknown TS_TEST_BACKEND: {backendName}")
        };
        using var model = (DiffusionGemmaModel)ModelBase.Create(modelPath, backend);
        int[] prompt = model.Tokenizer.Encode("<bos><|turn>user\nClassify the mood: I love this!<turn|>\n<|turn>model\n", false).ToArray();
        int[] labels = new[] { "A", "B", "C" }.Select(label =>
            model.Tokenizer.Encode(label, false).Single()).ToArray();
        // Non-multiple-of-16 width and heterogeneous/reordered labels catch position and union indexing bugs.
        int[] canvas = Enumerable.Repeat(model.MaskTokenId, 7).ToArray();
        canvas[0] = labels[0];
        int[] seedCopy = (int[])canvas.Clone();
        int[] promptCopy = (int[])prompt.Clone();
        int[] positions = [5, 1, 5];
        int[][] requested = [labels, [labels[2], labels[0]], [labels[1]]];
        float[][] selected = model.ReadStructured(prompt, canvas, positions, requested);
        float[][] reference = model.ReadStructuredReference(prompt, canvas, positions, requested);
        float[][] repeated = model.ReadStructured(prompt, canvas, positions, requested);
        double maxDifference = 0;
        for (int row = 0; row < selected.Length; row++)
        {
            Assert.Equal(requested[row].Length, selected[row].Length);
            Assert.InRange(Math.Abs(selected[row].Sum() - 1), 0, 1e-6);
            Assert.Equal(selected[row], repeated[row]);
            for (int col = 0; col < selected[row].Length; col++)
            {
                Assert.True(float.IsFinite(selected[row][col]));
                maxDifference = Math.Max(maxDifference, Math.Abs(selected[row][col] - reference[row][col]));
            }
        }
        output.WriteLine($"Selected vs full-vocabulary max probability difference: {maxDifference:G9}; backend={backend}");
        // Different matrix widths may select different quantized GEMM tilings; this is a numerical,
        // not bitwise, equivalence check. Repeated identical selected reads above must be bitwise equal.
        Assert.InRange(maxDifference, 0, 0.01);
        Assert.Equal(seedCopy, canvas);
        Assert.Equal(promptCopy, prompt);
        Assert.Throws<OperationCanceledException>(() => model.ReadStructured(prompt, canvas, positions,
            requested, new CancellationToken(canceled: true)));
        int[] otherPrompt = model.Tokenizer.Encode(
            "<bos><|turn>user\nClassify the mood: I dislike this.<turn|>\n<|turn>model\n", false).ToArray();
        _ = model.ReadStructured(otherPrompt, canvas, positions, requested);
        float[][] afterPromptSwap = model.ReadStructured(prompt, canvas, positions, requested);
        for (int row = 0; row < selected.Length; row++) Assert.Equal(selected[row], afterPromptSwap[row]);
        model.ClearStructuredCache();
        float[][] afterClear = model.ReadStructured(prompt, canvas, positions, requested);
        for (int row = 0; row < selected.Length; row++) Assert.Equal(selected[row], afterClear[row]);
    }
}
