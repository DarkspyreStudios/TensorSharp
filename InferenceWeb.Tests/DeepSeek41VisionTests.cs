// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41VisionTests
{
    [Theory]
    [InlineData("wide")]
    [InlineData("tall")]
    [InlineData("alpha_ignored")]
    [InlineData("downsample")]
    [InlineData("non_square")]
    [InlineData("one_pixel")]
    public void Preprocess_MatchesEveryBf16PatchValueFromOfficialPillowOracle(string name)
    {
        string root = FindFixtureDirectory();
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "manifest.json")));
        var expected = manifest.RootElement.GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("name").GetString() == name);
        var (patches, grid) = new DeepSeek41ImageProcessor().ProcessImage(
            Path.Combine(root, expected.GetProperty("image").GetString()!));
        Assert.Equal(expected.GetProperty("resized_width").GetInt32(), grid.Width);
        Assert.Equal(expected.GetProperty("resized_height").GetInt32(), grid.Height);
        Assert.Equal(expected.GetProperty("patch_grid")[0].GetInt32(), grid.PatchRows);
        Assert.Equal(expected.GetProperty("patch_grid")[1].GetInt32(), grid.PatchColumns);
        Assert.Equal(expected.GetProperty("llm_grid")[0].GetInt32(), grid.ImageRows);
        Assert.Equal(expected.GetProperty("llm_grid")[1].GetInt32(), grid.ImageColumns);
        Assert.Equal(expected.GetProperty("token_count").GetInt32(), grid.TokenCount);
        byte[] bytes = new byte[patches.Length * sizeof(float)];
        Buffer.BlockCopy(patches, 0, bytes, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian)
            for (int i = 0; i < bytes.Length; i += 4) Array.Reverse(bytes, i, 4);
        Assert.Equal(expected.GetProperty("patch_bytes").GetInt32(), bytes.Length);
        Assert.Equal(expected.GetProperty("patch_sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(-1, 5)]
    public void Plan_RejectsEmptyImage(int width, int height)
        => Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeek41ImageProcessor().Plan(width, height));

    [Theory]
    [InlineData(1, 100000)]
    [InlineData(100000, 1)]
    [InlineData(4096, 2160)]
    [InlineData(2160, 4096)]
    public void Plan_ExtremeAspectsStayWithinImageBudget(int width, int height)
    {
        var grid = new DeepSeek41ImageProcessor().Plan(width, height);
        Assert.InRange(grid.TokenCount, 4, 1024);
        Assert.Equal(0, grid.Width % 14);
        Assert.Equal(0, grid.Height % 14);
    }

    [Fact]
    public void Render_MultipleImagesAndHistoryUseOnePlaceholderEachWithoutMutatingHistory()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Compare these.", ImagePaths = new() { "a.png", "b.png" } },
            new() { Role = "assistant", Content = "Different colors." },
            new() { Role = "user", Content = "And this?", ImagePaths = new() { "c.png" } },
        };
        string rendered = ChatTemplate.RenderDeepSeek41(history);
        string image = ChatTemplate.DeepSeek41ImagePlaceholder;
        Assert.Contains("<｜User｜>" + image + image + "Compare these.", rendered);
        Assert.Contains("<｜User｜>" + image + "And this?", rendered);
        Assert.Equal(3, rendered.Split(image).Length - 1);
        Assert.Equal("Compare these.", history[0].Content);
        Assert.Equal(rendered, ChatTemplate.RenderDeepSeek41(history));
    }

    [Fact]
    public void Render_VideoFramesUseOrderedImageSpans()
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = "Frames at 0s and 1s. Describe the change.",
            IsVideo = true, ImagePaths = new() { "frame0.png", "frame1.png" } } };
        string rendered = ChatTemplate.RenderDeepSeek41(history);
        Assert.Contains(string.Concat(Enumerable.Repeat(ChatTemplate.DeepSeek41ImagePlaceholder, 2)), rendered);
        Assert.DoesNotContain("<video>", rendered);
    }

    /// <summary>
    /// The companion used to load with a hardcoded "CUDA", which on
    /// <c>--backend ggml_cpu</c> pulled a GPU into an explicitly CPU-only run.
    /// These names are the ggml registry names the native loader matches, so
    /// they are also what the text executor asks for. Only these two backends
    /// reach a V4.1 load at all (DeepSeek41Architecture.ValidateLoad).
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda, "CUDA")]
    [InlineData(BackendType.GgmlCpu, "CPU")]
    public void VisionCompanion_LoadsOnTheTextModelsOwnBackend(BackendType backend, string expected)
        => Assert.Equal(expected, DeepSeek41Model.ResolveVisionBackendName(backend));

    /// <summary>
    /// V4.1 refuses these two backends before any weight is read, so this is a
    /// drift guard on the shared registry-name table rather than a claim that
    /// the companion runs on Vulkan or Metal: the helper must name the backend
    /// the operator chose, never substitute a different one.
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlVulkan, "Vulkan")]
    [InlineData(BackendType.GgmlMetal, "Metal")]
    public void VisionBackendNameNeverSubstitutesAnotherBackend(BackendType backend, string expected)
        => Assert.Equal(expected, DeepSeek41Model.ResolveVisionBackendName(backend));

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Mlx)]
    public void VisionCompanion_RefusesBackendsWithNoGgmlRegistryName(BackendType backend)
    {
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41Model.ResolveVisionBackendName(backend));
        Assert.Contains(backend.ToString(), error.Message);
        Assert.Contains("ggml_cuda", error.Message);
    }

    [Fact]
    public void VisionInterface_IsSpecificToV41AndCompanionDiscoveryIsExact()
    {
        Assert.True(typeof(IVisionCapableModel).IsAssignableFrom(typeof(DeepSeek41Model)));
        Assert.False(typeof(IVisionCapableModel).IsAssignableFrom(typeof(DeepSeek4Model)));
        Assert.Equal(new[] { "deepseek41.vision.gguf" }, DeepSeek41Architecture.Descriptor.ProjectorFileHints);
    }

    [Fact]
    public void QueuedSpans_AreSortedAndPackedWithoutTextPlaceholders()
    {
        var queue = new DeepSeek41VisionQueue();
        queue.Add(new float[] { 30, 31 }, 1, 5);
        queue.Add(new float[] { 10, 11, 20, 21 }, 2, 1);
        var (mask, rows, values) = queue.Materialize(new[] { 4, 99, 99, 5, 6, 99 }, 2, 99);
        Assert.Equal(3, rows);
        Assert.Equal(new byte[] { 0, 1, 1, 0, 0, 1 }, mask);
        Assert.Equal(new float[] { 10, 11, 20, 21, 30, 31 }, values);
        queue.Clear();
        Assert.Equal(0, queue.Count);
    }

    [Theory]
    [InlineData("overlap")]
    [InlineData("past_end")]
    [InlineData("wrong_token")]
    [InlineData("wrong_size")]
    public void QueuedSpans_RejectMisalignmentBeforeNativeCall(string failure)
    {
        var queue = new DeepSeek41VisionQueue();
        queue.Add(failure == "wrong_size" ? new float[1] : new float[4], 2,
            failure == "past_end" ? 3 : 0);
        if (failure == "overlap") queue.Add(new float[2], 1, 1);
        int[] tokens = failure == "wrong_token" ? new[] { 99, 1, 99 } : new[] { 99, 99, 99 };
        Assert.Throws<ArgumentException>(() => queue.Materialize(tokens, 2, 99));
    }

    private static string FindFixtureDirectory()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "InferenceWeb.Tests", "Fixtures", "DeepSeek41Vision");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("Cannot locate committed DeepSeek41Vision fixtures.");
    }
}
