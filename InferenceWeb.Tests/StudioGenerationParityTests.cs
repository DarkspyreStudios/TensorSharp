using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Chat.Studio;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

/// <summary>Contract parity only: these tests do not load weights or claim visual model quality.</summary>
public sealed class StudioGenerationParityTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-video-parity-" + Guid.NewGuid().ToString("N"));

    // Capability getters are metadata only. No native model is created, used, or disposed.
    static Ltx25Model MetadataModel() => (Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model));

    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    [Fact]
    public void LtxCapabilityUsesTheModelQualityDefaultWhenHostLeavesSizeUnset()
    {
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", new());

        var native = Ltx25GenerationSettings.Resolve(new() { Seed = 42 });
        Assert.Equal(native.Width, capability.DefaultWidth);
        Assert.Equal(native.Height, capability.DefaultHeight);
        Assert.Equal(1024, capability.DefaultWidth);
        Assert.Equal(native.Fps, capability.DefaultFps);
        Assert.Equal(native.Frames, capability.DefaultFrameCount);
        Assert.True(capability.DefaultGenerateAudio);
        Assert.Equal(Ltx25GenerationSettings.Steps, capability.DefaultSteps);
        Assert.Equal(1, capability.DefaultCfgScale);
    }

    [Theory]
    [InlineData(1024, 1024, 144, 24, 145)]
    [InlineData(1280, 768, 17, 30, 17)]
    [InlineData(512, 512, 9, 24, 9)] // An explicit host choice remains authoritative.
    public void StudioCapabilitiesInheritChatHostDefaults(int width, int height, int frames, int fps, int effectiveFrames)
    {
        var options = Options(width, height, frames, fps);
        var shared = VideoGenerationParamsParser.CreateDefaults(options);
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", shared);

        Assert.Equal(width, capability.DefaultWidth);
        Assert.Equal(height, capability.DefaultHeight);
        Assert.Equal(fps, capability.DefaultFps);
        Assert.Equal(effectiveFrames, capability.DefaultFrameCount);
        Assert.Equal(8, capability.DefaultSteps);
        Assert.Equal(frames, shared.Frames); // Capability snapping never rewrites chat defaults.
    }

    [Fact]
    public void FixedLtxSamplingScheduleOverridesAnIncompatibleHostDefault()
    {
        var defaults = new VideoGenerationParams { Steps = 20 };
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", defaults);
        Assert.Equal(8, capability.DefaultSteps);
        Assert.Equal(capability.FixedSteps, capability.DefaultSteps);
        Assert.Equal(20, defaults.Steps);
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Steps = 20 }));
    }

    [Theory]
    [InlineData("minimax", 641, 385, 672, 416)]
    [InlineData("minimax", 0, 385, 416, 416)]
    [InlineData("minimax", 641, 0, 672, 672)]
    [InlineData("wan", 641, 385, 640, 384)]
    [InlineData("wan", 0, 385, 832, 384)]
    [InlineData("ltx", 641, 385, 641, 385)]
    public void InheritedDimensionsUseEachModelsNativeAlignment(string family, int width, int height, int expectedWidth, int expectedHeight)
    {
        IVideoGenerationModel model = family switch
        {
            "minimax" => (MiniMaxH3Model)RuntimeHelpers.GetUninitializedObject(typeof(MiniMaxH3Model)),
            "wan" => (WanVideoModel)RuntimeHelpers.GetUninitializedObject(typeof(WanVideoModel)),
            _ => MetadataModel(),
        };
        var capability = NativeStudioGenerationAdapter.CreateCapability(model, family, new() { Width = width, Height = height });
        Assert.Equal(expectedWidth, capability.DefaultWidth);
        Assert.Equal(expectedHeight, capability.DefaultHeight);
        if (family == "minimax")
        {
            var canvas = MiniMaxH3Pipeline.ResolveCanvas(new() { Width = width, Height = height });
            Assert.Equal(MiniMaxH3Geometry.AlignSpatial(canvas.Width), capability.DefaultWidth);
            Assert.Equal(MiniMaxH3Geometry.AlignSpatial(canvas.Height), capability.DefaultHeight);
        }
        if (family == "wan") Assert.Equal(WanVideoPipeline.SnapDim(height, capability.SpatialMultiple), capability.DefaultHeight);
        if (family == "ltx") Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Width = capability.DefaultWidth, Height = capability.DefaultHeight }));
    }

    [Fact]
    public void StudioImageToVideoMatchesChatRecipeConditioningAndNoiseForTheSameSeed()
    {
        var options = Options(1024, 1024, 144, 24);
        Directory.CreateDirectory(options.UploadDirectory);
        string image = Path.Combine(options.UploadDirectory, "banner.png");
        File.WriteAllBytes(image, [1, 2, 3, 4]); // Neither contract path decodes pixels here.
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", VideoGenerationParamsParser.CreateDefaults(options));
        var request = new StudioGenerationRequest
        {
            ModelId = "ltx", Prompt = "A person smiles gently.", ConditioningAssetId = "image",
            Width = capability.DefaultWidth, Height = capability.DefaultHeight, FrameCount = capability.DefaultFrameCount,
            Fps = capability.DefaultFps, Steps = capability.DefaultSteps, CfgScale = capability.DefaultCfgScale,
            GenerateAudio = capability.DefaultGenerateAudio, Seed = 42,
        };
        var studio = NativeStudioGenerationAdapter.Parameters(request, new Dictionary<string, string> { ["image"] = image }, null, default);
        var body = JsonSerializer.SerializeToElement(new { prompt = request.Prompt, frames = 145, imagePath = "banner.png", videoMode = "i2v", seed = 42 });
        var chat = VideoGenerationParamsParser.Parse(body, options, out string error);

        Assert.Null(error);
        Assert.Equal(chat.Mode, studio.Mode);
        Assert.Equal(chat.GenerateAudio, studio.GenerateAudio);
        Assert.Equal(chat.ImageBytes, File.ReadAllBytes(studio.ImagePath));
        var chatGuide = Assert.Single(Ltx25ImageConditioning.Resolve(chat, 145));
        var studioGuide = Assert.Single(Ltx25ImageConditioning.Resolve(studio, 145));
        Assert.Equal(chatGuide.FrameIndex, studioGuide.FrameIndex);
        Assert.Equal(1, studioGuide.Strength);
        var chatSettings = Ltx25GenerationSettings.Resolve(chat);
        var studioSettings = Ltx25GenerationSettings.Resolve(studio);
        Assert.Equal(chatSettings, studioSettings);
        Assert.True(studioSettings.Ancestral);
        Assert.Equal(11, studioSettings.TotalSteps);
        Assert.Equal(512, studioSettings.HalfResolution().Width);
        Assert.Equal(chatSettings.HalfResolution().InitialNoise().Video, studioSettings.HalfResolution().InitialNoise().Video);
    }

    ServerHostingOptions Options(int width, int height, int frames, int fps) => ServerOptionsBuilder.Build(
        ["--video-width", width.ToString(), "--video-height", height.ToString(), "--video-frames", frames.ToString(), "--fps", fps.ToString(), "--video-steps", "8"], root);
}
