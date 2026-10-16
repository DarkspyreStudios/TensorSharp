using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Chat.Studio;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>Metadata admission only: no model weights, codec process, or device is used.</summary>
public sealed class Ltx25MediaCapabilityTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TemporalGuideCapabilitiesRequireTheConditioningCodecButIngredientsDoesNot(
        bool conditioningCodec, bool referenceLora)
    {
        var previous = MediaCodecs.VideoEncoder;
        try
        {
            MediaCodecs.VideoEncoder = new ConditioningCodecStub(conditioningCodec);
            AssertCapabilities(MetadataModel(referenceLora), conditioningCodec, referenceLora);
        }
        finally { MediaCodecs.VideoEncoder = previous; }
    }

    [Fact]
    public void ProviderWithoutConditioningImplementationDoesNotAdvertiseTemporalGuides()
    {
        var previous = MediaCodecs.VideoEncoder;
        try
        {
            // An encoder implementing only MP4 export inherits the interface's
            // false default. MP4 export alone cannot satisfy LTX preprocessing.
            MediaCodecs.VideoEncoder = new ExportOnlyCodecStub();
            AssertCapabilities(MetadataModel(referenceLora: true), temporalGuides: false, referenceLora: true);
        }
        finally { MediaCodecs.VideoEncoder = previous; }
    }

    [Fact]
    public void CapabilitiesFollowTheActiveProviderWithoutReloadingModelWeights()
    {
        var previous = MediaCodecs.VideoEncoder;
        try
        {
            var model = MetadataModel(referenceLora: true);
            MediaCodecs.VideoEncoder = new ConditioningCodecStub(true);
            AssertCapabilities(model, temporalGuides: true, referenceLora: true);
            MediaCodecs.VideoEncoder = new ExportOnlyCodecStub();
            AssertCapabilities(model, temporalGuides: false, referenceLora: true);
        }
        finally { MediaCodecs.VideoEncoder = previous; }
    }

    private static void AssertCapabilities(Ltx25Model model, bool temporalGuides, bool referenceLora)
    {
        Assert.Equal(temporalGuides, model.SupportsImageConditioning);
        Assert.Equal(temporalGuides, model.SupportsEndImageConditioning);
        Assert.Equal(temporalGuides, model.SupportsKeyframeConditioning);
        Assert.Equal(referenceLora, model.SupportsReferenceConditioning);

        var capability = NativeStudioGenerationAdapter.CreateCapability(model, "ltx", new());
        Assert.Equal(temporalGuides, capability.ImageConditioning);
        Assert.Equal(temporalGuides, capability.LastFrameConditioning);
        Assert.Equal(temporalGuides, capability.TemporalGuidance);
        Assert.Equal(referenceLora, capability.IdentityReferences);
        Assert.True(capability.Available);
        Assert.True(capability.TextToVideo);
        Assert.True(capability.AudioOutput);
    }

    private static Ltx25Model MetadataModel(bool referenceLora)
    {
        // These getters only inspect the provider and whether a LoRA was loaded.
        // Deliberately avoid constructing or disposing a native model in this test.
        var model = (Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model));
        if (referenceLora)
            typeof(Ltx25Model).GetField("<ReferenceLoraPath>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(model, "metadata-only-ingredients.safetensors");
        return model;
    }

    private sealed class ExportOnlyCodecStub : IVideoEncoder
    {
        public string SaveMp4(string path, RgbImage[] frames, int fps) =>
            throw new InvalidOperationException("Capability discovery must not invoke the codec.");
    }

    private sealed class ConditioningCodecStub(bool supported) : IVideoEncoder
    {
        public bool SupportsH264ImageConditioning => supported;
        public string SaveMp4(string path, RgbImage[] frames, int fps) =>
            throw new InvalidOperationException("Capability discovery must not invoke the codec.");
    }
}
