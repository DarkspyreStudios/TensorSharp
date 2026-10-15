using System.Text.Json;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25SamplingTests
{
    private static Ltx25DenoiserInput Input(float[]? video = null, float[]? audio = null,
        float[]? videoTimesteps = null, float[]? audioTimesteps = null, float[]? keyframeMask = null) => new()
    {
        Video = video ?? [1f, 2f], VideoTokens = 1,
        Audio = audio ?? [3f, 4f], AudioTokens = 1,
        VideoContext = [5f, 6f], VideoContextTokens = 1,
        AudioContext = [7f, 8f], AudioContextTokens = 1,
        VideoPositions = Ltx25Geometry.VideoPositions(1, 1, 1, 24),
        AudioPositions = Ltx25Geometry.AudioPositions(1),
        FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        VideoSigma = .25f, AudioSigma = .5f,
        VideoTimesteps = videoTimesteps, AudioTimesteps = audioTimesteps, KeyframeMask = keyframeMask,
    };

    private static float[] Values(JsonElement value) => value.EnumerateArray().Select(item => item.GetSingle()).ToArray();

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 3e-7f,
                $"Element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Fact]
    public void JointEulerMatchesPinnedOriginalFloat32X0AndStepArithmetic()
    {
        using Stream stream = typeof(Ltx25SamplingTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.euler-float32.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        JsonElement root = fixture.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", root.GetProperty("source_revision").GetString());
        float[] initialVideo = Values(root.GetProperty("initial_video"));
        float[] initialAudio = Values(root.GetProperty("initial_audio"));
        var initial = Input(initialVideo, initialAudio);
        var progress = new List<(int Step, int Total)>();
        int called = 0;
        Ltx25SamplingOutput output = Ltx25Sampling.RunEuler(input =>
        {
            JsonElement[] expected = root.GetProperty("steps").EnumerateArray().ToArray();
            Close(called == 0 ? initialVideo : Values(expected[called - 1].GetProperty("video")), input.Video);
            Close(called == 0 ? initialAudio : Values(expected[called - 1].GetProperty("audio")), input.Audio);
            Assert.NotSame(initialVideo, input.Video);
            Assert.NotSame(initialAudio, input.Audio);
            Assert.Same(initial.VideoContext, input.VideoContext);
            Assert.Same(initial.AudioContext, input.AudioContext);
            Assert.Same(initial.VideoPositions, input.VideoPositions);
            Assert.Same(initial.AudioPositions, input.AudioPositions);
            Assert.Equal(initial.FrequencyPrecision, input.FrequencyPrecision);
            Assert.Equal(root.GetProperty("sigmas")[called].GetSingle(), input.VideoSigma);
            Assert.Equal(input.VideoSigma, input.AudioSigma);
            JsonElement step = expected[called++];
            return new(Values(step.GetProperty("video_velocity")), Values(step.GetProperty("audio_velocity")), 2);
        }, initial, Values(root.GetProperty("sigmas")), (step, total) => progress.Add((step, total)));

        JsonElement last = root.GetProperty("steps")[2];
        Close(Values(last.GetProperty("video")), output.Video);
        Close(Values(last.GetProperty("audio")), output.Audio);
        Assert.Equal(3, output.Steps);
        Assert.Equal(2, output.BlocksPerStep);
        Assert.Equal(3, called);
        Assert.Equal(new[] { (1, 3), (2, 3), (3, 3) }, progress);
        Close(Values(root.GetProperty("initial_video")), initialVideo);
        Close(Values(root.GetProperty("initial_audio")), initialAudio);
    }

    public static TheoryData<float[]> InvalidSchedules => new()
    {
        Array.Empty<float>(), new[] { 1f }, new[] { 1f, .5f }, new[] { 1f, .5f, .75f, 0f }, new[] { 1f, 0f, 0f },
        new[] { float.NaN, 0f }, new[] { float.PositiveInfinity, 0f }, new[] { 1.1f, 0f }, new[] { 1f, -.1f, 0f },
    };

    [Theory]
    [MemberData(nameof(InvalidSchedules))]
    public void InvalidScheduleFailsBeforePrediction(float[] sigmas)
    {
        Assert.Throws<ArgumentException>(() => Ltx25Sampling.RunEuler(
            _ => throw new Xunit.Sdk.XunitException("Must validate before invoking the predictor."), Input(), sigmas));
    }

    [Theory]
    [InlineData("video")]
    [InlineData("audio")]
    public void ConditioningIsRejectedInsteadOfBeingSilentlyDropped(string kind)
    {
        var input = Input(videoTimesteps: kind == "video" ? [.5f] : null,
            audioTimesteps: kind == "audio" ? [.5f] : null);
        Assert.Throws<NotSupportedException>(() => Ltx25Sampling.RunEuler(
            _ => throw new Xunit.Sdk.XunitException("Conditioning is outside this sampler's scope."), input, [1f, 0f]));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InvalidVelocityCannotProduceSuccessfulSampling(bool dimensions)
    {
        Assert.Throws<InvalidOperationException>(() => Ltx25Sampling.RunEuler(
            _ => new([0f, 0f], dimensions ? [0f] : [float.NaN, 0f], 1), Input(), [1f, 0f]));
    }

    [Fact]
    public void ChangingDiagnosticBlockCountCannotLookLikeACompleteSchedule()
    {
        int calls = 0;
        Assert.Throws<InvalidOperationException>(() => Ltx25Sampling.RunEuler(
            _ => new([0f, 0f], [0f, 0f], ++calls), Input(), [1f, .5f, 0f]));
    }

    [Fact]
    public void CancellationBeforePredictionDoesNotRunTheModel()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => Ltx25Sampling.RunEuler(
            _ => throw new Xunit.Sdk.XunitException("Cancelled sampling must not run."), Input(), [1f, 0f],
            cancellationToken: cancellation.Token));
    }

    [Fact]
    public void CancellationDuringPredictionDoesNotReportACompletedStep()
    {
        using var cancellation = new CancellationTokenSource();
        int progress = 0;
        Assert.Throws<OperationCanceledException>(() => Ltx25Sampling.RunEuler(_ =>
        {
            cancellation.Cancel();
            return new([0f, 0f], [0f, 0f], 1);
        }, Input(), [1f, 0f], (_, _) => ++progress, cancellation.Token));
        Assert.Equal(0, progress);
    }
}
