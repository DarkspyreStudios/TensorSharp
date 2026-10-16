using TensorSharp.Cuda;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed partial class Ltx25DenoiserTests
{
    [CudaTheory]
    [MemberData(nameof(Cases))]
    public void FusedHeadGatesExactlyPreserveCompleteOfficialGraph(int index)
    {
        string? previousAttention = Environment.GetEnvironmentVariable("TS_LTX25_CUDA_ATTN");
        string? previousHeadGate = Environment.GetEnvironmentVariable("TS_LTX25_CUDA_FUSED_HEAD_GATE");
        try
        {
            using var checkpoint = new Checkpoint();
            using var allocator = new CudaAllocator();
            var reference = Oracle.GetProperty("cases")[index];
            Environment.SetEnvironmentVariable("TS_LTX25_CUDA_ATTN", "1");
            Environment.SetEnvironmentVariable("TS_LTX25_CUDA_FUSED_HEAD_GATE", "0");
            Ltx25DenoiserOutput baseline;
            using (var model = new Ltx25Denoiser(checkpoint.Path, allocator))
            {
                baseline = model.Forward(Input(reference));
                Assert.Equal(0, model.OptimizedCudaHeadGates);
            }
            Environment.SetEnvironmentVariable("TS_LTX25_CUDA_FUSED_HEAD_GATE", "1");
            using var optimized = new Ltx25Denoiser(checkpoint.Path, allocator);
            var actual = optimized.Forward(Input(reference));
            Assert.Equal(6L * actual.ExecutedBlocks, optimized.OptimizedCudaHeadGates);
            Assert.Equal(baseline.VideoVelocity, actual.VideoVelocity);
            Assert.Equal(baseline.AudioVelocity, actual.AudioVelocity);
            Equal(Floats(reference, "videoVelocity"), actual.VideoVelocity, 2e-5, "fused gate official video");
            Equal(Floats(reference, "audioVelocity"), actual.AudioVelocity, 2e-5, "fused gate official audio");
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_LTX25_CUDA_ATTN", previousAttention);
            Environment.SetEnvironmentVariable("TS_LTX25_CUDA_FUSED_HEAD_GATE", previousHeadGate);
        }
    }

}
