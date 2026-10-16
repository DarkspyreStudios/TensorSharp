using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3DensePrecisionTests
{
    [CudaFact]
    public void DefaultBfloat16ProjectionPreservesF32ResidualAcrossSizeAndBandBoundaries()
    {
        // Neither a diagnostic override nor NVIDIA's global override may make
        // the previous approximate default appear to pass this regression.
        Assert.True(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_H3_CUDA_BF16_PEDANTIC")),
            "Clear TS_H3_CUDA_BF16_PEDANTIC to validate the default projection policy.");
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        const int inner = 1024, columns = 515, bandColumns = 257;
        var bytes = new byte[inner * columns * sizeof(ushort)];
        for (int column = 0; column < columns; ++column)
        {
            float weight = 1 << (column % 3);
            ushort bf16 = (ushort)(BitConverter.SingleToUInt32Bits(weight) >> 16);
            for (int k = 0; k < inner; ++k)
                BitConverter.TryWriteBytes(bytes.AsSpan((column * inner + k) * sizeof(ushort)), bf16);
        }

        using var allocator = new CudaAllocator();
        using var context = new DirectContext(allocator);
        IntPtr host = Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes, 0, host, bytes.Length);
            foreach (int rows in new[] { 1, 64 })
            {
                // TF32 rounds both magnitudes to 131072 and loses every +32
                // residual. F32 products and all possible partial sums here are
                // exact, even though the activations exceed Half's finite range.
                float[] values = Enumerable.Range(0, rows * inner)
                    .Select(i => i % 2 == 0 ? 131104f : -131072f).ToArray();
                using var input = context.FromFloats(values, rows, inner);
                using var output = context.NewF32(rows, columns);
                foreach (int budget in new[] { 64 << 20, bandColumns * inner * sizeof(float) })
                {
                    // At 64 rows each full 257-column band exceeds the old
                    // 16-million-product fast-math threshold; the final band
                    // has one column. Both paths must retain the same precision.
                    Ops.Fill(output, float.NaN);
                    Assert.True(CudaMiniMaxH3Ops.TryMappedDense(output, input, host, 30,
                        inner, columns, bytes.Length, maximumConvertedWeightBytes: budget));
                    Check(output, rows);
                }
                Assert.True(CudaMiniMaxH3Ops.TryLinear(output, input, host, 30,
                    inner, columns, bytes.Length));
                Check(output, rows);
            }
        }
        finally
        {
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Marshal.FreeHGlobal(host);
        }

        static void Check(Tensor output, int rows)
        {
            float[] actual = output.GetElementsAsFloat(rows * columns);
            for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                Assert.Equal((float)(inner / 2 * 32 * (1 << (column % 3))), actual[row * columns + column]);
        }
    }
}
