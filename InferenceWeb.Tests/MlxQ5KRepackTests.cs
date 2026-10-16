using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Runtime;
using Half = System.Half;

namespace InferenceWeb.Tests;

public sealed class MlxQ5KRepackTests
{
    [Theory]
    [InlineData(256, 1)]
    [InlineData(512, 3)]
    [InlineData(1280, 67)]
    [InlineData(1024, 2048)]
    public unsafe void ExactQuantsAndAffineParametersMatchIndependentEncoding(int inner, int rows)
    {
        const int guard = 11;
        int blocks = inner / 256, rawBytes = rows * blocks * 176, packedBytes = rows * blocks * 160;
        int parameterCount = rows * blocks * 8;
        byte[] raw = Enumerable.Repeat((byte)0x5a, rawBytes + guard * 2).ToArray();
        byte[] expected = Enumerable.Repeat((byte)0xa5, packedBytes + guard * 2).ToArray();
        Array.Clear(expected, guard, packedBytes);
        Half[] expectedScales = Enumerable.Repeat((Half)37, parameterCount + guard * 2).ToArray();
        Half[] expectedBiases = (Half[])expectedScales.Clone();
        var random = new Random(51389 + rows + inner);
        ushort[] halfBits = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x3000, 0x7bff, 0x0400];
        for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
        {
            int start = guard + (row * blocks + block) * 176;
            Array.Clear(raw, start, 176);
            Half d = BitConverter.UInt16BitsToHalf(halfBits[(row + block) % halfBits.Length]);
            Half min = BitConverter.UInt16BitsToHalf(halfBits[(row * 3 + block + 2) % halfBits.Length]);
            BitConverter.TryWriteBytes(raw.AsSpan(start), BitConverter.HalfToUInt16Bits(d));
            BitConverter.TryWriteBytes(raw.AsSpan(start + 2), BitConverter.HalfToUInt16Bits(min));
            byte[] scales = Enumerable.Range(0, 8).Select(_ => (byte)random.Next(64)).ToArray();
            byte[] minima = Enumerable.Range(0, 8).Select(_ => (byte)random.Next(64)).ToArray();
            for (int group = 0; group < 4; ++group)
            {
                raw[start + 4 + group] = (byte)(scales[group] | ((scales[group + 4] >> 4) << 6));
                raw[start + 8 + group] = (byte)(minima[group] | ((minima[group + 4] >> 4) << 6));
                raw[start + 12 + group] = (byte)((scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4));
            }
            for (int group = 0; group < 8; ++group)
            {
                int index = guard + (row * blocks + block) * 8 + group;
                expectedScales[index] = (Half)((float)d * scales[group]);
                expectedBiases[index] = (Half)(-(float)min * minima[group]);
            }
            // Start from known logical quants, independently encode GGML's
            // split low/high fields, and write each bit in MLX channel order.
            // The first block deliberately includes every possible 5-bit value.
            for (int channel = 0; channel < 256; ++channel)
            {
                int quant = row == 0 && block == 0 ? channel % 32 : random.Next(32);
                int group = channel / 32, lane = channel % 32;
                raw[start + 16 + lane] |= (byte)((quant >> 4) << group);
                raw[start + 48 + (group / 2) * 32 + lane] |= (byte)((quant & 15) << ((group & 1) * 4));
                int logical = row * inner + block * 256 + channel;
                for (int bit = 0; bit < 5; ++bit)
                {
                    int packedBit = logical * 5 + bit;
                    expected[guard + packedBit / 8] |= (byte)(((quant >> bit) & 1) << (packedBit % 8));
                }
            }
        }
        byte[] original = (byte[])raw.Clone();
        foreach (bool parallel in new[] { false, true })
        {
            // Nonzero destinations prove that every output byte is assigned,
            // without a hidden dependency on the former zero-fill pass.
            byte[] actual = Enumerable.Repeat((byte)0xa5, expected.Length).ToArray();
            Half[] scales = Enumerable.Repeat((Half)37, expectedScales.Length).ToArray();
            Half[] biases = (Half[])scales.Clone();
            fixed (byte* source = raw, destination = actual)
            fixed (Half* scale = scales, bias = biases)
                MlxQ5KRepack.Pack((IntPtr)(source + guard), (IntPtr)(destination + guard),
                    (IntPtr)(scale + guard), (IntPtr)(bias + guard), inner, rows, parallel);
            Assert.Equal(expected, actual);
            Assert.Equal(MemoryMarshal.AsBytes(expectedScales.AsSpan()).ToArray(), MemoryMarshal.AsBytes(scales.AsSpan()).ToArray());
            Assert.Equal(MemoryMarshal.AsBytes(expectedBiases.AsSpan()).ToArray(), MemoryMarshal.AsBytes(biases.AsSpan()).ToArray());
            Assert.Equal(original, raw);
        }
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(255, 1)]
    [InlineData(257, 1)]
    [InlineData(256, 0)]
    [InlineData(256, -1)]
    public void InvalidGeometryIsRejectedBeforeReading(int inner, int rows)
        => Assert.Throws<ArgumentException>(() => MlxQ5KRepack.Pack((IntPtr)1, (IntPtr)1, (IntPtr)1, (IntPtr)1, inner, rows));

    [Fact]
    public void NullBuffersAreRejectedBeforeReading()
    {
        for (int missing = 0; missing < 4; ++missing)
        {
            IntPtr[] pointers = [(IntPtr)1, (IntPtr)1, (IntPtr)1, (IntPtr)1];
            pointers[missing] = IntPtr.Zero;
            Assert.Throws<ArgumentException>(() => MlxQ5KRepack.Pack(pointers[0], pointers[1], pointers[2], pointers[3], 256, 1));
        }
    }

    [MlxTheory]
    [InlineData(1, 512, 37, false)]
    [InlineData(3, 512, 37, true)]
    [InlineData(33, 1024, 1024, false)]
    [InlineData(33, 1024, 1024, true)]
    public void ProductionRepackerPreservesMatmulAndGatherBits(int tokens, int inner, int rows, bool preserveFloat32)
    {
        byte[] raw = new byte[rows * (inner / 256) * 176];
        new Random(51823 + inner + rows).NextBytes(raw);
        for (int block = 0; block < raw.Length / 176; ++block)
        {
            BitConverter.TryWriteBytes(raw.AsSpan(block * 176), BitConverter.HalfToUInt16Bits((Half)(.002f + block % 7 * .0001f)));
            BitConverter.TryWriteBytes(raw.AsSpan(block * 176 + 2), BitConverter.HalfToUInt16Bits((Half)(.001f + block % 11 * .0001f)));
        }
        IntPtr source = Marshal.AllocHGlobal(raw.Length);
        using var allocator = new MlxAllocator();
        string? previous = Environment.GetEnvironmentVariable("TS_MLX_Q5K_REPACK");
        try
        {
            Marshal.Copy(raw, 0, source, raw.Length);
            var values = new float[tokens, inner];
            for (int row = 0; row < tokens; ++row)
                for (int col = 0; col < inner; ++col) values[row, col] = MathF.Sin(row * .19f + col * .03f) * (preserveFloat32 ? 70000f : .25f);
            using var input = Tensor.FromArray(allocator, values);
            using var output = new Tensor(allocator, DType.Float32, tokens, rows);
            using var indices = Tensor.FromArray(allocator, new[] { rows - 1, 0, rows / 2 });
            using var gathered = new Tensor(allocator, DType.Float32, 3, inner);
            byte[]? expectedMatmul = null, expectedGather = null;
            // Cache eviction forces the real production conversion each time.
            // The final control repeat also checks that mode changes do not
            // retain stale packed data or affine parameter representations.
            foreach (string mode in new[] { "0", "1", "0" })
            {
                Environment.SetEnvironmentVariable("TS_MLX_Q5K_REPACK", mode);
                MlxQuantizedOps.ReleaseQuantizedWeight(allocator, source);
                Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(output, input, source, source,
                    (int)GgmlTensorType.Q5_K, inner, rows, raw.Length, preserveFloat32));
                float[] actual = output.GetElementsAsFloat(tokens * rows);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                byte[] matmulBytes = MemoryMarshal.AsBytes(actual.AsSpan()).ToArray();
                Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(gathered, source, IntPtr.Zero,
                    (int)GgmlTensorType.Q5_K, inner, rows, raw.Length, indices));
                byte[] gatherBytes = MemoryMarshal.AsBytes(gathered.GetElementsAsFloat(3 * inner).AsSpan()).ToArray();
                expectedMatmul ??= matmulBytes; expectedGather ??= gatherBytes;
                Assert.Equal(expectedMatmul, matmulBytes); Assert.Equal(expectedGather, gatherBytes);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_Q5K_REPACK", previous);
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, source);
            Marshal.FreeHGlobal(source);
        }
    }
}
