// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The V4.1 trained cache quantization (FP8 E4M3, MXFP4, NVFP4) against the
/// reference implementation.
///
/// <para>These bins decide which compressed rows the sparse selection keeps, so
/// a single value on the wrong side of a boundary changes the attention key set
/// rather than only the last digit. Point TS_DSV41_QUANT_VECTORS at a raw
/// little-endian F32 file and this writes back one quantized file per mode for
/// eng/tests/dsv41-quant-vectors.py to compare against PyTorch.</para>
/// </summary>
public class Dsv41QuantizationTests
{
    [Fact]
    public void QuantizesTheHarnessVectorsForEveryMode()
    {
        string path = Environment.GetEnvironmentVariable("TS_DSV41_QUANT_VECTORS");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        byte[] raw = File.ReadAllBytes(path);
        int count = raw.Length / sizeof(float);
        for (int mode = 0; mode <= 2; mode++)
        {
            var values = new float[count];
            Buffer.BlockCopy(raw, 0, values, 0, count * sizeof(float));
            DeepSeek4CpuExecutor.QuantizeForTest(values, mode);
            var outBytes = new byte[count * sizeof(float)];
            Buffer.BlockCopy(values, 0, outBytes, 0, outBytes.Length);
            File.WriteAllBytes($"{path}.cs.mode{mode}.f32", outBytes);
        }
    }
}
