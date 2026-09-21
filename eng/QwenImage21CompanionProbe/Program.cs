// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

// Run twice, setting TS_QWEN_TE_FUSED=0/1 before process startup, then compare.
// No model/device scenario is considered passing unless this tool executes it.
if (args.Length == 3 && args[0] == "compare")
{
    float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length % sizeof(float) != 0) throw new InvalidDataException("Expected raw F32.");
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
    var expected = Read(args[1]);
    var actual = Read(args[2]);
    if (expected.Length != actual.Length) throw new InvalidDataException("Output lengths differ.");
    double square = 0, referenceSquare = 0, maxError = 0;
    for (int i = 0; i < expected.Length; i++)
    {
        if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
            throw new InvalidDataException("Non-finite hidden state.");
        double delta = actual[i] - expected[i];
        square += delta * delta;
        referenceSquare += (double)expected[i] * expected[i];
        maxError = Math.Max(maxError, Math.Abs(delta));
    }
    double relativeL2 = Math.Sqrt(square / Math.Max(referenceSquare, 1e-30));
    // Per-op vs fused reduction order differs; quantized execution is also
    // backend dependent. Threshold measures the entire conditioning tensor.
    bool passed = relativeL2 < 0.01;
    Console.WriteLine(JsonSerializer.Serialize(new { passed, values = expected.Length, relativeL2, maxError }));
    return passed ? 0 : 1;
}

if (args.Length < 3 || args[0] != "text")
{
    Console.Error.WriteLine("text <Qwen3VL-8B.gguf> <output.f32> [GgmlCpu|GgmlMetal|GgmlCuda] [prompt]");
    Console.Error.WriteLine("compare <reference.f32> <actual.f32>");
    return 2;
}
BackendType backend = args.Length > 3 ? Enum.Parse<BackendType>(args[3], true) : BackendType.GgmlCpu;
string prompt = args.Length > 4 ? args[4] : "A red cube on a white table.";
string output = Path.GetFullPath(args[2]);
Directory.CreateDirectory(Path.GetDirectoryName(output));
var timer = Stopwatch.StartNew();
using (var encoder = new QwenImageTextEncoder(args[1], backend, qwenImage21: true))
{
    int[] tokens = encoder.Tokenizer.Encode(QwenImage21Conditioner.SystemPrompt +
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
    timer.Restart();
    float[] hidden = encoder.EncodeHidden(tokens);
    if (hidden.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite hidden state.");
    double seconds = timer.Elapsed.TotalSeconds;
    byte[] bytes = new byte[hidden.Length * sizeof(float)];
    Buffer.BlockCopy(hidden, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes(output, bytes);
    Console.WriteLine(JsonSerializer.Serialize(new { tokens = tokens.Length, hidden = encoder.HiddenSize,
        seconds, backend = backend.ToString(), fusedRequested = Environment.GetEnvironmentVariable("TS_QWEN_TE_FUSED") != "0", output }));
}
GgmlBasicOps.ReleaseReuseComputeBuffers();
GgmlBasicOps.ClearHostBufferCache();
return 0;
