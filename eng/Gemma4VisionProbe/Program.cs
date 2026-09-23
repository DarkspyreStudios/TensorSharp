// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// Runs Gemma4VisionEncoder over a FROZEN pixel array and dumps the projected soft tokens.
//
// The point of taking pixels from a file rather than decoding an image here is that it makes the
// result a test of the TOWER, not of two independently-written resize routines. The numpy oracle
// (eng/diffusiongemma-vision-oracle.py) writes the same pixels it fed itself, so any difference in
// the dumped output is attributable to the tower's math alone.
//
//   dotnet run --project eng/Gemma4VisionProbe -c Release -- \
//       --projector <mmproj.gguf|shard.safetensors> \
//       --pixels <chw_f32.bin> --width W --height H \
//       --backend ggml_metal --out <out_f32.bin>
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;

internal static class Program
{
    private static int Main(string[] args)
    {
        string projector = null, pixelsPath = null, outPath = null, backend = "ggml_metal";
        string imagePath = null, dumpPixels = null;
        int width = 0, height = 0, repeat = 1, softTokens = 0;
        string warmup = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--projector": projector = args[++i]; break;
                case "--pixels": pixelsPath = args[++i]; break;
                case "--width": width = int.Parse(args[++i]); break;
                case "--height": height = int.Parse(args[++i]); break;
                case "--backend": backend = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--repeat": repeat = int.Parse(args[++i]); break;
                case "--image": imagePath = args[++i]; break;
                case "--soft-tokens": softTokens = int.Parse(args[++i]); break;
                case "--dump-pixels": dumpPixels = args[++i]; break;
                // Measurement probe for the cold-start question: run one throwaway encode at
                // the given "WxH" before the timed loop, so the first TIMED encode sees warm
                // Metal pipelines. Reported separately so the warm-up's own cost is visible.
                case "--warmup": warmup = args[++i]; break;
                default:
                    Console.Error.WriteLine($"unknown argument '{args[i]}'");
                    return 2;
            }
        }

        if (projector == null || (pixelsPath == null && imagePath == null))
        {
            Console.Error.WriteLine(
                "usage: --projector <path> --pixels <chw_f32.bin> --width W --height H " +
                "[--backend ggml_metal|ggml_cpu] [--out <f32.bin>] [--repeat N]");
            return 2;
        }

        float[] pixels;
        if (imagePath != null)
        {
            // Exercise the REAL preprocessing path, so the comparison includes resize and
            // letterboxing rather than only the tower.
            int st = softTokens > 0 ? softTokens : Gemma4ImageProcessor.DefaultSoftTokens;
            var proc = new Gemma4ImageProcessor(minTokens: st, maxTokens: st);
            var r = proc.ProcessImage(imagePath);
            pixels = r.pixels; width = r.width; height = r.height;
            Console.WriteLine($"preprocessed: {width}x{height} ({width / 48}x{height / 48} soft = {(width / 48) * (height / 48)})");
            if (dumpPixels != null)
            {
                byte[] pb = new byte[pixels.LongLength * sizeof(float)];
                Buffer.BlockCopy(pixels, 0, pb, 0, pb.Length);
                File.WriteAllBytes(dumpPixels, pb);
                Console.WriteLine($"wrote pixels {dumpPixels}");
            }
        }
        else
        {
        byte[] raw = File.ReadAllBytes(pixelsPath);
        long expected = 3L * width * height * sizeof(float);
        if (raw.LongLength != expected)
        {
            Console.Error.WriteLine(
                $"pixel file is {raw.LongLength} bytes, expected {expected} for 3x{height}x{width} float32.");
            return 2;
        }

        pixels = new float[3L * width * height];
        Buffer.BlockCopy(raw, 0, pixels, 0, raw.Length);
        }

        GgmlBackendType ggmlType = backend switch
        {
            "ggml_metal" => GgmlBackendType.Metal,
            "ggml_cpu" => GgmlBackendType.Cpu,
            "ggml_cuda" => GgmlBackendType.Cuda,
            _ => throw new ArgumentException($"unsupported backend '{backend}'"),
        };

        var ctx = new GgmlContext(new[] { 0 }, ggmlType);
        var allocator = new GgmlAllocator(ctx, 0);

        var swLoad = Stopwatch.StartNew();
        using var encoder = new Gemma4VisionEncoder(projector, allocator);
        swLoad.Stop();
        Console.WriteLine($"load: {swLoad.ElapsedMilliseconds} ms");

        if (warmup != null)
        {
            var wh = warmup.Split('x');
            int ww = int.Parse(wh[0]), wht = int.Parse(wh[1]);
            var swW = Stopwatch.StartNew();
            using (var warmResult = encoder.Encode(new float[3L * ww * wht], ww, wht)) { }
            swW.Stop();
            Console.WriteLine($"warmup {ww}x{wht}: {swW.Elapsed.TotalMilliseconds:F1} ms");
        }

        Tensor result = null;
        double bestMs = double.MaxValue, firstMs = 0;
        for (int r = 0; r < repeat; r++)
        {
            result?.Dispose();
            var sw = Stopwatch.StartNew();
            result = encoder.Encode(pixels, width, height);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            if (r == 0) firstMs = ms;
            if (ms < bestMs) bestMs = ms;
            Console.WriteLine($"encode[{r}]: {ms:F1} ms");
        }

        long rows = result.Sizes[0], dim = result.Sizes[1];
        float[] outData = result.GetElementsAsFloat((int)(rows * dim));

        double sum = 0, absSum = 0, absMax = 0;
        for (long i = 0; i < outData.LongLength; i++)
        {
            double v = outData[i];
            sum += v;
            absSum += Math.Abs(v);
            if (Math.Abs(v) > absMax) absMax = Math.Abs(v);
        }
        double mean = sum / outData.LongLength;
        double var = 0;
        for (long i = 0; i < outData.LongLength; i++)
        {
            double d = outData[i] - mean;
            var += d * d;
        }
        var /= outData.LongLength;

        Console.WriteLine($"output: [{rows}, {dim}]");
        Console.WriteLine($"  float64 sum     : {sum:R}");
        Console.WriteLine($"  float64 abs-sum : {absSum:R}");
        Console.WriteLine($"  mean            : {mean:R}");
        Console.WriteLine($"  std             : {Math.Sqrt(var):R}");
        Console.WriteLine($"  absmax          : {absMax:R}");
        Console.WriteLine($"  encode first/best ms: {firstMs:F1} / {bestMs:F1}");
        // The tower must never leave a resident device COPY of its weights behind:
        // the zero-copy host-pointer wrap in fused_gemma4_vision_block is what keeps
        // it from stealing VRAM from the language model it shares the device with.
        Console.WriteLine($"  device-copy cache resident: {TensorSharp.GGML.GgmlBasicOps.DeviceCopyCacheResidentBytes() / 1048576.0:F2} MB");
        if (TensorSharp.GGML.GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeB, out long totalB))
            Console.WriteLine($"  backend device memory: {(totalB - freeB) / 1048576.0:F0} MB used of {totalB / 1048576.0:F0} MB");

        if (outPath != null)
        {
            byte[] bytes = new byte[outData.LongLength * sizeof(float)];
            Buffer.BlockCopy(outData, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(outPath, bytes);
            Console.WriteLine($"wrote {outPath} ({bytes.LongLength} bytes)");
        }

        result.Dispose();
        return 0;
    }
}
