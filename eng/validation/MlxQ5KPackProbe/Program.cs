using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.MLX;
using Half = System.Half;

int iterations = args.Length > 0 ? int.Parse(args[0]) : 7;
if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations));
unsafe
{
    // Representative LTX attention and feed-forward projection geometries.
    foreach (var shape in new[] { (Inner: 2048, Rows: 2048), (Inner: 4096, Rows: 4096), (Inner: 4096, Rows: 16384), (Inner: 16384, Rows: 4096) })
    {
        int blocks = shape.Inner / 256;
        long rawBytes = (long)shape.Rows * blocks * 176, packedBytes = (long)shape.Rows * blocks * 160;
        long parameters = (long)shape.Rows * blocks * 8;
        IntPtr source = Alloc(rawBytes), baseline = Alloc(packedBytes), optimized = Alloc(packedBytes);
        IntPtr oldScales = Alloc(parameters * 2), oldBiases = Alloc(parameters * 2);
        IntPtr newScales = Alloc(parameters * 2), newBiases = Alloc(parameters * 2);
        try
        {
            byte* raw = (byte*)source;
            for (long block = 0; block < (long)shape.Rows * blocks; ++block)
            {
                byte* row = raw + block * 176;
                for (int i = 0; i < 176; ++i) row[i] = (byte)((block * 13 + i * 17) % 256);
                ((Half*)row)[0] = (Half).125f; ((Half*)row)[1] = (Half).0625f;
            }
            for (int i = 0; i < 2; ++i) foreach (int mode in new[] { 0, 1, 2 }) Run(mode);
            AssertSame(baseline, optimized, packedBytes);
            AssertSame(oldScales, newScales, parameters * 2); AssertSame(oldBiases, newBiases, parameters * 2);
            var times = new[] { new List<double>(), new List<double>(), new List<double>() };
            for (int i = 0; i < iterations; ++i)
                foreach (int mode in i % 2 == 0 ? new[] { 0, 1, 2 } : new[] { 2, 1, 0 })
                {
                    var watch = Stopwatch.StartNew(); Run(mode); watch.Stop();
                    times[mode].Add(watch.Elapsed.TotalMilliseconds);
                }
            double Median(List<double> values) { var sorted = values.Order().ToArray(); return sorted[sorted.Length / 2]; }
            double oldMs = Median(times[0]), serialMs = Median(times[1]), parallelMs = Median(times[2]);
            Console.WriteLine(JsonSerializer.Serialize(new { shape = new[] { shape.Rows, shape.Inner }, rawBytes, iterations,
                processorCount = Environment.ProcessorCount, maxWorkers = Math.Min(8, Environment.ProcessorCount),
                baselineMedianMs = oldMs, fastSerialMedianMs = serialMs, parallelMedianMs = parallelMs,
                serialSpeedup = oldMs / serialMs, parallelSpeedup = oldMs / parallelMs, samplesMs = times,
                packedAndAffineParametersBitExact = true,
                scope = "CPU repacking plus baseline required zeroing; excludes allocation, file paging, upload, F32 parameter preparation and inference" }));
            void Run(int mode)
            {
                if (mode == 0) Baseline(source, baseline, oldScales, oldBiases, shape.Inner, shape.Rows);
                else MlxQ5KRepack.Pack(source, optimized, newScales, newBiases, shape.Inner, shape.Rows, parallel: mode == 2);
            }
        }
        finally
        {
            foreach (IntPtr pointer in new[] { source, baseline, optimized, oldScales, oldBiases, newScales, newBiases })
                NativeMemory.AlignedFree((void*)pointer);
        }
    }
}

static unsafe IntPtr Alloc(long bytes)
{
    var pointer = (IntPtr)NativeMemory.AlignedAlloc((nuint)bytes, 64);
    return pointer != IntPtr.Zero ? pointer : throw new OutOfMemoryException();
}

static unsafe void AssertSame(IntPtr first, IntPtr second, long bytes)
{
    byte* a = (byte*)first, b = (byte*)second;
    for (long i = 0; i < bytes; ++i)
        if (a[i] != b[i]) throw new InvalidOperationException($"Packing mismatch at {i}");
}

// Original production scalar extraction, generic bit packing and required clear.
static unsafe void Baseline(IntPtr source, IntPtr packed, IntPtr scales, IntPtr biases, int inner, int rows)
{
    int blocks = inner / 256, groups = blocks * 8;
    byte* raw = (byte*)source, destination = (byte*)packed;
    Half* scale = (Half*)scales, bias = (Half*)biases;
    NativeMemory.Clear(destination, (nuint)((long)rows * blocks * 160));
    byte[] values = new byte[32];
    for (int row = 0; row < rows; ++row)
    for (int block = 0; block < blocks; ++block)
    {
        byte* input = raw + ((long)row * blocks + block) * 176;
        float d = (float)BitConverter.UInt16BitsToHalf((ushort)(input[0] | input[1] << 8));
        float min = (float)BitConverter.UInt16BitsToHalf((ushort)(input[2] | input[3] << 8));
        byte* p = input + 4, high = input + 16, low = input + 48;
        for (int group = 0; group < 8; ++group)
        {
            int s = group < 4 ? p[group] & 63 : (p[group + 4] & 15) | ((p[group - 4] >> 6) << 4);
            int m = group < 4 ? p[group + 4] & 63 : (p[group + 4] >> 4) | ((p[group] >> 6) << 4);
            byte* q = low + (group / 2) * 32;
            for (int i = 0; i < 32; ++i)
                values[i] = (byte)((group % 2 == 0 ? q[i] & 15 : q[i] >> 4) | (((high[i] >> group) & 1) << 4));
            scale[row * groups + block * 8 + group] = (Half)(d * s);
            bias[row * groups + block * 8 + group] = (Half)(-min * m);
            long offset = (long)row * blocks * 160 + block * 160 + group * 20;
            int bitPosition = 0;
            foreach (byte value in values)
            {
                int index = (int)(offset + bitPosition / 8), shift = bitPosition & 7;
                uint shifted = (uint)((value & 31) << shift);
                destination[index] |= (byte)shifted;
                if (shifted > 255) destination[index + 1] |= (byte)(shifted >> 8);
                if (shifted > 65535) destination[index + 2] |= (byte)(shifted >> 16);
                bitPosition += 5;
            }
        }
    }
}
