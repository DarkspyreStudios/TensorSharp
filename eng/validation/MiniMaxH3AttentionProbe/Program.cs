using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;

string? Value(string name) => args.SkipWhile(value => value != name).Skip(1).FirstOrDefault();
int[] sequences = (Value("--rows") ?? "64,1776,2364").Split(',').Select(int.Parse).ToArray();
int repeats = int.Parse(Value("--repeats") ?? "8");
if (sequences.Any(n => n <= 0) || repeats < 3) throw new ArgumentException("Positive sequence lengths and at least three repeats required");
const int heads = 56, head = 128, width = heads * head;
using var allocator = new CudaAllocator();
Tensor FromFloats(float[] values, int rows, int columns)
{
    using var flat = Tensor.FromArray(allocator, values);
    return flat.View(rows, columns);
}
var records = new List<object>();
foreach (int sequence in sequences)
{
    float[] q = Enumerable.Range(0, checked(sequence * width)).Select(i => MathF.Sin(i * .131f)).ToArray();
    float[] k = Enumerable.Range(0, q.Length).Select(i => MathF.Cos(i * .173f)).ToArray();
    float[] v = Enumerable.Range(0, q.Length).Select(i => MathF.Sin(i * .317f) * 80000f).ToArray();
    using var qt = FromFloats(q, sequence, width);
    using var kt = FromFloats(k, sequence, width);
    using var vt = FromFloats(v, sequence, width);
    using var original = new Tensor(allocator, DType.Float32, sequence, width);
    using var candidate = new Tensor(allocator, DType.Float32, sequence, width);
    float scale = 1f / MathF.Sqrt(head);
    double Run(bool batched)
    {
        allocator.Synchronize();
        var timer = Stopwatch.StartNew();
        bool accepted = batched
            ? CudaMiniMaxH3Ops.TryBatchedAttention(candidate, qt, kt, vt, heads, head, scale)
            : CudaWanOps.TryAttention(original, qt, kt, vt, null, heads, sequence, sequence, head, scale);
        if (!accepted) throw new InvalidOperationException("Requested attention kernel was unavailable; no fallback benchmark allowed");
        allocator.Synchronize();
        return timer.Elapsed.TotalMilliseconds;
    }
    Run(false); Run(true); Run(true); Run(false);
    var before = new List<double>(); var after = new List<double>();
    for (int i = 0; i < repeats; ++i)
        if (i % 2 == 0) { before.Add(Run(false)); after.Add(Run(true)); }
        else { after.Add(Run(true)); before.Add(Run(false)); }
    float[] reference = original.GetElementsAsFloat(q.Length), actual = candidate.GetElementsAsFloat(q.Length);
    double sumError = 0, sumReference = 0;
    for (int i = 0; i < actual.Length; ++i)
    {
        if (!float.IsFinite(actual[i]) || !float.IsFinite(reference[i])) throw new InvalidDataException("Nonfinite attention output");
        double error = actual[i] - (double)reference[i];
        sumError += error * error; sumReference += (double)reference[i] * reference[i];
    }
    double relativeError = Math.Sqrt(sumError / sumReference);
    if (relativeError > 2e-5) throw new InvalidDataException($"Streaming versus batched attention error {relativeError}");
    double Median(List<double> list) { var sorted = list.Order().ToArray(); return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2; }
    records.Add(new { sequence, heads, head, before, after, beforeMedianMs = Median(before), afterMedianMs = Median(after),
        speedup = Median(before) / Median(after), relativeError });
}
string json = JsonSerializer.Serialize(new
{
    completed = true, records,
    scope = "Synchronized synthetic F32 attention only; baseline streaming scalar kernel, candidate pedantic F32 batched GEMM",
    limitations = "Different summation order; synthetic agreement is not full50 model or end-to-end quality evidence; long-sequence existing generic GEMM route is not this baseline",
}, new JsonSerializerOptions { WriteIndented = true });
if (Value("--output") is string output) { Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!); File.WriteAllText(output, json); }
Console.WriteLine(json);
