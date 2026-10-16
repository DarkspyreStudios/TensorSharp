using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

// Fixed-latent decoder isolate. Run identical copies beside frozen baseline and
// candidate assemblies. This excludes the rest of the generation pipeline.
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
string models = Path.GetFullPath(Required("models"));
string output = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "cuda").ToLowerInvariant();
if (backend is not ("cuda" or "mlx")) throw new ArgumentException("--backend must be cuda or mlx");
string arm = options.GetValueOrDefault("arm", "mapped");
if (arm is not ("mapped" or "expanded")) throw new ArgumentException("--arm must be mapped or expanded");
int warmRuns = Number("warm-runs", 2);
if (warmRuns < 0) throw new ArgumentException("--warm-runs cannot be negative");
if (backend == "cuda") Environment.SetEnvironmentVariable("TS_H3_CUDA_MAPPED_VAE_WEIGHTS", arm == "mapped" ? "1" : "0");

string[] dimensionOptions = ["latent-frames", "latent-height", "latent-width"];
bool customShape = dimensionOptions.Any(options.ContainsKey);
if (customShape && !dimensionOptions.All(options.ContainsKey))
    throw new ArgumentException("Specify all of --latent-frames, --latent-height and --latent-width");
if (options.ContainsKey("latent") && !customShape)
    throw new ArgumentException("Raw --latent input requires explicit latent dimensions; layout is THWC Float32");
string decodeMode = options.GetValueOrDefault("decode-mode", customShape ? "clip" : "tokens");
if (decodeMode is not ("tokens" or "clip")) throw new ArgumentException("--decode-mode must be tokens or clip");

float[] ReadFloats(string path)
{
    byte[] bytes = File.ReadAllBytes(path);
    if (bytes.Length % sizeof(float) != 0) throw new ArgumentException("Float32 input has an incomplete element: " + path);
    float[] data = new float[bytes.Length / sizeof(float)];
    Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
    return data;
}
string FloatHash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
string FileHash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}

int channels = MiniMaxH3VideoVae.LatentChannels, time, height, width;
float[] latent;
float[]? expected = null;
string inputSource;
if (customShape)
{
    time = Number("latent-frames", 0); height = Number("latent-height", 0); width = Number("latent-width", 0);
    if (time <= 0 || height <= 0 || width <= 0) throw new ArgumentException("Latent dimensions must be positive");
    int count = checked(time * height * width * channels);
    if (options.TryGetValue("latent", out string? path))
    {
        inputSource = Path.GetFullPath(path);
        latent = ReadFloats(inputSource);
        if (latent.Length != count) throw new ArgumentException($"Raw latent must contain {count} Float32 values in THWC order");
    }
    else
    {
        inputSource = "Deterministic xorshift32(seed=1) THWC values in [-0.6,0.6); no independent oracle";
        latent = new float[count];
        uint state = 1;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            latent[i] = ((state >> 8) * (1f / 16777216f) - .5f) * 1.2f;
        }
    }
}
else
{
    string fixture = Path.Combine(models, "fixtures", "h3_video_latent_in.bin");
    int[] shape = File.ReadAllText(fixture + ".shape").Split(',').Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToArray();
    if (shape.Length != 5 || shape[0] != 1 || shape[1] != channels || shape.Skip(2).Any(value => value <= 0))
        throw new ArgumentException("Fixture latent shape must be [1,24,T,H,W]");
    time = shape[2]; height = shape[3]; width = shape[4];
    float[] chw = ReadFloats(fixture);
    if (chw.Length != checked(channels * time * height * width)) throw new ArgumentException("Fixture shape/length mismatch");
    latent = new float[chw.Length];
    for (int c = 0; c < channels; c++) for (int t = 0; t < time; t++) for (int h = 0; h < height; h++) for (int w = 0; w < width; w++)
        latent[((t * height + h) * width + w) * channels + c] = chw[((c * time + t) * height + h) * width + w];
    inputSource = fixture + " (BCTHW reordered to THWC)";
    if (decodeMode == "tokens") expected = ReadFloats(Path.Combine(models, "fixtures", "h3_video_pixels_out.bin"));
}
if (latent.Any(value => !float.IsFinite(value))) throw new ArgumentException("Input latent must be finite");

var records = new List<object>();
var report = new Dictionary<string, object?>
{
    ["completed"] = false,
    ["backend"] = backend,
    ["arm"] = backend == "cuda" ? arm : "Not applicable to MLX",
    ["decodeMode"] = decodeMode,
    ["scope"] = "Isolated fixed-latent video decoder. Excludes text encoder, denoiser, conditioning encoders, audio and codecs. First decode includes deferred decoder weight loading; clip mode includes production temporal padding and spatial tiling.",
    ["latentShape"] = new[] { time, height, width, channels },
    ["outputPixelShape"] = new[] { MiniMaxH3VideoVae.OutChannels, decodeMode == "clip" ? MiniMaxH3VideoVae.DecodedFrameCount(time) : time * MiniMaxH3VideoVae.PatchTemporal, height * MiniMaxH3VideoVae.PatchSpatial, width * MiniMaxH3VideoVae.PatchSpatial },
    ["inputSource"] = inputSource,
    ["inputSha256"] = FloatHash(latent),
    ["warmRuns"] = warmRuns,
    ["independentOracleAvailable"] = expected != null,
    ["oracleBound"] = expected != null ? "Existing independent video fixture cosine > 0.999; finite outputs; byte-exact cold/warm repeat" : "No independent quality oracle; finite outputs and byte-exact cold/warm repeat only; compare output hashes between baseline and candidate",
    ["managedBinaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(MiniMaxH3VideoVae).Assembly, typeof(MlxAllocator).Assembly, typeof(CudaAllocator).Assembly }
        .Distinct().Select(assembly => new { path = assembly.Location, sha256 = FileHash(assembly.Location) }).ToArray(),
    ["records"] = records,
};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();

int Run()
{
    IAllocator allocator = backend == "mlx" ? new MlxAllocator() : new CudaAllocator(0);
    object? Memory() => allocator is MlxAllocator mlx ? mlx.GetMemorySnapshot() : null;
    try
    {
        report["memoryBeforeConstructor"] = Memory();
        var constructor = Stopwatch.StartNew();
        using var model = new MiniMaxH3VideoVae(Path.Combine(models, "minimax_h3_video_vae_fp16.safetensors"),
            backend == "mlx" ? BackendType.Mlx : BackendType.Cuda, allocator);
        constructor.Stop();
        report["constructorSeconds"] = constructor.Elapsed.TotalSeconds;
        float[]? first = null;
        bool passed = true;
        for (int run = 0; run <= warmRuns; run++)
        {
            object? before = Memory();
            var timer = Stopwatch.StartNew();
            float[] decoded = decodeMode == "clip" ? model.Decode(latent, time, height, width) : model.DecodeTokens(latent, time, height, width);
            timer.Stop();
            object? after = Memory();
            float[] pixels = decodeMode == "clip" ? decoded : MiniMaxH3VideoVae.UnpackPatches(decoded, time, height, width);
            bool finite = pixels.All(float.IsFinite);
            bool repeat = first == null || MemoryMarshal.AsBytes(first.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(decoded.AsSpan()));
            double? cosine = null, maximum = null;
            if (expected != null && pixels.Length != expected.Length) throw new InvalidOperationException("Oracle output length mismatch");
            if (expected != null && finite)
            {
                double dot = 0, squareReference = 0, squareActual = 0, max = 0;
                for (int i = 0; i < pixels.Length; i++)
                {
                    dot += (double)pixels[i] * expected[i]; squareReference += (double)expected[i] * expected[i]; squareActual += (double)pixels[i] * pixels[i];
                    max = Math.Max(max, Math.Abs((double)pixels[i] - expected[i]));
                }
                cosine = dot / (Math.Sqrt(squareReference) * Math.Sqrt(squareActual) + 1e-30);
                maximum = max;
                passed &= cosine > .999;
            }
            passed &= finite && repeat;
            records.Add(new
            {
                run, cacheScope = run == 0 ? "First decode including deferred weight load and cold resident weights" : "Same loaded model, resident weights",
                seconds = timer.Elapsed.TotalSeconds, finite, cosine, maxAbs = maximum, identicalToCold = repeat,
                decodedSha256 = FloatHash(decoded), pixelsSha256 = FloatHash(pixels), memoryBefore = before, memoryAfter = after,
                processWorkingSetBytes = Process.GetCurrentProcess().WorkingSet64,
            });
            first ??= decoded;
            report["passed"] = passed;
            Save();
            Console.WriteLine(JsonSerializer.Serialize(records[^1]));
        }
        report["completed"] = true;
        Save();
        return passed ? 0 : 1;
    }
    finally { (allocator as IDisposable)?.Dispose(); }
}

try { return backend == "mlx" ? MlxWorker.Shared.Invoke(Run) : Run(); }
catch (Exception error)
{
    report["passed"] = false; report["error"] = error.ToString(); Save(); throw;
}
