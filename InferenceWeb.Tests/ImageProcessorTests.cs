using ImageMagick;

namespace InferenceWeb.Tests;

public class ImageProcessorTests
{
    /// <summary>Matches Pillow RGB bicubic samples, including clipping between the two resize passes.</summary>
    [Theory]
    [InlineData(7, 5, 4, 3, "ERZ8WBZ/phZ/7hZ8EX9/WH9/pn9/7n9/Eeh8WOh/puh/7uh8")]
    [InlineData(3, 5, 8, 6, "AAAABAABLABxZADwmgDw0wBx+wAB/wAAAC3YBC3RLC2JZC03mi030y2J+y3R/y3YAGRgBGRiLGR8ZGSZmmSZ02R8+2Ri/2RgAJpgBJpiLJp8ZJqZmpqZ05p8+5pi/5pgANLYBNLRLNKJZNI3mtI309KJ+9LR/9LYAP8ABP8BLP9xZP/wmv/w0/9x+/8B//8A")]
    [InlineData(7, 5, 7, 5, "AAAAKgD/VQAAfwD/qgAA1AD//wAAAD//Kj8AVT//fz8Aqj//1D8A/z//AH8AKn//VX8Af3//qn8A1H///38AAL//Kr8AVb//f78Aqr//1L8A/7//AP8AKv//Vf8Af///qv8A1P////8A")]
    public void BicubicResizeMatchesPillowRgb(int width, int height, int targetWidth, int targetHeight, string expectedBase64)
    {
        byte[] rgba = new byte[width * height * 4];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int offset = (y * width + x) * 4;
                rgba[offset] = (byte)(x * 255 / (width - 1));
                rgba[offset + 1] = (byte)(y * 255 / (height - 1));
                rgba[offset + 2] = (byte)(((x + y) % 2) * 255);
                rgba[offset + 3] = 255;
            }

        // Frozen RGB output from Pillow 12.1.1 Image.resize(..., Resampling.BICUBIC).
        // The red/green ramps and blue checker expose rounding, clipping and axis ordering.
        byte[] expected = Convert.FromBase64String(expectedBase64);
        float[] actual = ImageProcessorUtils.ResizeRgbaToChannelFirstBicubic(
            rgba, width, height, targetWidth, targetHeight, [0, 0, 0], [1, 1, 1]);
        int plane = targetWidth * targetHeight;
        Assert.Equal(expected.Length, actual.Length);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < plane; i++)
                Assert.Equal(expected[i * 3 + c] / 255f, actual[c * plane + i]);
    }

    private const string EmbeddedJpegBase64 =
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD7V/Z2+C3w91v9n74ZajqPgTwzf6heeF9MuLm7utHt5JZ5XtImd3dkJZmJJJJySSTRRRXyOL/3ip/if5nwmO/3qr/il+bP/9k=";

    [Fact]
    public void Gemma4ImageProcessorProcessImageSupportsJpeg()
    {
        string path = WriteEmbeddedJpeg();
        try
        {
            var processor = new Gemma4ImageProcessor(patchSize: 1, nMerge: 1, minTokens: 1, maxTokens: 4);
            var (pixels, width, height) = processor.ProcessImage(path);

            Assert.Equal(2, width);
            Assert.Equal(2, height);
            Assert.Equal(12, pixels.Length);
            Assert.All(pixels, value => Assert.InRange(value, -1f, 1f));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Gemma4ImageProcessorPreservesAspectRatioAndDoesNotUpscale()
    {
        // Regression test for the gemma4uv image-preprocessing bug.
        //
        // A 750x500 photo has 375k pixels, already inside the [40,280]-token budget
        // (92160..645120 px at patch16*merge3). The reference (llama.cpp
        // calc_size_preserved_ratio + PAD_CEIL) therefore keeps the aspect-aligned
        // native size 768x480 (16x10 = 160 patches) and letter-boxes with black.
        // The old SmartResize always rescaled to the *max* pixel budget, producing a
        // stretched, upscaled 960x624 (260 patches) image -> the model then reported a
        // "very wide", "blurry" picture and hallucinated cut/sliced apples.
        string path = WriteSyntheticPng(750, 500);
        try
        {
            var processor = new Gemma4ImageProcessor(
                patchSize: 16, nMerge: 3, minTokens: 40, maxTokens: 280,
                imageMean: new[] { 0f, 0f, 0f }, imageStd: new[] { 1f, 1f, 1f });
            var (pixels, width, height) = processor.ProcessImage(path);

            Assert.Equal(768, width);   // not the old upscaled 960
            Assert.Equal(480, height);  // not the old upscaled 624
            Assert.Equal(3 * width * height, pixels.Length);

            // Content (red) is centred (720x480, offsetX=24); the 24px left/right
            // borders are black letter-box padding -> normalized 0 (mean=0, std=1).
            int rowMid = (height / 2) * width;
            Assert.InRange(pixels[rowMid + 0], -0.01f, 0.01f);             // left pad -> black
            Assert.True(pixels[rowMid + width / 2] > 0.9f);                // centre -> red (~1)
            Assert.InRange(pixels[rowMid + (width - 1)], -0.01f, 0.01f);   // right pad -> black
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Qwen35ImageProcessorComputeImageTokenCountSupportsJpeg()
    {
        string path = WriteEmbeddedJpeg();
        try
        {
            var processor = new Qwen35ImageProcessor(patchSize: 1, mergeSize: 1, shortestEdge: 1, longestEdge: 16);
            int tokenCount = processor.ComputeImageTokenCount(path);

            Assert.Equal(4, tokenCount);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void NemotronImageProcessorCapsTilesForLatencyByDefault()
    {
        string path = WriteSyntheticPng(96, 64);
        string? oldMaxTiles = Environment.GetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES");
        try
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", null);
            var processor = CreateNemotronTestProcessor(maxTiles: 12);
            var tiles = processor.ProcessImage(path);

            Assert.Single(tiles);
            Assert.All(tiles, tile =>
            {
                Assert.Equal(64, tile.Width);
                Assert.Equal(64, tile.Height);
                Assert.Equal(3 * 64 * 64, tile.Pixels.Length);
            });
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", oldMaxTiles);
            File.Delete(path);
        }
    }

    [Fact]
    public void NemotronImageProcessorHonorsFullTileOverride()
    {
        string path = WriteSyntheticPng(96, 64);
        string? oldMaxTiles = Environment.GetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES");
        try
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", "12");
            var processor = CreateNemotronTestProcessor(maxTiles: 12);
            var tiles = processor.ProcessImage(path);

            Assert.Equal(7, tiles.Count); // 3x2 tiles plus thumbnail.
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", oldMaxTiles);
            File.Delete(path);
        }
    }

    [Fact]
    public void UserSuppliedJpegSmokeTestWhenConfigured()
    {
        string? path = Environment.GetEnvironmentVariable("TENSORSHARP_JPEG_SMOKE_PATH");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(path);
        var qwen = new Qwen35ImageProcessor(patchSize: 1, mergeSize: 1, shortestEdge: 1, longestEdge: width * height);
        int qwenTokens = qwen.ComputeImageTokenCount(path);

        var gemma4 = new Gemma4ImageProcessor(patchSize: 1, nMerge: 1, minTokens: 1, maxTokens: width * height);
        var (gemma4Pixels, gemma4Width, gemma4Height) = gemma4.ProcessImage(path);

        Assert.True(width > 0);
        Assert.True(height > 0);
        Assert.True(qwenTokens > 0);
        Assert.Equal(width, gemma4Width);
        Assert.Equal(height, gemma4Height);
        Assert.Equal(3 * width * height, gemma4Pixels.Length);
    }

    [Fact]
    public void Qwen35ImageProcessorReadDimensionsSupportsHeic()
    {
        string? heicPath = TryWriteSyntheticHeic(80, 48);
        if (heicPath == null)
            return;

        try
        {
            var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(heicPath);
            Assert.Equal(80, width);
            Assert.Equal(48, height);
        }
        finally
        {
            File.Delete(heicPath);
        }
    }

    [Fact]
    public void UserSuppliedHeicSmokeTestWhenConfigured()
    {
        string? path = Environment.GetEnvironmentVariable("TENSORSHARP_HEIC_SMOKE_PATH");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(path);

        Assert.True(width > 0);
        Assert.True(height > 0);
    }

    private static string WriteEmbeddedJpeg()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jpg");
        File.WriteAllBytes(path, Convert.FromBase64String(EmbeddedJpegBase64));
        return path;
    }

    private static NemotronImageProcessor CreateNemotronTestProcessor(int maxTiles)
    {
        return new NemotronImageProcessor(
            imageSize: 64,
            patchSize: 16,
            numChannels: 3,
            maxTiles: maxTiles,
            minNumPatches: 0,
            maxNumPatches: 0,
            useThumbnail: true,
            projectorScaleFactor: 2,
            imageMean: null,
            imageStd: null);
    }

    private static string WriteSyntheticPng(uint width, uint height)
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        using var image = new MagickImage(MagickColors.Red, width, height);
        image.Format = MagickFormat.Png;
        image.Write(path);
        return path;
    }

    // Synthesize a HEIC file on disk so the round-trip exercise does not require
    // a vendored binary blob. Returns null when the bundled Magick.NET native
    // binaries cannot encode HEIC on this platform (e.g. when x265 is not
    // available), which keeps the test environment-resilient.
    private static string? TryWriteSyntheticHeic(uint width, uint height)
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.heic");
        try
        {
            using var image = new MagickImage(MagickColors.Crimson, width, height);
            image.Format = MagickFormat.Heic;
            image.Write(path);

            if (!File.Exists(path) || new FileInfo(path).Length == 0)
            {
                if (File.Exists(path)) File.Delete(path);
                return null;
            }

            return path;
        }
        catch
        {
            if (File.Exists(path)) File.Delete(path);
            return null;
        }
    }
}
