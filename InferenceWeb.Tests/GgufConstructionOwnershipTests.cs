using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgufConstructionOwnershipTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(),
        "gguf-construction-" + Guid.NewGuid().ToString("N"));

    public GgufConstructionOwnershipTests() => Directory.CreateDirectory(_directory);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedReaderAndShardsRemainOwnedUntilExplicitRelease(bool split)
    {
        string[] paths = WriteMalformedArtifact(split);
        GgufFile? retained = null;
        try
        {
            var failure = Assert.Throws<InvalidDataException>(() =>
                new GgufFile(paths[0], reader => retained = reader));
            Assert.Contains("magic: 0xDEADBEEF", failure.Message);
            Assert.NotNull(retained);
            foreach (string path in paths)
            {
                // The retained read handle prevents a writable exclusive reopen.
                Assert.Throws<IOException>(() => OpenExclusive(path).Dispose());
            }
            retained.Dispose();
            foreach (string path in paths)
            {
                using var reopened = OpenExclusive(path);
                Assert.True(reopened.CanWrite);
            }
        }
        finally
        {
            retained?.Dispose();
        }
    }

    [Fact]
    public void StandaloneShardFailureReleasesTheWholeReaderGraph()
    {
        string[] paths = WriteMalformedArtifact(split: true);
        var failure = Assert.Throws<InvalidDataException>(() => new GgufFile(paths[0]));
        Assert.Contains("magic: 0xDEADBEEF", failure.Message);
        foreach (string path in paths)
        {
            using var reopened = OpenExclusive(path);
            Assert.True(reopened.CanWrite);
        }
    }

    private string[] WriteMalformedArtifact(bool split)
    {
        string malformed = Path.Combine(_directory, split ? "projector-00002-of-00002.gguf" : "projector.gguf");
        using (var writer = new BinaryWriter(File.Create(malformed))) writer.Write(0xDEADBEEFu);
        if (!split) return [malformed];

        string first = Path.Combine(_directory, "projector-00001-of-00002.gguf");
        using (var writer = new BinaryWriter(File.Create(first)))
        {
            writer.Write(0x46554747u);
            writer.Write(3u);
            writer.Write(0ul);
            writer.Write(2ul);
            WriteMetadata(writer, "split.count", 2);
            WriteMetadata(writer, "split.no", 0);
        }
        return [first, malformed];
    }

    private static void WriteMetadata(BinaryWriter writer, string name, uint value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(name);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
        writer.Write((uint)GgufValueType.Uint32);
        writer.Write(value);
    }

    private static FileStream OpenExclusive(string path)
        => File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
