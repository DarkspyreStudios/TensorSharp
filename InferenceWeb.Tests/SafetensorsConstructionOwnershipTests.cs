using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class SafetensorsConstructionOwnershipTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(),
        "safetensors-construction-" + Guid.NewGuid().ToString("N"));

    public SafetensorsConstructionOwnershipTests() => Directory.CreateDirectory(_directory);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedHeaderStreamFollowsItsActualConstructionOwner(bool parentOwned)
    {
        string path = Path.Combine(_directory, "projector.safetensors");
        using (var writer = new BinaryWriter(File.Create(path))) writer.Write(0ul);
        SafetensorsFile? retained = null;
        try
        {
            var failure = Assert.Throws<InvalidDataException>(() => parentOwned
                ? new SafetensorsFile(path, reader => retained = reader)
                : new SafetensorsFile(path));
            Assert.Contains("header length 0 is out of range", failure.Message);
            if (parentOwned)
            {
                Assert.NotNull(retained);
                Assert.Throws<IOException>(() => OpenExclusive(path).Dispose());
                retained.Dispose();
            }
            else
            {
                Assert.Null(retained);
            }
            using var reopened = OpenExclusive(path);
            Assert.True(reopened.CanWrite);
        }
        finally
        {
            retained?.Dispose();
        }
    }

    private static FileStream OpenExclusive(string path)
        => File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
