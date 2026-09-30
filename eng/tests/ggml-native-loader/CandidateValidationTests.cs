using System.Security.Cryptography;
using TensorSharp.GGML;
using Xunit;

namespace TensorSharp.NativeLoader.Tests;

public sealed class CandidateValidationTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ggml-candidate-" + Guid.NewGuid().ToString("N"));
    private readonly GgmlNativeCandidate candidate;

    public CandidateValidationTests()
    {
        Directory.CreateDirectory(root);
        // These bytes exercise manifest validation only, never native initialization.
        File.WriteAllText(Path.Combine(root, GgmlNativeLoader.EntryLibraryName), "manifest fixture");
        candidate = new(root, GgmlNativeLoader.TensorSharpBuild, GgmlNativeLoader.RuntimeIdentifier,
            "cpu", GgmlBackendType.Cpu, [Describe(GgmlNativeLoader.EntryLibraryName)]);
    }

    [Fact]
    public void MatchingFilesPassWithoutLoadingNativeCode()
    {
        Assert.Null(GgmlNativeLoader.Check(candidate));
        Assert.Null(GgmlNativeLoader.Current);
    }

    [Fact]
    public void WrongRidAndBuildAreRefused()
    {
        Refused(candidate with { Rid = "another-rid" }, GgmlNativeRefusalCodes.IncompatibleHardware);
        Refused(candidate with { TensorSharpBuild = "another-build" }, GgmlNativeRefusalCodes.NotSelected);
    }

    [Fact]
    public void BackendAndVariantMustBeDeclared()
    {
        Refused(candidate with { Backend = (GgmlBackendType)999 }, GgmlNativeRefusalCodes.IncompatibleHardware);
        Refused(candidate with { Variant = " " }, GgmlNativeRefusalCodes.NotSelected);
    }

    [Fact]
    public void MissingLibraryIsRefusedWithAndWithoutManifest()
    {
        File.Delete(Path.Combine(root, GgmlNativeLoader.EntryLibraryName));
        Refused(candidate, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [] }, GgmlNativeRefusalCodes.LoadFailed);
    }

    [Fact]
    public void ChangedSizeOrHashIsRefused()
    {
        GgmlNativeFile file = candidate.Files[0];
        Refused(candidate with { Files = [file with { Size = file.Size + 1 }] }, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [file with { Sha256 = new string('0', 64) }] }, GgmlNativeRefusalCodes.HashMismatch);
    }

    [Theory]
    [InlineData("")]
    [InlineData("../escaped")]
    [InlineData("nested/../escaped")]
    [InlineData("nested//escaped")]
    [InlineData("nested/./escaped")]
    [InlineData("/absolute")]
    [InlineData("C:/absolute")]
    [InlineData("nested\\escaped")]
    [InlineData("nul\0byte")]
    public void UnsafeRelativePathsAreRefused(string path)
    {
        Refused(candidate with { Files = [candidate.Files[0], new(path, 0, new string('0', 64))] },
            GgmlNativeRefusalCodes.HashMismatch);
    }

    [Fact]
    public void DuplicateOrMalformedManifestEntriesAreRefused()
    {
        GgmlNativeFile file = candidate.Files[0];
        Refused(candidate with { Files = [file, file] }, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [file, null!] }, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [file with { Size = -1 }] }, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [file with { Sha256 = "invalid" }] }, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [], Directory = "relative" }, GgmlNativeRefusalCodes.LoadFailed);
        Refused(candidate with { Directory = root + "\0" }, GgmlNativeRefusalCodes.LoadFailed);
    }

    [Fact]
    public void FileListMustNameTheEntryLibrary()
    {
        Refused(candidate with { Files = [new("notice.txt", 0, new string('0', 64))] }, GgmlNativeRefusalCodes.HashMismatch);
    }

    [Fact]
    public void LinkedEntryIsRefusedWithoutAFileList()
    {
        string entry = Path.Combine(root, GgmlNativeLoader.EntryLibraryName);
        string target = Path.Combine(root, "target");
        File.Move(entry, target);
        File.CreateSymbolicLink(entry, target);
        Refused(candidate, GgmlNativeRefusalCodes.HashMismatch);
        Refused(candidate with { Files = [] }, GgmlNativeRefusalCodes.HashMismatch);
    }

    [Fact]
    public void LinkedCandidateDirectoryIsRefused()
    {
        string link = Path.Combine(root, "linked");
        Directory.CreateSymbolicLink(link, root);
        Refused(candidate with { Directory = link }, GgmlNativeRefusalCodes.HashMismatch);
    }

    [Fact]
    public void LinkedParentOfCandidateDirectoryIsRefused()
    {
        string nested = Path.Combine(root, "actual", "nested");
        Directory.CreateDirectory(nested);
        File.Copy(Path.Combine(root, GgmlNativeLoader.EntryLibraryName), Path.Combine(nested, GgmlNativeLoader.EntryLibraryName));
        string link = Path.Combine(root, "linked");
        Directory.CreateSymbolicLink(link, Path.Combine(root, "actual"));
        Refused(candidate with { Directory = Path.Combine(link, "nested") }, GgmlNativeRefusalCodes.HashMismatch);
    }

    [Fact]
    public void LinkedDirectoryInsideManifestIsRefused()
    {
        string actual = Path.Combine(root, "actual");
        Directory.CreateDirectory(actual);
        File.WriteAllText(Path.Combine(actual, "dependency"), "dependency fixture");
        Directory.CreateSymbolicLink(Path.Combine(root, "linked"), actual);
        Refused(candidate with { Files = [candidate.Files[0], Describe("linked/dependency")] }, GgmlNativeRefusalCodes.HashMismatch);
    }

    private GgmlNativeFile Describe(string path)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(root, path));
        return new(path, bytes.Length, Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }

    private static void Refused(GgmlNativeCandidate value, string code) =>
        Assert.Equal(code, Assert.IsType<GgmlNativeRefusal>(GgmlNativeLoader.Check(value)).Code);

    public void Dispose() => Directory.Delete(root, recursive: true);
}
