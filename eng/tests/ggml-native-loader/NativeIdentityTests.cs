using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.GGML;
using Xunit;

namespace TensorSharp.NativeLoader.Tests;

public sealed class NativeIdentityTests
{
    private static string Raw => $"format=1;tensorsharp={GgmlNativeLoader.TensorSharpBuild};source={new string('a', 40)};ggml={new string('b', 40)};variant=cpu;rid={GgmlNativeLoader.RuntimeIdentifier};cpu=portable;abi={GgmlNativeLoader.NativeAbi}";

    [Fact]
    public void ExactSourceAndAbiFieldsRoundTrip()
    {
        GgmlNativeBuildIdentity identity = GgmlNativeBuildIdentity.Parse(Raw);
        Assert.Equal(GgmlNativeLoader.NativeAbi, identity.NativeAbi);
        Assert.Equal(new string('a', 40), identity.SourceCommit);
        Assert.Equal(new string('b', 40), identity.GgmlCommit);
        Assert.Equal(Raw, identity.Raw);
    }

    [Theory]
    [InlineData("format", "2")]
    [InlineData("source", "unknown")]
    [InlineData("ggml", "unknown")]
    [InlineData("abi", "unknown")]
    [InlineData("abi", "ABCDEF")]
    [InlineData("rid", "")]
    [InlineData("cpu", "")]
    [InlineData("variant", "")]
    public void InexactOrMissingFieldsCannotBecomeAnIdentity(string field, string value)
    {
        string raw = string.Join(';', Raw.Split(';').Select(pair => pair.StartsWith(field + "=", StringComparison.Ordinal) ? field + "=" + value : pair));
        Assert.Throws<FormatException>(() => GgmlNativeBuildIdentity.Parse(raw));
    }

    [Theory]
    [InlineData(";abi=duplicate")]
    [InlineData(";malformed")]
    [InlineData(";unknown=\nvalue")]
    [InlineData(";;unknown=value")]
    [InlineData(";")]
    public void DuplicateMalformedOrNonAsciiFieldsAreRejected(string extra) =>
        Assert.Throws<FormatException>(() => GgmlNativeBuildIdentity.Parse(Raw + extra));

    [Fact]
    public void ManagedAndNativeBuildToolsProduceTheSameIdentity()
    {
        string root = RepositoryRoot();
        using var process = new Process
        {
            StartInfo = new("cmake")
            {
                WorkingDirectory = root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            }
        };
        process.StartInfo.ArgumentList.Add("-P");
        process.StartInfo.ArgumentList.Add(Path.Combine(root, "eng", "print-ggml-native-abi.cmake"));
        Assert.True(process.Start());
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(30_000));
        Assert.Equal(0, process.ExitCode);
        Assert.Equal("-- GgmlNativeAbi=" + GgmlNativeLoader.NativeAbi, output.Trim());
        Assert.Equal(string.Empty, error);
        Assert.Equal(64, GgmlNativeLoader.NativeAbi.Length);
        Assert.Equal(File.ReadAllText(Path.Combine(root, "eng", "ggml-revision")).Trim(), GgmlNativeLoader.GgmlCommit);
    }

    [Fact]
    public void ResolverRegistrationAndStaticConstructionDoNotBindNativeImports()
    {
        Type native = typeof(GgmlNativeLoader).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!;
        RuntimeHelpers.RunClassConstructor(native.TypeHandle);
        native.GetMethod("EnsureImportResolverRegistered", BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, null);
        Assert.False((bool)typeof(GgmlNativeLoader).GetField("s_defaultProbingBound", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!);
        Assert.Null(GgmlNativeLoader.Current);
    }

    private static string RepositoryRoot()
    {
        for (DirectoryInfo? directory = new(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "eng", "print-ggml-native-abi.cmake")))
                return directory.FullName;
        throw new DirectoryNotFoundException("The native identity test runs inside its TensorSharp worktree.");
    }
}
