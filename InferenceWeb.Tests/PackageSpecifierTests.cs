using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

public class PackageSpecifierTests
{
    private static PackageInstaller Installer(params string[] allowed) => new(
        new CodeExecOptions { AllowInstall = true, AllowedPackages = allowed }, (ISkillSandbox)null);

    [Theory]
    [InlineData("@scope/tool", "@scope/tool")]
    [InlineData("@scope/tool@1.2.3", "@scope/tool")]
    [InlineData("@scope/tool@latest", "@scope/tool")]
    [InlineData("typescript@^5.0.0", "typescript")]
    [InlineData("typescript@~5.0", "typescript")]
    [InlineData("some-tool@1.0.0-beta.1", "some-tool")]
    public void NpmRegistrySpecsPreserveScopeAndSeparateVersion(string spec, string name)
    {
        Assert.True(Installer().TryValidate(new[] { spec }, out var error, CodeLanguage.JavaScript), error);
        Assert.Equal(name, PackageInstaller.BareName(spec, CodeLanguage.JavaScript));
    }

    [Theory]
    [InlineData("https://example.com/package.tgz")]
    [InlineData("file:../package")]
    [InlineData("owner/repo")]
    [InlineData("@scope/../package")]
    [InlineData("tool@npm:other")]
    [InlineData("tool@https://example.com")]
    [InlineData("--registry=example.com")]
    [InlineData("tool\n")]
    [InlineData("@scope/tool@")]
    [InlineData("numpy==2.1.0")]
    public void NpmCannotRedirectInstallerOrUsePythonSpec(string spec)
    {
        Assert.False(Installer().TryValidate(new[] { spec }, out _, CodeLanguage.JavaScript));
    }

    [Fact]
    public void ScopedAllowListUsesCompleteNameWithoutVersion()
    {
        var installer = Installer("@scope/tool");
        Assert.True(installer.TryValidate(new[] { "@scope/tool@latest" }, out _, CodeLanguage.JavaScript));
        Assert.False(installer.TryValidate(new[] { "@other/tool@latest" }, out _, CodeLanguage.JavaScript));
        Assert.False(installer.TryValidate(new[] { "tool" }, out _, CodeLanguage.JavaScript));
    }

    [Fact]
    public void PythonValidationStillUsesPythonSyntax()
    {
        var installer = Installer("numpy");
        Assert.True(installer.TryValidate(new[] { "numpy==2.1.0" }, out _));
        Assert.False(installer.TryValidate(new[] { "numpy@latest" }, out _));
        Assert.False(installer.TryValidate(new[] { "@scope/tool" }, out _));
        Assert.False(installer.TryValidate(new[] { "numpy\n" }, out _));
    }
}
