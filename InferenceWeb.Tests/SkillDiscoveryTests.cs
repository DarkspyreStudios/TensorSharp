// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.IO;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

public sealed class SkillDiscoveryTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-skill-discovery-" + Guid.NewGuid().ToString("N"));

    public SkillDiscoveryTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    private static string AddSkills(string directory)
    {
        string root = Path.Combine(directory, ".agents", "skills");
        Directory.CreateDirectory(root);
        return root;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepositoryDiscoveryHonorsNearestRootPrecedenceAndGitBoundary(bool worktree)
    {
        AddSkills(_directory);
        string repo = Path.Combine(_directory, "repo");
        string repoSkills = AddSkills(repo);
        if (worktree)
            File.WriteAllText(Path.Combine(repo, ".git"), "gitdir: ../elsewhere");
        else
            Directory.CreateDirectory(Path.Combine(repo, ".git"));
        string nested = Path.Combine(repo, "project", "src");
        string projectSkills = AddSkills(Path.GetDirectoryName(nested)!);
        string nestedSkills = AddSkills(nested);

        Assert.Equal(new[] { nestedSkills, projectSkills, repoSkills }, SkillDiscovery.RepositoryRoots(nested));
    }

    [Fact]
    public void OutsideRepositoryOnlyWorkingDirectoryIsConsidered()
    {
        AddSkills(_directory);
        string nested = Path.Combine(_directory, "project");
        string nestedSkills = AddSkills(nested);

        Assert.Equal(new[] { nestedSkills }, SkillDiscovery.RepositoryRoots(nested));
    }

    [Fact]
    public void MissingDirectoryReturnsNoRoots()
    {
        Assert.Empty(SkillDiscovery.RepositoryRoots(Path.Combine(_directory, "missing")));
        Assert.Empty(SkillDiscovery.RepositoryRoots(string.Empty));
    }

    [Fact]
    public void RepositoryAndBinaryRootsAreDefaultsButExplicitRootsRemainExclusive()
    {
        using var environment = new EnvScope();
        environment.Set(SkillHostOptions.RootsEnvVar, null);
        string repositoryRoot = AddSkills(_directory);
        string binary = Path.Combine(_directory, "bin");

        SkillHostOptions defaults = SkillHostOptions.Parse(Array.Empty<string>())
            .ApplyEnvironmentAndDefaults(binary, _directory);
        Assert.Equal(new[] { repositoryRoot, Path.Combine(binary, "skills") }, defaults.Roots);
        // Hosts previously recognized a default only when the root list had one entry.
        // Multiple discovered roots must still permit first-run binary-root creation.
        defaults.ValidateRoots();
        Assert.True(Directory.Exists(Path.Combine(binary, "skills")));

        SkillHostOptions explicitRoots = SkillHostOptions.Parse(new[] { "--skills-dir", "/operator-selected" })
            .ApplyEnvironmentAndDefaults(binary, _directory);
        Assert.Equal(new[] { "/operator-selected" }, explicitRoots.Roots);

        environment.Set(SkillHostOptions.RootsEnvVar, "/environment-selected");
        SkillHostOptions environmentRoots = SkillHostOptions.Parse(Array.Empty<string>())
            .ApplyEnvironmentAndDefaults(binary, _directory);
        Assert.Equal(new[] { "/environment-selected" }, environmentRoots.Roots);
    }

    [Fact]
    public void ExplicitBinaryRootIsNotCreatedAutomatically()
    {
        using var environment = new EnvScope();
        environment.Set(SkillHostOptions.RootsEnvVar, null);
        string binary = Path.Combine(_directory, "bin");
        string missing = Path.Combine(binary, "skills");
        SkillHostOptions options = SkillHostOptions.Parse(new[] { "--skills-dir", missing })
            .ApplyEnvironmentAndDefaults(binary, _directory);

        Assert.Throws<ArgumentException>(() => options.ValidateRoots());
        Assert.False(Directory.Exists(missing));
    }
}
