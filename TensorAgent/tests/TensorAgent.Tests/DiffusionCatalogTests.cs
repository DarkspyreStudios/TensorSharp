// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// A representative Qwen-Image-2.1 entry used only by the test assembly.
///
/// <para>
/// TensorAgent intentionally offers no diffusion checkpoint in its built-in catalog.
/// The companion publisher is still production infrastructure, though, and needs a
/// complete multi-file model to exercise it without making a download appear to be
/// supported. Live media tests also use this fixture to give explicitly supplied local
/// files the names and roles the pipeline expects. The names, sizes and hashes are the
/// ones <c>config/qwen-image-2.1.json</c> downloads.
/// </para>
/// </summary>
internal static class DiffusionModelFixture
{
    internal static CatalogModel QwenImage21 { get; } = new()
    {
        Id = "test-qwen-image-2.1",
        DisplayName = "Qwen-Image-2.1 test fixture",
        Family = CatalogFamily.QwenImage,
        Kind = CatalogArchitectureKind.Diffusion,
        Parameters = "Qwen-Image-2.1 DiT + Qwen3-VL-8B text encoder",
        Quantization = "Q4_K_M (DiT) / Q4_K_M (text encoder)",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "qwen_image_2.1_Q4_K_M.gguf", string.Empty,
                4_189_343_904, "dc956c958fbfa1d5c64ec316d7e865283d17d97a9eb332a4a74a4d63afaae9a5"),
            new CatalogFile(CatalogFileRole.TextEncoder, "Qwen3VL-8B-Instruct-Q4_K_M.gguf", string.Empty,
                5_027_784_800, "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
            new CatalogFile(CatalogFileRole.Vae, "qwen_image_2.1_vae_bf16.safetensors", string.Empty,
                675_509_688, "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
            // Optional for loading and text-to-image; editing refuses to run without it.
            new CatalogFile(CatalogFileRole.VisionProjector,
                "mmproj-Qwen3VL-8B-Instruct-F16.gguf", string.Empty,
                1_159_029_824, "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38",
                Optional: true),
        },
        Modalities = CatalogModalities.Image | CatalogModalities.ImageOutput,
        MinDeviceMemoryGB = 24,
        ContextLength = 0,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
        Experimental = true,
        License = "Apache-2.0",
        Notes = "Test-only diffusion fixture; not a built-in TensorAgent model.",
    };
}

/// <summary>
/// Tests that write the process's own environment, kept out of everything else's way.
///
/// <para>
/// <c>DiffusionCompanions</c> publishes to environment variables because that is the
/// only channel the Qwen-Image DiT reads, and an environment is one object shared by
/// every test in the assembly. Any class that constructs an <c>AgentAppHost</c>
/// publishes too, so without this the two race and the loser fails somewhere else
/// entirely — which is exactly what happened before it was added.
/// </para>
/// </summary>
[CollectionDefinition(ProcessEnvironmentCollection.Name, DisableParallelization = true)]
public sealed class ProcessEnvironmentCollection
{
    public const string Name = "process environment";
}

/// <summary>
/// Whether a representative image-generation model's four files are the four files
/// the pipeline goes looking for, under names it will recognise.
///
/// <para>
/// A companion whose name the pipeline's directory scan does not match can be present
/// and valid yet still be silently ignored. Keeping a representative definition here
/// catches that integration failure without requiring a built-in catalog entry.
/// </para>
/// <para>
/// The scans are stated here rather than called, because they are private to
/// <c>QwenImageModel</c>. <see cref="TheScansThisFileMirrorsAreStillTheOnesTheModelPerforms"/>
/// is what keeps the two from drifting apart.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class DiffusionCatalogTests
{
    private static CatalogModel QwenImage21 =>
        DiffusionModelFixture.QwenImage21;

    /// <summary>What <c>DiffusionCompanions</c> publishes, and all it publishes.</summary>
    private static readonly string[] CompanionVariables =
    [
        "TS_QWEN_IMAGE_VAE", "TS_QWEN_IMAGE_TE", "TS_QWEN_IMAGE_MMPROJ",
    ];

    /// <summary>
    /// Variables only the retired Qwen-Image-Edit-2511 pipeline read. Nothing may publish
    /// them: nothing reads the area cap any more, and a set <c>TS_QWEN_IMAGE_LORA</c> makes
    /// Qwen-Image-2.1 refuse to run at all.
    /// </summary>
    private static readonly string[] RetiredVariables =
    [
        "TS_QWEN_IMAGE_LORA", "TS_QWEN_IMAGE_MAX_AREA",
    ];

    private static string NameOf(CatalogModel model, CatalogFileRole role) =>
        model.Files.Single(f => f.Role == role).FileName;

    /// <summary>The VAE scan in <c>QwenImageModel</c>'s constructor: a safetensors file
    /// naming the 2.1 VAE.</summary>
    private static bool VaeScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return n.Contains("qwen_image_2.1_vae") && n.EndsWith(".safetensors");
    }

    /// <summary>The text-encoder scan: the largest GGUF naming Qwen3-VL-8B that is not
    /// a projector.</summary>
    private static bool TextEncoderScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj") && n.EndsWith(".gguf");
    }

    /// <summary>The vision-projector scan: a GGUF naming both a projector and
    /// Qwen3-VL-8B.</summary>
    private static bool VisionProjectorScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return n.Contains("mmproj") && (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && n.EndsWith(".gguf");
    }

    [Fact]
    public void TheQwenImage21FixtureCarriesEveryNetworkTheDiTDoesNotContain()
    {
        CatalogModel model = QwenImage21;
        foreach (CatalogFileRole role in new[]
                 {
                     CatalogFileRole.Weights, CatalogFileRole.TextEncoder,
                     CatalogFileRole.Vae, CatalogFileRole.VisionProjector,
                 })
        {
            Assert.True(model.Files.Any(f => f.Role == role), $"{model.Id} has no {role}");
        }

        // The VAE and the text encoder are not optional: QwenImageModel's constructor
        // throws FileNotFoundException without them, so an entry that marks either one
        // optional is an entry that can finish downloading and then refuse to load.
        Assert.False(model.Files.Single(f => f.Role == CatalogFileRole.Vae).Optional);
        Assert.False(model.Files.Single(f => f.Role == CatalogFileRole.TextEncoder).Optional);
    }

    [Fact]
    public void EveryFixtureCompanionIsNamedSomethingThePipelinesOwnScanWillMatch()
    {
        CatalogModel model = QwenImage21;

        Assert.True(VaeScanFinds(NameOf(model, CatalogFileRole.Vae)),
            $"the VAE '{NameOf(model, CatalogFileRole.Vae)}' is not a name the 2.1 VAE scan looks for");

        Assert.True(TextEncoderScanFinds(NameOf(model, CatalogFileRole.TextEncoder)),
            $"the text encoder '{NameOf(model, CatalogFileRole.TextEncoder)}' is not a name the Qwen3-VL-8B scan looks for");

        Assert.True(VisionProjectorScanFinds(NameOf(model, CatalogFileRole.VisionProjector)),
            $"the vision projector '{NameOf(model, CatalogFileRole.VisionProjector)}' is not a name the mmproj scan "
            + "looks for — it needs 'mmproj' AND 'qwen3vl-8b' (or 'qwen3-vl-8b') in it, or editing refuses to run "
            + "for want of a projector that is sitting right there");
    }

    [Fact]
    public void NoTwoFixtureCompanionsAnswerToTheSameScan()
    {
        // The three scans run over one directory, so a name that satisfies two of them
        // hands one network to the wrong loader. The text-encoder scan in particular
        // takes the LARGEST matching GGUF, which is what the projector would be if its
        // name did not say "mmproj".
        CatalogModel model = QwenImage21;
        string weights = NameOf(model, CatalogFileRole.Weights);
        string textEncoder = NameOf(model, CatalogFileRole.TextEncoder);
        string projector = NameOf(model, CatalogFileRole.VisionProjector);

        Assert.False(TextEncoderScanFinds(projector), $"the projector '{projector}' would be loaded as the text encoder");
        Assert.False(TextEncoderScanFinds(weights), $"the DiT '{weights}' would be loaded as the text encoder");
        Assert.False(VisionProjectorScanFinds(textEncoder), $"the text encoder '{textEncoder}' would be loaded as the projector");
        Assert.False(VisionProjectorScanFinds(weights), $"the DiT '{weights}' would be loaded as the projector");
        Assert.False(VaeScanFinds(weights));
        Assert.False(VaeScanFinds(textEncoder));
        Assert.False(VaeScanFinds(projector));
    }

    [Fact]
    public void TheScansThisFileMirrorsAreStillTheOnesTheModelPerforms()
    {
        // A weak guard and an honest one: it cannot tell that a predicate's logic
        // changed, only that the strings it matches on are still there. That is enough
        // to catch the rename this file exists to prevent, and the alternative — making
        // the resolvers public on a shared assembly so a phone test can call them — is
        // a worse trade.
        string model = ReadSource("TensorSharp.Models/Models/QwenImage/QwenImageModel.cs");
        foreach (string literal in new[]
                 {
                     "\"TS_QWEN_IMAGE_VAE\"", "\"TS_QWEN_IMAGE_TE\"", "\"TS_QWEN_IMAGE_MMPROJ\"",
                     "n.Contains(\"qwen_image_2.1_vae\") && n.EndsWith(\".safetensors\")",
                     "(n.Contains(\"qwen3vl-8b\") || n.Contains(\"qwen3-vl-8b\")) && !n.Contains(\"mmproj\") && n.EndsWith(\".gguf\")",
                     "n.Contains(\"mmproj\") && (n.Contains(\"qwen3vl-8b\") || n.Contains(\"qwen3-vl-8b\")) && n.EndsWith(\".gguf\")",
                 })
        {
            Assert.Contains(literal, model, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EveryInstalledCompanionIsPublishedUnderTheVariableTheModelReads()
    {
        using var installation = new FakeInstall(QwenImage21, install: model => model.Files);

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_QWEN_IMAGE_VAE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.TextEncoder), published["TS_QWEN_IMAGE_TE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.VisionProjector), published["TS_QWEN_IMAGE_MMPROJ"]);
        foreach (string variable in CompanionVariables)
            Assert.Equal(published[variable], Environment.GetEnvironmentVariable(variable));
        Assert.Equal(CompanionVariables.Order(StringComparer.Ordinal), published.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void NothingOnlyTheRetiredPipelineReadIsEverPublished()
    {
        // A host that still published the LoRA path would stop every Qwen-Image-2.1 run
        // with "does not support LoRA adapters" the moment such a file was installed.
        // The variables are process-wide and a developer shell may still export them, so
        // start from a known state and hand the caller's values back afterwards.
        var saved = RetiredVariables.ToDictionary(v => v, Environment.GetEnvironmentVariable);
        try
        {
            foreach (string variable in RetiredVariables)
                Environment.SetEnvironmentVariable(variable, null);
            using var installation = new FakeInstall(QwenImage21, install: model => model.Files);

            IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

            foreach (string variable in RetiredVariables)
            {
                Assert.False(published.ContainsKey(variable), $"{variable} was published");
                Assert.Null(Environment.GetEnvironmentVariable(variable));
            }
        }
        finally
        {
            foreach ((string variable, string? value) in saved)
                Environment.SetEnvironmentVariable(variable, value);
        }
    }

    [Fact]
    public void ACompanionThatWasNeverDownloadedIsClearedRatherThanPointedAt()
    {
        // Companion files may be absent from a partial or deliberately minimal local
        // installation (the projector is optional), and the environment is
        // process-wide. Leaving a variable from a previous selection would point the
        // next load at unrelated state.
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ", "/somewhere/from/before.gguf");
        using var installation = new FakeInstall(QwenImage21,
            install: model => model.Files.Where(f => f.Role != CatalogFileRole.VisionProjector));

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

        Assert.Null(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ"));
        Assert.False(published.ContainsKey("TS_QWEN_IMAGE_MMPROJ"));
        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_QWEN_IMAGE_VAE"]);
    }

    [Fact]
    public void SelectingSomethingThatIsNotADiffusionModelLeavesNothingBehind()
    {
        using var installation = new FakeInstall(QwenImage21, install: model => model.Files);
        DiffusionCompanions.Publish(QwenImage21, installation.Store);
        DiffusionCompanions.Publish(null, installation.Store);

        foreach (string variable in CompanionVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
    }

    /// <summary>The repo's own copy of a file, so a test can read the source it mirrors.</summary>
    private static string ReadSource(string relativePath)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;

        Assert.True(directory is not null,
            $"no repository root above {AppContext.BaseDirectory}; this test reads {relativePath} from the working tree");
        string full = Path.Combine(directory!.FullName, relativePath);
        Assert.True(File.Exists(full), $"{full} is missing");
        return File.ReadAllText(full);
    }

    /// <summary>
    /// A model directory holding a chosen subset of an entry's files, so the publisher
    /// can be exercised against what is really on disk. The files are empty: nothing
    /// here loads them, and the point is which paths exist.
    /// </summary>
    private sealed class FakeInstall : IDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-diffusion-" + Guid.NewGuid().ToString("N"));
        private readonly CatalogModel _model;

        public FakeInstall(CatalogModel model, Func<CatalogModel, IEnumerable<CatalogFile>> install)
        {
            _model = model;
            Store = new ModelStore(_root);
            Directory.CreateDirectory(Store.DirectoryFor(model));
            foreach (CatalogFile file in install(model))
                File.WriteAllBytes(Store.PathFor(model, file), []);
        }

        public ModelStore Store { get; }

        public string PathOf(CatalogFileRole role) =>
            Store.PathFor(_model, _model.Files.Single(f => f.Role == role));

        public void Dispose()
        {
            // The publisher writes process-wide state; a test that left it set would
            // decide what the next one sees.
            DiffusionCompanions.Publish(null, Store);
            try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
        }
    }

    /// <summary>
    /// A remembered selection can outlive the catalog entry it names. Constructing a
    /// host for that state must clear process-wide companion paths rather than leave a
    /// previous diffusion run wired into an unrelated model.
    /// </summary>
    [Fact]
    public void BuildingTheHostClearsCompanionsForARemovedDiffusionSelection()
    {
        CatalogModel model = QwenImage21;
        // The id the catalog's Qwen-Image-Edit entry used before it was withdrawn: a
        // device that installed it still has it saved as its selection.
        const string removedId = "qwen-image-edit-2511-q2k";
        Assert.Null(ModelCatalog.Find(removedId));

        string root = Path.Combine(Path.GetTempPath(), "tensoragent-wired-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache")) { DeviceMemoryGB = 16 };
        paths.EnsureCreated();

        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = removedId;
        settings.Save(chosen);

        using var installation = new FakeInstall(model, install: candidate => candidate.Files);
        Assert.NotEmpty(DiffusionCompanions.Publish(model, installation.Store));

        using (var host = new AgentAppHost(paths))
        {
            // Only the published companions are the host's to clear; nothing sets the
            // retired variables any more (NothingOnlyTheRetiredPipelineReadIsEverPublished).
            foreach (string variable in CompanionVariables)
                Assert.Null(Environment.GetEnvironmentVariable(variable));
        }

        try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
    }
}
