"""Offline checks of the actual PowerShell verifier's package policies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (
    "TensorSharp.Core", "TensorSharp.Runtime.Logging", "TensorSharp.Runtime",
    "TensorSharp.AgentHost", "TensorSharp.Backends.Cuda", "TensorSharp.Backends.GGML",
    "TensorSharp.Backends.MLX", "TensorSharp.Distributed", "TensorSharp.Models",
    "TensorSharp.Chat", "TensorSharp.Server", "TensorSharp.Server.Host", "TensorSharp.Cli",
)

# Evaluate only policy statements parsed from the shipping script. No script
# entrypoint, package IO, MSBuild, pack, native compiler or network call runs.
POLICY_READER = r"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0
function dotnet { throw 'Offline tests must not execute dotnet.' }
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:VERIFY_PACKAGES_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw ($errors -join "`n") }
function Assignment([string]$name) {
    $nodes = @($ast.FindAll({ param($node)
        $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $node.Left.Extent.Text -eq $name
    }.GetNewClosure(), $true))
    if ($nodes.Count -ne 1) { throw "Expected one assignment for $name." }
    return $nodes[0].Extent.Text
}
Invoke-Expression (Assignment '$PublicPackages')
$skip = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Clauses[0].Item1.Extent.Text -eq '$SkipNativeBuild'
}, $true))
if ($skip.Count -ne 1) { throw 'Expected one native-skip policy.' }
$ExtraPackArgs = @(); $SkipNativeBuild = $false
Invoke-Expression $skip[0].Extent.Text
$defaultArgs = @($ExtraPackArgs)
$ExtraPackArgs = @(); $SkipNativeBuild = $true
Invoke-Expression $skip[0].Extent.Text
$skipArgs = @($ExtraPackArgs)
$pack = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.CommandAst] -and
    $node.CommandElements[0].Extent.Text -eq 'Invoke-CheckedDotNet'
}, $true))
if ($pack.Count -ne 1) { throw 'Expected one checked pack invocation.' }
function Invoke-CheckedDotNet([string[]]$Arguments) { $script:capturedArgs = @($Arguments) }
$projectPath = 'controlled.csproj'; $PackageOutput = 'controlled-output'
$Configuration = $env:VERIFY_CONFIGURATION
Invoke-Expression $pack[0].Extent.Text
$skipPackArgs = @($capturedArgs)
$ExtraPackArgs = @()
Invoke-Expression $pack[0].Extent.Text
$defaultPackArgs = @($capturedArgs)
$assert = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Assert-SameSet'
}, $true))
if ($assert.Count -ne 1) { throw 'Expected one dependency assertion.' }
Invoke-Expression $assert[0].Extent.Text
$cases = @()
foreach ($case in ($env:VERIFY_DEPENDENCY_CASES | ConvertFrom-Json)) {
    $nupkg = [pscustomobject]@{ Dependencies = @($case.Actual | ForEach-Object {
        [pscustomobject]@{ Id = $_; Version = '2.8.6.8' }
    }) }
    Invoke-Expression (Assignment '$internalDependencies')
    $failure = $null
    try { Assert-SameSet -PackageId 'controlled' -Actual $internalDependencies -Expected $case.Expected }
    catch { $failure = $_.Exception.Message }
    $cases += [pscustomobject]@{ Name = $case.Name; Actual = @($internalDependencies); Failure = $failure }
}
[pscustomobject]@{
    Packages = @($PublicPackages | ForEach-Object {
        [pscustomobject]@{ Id = $_.Id; Project = $_.Project;
            Dependencies = @($_.TensorSharpDependencies); Embedded = @($_.EmbeddedAssemblies) }
    }); DefaultArgs = $defaultArgs; SkipArgs = $skipArgs; Cases = $cases
    DefaultPackArgs = $defaultPackArgs; SkipPackArgs = $skipPackArgs
} | ConvertTo-Json -Depth 8 -Compress
"""


def package_id(project):
    root = ET.parse(project).getroot()
    return root.findtext(".//PackageId") or "Darkspyre." + project.stem


class VerifyPackagesPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            raise RuntimeError("PowerShell is required for verifier policy tests.")
        cases = [
            {"Name": "valid", "Actual": ["Darkspyre.TensorSharp.Tensors", "System.Text.Json"],
             "Expected": ["Darkspyre.TensorSharp.Tensors"]},
            {"Name": "unexpected", "Actual": ["Darkspyre.TensorSharp.Unexpected"], "Expected": []},
            {"Name": "native", "Actual": ["Darkspyre.TensorSharp.Backends.GGML.Native.win-arm64"], "Expected": []},
            {"Name": "legacy", "Actual": ["TensorSharp.Tensors"], "Expected": []},
            {"Name": "embedded", "Actual": ["AdvUtils"], "Expected": []},
            {"Name": "missing", "Actual": ["System.Text.Json"], "Expected": ["Darkspyre.TensorSharp.Tensors"]},
            {"Name": "external", "Actual": ["System.Text.Json", "Microsoft.Extensions.Logging"], "Expected": []},
        ]
        env = os.environ | {
            "VERIFY_PACKAGES_SCRIPT": str(ROOT / "eng/verify-packages.ps1"),
            "VERIFY_DEPENDENCY_CASES": json.dumps(cases),
            "VERIFY_CONFIGURATION": os.environ.get("VERIFY_CONFIGURATION", "Release"),
        }
        result = subprocess.run([pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", POLICY_READER],
                                env=env, capture_output=True, text=True, timeout=30, check=True)
        cls.policy = json.loads(result.stdout)
        cls.cases = {case["Name"]: case for case in cls.policy["Cases"]}

    def test_inventory_matches_exact_committed_package_ids(self):
        expected = {f"{name}/{name}.csproj": package_id(ROOT / name / f"{name}.csproj") for name in PROJECTS}
        self.assertEqual(len(expected), len(self.policy["Packages"]))
        self.assertEqual(expected, {item["Project"]: item["Id"] for item in self.policy["Packages"]})

    def test_dependencies_match_direct_nonprivate_project_references(self):
        for package in self.policy["Packages"]:
            project = ROOT / package["Project"]
            expected = []
            for reference in ET.parse(project).getroot().findall(".//ProjectReference"):
                if reference.get("PrivateAssets") != "all":
                    target = (project.parent / reference.attrib["Include"].replace("\\", "/")).resolve()
                    expected.append(package_id(target))
            with self.subTest(package=package["Id"]):
                self.assertCountEqual(expected, package["Dependencies"])

    def test_embedded_advutils_is_not_a_package_dependency(self):
        core = next(item for item in self.policy["Packages"] if item["Project"].startswith("TensorSharp.Core/"))
        self.assertEqual(["AdvUtils.dll"], core["Embedded"])
        self.assertEqual([], core["Dependencies"])
        self.assertIsNotNone(self.cases["embedded"]["Failure"])

    def test_native_skip_covers_all_three_backends(self):
        self.assertCountEqual(["-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true",
                               "-p:TensorSharpSkipCudaNative=true"], self.policy["SkipArgs"])

    def test_default_does_not_skip_native_builds(self):
        self.assertEqual([], self.policy["DefaultArgs"])

    def test_checked_pack_call_preserves_configuration_and_skip_arguments(self):
        expected = ["pack", "controlled.csproj", "-c", os.environ.get("VERIFY_CONFIGURATION", "Release"),
                    "-o", "controlled-output"]
        self.assertEqual(expected, self.policy["DefaultPackArgs"])
        self.assertEqual(expected + self.policy["SkipArgs"], self.policy["SkipPackArgs"])

    def test_valid_internal_dependencies_and_external_packages_pass(self):
        self.assertIsNone(self.cases["valid"]["Failure"])
        self.assertEqual(["Darkspyre.TensorSharp.Tensors"], self.cases["valid"]["Actual"])
        self.assertIsNone(self.cases["external"]["Failure"])
        self.assertEqual([], self.cases["external"]["Actual"])

    def test_unexpected_and_legacy_supplier_dependencies_fail(self):
        for name in ("unexpected", "native", "legacy"):
            with self.subTest(name=name):
                self.assertIsNotNone(self.cases[name]["Failure"])

    def test_missing_supplier_dependency_fails(self):
        self.assertIsNotNone(self.cases["missing"]["Failure"])


if __name__ == "__main__":
    unittest.main()
