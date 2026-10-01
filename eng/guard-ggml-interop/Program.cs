using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

if (args.Length != 2 || args[0] is not ("--rewrite" or "--verify" or "--inventory"))
    throw new ArgumentException("Usage: guard-ggml-interop <--rewrite|--verify|--inventory> <TensorSharp-root>");
string root = Path.GetFullPath(args[1]);
bool inspect = args[0] is "--verify" or "--inventory";
int imports = 0;
var failures = new List<string>();
var entryPoints = new HashSet<string>(StringComparer.Ordinal);
foreach (string file in Directory.EnumerateFiles(Path.Combine(root, "TensorSharp.Backends.GGML"), "*.cs").Order(StringComparer.Ordinal))
{
    string source = File.ReadAllText(file);
    SyntaxNode tree = CSharpSyntaxTree.ParseText(source).GetRoot();
    var declarations = tree.DescendantNodes().OfType<MethodDeclarationSyntax>().Where(IsImport).ToArray();
    if (declarations.Length == 0) continue;
    imports += declarations.Length;
    foreach (MethodDeclarationSyntax method in declarations.Reverse())
    {
        if (args[0] == "--inventory")
        {
            AttributeArgumentSyntax? entryPoint = method.AttributeLists.SelectMany(l => l.Attributes).Where(IsImportAttribute)
                .SelectMany(a => a.ArgumentList!.Arguments).SingleOrDefault(a => a.NameEquals?.Name.Identifier.ValueText == "EntryPoint");
            if (entryPoint?.Expression is not LiteralExpressionSyntax literal || !literal.IsKind(SyntaxKind.StringLiteralExpression) ||
                string.IsNullOrWhiteSpace(literal.Token.ValueText))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: inventory requires an explicit literal entrypoint");
            else
                entryPoints.Add(literal.Token.ValueText);
        }
        if (method.Identifier.ValueText.StartsWith("Native_", StringComparison.Ordinal))
        {
            string wrapperName = method.Identifier.ValueText[7..];
            TypeDeclarationSyntax type = method.Ancestors().OfType<TypeDeclarationSyntax>().First();
            MethodDeclarationSyntax? guarded = type.Members.OfType<MethodDeclarationSyntax>().FirstOrDefault(m => m.Identifier.ValueText == wrapperName);
            if (!method.Modifiers.Any(SyntaxKind.PrivateKeyword) || guarded?.Body == null ||
                !guarded.Body.DescendantNodes().OfType<InvocationExpressionSyntax>().Any(i => i.Expression.ToString() == "GgmlNativeLoader.EnterNativeCall"))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: no mandatory native-call guard");
            var shape = method.WithIdentifier(SyntaxFactory.Identifier(wrapperName));
            string? family = AcquireFamily(wrapperName) ?? ReleaseFamily(wrapperName) ?? UseFamily(shape);
            string bodyText = guarded?.Body?.ToString() ?? string.Empty;
            if (family != null && !bodyText.Contains("\"" + family + "\"", StringComparison.Ordinal))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: missing resource-family guard");
            if (AcquireFamily(wrapperName) != null && !bodyText.Contains("GgmlNativeLoader.TrackNativeHandle", StringComparison.Ordinal))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: allocation is not retained");
            if (AcquireFamily(wrapperName) != null && BackendParameter(method) != null && !bodyText.Contains("GgmlNativeLoader.PrepareModelBackend", StringComparison.Ordinal))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: whole-model backend bypasses configured owner");
            if (ReleaseFamily(wrapperName) != null && (!bodyText.Contains("GgmlNativeLoader.BeginNativeHandleRelease", StringComparison.Ordinal) || !bodyText.Contains("resource.Complete()", StringComparison.Ordinal)))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: release is not exclusive and completed");
            if (method.ReturnType.ToString() == "IntPtr" && AcquireFamily(wrapperName) == null &&
                wrapperName is not ("TSGgml_GetLastError" or "TSGgml_GetBackendFailureText" or "TSGgml_GetBuildIdentity"))
                failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: unclassified returned native handle");
            foreach (InvocationExpressionSyntax invocation in tree.DescendantNodes().OfType<InvocationExpressionSyntax>().Where(i => i.Expression.ToString() == method.Identifier.ValueText))
                if (invocation.Ancestors().OfType<MethodDeclarationSyntax>().First().Identifier.ValueText != wrapperName)
                    failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: raw import invoked outside its guard");
            continue;
        }
        if (inspect)
        {
            failures.Add($"{Path.GetFileName(file)}:{method.Identifier}: unguarded import");
            continue;
        }
        string name = method.Identifier.ValueText;
        string arguments = string.Join(", ", method.ParameterList.Parameters.Select(p =>
            (p.Modifiers.Any(SyntaxKind.OutKeyword) ? "out " : p.Modifiers.Any(SyntaxKind.RefKeyword) ? "ref " : p.Modifiers.Any(SyntaxKind.InKeyword) ? "in " : "") + p.Identifier.Text));
        string call = "Native_" + name + "(" + arguments + ")";
        string? useFamily = UseFamily(method);
        string body = useFamily == null ? "using var call = GgmlNativeLoader.EnterNativeCall();\n" :
            $"using var call = GgmlNativeLoader.EnterNativeCall(\"{useFamily}\", {method.ParameterList.Parameters[0].Identifier.Text});\n";
        if (AcquireFamily(name) != null && BackendParameter(method) is string backend)
            body = $"{backend} = GgmlNativeLoader.PrepareModelBackend({backend});\n" + body;
        if (AcquireFamily(name) is string acquire)
            body += $"return GgmlNativeLoader.TrackNativeHandle(\"{acquire}\", {call});";
        else if (ReleaseFamily(name) is string release)
            body += $"using var resource = GgmlNativeLoader.BeginNativeHandleRelease(\"{release}\", {method.ParameterList.Parameters[0].Identifier.Text});\n{call};\nresource.Complete();";
        else
            body += (method.ReturnType.ToString() == "void" ? call + ";" : "return " + call + ";");
        var wrapper = method.WithAttributeLists(default)
            .WithModifiers(SyntaxFactory.TokenList(method.Modifiers.Where(m => !m.IsKind(SyntaxKind.ExternKeyword) && !m.IsKind(SyntaxKind.PartialKeyword))))
            .WithBody(SyntaxFactory.ParseStatement("{" + body + "}") as BlockSyntax)
            .WithSemicolonToken(default);
        var rawAttributes = method.AttributeLists.Select(list => list.WithAttributes(SyntaxFactory.SeparatedList(list.Attributes.Select(attribute =>
        {
            if (attribute.Name.ToString() is not ("LibraryImport" or "DllImport") || attribute.ArgumentList!.Arguments.Any(a => a.NameEquals?.Name.Identifier.ValueText == "EntryPoint"))
                return attribute;
            return attribute.WithArgumentList(attribute.ArgumentList.AddArguments(SyntaxFactory.AttributeArgument(SyntaxFactory.LiteralExpression(SyntaxKind.StringLiteralExpression, SyntaxFactory.Literal(name)))
                .WithNameEquals(SyntaxFactory.NameEquals("EntryPoint"))));
        }))));
        var raw = method.WithIdentifier(SyntaxFactory.Identifier("Native_" + name)).WithAttributeLists(SyntaxFactory.List(rawAttributes))
            .WithModifiers(SyntaxFactory.TokenList(new[] { SyntaxFactory.Token(SyntaxKind.PrivateKeyword) }.Concat(method.Modifiers.Where(m => !m.IsKind(SyntaxKind.PublicKeyword) && !m.IsKind(SyntaxKind.InternalKeyword) && !m.IsKind(SyntaxKind.ProtectedKeyword) && !m.IsKind(SyntaxKind.PrivateKeyword)))))
            .WithoutLeadingTrivia();
        int lineStart = source.LastIndexOf('\n', method.SpanStart) + 1;
        string indent = source[lineStart..method.SpanStart];
        string replacement = wrapper.WithoutLeadingTrivia().NormalizeWhitespace(eol: "\n").ToFullString() + "\n\n" + raw.NormalizeWhitespace(eol: "\n").ToFullString();
        replacement = replacement.Replace("\n", "\n" + indent);
        source = source[..method.SpanStart] + replacement + source[method.Span.End..];
    }
    if (args[0] == "--rewrite")
    {
        source = string.Join("\n", source.Split('\n').Select(line => line.TrimEnd(' ', '\t', '\r')));
        if (CSharpSyntaxTree.ParseText(source).GetDiagnostics().Any(d => d.Severity == DiagnosticSeverity.Error))
            throw new InvalidOperationException("The rewrite produced invalid C# syntax: " + file);
        bool bom = File.ReadAllBytes(file).Take(3).SequenceEqual(new byte[] { 0xef, 0xbb, 0xbf });
        File.WriteAllText(file, source, new System.Text.UTF8Encoding(bom));
    }
}
if (inspect)
{
    foreach (string failure in failures) Console.Error.WriteLine(failure);
    if (args[0] == "--inventory" && failures.Count == 0)
    {
        string[] extensions = [".cpp", ".h", ".hpp", ".inc", ".cu", ".cuh"];
        var inputs = Directory.EnumerateFiles(Path.Combine(root, "TensorSharp.GGML.Native"))
            .Where(path => extensions.Contains(Path.GetExtension(path)))
            .Concat(Directory.EnumerateFiles(Path.Combine(root, "TensorSharp.Backends.GGML"), "*.cs"))
            .Append(Path.Combine(root, "eng", "ggml-revision"))
            .Select(path => (Path: path, Relative: Path.GetRelativePath(root, path).Replace('\\', '/')))
            .OrderBy(item => item.Relative, StringComparer.Ordinal);
        var manifest = new StringBuilder();
        foreach (var input in inputs)
        {
            byte[] bytes = File.ReadAllBytes(input.Path);
            byte[] normalized = bytes.Where((value, index) => value != 13 || index + 1 == bytes.Length || bytes[index + 1] != 10).ToArray();
            manifest.Append(input.Relative).Append('=').Append(Convert.ToHexStringLower(SHA256.HashData(normalized))).Append('\n');
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            schema = "tensorsharp-ggml-required-exports/1",
            nativeAbi = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(manifest.ToString()))),
            ggmlCommit = File.ReadAllText(Path.Combine(root, "eng", "ggml-revision")).Trim(),
            exports = entryPoints.Order(StringComparer.Ordinal).ToArray(),
        }, new JsonSerializerOptions { WriteIndented = true }));
    }
    else if (args[0] == "--verify")
        Console.WriteLine($"Inspected {imports} native imports; {failures.Count} guard failures.");
    return failures.Count == 0 ? 0 : 1;
}
return failures.Count == 0 ? 0 : 3;

static bool IsImport(MethodDeclarationSyntax method) => method.AttributeLists.SelectMany(l => l.Attributes).Any(IsImportAttribute);

static bool IsImportAttribute(AttributeSyntax a) =>
    a.Name.ToString() is "LibraryImport" or "DllImport" && a.ArgumentList?.Arguments.FirstOrDefault()?.Expression.ToString() is "DllName" or "\"GgmlOps\"";

static string? BackendParameter(MethodDeclarationSyntax method) => method.ParameterList.Parameters.FirstOrDefault(p =>
    p.Type?.ToString() == "string" && p.Identifier.ValueText is "backend" or "backendName")?.Identifier.Text;

static string? AcquireFamily(string name) => name switch
{
    "TSGgml_Dsv4LoadModel" or "TSGgml_Dsv4LoadModelDspark" => "deepseek-model",
    "TSGgml_GlmLoadModel" => "glm-model",
    "TSGgml_Dsv41VisionLoad" => "deepseek-vision",
    "TSGgml_EmbeddingLoad" => "embedding-model",
    "TSGgml_PagedKvPoolCreate" => "paged-kv-pool",
    "TSGgml_Qwen4ExpMtpCreate" => "qwen4-mtp",
    "TSGgml_Qwen4ExpStateSnapshotCreate" => "qwen4-snapshot",
    "TSGgml_AlignedAlloc" => "aligned-allocation",
    _ => null,
};

static string? ReleaseFamily(string name) => name switch
{
    "TSGgml_Dsv4Free" => "deepseek-model",
    "TSGgml_GlmFree" => "glm-model",
    "TSGgml_Dsv41VisionFree" => "deepseek-vision",
    "TSGgml_EmbeddingFree" => "embedding-model",
    "TSGgml_PagedKvPoolFree" => "paged-kv-pool",
    "TSGgml_Qwen4ExpMtpFree" => "qwen4-mtp",
    "TSGgml_Qwen4ExpStateSnapshotFree" => "qwen4-snapshot",
    "TSGgml_AlignedFree" => "aligned-allocation",
    _ => null,
};

static string? UseFamily(MethodDeclarationSyntax method)
{
    string name = method.Identifier.ValueText;
    if (ReleaseFamily(name) != null || AcquireFamily(name) != null || method.ParameterList.Parameters.FirstOrDefault() is not { } first ||
        first.Type?.ToString() != "IntPtr" || first.Identifier.ValueText != "handle")
        return null;
    foreach (var (prefix, family) in new[] { ("TSGgml_Dsv41Vision", "deepseek-vision"), ("TSGgml_Dsv4", "deepseek-model"), ("TSGgml_Glm", "glm-model"),
        ("TSGgml_Embedding", "embedding-model"),
        ("TSGgml_PagedKvPool", "paged-kv-pool"), ("TSGgml_Qwen4ExpMtp", "qwen4-mtp"),
        ("TSGgml_Qwen4ExpStateSnapshot", "qwen4-snapshot") })
        if (name.StartsWith(prefix, StringComparison.Ordinal)) return family;
    throw new InvalidOperationException("Unclassified native handle use: " + name);
}
