// Probe generic JSON grammar with actual checkpoint vocabulary; no tensor reads.
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

if (args.Length != 2) throw new ArgumentException("Usage: dsv41-json-unicode header.json report.json");
byte[] metadataBytes = File.ReadAllBytes(args[0]);
using var header = JsonDocument.Parse(metadataBytes);
var metadata = header.RootElement.GetProperty("metadata");
T Read<T>(string key) => metadata.GetProperty(key).Deserialize<T>()!;
var tokenizer = new BpeTokenizer(Read<string[]>("tokenizer.ggml.tokens"), Read<int[]>("tokenizer.ggml.token_type"),
    Read<string[]>("tokenizer.ggml.merges"), Read<int>("tokenizer.ggml.bos_token_id"),
    new[] { Read<int>("tokenizer.ggml.eos_token_id") }, false, false, Read<string>("tokenizer.ggml.pre"));
byte[] Bytes(int id)
{
    var bytes = new List<byte>(); tokenizer.AppendTokenBytes(id, bytes); return bytes.ToArray();
}
var rawByteTokens = new Dictionary<byte, int>();
var special = ((ISpecialTokenVocabulary)tokenizer).SpecialTokenIds.ToHashSet();
for (int id = 0; id < tokenizer.VocabSize && rawByteTokens.Count < 256; ++id)
{
    if (special.Contains(id)) continue;
    byte[] bytes = Bytes(id);
    if (bytes.Length == 1) rawByteTokens.TryAdd(bytes[0], id);
}
int[] SplitRocket(string prefix) => tokenizer.Encode(prefix, false)
    .Concat(new byte[] { 0xF0, 0x9F, 0x9A, 0x80 }.Select(value => rawByteTokens[value]))
    .Concat(tokenizer.Encode("\"}", false)).ToArray();
const string minimal = "{\"emoji\":\"🚀\"}";
const string full = "{\"city\":\"北京\",\"message\":\"你好，世界\",\"emoji\":\"🚀\"}";
var cases = new List<(string Name, string Text, int[] Ids)>
{
    ("literal-rocket", minimal, tokenizer.Encode(minimal, false).ToArray()),
    ("literal-chinese-object", full, tokenizer.Encode(full, false).ToArray()),
    ("rocket-four-byte-tokens", minimal, SplitRocket("{\"emoji\":\"")),
    ("chinese-object-four-byte-rocket", full, SplitRocket("{\"city\":\"北京\",\"message\":\"你好，世界\",\"emoji\":\"")),
};
foreach (string escaped in new[] { "\\ud83d\\ude80", "\\uD83D\\uDE80" })
{
    string text = "{\"emoji\":\"" + escaped + "\"}";
    cases.Add((escaped.Contains('D') ? "ascii-upper-surrogates" : "ascii-lower-surrogates", text, tokenizer.Encode(text, false).ToArray()));
}
string mixed = "{\"city\":\"北京\",\"message\":\"你好，世界\",\"emoji\":\"\\ud83d\\ude80\"}";
cases.Add(("chinese-object-surrogate-escape", mixed, tokenizer.Encode(mixed, false).ToArray()));
var reports = new List<object>();
int failures = 0;
foreach (var (name, text, ids) in cases)
{
    var constraint = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
    var steps = new List<object>();
    bool allowed = true;
    foreach (int id in ids)
    {
        ulong[] mask = constraint.CurrentMask();
        bool bit = (mask[id >> 6] & (1UL << (id & 63))) != 0;
        var logits = new float[tokenizer.VocabSize];
        logits[id] = 100;
        constraint.ApplyMask(logits, allowEos: false);
        bool finite = float.IsFinite(logits[id]);
        constraint.Accept(id);
        allowed &= bit && finite && !constraint.IsDead;
        steps.Add(new { token_id = id, vocabulary_piece = tokenizer.Vocab[id],
            utf8_hex = Convert.ToHexString(Bytes(id)), mask_allows = bit, logit_is_finite = finite, dead_after_accept = constraint.IsDead });
    }
    byte[] output = ids.SelectMany(Bytes).ToArray();
    string reconstructed = new UTF8Encoding(false, true).GetString(output);
    using var parsed = JsonDocument.Parse(reconstructed);
    bool contentCorrect = parsed.RootElement.GetProperty("emoji").GetString() == "🚀";
    if (name.StartsWith("chinese", StringComparison.Ordinal) || name == "literal-chinese-object")
        contentCorrect &= parsed.RootElement.GetProperty("city").GetString() == "北京" &&
                          parsed.RootElement.GetProperty("message").GetString() == "你好，世界";
    bool pass = allowed && constraint.IsComplete && reconstructed == text && contentCorrect;
    if (!pass) ++failures;
    reports.Add(new { name, pass, all_tokens_allowed = allowed, complete = constraint.IsComplete,
        text, reconstructed, parsed_emoji = parsed.RootElement.GetProperty("emoji").GetString(), steps });
}
var negative = new List<object>();
foreach (var (name, prefix, rejectedByte) in new[]
{
    ("raw-newline-in-string", Array.Empty<byte>(), (byte)0x0A),
    ("isolated-continuation", Array.Empty<byte>(), (byte)0x80),
    ("overlong-after-F0", new byte[] { 0xF0 }, (byte)0x80),
    ("ascii-before-rocket-completes", new byte[] { 0xF0, 0x9F, 0x9A }, (byte)0x22),
})
{
    var constraint = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
    foreach (int id in tokenizer.Encode("{\"emoji\":\"", false)) constraint.Accept(id);
    foreach (byte value in prefix) constraint.Accept(rawByteTokens[value]);
    int rejectedId = rawByteTokens[rejectedByte];
    ulong[] mask = constraint.CurrentMask();
    bool rejected = (mask[rejectedId >> 6] & (1UL << (rejectedId & 63))) == 0;
    if (!rejected) ++failures;
    negative.Add(new { name, pass = rejected, prefix_hex = Convert.ToHexString(prefix), rejected_token_id = rejectedId });
}
var malformed = new List<object>();
foreach (byte[] raw in new[]
{
    new byte[] { 0xC0, 0xAF }, new byte[] { 0xE0, 0x80, 0xAF },
    new byte[] { 0xF0, 0x80, 0x80, 0xAF }, new byte[] { 0xED, 0xA0, 0x80 },
    new byte[] { 0xF4, 0x90, 0x80, 0x80 },
})
{
    int[] ids = tokenizer.Encode("{\"emoji\":\"", false).Concat(raw.Select(value => rawByteTokens[value]))
        .Concat(tokenizer.Encode("\"}", false)).ToArray();
    var constraint = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
    bool allowed = true;
    var steps = new List<object>();
    foreach (int id in ids)
    {
        ulong[] mask = constraint.CurrentMask();
        bool bit = (mask[id >> 6] & (1UL << (id & 63))) != 0;
        allowed &= bit;
        constraint.Accept(id);
        steps.Add(new { token_id = id, utf8_hex = Convert.ToHexString(Bytes(id)), mask_allows = bit,
            dead_after_accept = constraint.IsDead });
    }
    byte[] bytes = ids.SelectMany(Bytes).ToArray();
    bool strictUtf8 = true;
    try { _ = new UTF8Encoding(false, true).GetString(bytes); }
    catch (DecoderFallbackException) { strictUtf8 = false; }
    bool pass = !allowed && !constraint.IsComplete && !strictUtf8;
    if (!pass) ++failures;
    malformed.Add(new { utf8_hex = Convert.ToHexString(raw), pass, all_tokens_allowed = allowed,
        complete = constraint.IsComplete, strict_utf8_valid = strictUtf8,
        replacement_decoded = Encoding.UTF8.GetString(bytes), steps });
}
var report = new
{
    scope = "Actual Q2_K tokenizer plus generic JSON grammar only; no model inference and no DSML grammar.",
    metadata_path = Path.GetFullPath(args[0]), metadata_sha256 = Convert.ToHexString(SHA256.HashData(metadataBytes)).ToLowerInvariant(),
    runtime_path = typeof(TokenSampler).Assembly.Location,
    runtime_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(TokenSampler).Assembly.Location))).ToLowerInvariant(),
    source_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Program.cs")))).ToLowerInvariant(),
    vocab_size = tokenizer.VocabSize, pretokenizer = Read<string>("tokenizer.ggml.pre"),
    rocket_utf8_byte_token_ids = new byte[] { 0xF0, 0x9F, 0x9A, 0x80 }.Select(value => rawByteTokens[value]).ToArray(),
    failures, reports, negative, malformed
};
File.WriteAllText(args[1], JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Generic JSON actual-tokenizer Unicode probe: {cases.Count + negative.Count + malformed.Count - failures}/{cases.Count + negative.Count + malformed.Count} passed.");
return failures == 0 ? 0 : 1;
