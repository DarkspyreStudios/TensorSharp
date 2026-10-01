#nullable enable

using System.Text;
using TensorSharp.Runtime;

internal static class DeepSeek41LifetimeFixture
{
    internal const int Vocabulary = 129265;
    internal const int ImageToken = 129264;

    internal static ulong WriteText(string path)
    {
        var file = new FixtureFile("deepseek41");
        string[] tokens = Enumerable.Range(0, Vocabulary).Select(index => "t" + index).ToArray();
        tokens[ImageToken] = ChatTemplate.DeepSeek41ImagePlaceholder;
        ulong fingerprint = 14695981039346656037;
        foreach (string token in tokens)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(token);
            for (int index = 0; index < 8; index++) fingerprint = unchecked((fingerprint ^ (byte)((ulong)bytes.Length >> (8 * index))) * 1099511628211);
            foreach (byte value in bytes) fingerprint = unchecked((fingerprint ^ value) * 1099511628211);
        }
        file.String("tokenizer.ggml.model", "gpt2");
        file.String("tokenizer.ggml.pre", "joyai-llm");
        file.Strings("tokenizer.ggml.tokens", tokens);
        file.Strings("tokenizer.ggml.merges", []);
        file.Ints("tokenizer.ggml.token_type", Enumerable.Repeat(1, Vocabulary).ToArray());
        file.UInt("tokenizer.ggml.bos_token_id", 0);
        file.UInt("tokenizer.ggml.eos_token_id", 1);
        file.Bool("tokenizer.ggml.add_bos_token", false);
        file.Bool("tokenizer.ggml.add_eos_token", false);
        foreach (var (key, value) in new (string, uint)[]
        {
            ("block_count", 1), ("context_length", 32), ("embedding_length", 32),
            ("attention.head_count", 1), ("attention.head_count_kv", 1),
            ("attention.key_length", 64), ("attention.value_length", 64), ("rope.dimension_count", 32),
            ("attention.q_lora_rank", 32), ("attention.output_group_count", 1), ("attention.output_lora_rank", 32),
            ("attention.sliding_window", 8), ("expert_count", 1), ("expert_used_count", 1),
            ("expert_shared_count", 1), ("expert_feed_forward_length", 32), ("expert_gating_func", 4),
            ("hash_layer_count", 0), ("attention.indexer.head_count", 1), ("attention.indexer.key_length", 64),
            ("attention.indexer.top_k", 1), ("hyper_connection.count", 4),
            ("hyper_connection.sinkhorn_iterations", 20), ("rope.scaling.original_context_length", 32),
            ("engram.head_count", 1), ("engram.key_length", 32), ("engram.max_ngram_size", 2), ("engram.pad_id", 0)
        }) file.UInt("deepseek41." + key, value);
        foreach (var (key, value) in new (string, float)[]
        {
            ("attention.layer_norm_rms_epsilon", 1e-5f), ("expert_weights_scale", 1f),
            ("hyper_connection.epsilon", 1e-6f), ("rope.freq_base", 10000f),
            ("attention.compress_rope_freq_base", 20000f), ("rope.scaling.factor", 1f),
            ("rope.scaling.yarn_beta_fast", 32f), ("rope.scaling.yarn_beta_slow", 1f)
        }) file.Float("deepseek41." + key, value);
        file.String("deepseek41.rope.scaling.type", "yarn");
        file.Bool("deepseek41.expert_weights_norm", true);
        file.Ints("deepseek41.attention.compress_ratios", [2]);
        file.Int("deepseek41.attention.indexer.candidate_source_layer", -1);
        file.Ints("deepseek41.engram.layer_ids", [0]);
        file.Ints("deepseek41.engram.token_map", new int[Vocabulary]);
        file.Ulongs("deepseek41.engram.multipliers", [1, 3]);
        file.Ints("deepseek41.engram.primes", [2]);
        file.Ints("deepseek41.engram.offsets", [0]);
        file.Tensor("token_embd.weight", [32, Vocabulary]);
        file.Tensor("output_norm.weight", [32], 1);
        file.Tensor("output.weight", [32, Vocabulary]);
        foreach (string phase in new[] { "attn", "ffn" })
        {
            file.Tensor("blk.0." + phase + "_norm.weight", [32], 1);
            file.Tensor("blk.0.hc_" + phase + "_fn.weight", [128, 24]);
            file.Tensor("blk.0.hc_" + phase + "_base.weight", [24]);
            file.Tensor("blk.0.hc_" + phase + "_scale.weight", [3], .2f);
        }
        foreach (var (name, shape) in new (string, int[])[]
        {
            ("attn_q_a.weight", [32, 32]), ("attn_q_a_norm.weight", [32]),
            ("attn_q_b.weight", [32, 64]), ("attn_kv.weight", [32, 64]),
            ("attn_kv_a_norm.weight", [64]), ("attn_output_a.weight", [64, 32]),
            ("attn_output_b.weight", [32, 32]), ("attn_sinks.weight", [1]),
            ("attn_compressor_kv.weight", [32, 64]), ("attn_compressor_gate.weight", [32, 64]),
            ("attn_compressor_norm.weight", [64]), ("indexer.attn_k.weight", [64, 64]),
            ("indexer.k_norm.weight", [64]), ("indexer.attn_q_b.weight", [32, 64]), ("indexer.proj.weight", [32, 1]),
            ("ffn_gate_inp.weight", [32, 1]), ("exp_probs_b.bias", [1]),
            ("engram_embd.weight", [32, 2]), ("engram_k.weight", [32, 4]), ("engram_q.weight", [32, 4]),
            ("engram_wkv.weight", [32, 160])
        }) file.Tensor("blk.0." + name, shape, name.Contains("norm", StringComparison.Ordinal) ? 1 : .01f);
        foreach (string projection in new[] { "gate", "up", "down" })
        {
            file.Tensor("blk.0.ffn_" + projection + "_exps.weight", [32, 32, 1]);
            file.Tensor("blk.0.ffn_" + projection + "_shexp.weight", [32, 32]);
        }
        file.Write(path);
        if (new FileInfo(path).Length >= 40 * 1024 * 1024) throw new InvalidDataException("The text lifetime fixture exceeds its approved bound.");
        return fingerprint;
    }

    internal static void WriteVision(string path, ulong fingerprint, int textDimension)
    {
        var file = new FixtureFile("deepseek41_vision");
        foreach (var (key, value) in new (string, uint)[]
        {
            ("num_hidden_layers", 1), ("hidden_size", 4), ("num_attention_heads", 1),
            ("intermediate_size", 4), ("patch_size", 1), ("downsample_ratio", 1),
            ("max_image_tokens", 4), ("min_pixels", 1), ("max_wh_ratio", 0)
        }) file.UInt("deepseek41.vision." + key, value);
        file.Float("deepseek41.vision.rope_theta", 10000);
        file.UInt("deepseek41.hidden_size", (uint)textDimension);
        file.UInt("deepseek41.num_hidden_layers", 1);
        file.UInt("deepseek41.image_token_id", ImageToken);
        file.Ulong("deepseek41.tokenizer_hash", fingerprint);
        foreach (var (name, shape) in new (string, int[])[]
        {
            ("vision.patch_embed.proj.weight", [3, 4]), ("vision.patch_embed.proj.bias", [4]),
            ("vision.blocks.0.norm1.weight", [4]), ("vision.blocks.0.norm2.weight", [4]),
            ("vision.blocks.0.attn.wqkv.weight", [4, 12]), ("vision.blocks.0.attn.wqkv.bias", [12]),
            ("vision.blocks.0.attn.wo.weight", [4, 4]), ("vision.blocks.0.attn.wo.bias", [4]),
            ("vision.blocks.0.mlp.w1.weight", [4, 8]), ("vision.blocks.0.mlp.w2.weight", [4, 4]),
            ("vision.norm.weight", [4]), ("aligner.w1.weight", [4, textDimension]), ("aligner.w1.bias", [textDimension]),
            ("aligner.w2.weight", [textDimension, textDimension]), ("aligner.w2.bias", [textDimension]),
            ("image_start", [textDimension]), ("image_end", [textDimension]), ("image_newline", [textDimension]),
            ("layers.0.ffn.gate.bias_vl", [1])
        }) file.Tensor(name, shape, name.Contains("norm", StringComparison.Ordinal) ? 1 : .01f);
        file.Write(path);
        if (new FileInfo(path).Length >= 40 * 1024) throw new InvalidDataException("The vision lifetime fixture exceeds its approved bound.");
    }

    private sealed class FixtureFile
    {
        private readonly List<(string Key, Action<BinaryWriter> Write)> _metadata = [];
        private readonly List<(string Name, int[] Shape, float Value)> _tensors = [];
        internal FixtureFile(string architecture) => String("general.architecture", architecture);
        internal void String(string key, string value) => _metadata.Add((key, writer => { writer.Write(8u); WriteString(writer, value); }));
        internal void UInt(string key, uint value) => _metadata.Add((key, writer => { writer.Write(4u); writer.Write(value); }));
        internal void Int(string key, int value) => _metadata.Add((key, writer => { writer.Write(5u); writer.Write(value); }));
        internal void Float(string key, float value) => _metadata.Add((key, writer => { writer.Write(6u); writer.Write(value); }));
        internal void Bool(string key, bool value) => _metadata.Add((key, writer => { writer.Write(7u); writer.Write(value); }));
        internal void Ulong(string key, ulong value) => _metadata.Add((key, writer => { writer.Write(10u); writer.Write(value); }));
        internal void Strings(string key, string[] values) => Array(key, 8, values.Length, writer => { foreach (string value in values) WriteString(writer, value); });
        internal void Ints(string key, int[] values) => Array(key, 5, values.Length, writer => { foreach (int value in values) writer.Write(value); });
        internal void Ulongs(string key, ulong[] values) => Array(key, 10, values.Length, writer => { foreach (ulong value in values) writer.Write(value); });
        private void Array(string key, uint type, int count, Action<BinaryWriter> write) => _metadata.Add((key, writer => { writer.Write(9u); writer.Write(type); writer.Write((ulong)count); write(writer); }));
        internal void Tensor(string name, int[] shape, float value = .01f) => _tensors.Add((name, shape, value));
        internal void Write(string path)
        {
            using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
            writer.Write(0x46554747u);
            writer.Write(3u);
            writer.Write((ulong)_tensors.Count);
            writer.Write((ulong)_metadata.Count);
            foreach (var (key, write) in _metadata) { WriteString(writer, key); write(writer); }
            ulong offset = 0;
            foreach (var (name, shape, _) in _tensors)
            {
                WriteString(writer, name);
                writer.Write((uint)shape.Length);
                foreach (int length in shape) writer.Write((ulong)length);
                writer.Write(0u);
                writer.Write(offset);
                offset = Align(offset + (ulong)shape.Aggregate(1L, (count, length) => count * length) * 4);
            }
            while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
            long dataStart = writer.BaseStream.Position;
            foreach (var (_, shape, value) in _tensors)
            {
                long count = shape.Aggregate(1L, (total, length) => total * length);
                for (long index = 0; index < count; index++) writer.Write(value);
                while ((writer.BaseStream.Position - dataStart) % 32 != 0) writer.Write((byte)0);
            }
        }
        private static ulong Align(ulong value) => (value + 31) & ~31ul;
        private static void WriteString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }
    }
}
