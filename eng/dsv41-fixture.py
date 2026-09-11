#!/usr/bin/env python3
"""Write a small, deterministic V4.1 GGUF for native/reference trace comparison.

Exercises both compression ratios, shared caches, candidate filtering, delayed
hyper-connections, Engram, and mixed Q2_K/Q8_0/F32 weight storage. Not a language
model. Requires numpy, tokenizers, and gguf.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize, dequantize
from tokenizers import Tokenizer, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--f32", action="store_true", help="Store identical dequantized weights as F32 to isolate architecture from quantized matmul rounding")
    parser.add_argument("--cuda-index", action="store_true", help="Use 128-dimensional, 32-head indexing supported by the CUDA index kernel")
    parser.add_argument("--index-topk", type=int, default=2, help="Use 512 to exercise the real checkpoint's gather/bucket boundary")
    parser.add_argument("--token-count", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.index_topk <= 1024 or not 1 <= args.token_count <= 4096:
        raise ValueError("Fixture requires index-topk1..1024 and token-count1..4096")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4109)
    config = dict(model_type="deepseek_v41", text_config=dict(
        vocab_size=256, hidden_size=256, num_hidden_layers=5, num_attention_heads=4,
        num_key_value_heads=1, head_dim=64, qk_rope_head_dim=32, q_lora_rank=64,
        o_lora_rank=64, o_groups=2, rms_norm_eps=1e-20, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=256,
        scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling_factor=1.5,
        swiglu_limit=10.0, sliding_window=8, max_position_embeddings=4096,
        rope_theta=10000.0, compress_rope_theta=20000.0,
        rope_scaling=dict(rope_type="yarn", factor=2, beta_fast=32, beta_slow=1, original_max_position_embeddings=2048),
        compress_ratios=[0, 2, 2, 1, 1], kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4], index_n_heads=8, index_head_dim=64, index_topk=2,
        candidate_source_layer_id=3, candidate_topk_blocks=2, candidate_block_size=2,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=[1], engram_num_embeddings=[408], engram_max_ngram_size=3,
        engram_vocab_size=97, engram_n_heads=2, engram_head_dim=32, engram_pad_token_id=2,
        engram_compressed_vocab_size=256))
    c = config["text_config"]
    c["index_topk"] = args.index_topk
    c["candidate_topk_blocks"] = max(2, (args.index_topk + 1) // c["candidate_block_size"] + 1)
    if args.cuda_index:
        c["index_n_heads"], c["index_head_dim"] = 32, 128
    tokens = [f"t{i}" for i in range(c["vocab_size"])]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="t2"))
    module_spec = importlib.util.spec_from_file_location("dsv41_prepare", Path(__file__).with_name("dsv41-prepare.py"))
    prepare = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(prepare)
    sidecar, provenance, _ = prepare.prepare(config, tokenizer)
    (args.output_dir / "deepseek41.engram.bin").write_bytes(sidecar)
    (args.output_dir / "deepseek41.config.json").write_text(json.dumps(dict(config=config, fixture=True, **provenance), indent=2) + "\n")
    writer = GGUFWriter(args.output_dir / "deepseek41-fixture.gguf", "deepseek41")
    writer.add_name("DeepSeek V4.1 deterministic numerical fixture")
    uints = {
        "block_count": c["num_hidden_layers"], "context_length": c["max_position_embeddings"],
        "embedding_length": c["hidden_size"], "attention.head_count": c["num_attention_heads"],
        "attention.head_count_kv": 1, "attention.key_length": c["head_dim"], "attention.value_length": c["head_dim"],
        "rope.dimension_count": c["qk_rope_head_dim"], "attention.q_lora_rank": c["q_lora_rank"],
        "attention.sliding_window": c["sliding_window"], "expert_count": c["n_routed_experts"],
        "expert_used_count": c["num_experts_per_tok"], "expert_shared_count": 1,
        "expert_feed_forward_length": c["moe_intermediate_size"], "expert_gating_func": 4,
        "attention.indexer.head_count": c["index_n_heads"], "attention.indexer.key_length": c["index_head_dim"],
        "attention.indexer.top_k": c["index_topk"], "attention.output_group_count": c["o_groups"],
        "attention.output_lora_rank": c["o_lora_rank"], "hyper_connection.count": c["hc_mult"],
        "hyper_connection.sinkhorn_iterations": c["hc_sinkhorn_iters"], "hash_layer_count": 0,
        "rope.scaling.original_context_length": c["rope_scaling"]["original_max_position_embeddings"],
        "engram.head_count": c["engram_n_heads"], "engram.key_length": c["engram_head_dim"],
        "engram.max_ngram_size": c["engram_max_ngram_size"],
    }
    floats = {
        "attention.layer_norm_rms_epsilon": c["rms_norm_eps"], "expert_weights_scale": c["routed_scaling_factor"],
        "hyper_connection.epsilon": c["hc_eps"], "rope.freq_base": c["rope_theta"],
        "attention.compress_rope_freq_base": c["compress_rope_theta"], "rope.scaling.factor": c["rope_scaling"]["factor"],
        "rope.scaling.yarn_beta_fast": 32.0, "rope.scaling.yarn_beta_slow": 1.0,
    }
    for key, value in uints.items():
        writer.add_uint32("deepseek41." + key, value)
    for key, value in floats.items():
        writer.add_float32("deepseek41." + key, value)
    writer.add_string("deepseek41.rope.scaling.type", "yarn")
    writer.add_bool("deepseek41.expert_weights_norm", True)
    writer.add_array("deepseek41.attention.compress_ratios", c["compress_ratios"])
    writer.add_array("deepseek41.engram.layer_ids", c["engram_layer_ids"])
    writer.add_array("deepseek41.swiglu_clamp_exp", [10.0] * 5)
    writer.add_array("deepseek41.swiglu_clamp_shexp", [10.0] * 5)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("joyai-llm")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    def add(name, shape, mode="quant", scale=None):
        scale = (1 / shape[-1] ** .5) if scale is None else scale
        if mode == "norm":
            data = (1 + rng.normal(0, .02, shape)).astype(np.float32)
        elif mode == "scale":
            data = np.array([.2, .3, .4], dtype=np.float32)
        else:
            data = rng.normal(0, scale, shape).astype(np.float32)
        if mode == "quant" and shape[-1] % 256 == 0:
            # Q2_K's public Python converter only dequantizes. Construct valid
            # blocks directly: scales/mins=3, with centered two-bit payloads.
            blocks = int(np.prod(shape)) // 256
            encoded = np.empty((blocks, 84), dtype=np.uint8)
            encoded[:, :16] = 0x33
            encoded[:, 16:80] = rng.integers(0, 256, (blocks, 64), dtype=np.uint8)
            d = scale / (3 * 1.25 ** .5)
            encoded[:, 80:82] = np.frombuffer(np.float16(d).tobytes(), dtype=np.uint8)
            encoded[:, 82:84] = np.frombuffer(np.float16(1.5 * d).tobytes(), dtype=np.uint8)
            encoded = encoded.reshape(*shape[:-1], shape[-1] // 256 * 84)
            if args.f32:
                writer.add_tensor(name, dequantize(encoded, GGMLQuantizationType.Q2_K))
            else:
                writer.add_tensor(name, encoded, raw_dtype=GGMLQuantizationType.Q2_K)
        elif mode == "quant":
            encoded = quantize(data, GGMLQuantizationType.Q8_0)
            if args.f32:
                writer.add_tensor(name, dequantize(encoded, GGMLQuantizationType.Q8_0))
            else:
                writer.add_tensor(name, encoded, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data)

    dim, hc, qrank, head, heads, rank, groups = 256, 4, 64, 64, 4, 64, 2
    add("token_embd.weight", (256, dim), scale=.3)
    add("output_norm.weight", (dim,), "norm")
    add("output.weight", (256, dim))
    for layer in range(5):
        p = f"blk.{layer}."
        for sublayer in ("attn", "ffn"):
            add(p + sublayer + "_norm.weight", (dim,), "norm")
            add(p + f"hc_{sublayer}_fn.weight", ((2 + hc) * hc, hc * dim), "float")
            add(p + f"hc_{sublayer}_base.weight", ((2 + hc) * hc,), "float", .2)
            add(p + f"hc_{sublayer}_scale.weight", (3,), "scale")
        add(p + "attn_q_a.weight", (qrank, dim))
        add(p + "attn_q_a_norm.weight", (qrank,), "norm")
        add(p + "attn_q_b.weight", (head * heads, qrank))
        add(p + "attn_kv.weight", (head, dim))
        add(p + "attn_kv_a_norm.weight", (head,), "norm")
        add(p + "attn_output_a.weight", (groups, rank, head * heads // groups))
        add(p + "attn_output_b.weight", (dim, rank * groups))
        add(p + "attn_sinks.weight", (heads,), "float", .2)
        if layer in c["kv_source_layer_ids"]:
            add(p + "attn_compressor_kv.weight", (head, dim), "float")
            add(p + "attn_compressor_norm.weight", (head,), "norm")
            if c["compress_ratios"][layer] > 1:
                add(p + "attn_compressor_gate.weight", (head, dim), "float")
            add(p + "indexer.attn_k.weight", (c["index_head_dim"], head))
            add(p + "indexer.k_norm.weight", (c["index_head_dim"],), "norm")
        if layer in c["index_source_layer_ids"]:
            add(p + "indexer.attn_q_b.weight", (c["index_n_heads"] * c["index_head_dim"], qrank))
            add(p + "indexer.proj.weight", (c["index_n_heads"], dim))
        add(p + "ffn_gate_inp.weight", (4, dim), "float")
        add(p + "exp_probs_b.bias", (4,), "float", .1)
        for projection in ("gate", "up", "down"):
            add(p + f"ffn_{projection}_exps.weight", (4, dim, dim))
            add(p + f"ffn_{projection}_shexp.weight", (dim, dim))
        if layer == 1:
            add(p + "engram_embd.weight", (408, 32), scale=.3)
            add(p + "engram_k.weight", (hc, dim), "norm")
            add(p + "engram_q.weight", (hc, dim), "norm")
            add(p + "engram_wkv.weight", ((hc + 1) * dim, 128))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    seed_tokens = [0, 15, 32, 64, 128, 13, 254, 18, 22, 26, 16, 4, 11, 3, 18, 9]
    tokens = [seed_tokens[i % len(seed_tokens)] for i in range(args.token_count)]
    (args.output_dir / "tokens.json").write_text(json.dumps(tokens) + "\n")
    print(f"Wrote {args.output_dir / 'deepseek41-fixture.gguf'}")


if __name__ == "__main__":
    main()
