// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "dsv41_engram.h"
#include "gguf.h"

namespace tsg_dsv41 {

// Engram's compact lookup data is part of the GGUF metadata. TensorShape returns
// the GGML-order dimensions of a tensor across all shards, or an empty vector.
// No Engram table payload is read while validating this metadata.
template<typename TensorShape>
engram_data load_engram_gguf(gguf_context * file, TensorShape tensor_shape) {
    auto fail = [](const std::string & key) -> void {
        throw std::runtime_error("Missing or invalid DeepSeek V4.1 GGUF metadata: " + key);
    };
    auto integer = [&](const char * key, int64_t fallback = -2) -> int64_t {
        const int64_t id = gguf_find_key(file, key);
        if (id < 0) { if (fallback != -2) return fallback; fail(key); }
        switch (gguf_get_kv_type(file, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, id);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(file, id);
            case GGUF_TYPE_UINT64: {
                const uint64_t value = gguf_get_val_u64(file, id);
                if (value > uint64_t(INT64_MAX)) fail(key);
                return int64_t(value);
            }
            case GGUF_TYPE_INT64: return gguf_get_val_i64(file, id);
            default: fail(key); return 0;
        }
    };
    auto u32 = [&](const char * key, int64_t fallback = -2) -> uint32_t {
        const int64_t value = integer(key, fallback);
        if (value < 0 || uint64_t(value) > UINT32_MAX) fail(key);
        return uint32_t(value);
    };
    auto array = [&](const char * key, size_t max_count) -> std::vector<uint64_t> {
        const int64_t id = gguf_find_key(file, key);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY) fail(key);
        const size_t count = gguf_get_arr_n(file, id);
        if (!count || count > max_count) fail(key);
        const void * raw = gguf_get_arr_data(file, id);
        std::vector<uint64_t> values(count);
        for (size_t i = 0; i < count; ++i) {
            switch (gguf_get_arr_type(file, id)) {
                case GGUF_TYPE_UINT32: values[i] = static_cast<const uint32_t *>(raw)[i]; break;
                case GGUF_TYPE_INT32: {
                    const auto v = static_cast<const int32_t *>(raw)[i];
                    if (v < 0) fail(key);
                    values[i] = uint64_t(v); break;
                }
                case GGUF_TYPE_UINT64: values[i] = static_cast<const uint64_t *>(raw)[i]; break;
                case GGUF_TYPE_INT64: {
                    const auto v = static_cast<const int64_t *>(raw)[i];
                    if (v < 0) fail(key);
                    values[i] = uint64_t(v); break;
                }
                default: fail(key);
            }
        }
        return values;
    };
    engram_data result;
    const int64_t tokens = gguf_find_key(file, "tokenizer.ggml.tokens");
    if (tokens < 0 || gguf_get_kv_type(file, tokens) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(file, tokens) != GGUF_TYPE_STRING ||
        !gguf_get_arr_n(file, tokens) || gguf_get_arr_n(file, tokens) > 1048576)
        fail("tokenizer.ggml.tokens");
    result.vocab_size = uint32_t(gguf_get_arr_n(file, tokens));
    result.tokenizer_hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < result.vocab_size; ++i)
        result.tokenizer_hash = engram_data::fingerprint_token(result.tokenizer_hash, gguf_get_arr_str(file, tokens, i));
    const auto embedding = tensor_shape("token_embd.weight");
    if (embedding.size() < 2 || embedding[1] != result.vocab_size)
        fail("token embedding vocabulary does not match tokenizer");
    result.pad_id = u32("deepseek41.engram.pad_id");
    result.max_ngram_size = u32("deepseek41.engram.max_ngram_size");
    result.n_heads = u32("deepseek41.engram.head_count");
    result.head_dim = u32("deepseek41.engram.key_length");
    if (result.max_ngram_size < 2 || result.max_ngram_size > 16 ||
        !result.n_heads || result.n_heads > 128 || !result.head_dim || result.head_dim > 65536)
        fail("deepseek41.engram dimensions");
    const auto ids = array("deepseek41.engram.layer_ids", 128);
    const size_t columns = result.hash_columns();
    const auto multipliers = array("deepseek41.engram.multipliers", ids.size() * result.max_ngram_size);
    const auto primes = array("deepseek41.engram.primes", ids.size() * columns);
    const auto offsets = array("deepseek41.engram.offsets", ids.size() * columns);
    if (multipliers.size() != ids.size() * result.max_ngram_size ||
        primes.size() != ids.size() * columns || offsets.size() != ids.size() * columns)
        fail("deepseek41.engram flattened array lengths");
    const auto map = array("deepseek41.engram.token_map", result.vocab_size);
    for (uint64_t value : map) {
        if (value >= result.vocab_size) fail("deepseek41.engram.token_map");
        result.token_map.push_back(int32_t(value));
        result.compressed_vocab_size = std::max(result.compressed_vocab_size, uint32_t(value + 1));
    }
    const uint32_t n_layer = u32("deepseek41.block_count");
    if (!n_layer || n_layer > 128) fail("deepseek41.block_count");
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= n_layer) fail("deepseek41.engram.layer_ids");
        engram_data::layer_layout layer;
        layer.id = int32_t(ids[i]);
        const auto shape = tensor_shape("blk." + std::to_string(layer.id) + ".engram_embd.weight");
        if (shape.size() < 2 || shape[0] != result.head_dim || shape[1] <= 0 ||
            (shape.size() > 2 && std::any_of(shape.begin() + 2, shape.end(), [](int64_t d) { return d != 1; })))
            fail("Engram embedding tensor dimensions at layer " + std::to_string(layer.id));
        layer.rows = uint64_t(shape[1]);
        for (size_t j = 0; j < result.max_ngram_size; ++j)
            layer.multipliers.push_back(multipliers[i * result.max_ngram_size + j]);
        for (size_t j = 0; j < columns; ++j) {
            if (primes[i * columns + j] > UINT32_MAX) fail("deepseek41.engram.primes");
            layer.primes.push_back(uint32_t(primes[i * columns + j]));
            layer.offsets.push_back(offsets[i * columns + j]);
        }
        result.layers.push_back(std::move(layer));
    }
    const auto ratios = array("deepseek41.attention.compress_ratios", 256);
    if (ratios.size() < n_layer) fail("deepseek41.attention.compress_ratios");
    int32_t candidate = -1;
    for (uint32_t i = 0; i < n_layer; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        if (!tensor_shape(prefix + "engram_embd.weight").empty() &&
            std::find(ids.begin(), ids.end(), i) == ids.end())
            fail("Engram embedding tensor is absent from deepseek41.engram.layer_ids");
        if (!tensor_shape(prefix + "attn_compressor_kv.weight").empty())
            result.kv_source_layer_ids.push_back(int32_t(i));
        if (!tensor_shape(prefix + "indexer.attn_q_b.weight").empty()) {
            result.index_source_layer_ids.push_back(int32_t(i));
            if (candidate < 0 && ratios[i] == 1) candidate = int32_t(i);
        }
    }
    // Published V4.1 Flash omits these config values. Its first ratio-1
    // indexer owns the candidate mask; optional explicit metadata supports
    // checkpoints and small fixtures with a different pruning geometry.
    const int64_t source = integer("deepseek41.attention.indexer.candidate_source_layer", candidate);
    if (source < -1 || source >= n_layer) fail("deepseek41.attention.indexer.candidate_source_layer");
    result.candidate_source_layer_id = int32_t(source);
    if (source >= 0) {
        result.candidate_topk_blocks = u32("deepseek41.attention.indexer.candidate_top_k", 2048);
        result.candidate_block_size = u32("deepseek41.attention.indexer.candidate_block_size", 8);
        if (std::find(result.index_source_layer_ids.begin(), result.index_source_layer_ids.end(), source) ==
            result.index_source_layer_ids.end()) fail("candidate source must own an indexer");
    }
    result.validate(result.vocab_size);
    return result;
}

} // namespace tsg_dsv41
