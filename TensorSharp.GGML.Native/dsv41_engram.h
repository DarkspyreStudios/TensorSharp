// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tsg_dsv41 {

// Engram lookup metadata loaded directly from the checkpoint GGUF.
// The token map and hash parameters are model data, never generated at runtime.
struct engram_data {
    struct layer_layout {
        int32_t id = 0;
        uint64_t rows = 0;
        std::vector<uint64_t> multipliers;
        std::vector<uint32_t> primes;
        std::vector<uint64_t> offsets;
    };

    uint32_t vocab_size = 0, compressed_vocab_size = 0, pad_id = 0;
    uint32_t max_ngram_size = 0, n_heads = 0, head_dim = 0;
    uint64_t tokenizer_hash = 0;
    int32_t candidate_source_layer_id = -1;
    uint32_t candidate_topk_blocks = 0, candidate_block_size = 0;
    std::vector<int32_t> kv_source_layer_ids, index_source_layer_ids;
    std::vector<int32_t> token_map;
    std::vector<layer_layout> layers;

    uint32_t hash_columns() const { return (max_ngram_size - 1) * n_heads; }

    static uint64_t fingerprint_token(uint64_t hash, const std::string & token) {
        uint64_t size = token.size();
        for (int i = 0; i < 8; ++i) {
            hash = (hash ^ uint8_t(size >> (8 * i))) * UINT64_C(1099511628211);
        }
        for (unsigned char byte : token) hash = (hash ^ byte) * UINT64_C(1099511628211);
        return hash;
    }

    void validate(uint32_t expected_vocab) const {
        if (vocab_size != expected_vocab || !vocab_size || vocab_size > 1048576 ||
            !compressed_vocab_size || compressed_vocab_size > vocab_size || pad_id >= compressed_vocab_size ||
            layers.empty() || layers.size() > 128 || max_ngram_size < 2 || max_ngram_size > 16 ||
            !n_heads || n_heads > 128 || !head_dim || head_dim > 65536 ||
            kv_source_layer_ids.empty() || kv_source_layer_ids.size() > 128 ||
            index_source_layer_ids.empty() || index_source_layer_ids.size() > 128 ||
            candidate_source_layer_id < -1 ||
            (candidate_source_layer_id >= 0 && (!candidate_topk_blocks || !candidate_block_size ||
                candidate_topk_blocks > INT32_MAX || candidate_block_size > INT32_MAX)))
            throw std::runtime_error("DeepSeek V4.1 GGUF Engram metadata has invalid dimensions");
        auto check_sources = [](const std::vector<int32_t> & ids) {
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] < 0 || (i && ids[i] <= ids[i - 1]))
                    throw std::runtime_error("Invalid DeepSeek V4.1 shared-cache source layers");
        };
        check_sources(kv_source_layer_ids);
        check_sources(index_source_layer_ids);
        if (token_map.size() != vocab_size)
            throw std::runtime_error("DeepSeek V4.1 GGUF Engram token map does not match the tokenizer vocabulary");
        std::vector<bool> seen(compressed_vocab_size);
        for (int32_t value : token_map) {
            if (value < 0 || uint32_t(value) >= compressed_vocab_size)
                throw std::runtime_error("DeepSeek V4.1 compressed token id is out of bounds");
            seen[value] = true;
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::runtime_error("DeepSeek V4.1 compressed token map has missing ids");
        const uint32_t columns = hash_columns();
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & layer = layers[i];
            if (layer.id < 0 || (i && layer.id <= layers[i - 1].id) ||
                !layer.rows || layer.rows > uint64_t(std::numeric_limits<int32_t>::max()) ||
                layer.multipliers.size() != max_ngram_size ||
                layer.primes.size() != columns || layer.offsets.size() != columns)
                throw std::runtime_error("Invalid DeepSeek V4.1 Engram table dimensions");
            for (uint64_t multiplier : layer.multipliers)
                if (!(multiplier & 1) || multiplier > uint64_t(INT64_MAX) / compressed_vocab_size)
                    throw std::runtime_error("Invalid DeepSeek V4.1 Engram hash multiplier");
            uint64_t total = 0;
            for (uint32_t j = 0; j < columns; ++j) {
                if (layer.primes[j] < 2 || layer.offsets[j] != total)
                    throw std::runtime_error("Invalid DeepSeek V4.1 Engram bucket layout");
                total += layer.primes[j];
            }
            if (total != layer.rows)
                throw std::runtime_error("DeepSeek V4.1 Engram bucket sizes do not match table rows");
        }
    }

    // Returned layout is [layer][token][hash column]. History is per sequence.
    // A negative input token marks an image position and breaks all lookbacks.
    std::vector<int32_t> hash_tokens(const int32_t * tokens, size_t count, size_t start_pos,
                                    std::vector<int32_t> & history) const {
        if (start_pos > history.size() || count > SIZE_MAX - start_pos)
            throw std::runtime_error("DeepSeek V4.1 Engram history is not contiguous");
        for (size_t i = 0; i < count; ++i) {
            if (tokens[i] >= 0 && uint32_t(tokens[i]) >= vocab_size)
                throw std::runtime_error("DeepSeek V4.1 token is out of bounds");
        }
        history.resize(start_pos + count);
        for (size_t i = 0; i < count; ++i)
            history[start_pos + i] = tokens[i] < 0 ? -1 : token_map[tokens[i]];
        const uint32_t columns = hash_columns();
        std::vector<int32_t> hashes(layers.size() * count * columns);
        for (size_t li = 0; li < layers.size(); ++li) {
            const auto & layer = layers[li];
            for (size_t i = 0; i < count; ++i) {
                const size_t pos = start_pos + i;
                uint64_t rolling = 0;
                bool blocked = false;
                for (uint32_t shift = 0; shift < max_ngram_size; ++shift) {
                    blocked = blocked || pos < shift || history[pos - shift] < 0;
                    const int32_t token = blocked ? pad_id : history[pos - shift];
                    rolling ^= uint64_t(token) * layer.multipliers[shift];
                    if (!shift) continue;
                    for (uint32_t head = 0; head < n_heads; ++head) {
                        const uint32_t column = (shift - 1) * n_heads + head;
                        hashes[(li * count + i) * columns + column] = int32_t(
                            rolling % layer.primes[column] + layer.offsets[column]);
                    }
                }
            }
        }
        return hashes;
    }

    // `dequantize` converts exactly head_dim values from one quantized GGUF row.
    // No full-table allocation or device upload is needed for these sparse reads.
    template<typename Dequantize>
    void lookup(size_t layer_index, const void * table, uint64_t table_rows, size_t row_bytes,
                const int32_t * hashes, size_t count, float * output, Dequantize dequantize) const {
        lookup(layer_index, table, table_rows, row_bytes, hashes, count, output, dequantize,
            [](size_t rows, auto row) { for (size_t i = 0; i < rows; ++i) row(i); });
    }

    template<typename Dequantize, typename ParallelFor>
    void lookup(size_t layer_index, const void * table, uint64_t table_rows, size_t row_bytes,
                const int32_t * hashes, size_t count, float * output,
                Dequantize dequantize, ParallelFor parallel_for) const {
        auto row = prepare_lookup(layer_index, table, table_rows, row_bytes, hashes, count, output,
            std::move(dequantize));
        parallel_for(count * hash_columns(), std::move(row));
    }

    // Validate the entire lookup before scheduling any work. The returned
    // callable owns its scalar arguments and dequantizer, so callers may stage
    // several tables before submitting them together. The table, hashes, and
    // output storage must remain alive, and the validated hashes unchanged,
    // until every submitted row completes.
    template<typename Dequantize>
    auto prepare_lookup(size_t layer_index, const void * table, uint64_t table_rows, size_t row_bytes,
                        const int32_t * hashes, size_t count, float * output, Dequantize dequantize) const {
        if (layer_index >= layers.size() || table_rows != layers[layer_index].rows || !table || !row_bytes ||
            table_rows > SIZE_MAX / row_bytes || !hash_columns() || !head_dim ||
            (count && (!hashes || !output)) || count > SIZE_MAX / hash_columns() / head_dim / sizeof(float))
            throw std::runtime_error("DeepSeek V4.1 Engram lookup table does not match metadata");
        const auto * data = static_cast<const uint8_t *>(table);
        const size_t n_rows = count * hash_columns();
        for (size_t i = 0; i < n_rows; ++i) {
            if (hashes[i] < 0 || uint64_t(hashes[i]) >= table_rows)
                throw std::runtime_error("DeepSeek V4.1 Engram lookup is out of bounds");
        }
        // Each callback writes a disjoint row. Do not capture references to
        // this stack frame: deferred callbacks outlive prepare_lookup().
        return [data, row_bytes, hashes, output, dimension = head_dim,
                dequantize = std::move(dequantize)](size_t i) mutable {
            dequantize(data + size_t(hashes[i]) * row_bytes, output + i * dimension, dimension);
        };
    }
};

} // namespace tsg_dsv41
