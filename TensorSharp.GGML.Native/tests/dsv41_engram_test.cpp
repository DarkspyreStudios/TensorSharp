// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// c++ -std=c++17 -O2 dsv41_engram_test.cpp -o dsv41_engram_test
#include "../dsv41_engram.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

using tsg_dsv41::engram_data;

template<typename Function> static void expect_error(Function function) {
    bool caught = false;
    try { function(); } catch (const std::runtime_error &) { caught = true; }
    assert(caught);
}

static void check_chunk_boundaries(const engram_data & data, const std::vector<int32_t> & tokens) {
    std::vector<int32_t> history;
    const auto full = data.hash_tokens(tokens.data(), tokens.size(), 0, history);
    for (size_t split = 0; split <= tokens.size(); ++split) {
        history.clear();
        const auto first = data.hash_tokens(tokens.data(), split, 0, history);
        const auto second = data.hash_tokens(tokens.data() + split, tokens.size() - split, split, history);
        const size_t columns = data.hash_columns();
        for (size_t layer = 0; layer < data.layers.size(); ++layer) {
            assert(std::equal(first.begin() + layer * split * columns,
                              first.begin() + (layer + 1) * split * columns,
                              full.begin() + layer * tokens.size() * columns));
            assert(std::equal(second.begin() + layer * (tokens.size() - split) * columns,
                              second.begin() + (layer + 1) * (tokens.size() - split) * columns,
                              full.begin() + (layer * tokens.size() + split) * columns));
        }
    }
    history.clear();
    data.hash_tokens(tokens.data(), tokens.size(), 0, history);
    const auto replay = data.hash_tokens(tokens.data() + 1, tokens.size() - 1, 1, history);
    for (size_t layer = 0; layer < data.layers.size(); ++layer)
        assert(std::equal(replay.begin() + layer * (tokens.size() - 1) * data.hash_columns(),
                          replay.begin() + (layer + 1) * (tokens.size() - 1) * data.hash_columns(),
                          full.begin() + (layer * tokens.size() + 1) * data.hash_columns()));
}

static void synthetic_test() {
    engram_data data;
    data.vocab_size = 6;
    data.compressed_vocab_size = 4;
    data.pad_token_id = 1;
    data.max_ngram_size = 3;
    data.n_heads = 2;
    data.head_dim = 2;
    data.token_map = {0, 1, 2, 2, 3, 1};
    data.layers.push_back({1, 26, {7, 11, 13}, {3, 5, 7, 11}, {0, 3, 8, 15}});
    const std::vector<int32_t> tokens = {2, 3, 4, -1, 5, 0};
    std::vector<int32_t> history;
    const auto hashes = data.hash_tokens(tokens.data(), tokens.size(), 0, history);
    // Direct n-gram XOR reference, independently expressed without state/cache.
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        for (unsigned ngram = 2; ngram <= 3; ++ngram) {
            uint64_t expected = 0;
            bool blocked = false;
            for (unsigned lookback = 0; lookback < ngram; ++lookback) {
                blocked = blocked || lookback > pos || tokens[pos - lookback] < 0;
                const int token = blocked ? data.pad_token_id : tokens[pos - lookback];
                expected ^= uint64_t(data.token_map[token]) * data.layers[0].multipliers[lookback];
            }
            for (unsigned head = 0; head < 2; ++head) {
                const unsigned column = (ngram - 2) * 2 + head;
                assert(hashes[pos * 4 + column] == int(expected % data.layers[0].primes[column] + data.layers[0].offsets[column]));
            }
        }
    }
    check_chunk_boundaries(data, tokens);
    std::vector<float> table(52), output(tokens.size() * 8);
    for (size_t i = 0; i < table.size(); ++i) table[i] = float(i) / 7;
    auto copy_row = [](const void * row, float * out, unsigned dim) { std::memcpy(out, row, dim * sizeof(float)); };
    data.lookup(0, table.data(), 26, 2 * sizeof(float), hashes.data(), tokens.size(), output.data(), copy_row);
    for (size_t i = 0; i < hashes.size(); ++i)
        for (size_t j = 0; j < 2; ++j) assert(output[2 * i + j] == table[2 * hashes[i] + j]);
    expect_error([&] { data.lookup(0, table.data(), 25, 8, hashes.data(), tokens.size(), output.data(), copy_row); });
    int32_t invalid = 26;
    expect_error([&] { data.lookup(0, table.data(), 26, 8, &invalid, 1, output.data(), copy_row); });
    int32_t invalid_token = 6;
    expect_error([&] { data.hash_tokens(&invalid_token, 1, 0, history); });
    expect_error([&] { data.hash_tokens(tokens.data(), 1, 100, history); });
    check_chunk_boundaries(data, {1, 1, 1, 1, 1});
    check_chunk_boundaries(data, {-1, -1, 2, 3});
}

static void canonical_test(const std::string & path) {
    const auto data = engram_data::load(path, 129280, UINT64_C(1610858572546052822));
    assert(data.compressed_vocab_size == 99092 && data.layers.size() == 2);
    assert(data.kv_source_layer_ids == std::vector<int32_t>({2, 8, 14, 20}));
    assert(data.index_source_layer_ids == std::vector<int32_t>({2, 8, 14, 20, 24, 28, 32, 36}));
    assert(data.candidate_source_layer_id == 20 && data.candidate_block_size == 8 && data.candidate_topk_blocks == 2048);
    const std::vector<int32_t> tokens = {0, 100, 101, 102, -1, 103, 104};
    // NumPy reference using the official tokenizer normalization and PCG64 seeds.
    const int32_t expected[2][7][4] = {
        {{5702652,121476532,131476717,380066193}, {9357813,120674457,142523656,371635201},
         {14967493,120087816,140350287,379577895}, {8085103,120276459,133334380,369222544},
         {4299726,112312540,129401291,370096053}, {5558068,126975926,140551575,383461561},
         {13583392,116008417,137862774,377607687}},
        {{14361964,120604547,132225184,372375971}, {15410290,112117398,140943284,383683101},
         {3011271,114792526,137397375,378010528}, {5098593,122052292,143543847,368813437},
         {6788701,120694389,131862336,381853422}, {10655595,119663028,134680611,369662410},
         {4014438,123436036,130889979,374480667}}
    };
    const int columns[4] = {0, 7, 8, 23};
    std::vector<int32_t> history;
    const auto actual = data.hash_tokens(tokens.data(), tokens.size(), 0, history);
    for (size_t layer = 0; layer < 2; ++layer)
        for (size_t token = 0; token < tokens.size(); ++token)
            for (size_t column = 0; column < 4; ++column)
                assert(actual[(layer * tokens.size() + token) * 24 + columns[column]] == expected[layer][token][column]);
    check_chunk_boundaries(data, tokens);
    expect_error([&] { engram_data::load(path, 129280, 0); });
    expect_error([&] { engram_data::load(path, 10, data.tokenizer_hash); });
}

int main(int argc, char ** argv) {
    synthetic_test();
    if (argc > 1) canonical_test(argv[1]);
    std::cout << "DeepSeek V4.1 Engram tests passed" << (argc > 1 ? " (including official tokenizer fixtures)" : "") << "\n";
}
