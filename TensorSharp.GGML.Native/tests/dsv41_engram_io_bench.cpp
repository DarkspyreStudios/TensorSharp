// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Portable warm-memory lookup control, plus a Linux-only filesystem benchmark
// that creates/removes its own scratch file. Never evicts live model pages.
#include "../dsv41_engram.h"
#include "../dsv41_engram_io.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

using row_read = std::function<void(size_t)>;
static std::vector<row_read> prepare_tables(const tsg_dsv41::engram_data & layout, const void * table,
        size_t table_rows, const int32_t * hashes, size_t tokens, float * output) {
    const size_t rows = tokens * layout.hash_columns(), head = layout.head_dim;
    const size_t row_bytes = head * sizeof(int16_t);
    std::vector<row_read> reads;
    for (size_t e = 0; e < layout.layers.size(); ++e)
        reads.emplace_back(layout.prepare_lookup(e, static_cast<const char *>(table) + e * table_rows * row_bytes,
            table_rows, row_bytes, hashes + e * rows, tokens, output + e * rows * head,
            [](const void * source, float * target, size_t count) {
                const auto * values = static_cast<const int16_t *>(source);
                for (size_t i = 0; i < count; ++i) target[i] = float(values[i]) / 127;
            }));
    return reads;
}

static void run_tables(tsg_dsv41::engram_io_pool & pool, const std::vector<row_read> & reads,
        size_t rows, bool joint) {
    if (joint) pool.run(rows * reads.size(), [&](size_t task) { reads[task % reads.size()](task / reads.size()); });
    else for (const auto & read : reads) pool.run(rows, read);
}

static int warm_joint() {
    using clock = std::chrono::steady_clock;
    constexpr size_t table_rows = 65536, head = 256, repetitions = 200, samples = 12;
    tsg_dsv41::engram_data layout;
    layout.max_ngram_size = 4; layout.n_heads = 8; layout.head_dim = head;
    layout.layers.push_back({1, table_rows, {}, {}, {}});
    layout.layers.push_back({14, table_rows, {}, {}, {}});
    std::vector<int16_t> table(2 * table_rows * head);
    std::mt19937 random(4101);
    for (auto & value : table) value = int16_t(int(random() % 65536) - 32768);
    tsg_dsv41::engram_io_pool pool(16);
    for (size_t tokens : {size_t(1), size_t(3), size_t(257)}) {
        const size_t rows = tokens * layout.hash_columns();
        std::vector<int32_t> hashes(2 * rows);
        for (auto & hash : hashes) hash = int32_t(random() % table_rows);
        std::vector<float> expected(hashes.size() * head), actual(expected.size());
        for (size_t e = 0; e < 2; ++e) for (size_t row = 0; row < rows; ++row)
            for (size_t column = 0; column < head; ++column)
                expected[(e * rows + row) * head + column] = float(table[(e * table_rows + hashes[e * rows + row]) * head + column]) / 127;
        const auto reads = prepare_tables(layout, table.data(), table_rows, hashes.data(), tokens, actual.data());
        for (size_t i = 0; i < 50; ++i) { run_tables(pool, reads, rows, false); run_tables(pool, reads, rows, true); }
        std::vector<double> durations[2];
        const size_t repeats = tokens > 3 ? 10 : repetitions;
        for (size_t sample = 0; sample < samples; ++sample) for (size_t order = 0; order < 2; ++order) {
            const size_t index = (sample + order) % 2;
            const auto start = clock::now();
            for (size_t i = 0; i < repeats; ++i) run_tables(pool, reads, rows, index != 0);
            durations[index].push_back(std::chrono::duration<double, std::micro>(clock::now() - start).count() / repeats);
            require(std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                    "Warmed joint lookup changed row bytes");
        }
        for (size_t index = 0; index < 2; ++index) {
            auto sorted = durations[index]; std::sort(sorted.begin(), sorted.end());
            std::printf("{\"mode\":\"warm_joint\",\"schedule\":\"%s\",\"tables\":2,\"tokens\":%zu,"
                        "\"rows_per_table\":%zu,\"head_dim\":%zu,\"threads\":%u,\"repetitions_per_sample\":%zu,"
                        "\"exact_output\":true,\"median_us\":%.6f,\"samples_us\":[",
                        index ? "joint" : "separate", tokens, rows, head, pool.threads(), repeats,
                        (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]) / 2);
            for (size_t i = 0; i < durations[index].size(); ++i) std::printf("%s%.6f", i ? "," : "", durations[index][i]);
            std::printf("]}\n");
        }
    }
    return 0;
}

static int warm_memory() {
    using clock = std::chrono::steady_clock;
    constexpr size_t table_rows = 65536, head = 256, repetitions = 200, samples = 12;
    tsg_dsv41::engram_data layout;
    layout.max_ngram_size = 4; layout.n_heads = 8; layout.head_dim = head;
    layout.layers.push_back({1, table_rows, {}, {}, {}});
    std::vector<int16_t> table(table_rows * head);
    std::mt19937 random(4101);
    for (auto & value : table) value = int16_t(int(random() % 65536) - 32768);
    auto dequantize = [](const void * source, float * output, size_t count) {
        const auto * values = static_cast<const int16_t *>(source);
        for (size_t i = 0; i < count; ++i) output[i] = float(values[i]) / 127;
    };
    tsg_dsv41::engram_io_pool serial(1), parallel(16);
    for (size_t tokens : {size_t(1), size_t(3)}) {
        std::vector<int32_t> hashes(tokens * layout.hash_columns());
        for (auto & hash : hashes) hash = int32_t(random() % table_rows);
        std::vector<float> expected(hashes.size() * head), actual(expected.size());
        layout.lookup(0, table.data(), table_rows, head * sizeof(int16_t), hashes.data(), tokens,
            expected.data(), dequantize);
        auto lookup = [&](tsg_dsv41::engram_io_pool & pool) {
            layout.lookup(0, table.data(), table_rows, head * sizeof(int16_t), hashes.data(), tokens,
                actual.data(), dequantize, [&](size_t count, auto row) { pool.run(count, row); });
        };
        // Fault the selected pages and start the persistent workers before
        // timing. Alternate serial/parallel block order; report every sample.
        for (size_t i = 0; i < 50; ++i) { lookup(serial); lookup(parallel); }
        std::vector<double> durations[2];
        for (size_t sample = 0; sample < samples; ++sample) {
            for (size_t pass = 0; pass < 2; ++pass) {
                const size_t index = (sample + pass) % 2;
                auto & pool = index ? parallel : serial;
                const auto start = clock::now();
                for (size_t i = 0; i < repetitions; ++i) lookup(pool);
                durations[index].push_back(std::chrono::duration<double, std::micro>(clock::now() - start).count() / repetitions);
                require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                        "Warmed small lookup changed row bytes");
            }
        }
        for (size_t index = 0; index < 2; ++index) {
            auto sorted = durations[index]; std::sort(sorted.begin(), sorted.end());
            std::printf("{\"mode\":\"warm_memory\",\"tokens\":%zu,\"selected_rows\":%zu,\"head_dim\":%zu,"
                        "\"threads\":%u,\"repetitions_per_sample\":%zu,\"exact_output\":true,\"median_us\":%.6f,\"samples_us\":[",
                        tokens, hashes.size(), head, index ? parallel.threads() : serial.threads(), repetitions,
                        (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2);
            for (size_t i = 0; i < durations[index].size(); ++i)
                std::printf("%s%.6f", i ? "," : "", durations[index][i]);
            std::printf("]}\n");
        }
    }
    return 0;
}

int main(int argc, char ** argv) try {
    const bool joint = argc == 3 && std::strcmp(argv[1], "--joint") == 0;
    if (argc != 2 && !joint) { std::fprintf(stderr, "usage: %s SCRATCH_DIRECTORY | --joint SCRATCH_DIRECTORY | --warm-memory | --warm-joint\n", argv[0]); return 2; }
    if (std::strcmp(argv[1], "--warm-memory") == 0) return warm_memory();
    if (std::strcmp(argv[1], "--warm-joint") == 0) return warm_joint();
#if defined(__linux__)
    constexpr size_t bytes = 64 * 1024 * 1024;
    const long system_page = sysconf(_SC_PAGESIZE);
    require(system_page > 0 && bytes % size_t(system_page) == 0, "Unsupported benchmark page size");
    const size_t page = size_t(system_page);
    std::string pattern = std::string(argv[joint ? 2 : 1]) + "/dsv41-engram-io-XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end()); path.push_back(0);
    const int fd = mkstemp(path.data());
    require(fd >= 0, "Cannot create benchmark scratch file");
    struct cleanup {
        int fd; const char * path;
        ~cleanup() { close(fd); unlink(path); }
    } cleanup{fd, path.data()};
    std::mt19937 random(4101);
    std::vector<int16_t> source(bytes / sizeof(int16_t));
    for (auto & value : source) value = int16_t(int(random() % 65536) - 32768);
    for (size_t offset = 0; offset < bytes;) {
        const ssize_t wrote = write(fd, reinterpret_cast<char *>(source.data()) + offset, bytes - offset);
        require(wrote > 0, "Cannot write benchmark file"); offset += size_t(wrote);
    }
    require(fsync(fd) == 0, "Cannot flush benchmark file");
    auto dequantize = [](const void * source, float * output, size_t count) {
        const auto * values = static_cast<const int16_t *>(source);
        for (size_t i = 0; i < count; ++i) output[i] = float(values[i]) / 127;
    };
    tsg_dsv41::engram_io_pool serial(1), parallel(16);
    // Reuse established workers, like the model. Only the benchmark's own
    // file is advised away; remote filesystem/server caches are not controlled.
    for (int i = 0; i < 5; ++i) parallel.run(24, [](size_t) {});
    struct scenario { size_t tokens; uint32_t max_ngram, heads, head_dim; };
    const scenario cases[] = {{1, 4, 8, 256}, {3, 4, 8, 256},
        {joint ? size_t(257) : size_t(256), joint ? 4u : 3u, joint ? 8u : 2u, joint ? 256u : 32u}};
    for (const auto & test : cases) {
        tsg_dsv41::engram_data layout;
        layout.max_ngram_size = test.max_ngram; layout.n_heads = test.heads; layout.head_dim = test.head_dim;
        const size_t tables = joint ? 2 : 1;
        const size_t row_bytes = layout.head_dim * sizeof(int16_t), table_rows = bytes / tables / row_bytes;
        require(table_rows * tables * row_bytes == bytes, "Scratch table geometry does not fill file");
        layout.layers.push_back({1, table_rows, {}, {}, {}});
        if (joint) layout.layers.push_back({14, table_rows, {}, {}, {}});
        const size_t rows = test.tokens * layout.hash_columns();
        std::vector<int32_t> hashes(tables * rows);
        for (auto & hash : hashes) hash = int32_t(random() % table_rows);
        std::vector<unsigned char> selected(bytes / page, 0);
        for (size_t row = 0; row < hashes.size(); ++row) {
            const size_t offset = (row / rows * table_rows + size_t(hashes[row])) * row_bytes;
            const size_t first = offset / page;
            const size_t last = (offset + row_bytes - 1) / page;
            for (size_t i = first; i <= last; ++i) selected[i] = 1;
        }
        const size_t selected_pages = std::count(selected.begin(), selected.end(), 1);
        std::vector<float> expected(hashes.size() * layout.head_dim), actual(expected.size());
        for (size_t e = 0; e < tables; ++e)
            layout.lookup(e, source.data() + e * table_rows * layout.head_dim, table_rows, row_bytes,
                hashes.data() + e * rows, test.tokens, expected.data() + e * rows * layout.head_dim, dequantize);
        for (bool warm : {false, true}) {
            // Three alternating cold pairs cover the new decode branch. One
            // warmed pair retains the earlier control without repeatedly
            // warming gigabytes; --warm-memory measures warm overhead in detail.
            const size_t samples = warm ? 1 : 3;
            for (size_t sample = 0; sample < samples; ++sample) for (size_t order = 0; order < 2; ++order) {
                const size_t index = (sample + order) % 2;
                auto & pool = (joint || index) ? parallel : serial;
                require(posix_fadvise(fd, 0, bytes, POSIX_FADV_DONTNEED) == 0, "Cannot evict private benchmark file pages");
                void * mapped = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
                require(mapped != MAP_FAILED, "Cannot map benchmark file");
                struct unmap_on_exit {
                    void * address; size_t length;
                    ~unmap_on_exit() { munmap(address, length); }
                } mapping{mapped, bytes};
                std::vector<unsigned char> residency(bytes / page);
                require(mincore(mapped, bytes, residency.data()) == 0, "Cannot inspect benchmark page residency");
                size_t resident = 0, selected_resident = 0;
                for (size_t i = 0; i < residency.size(); ++i) {
                    resident += residency[i] & 1;
                    selected_resident += (residency[i] & 1) && selected[i];
                }
                double warm_seconds = 0;
                uint64_t warm_checksum = 0;
                if (warm) {
                    const auto start = std::chrono::steady_clock::now();
                    warm_checksum = pool.warm(mapped, bytes);
                    warm_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
                std::fill(actual.begin(), actual.end(), -98765.0f);
                // Prepare both schedules identically, without touching row
                // data, so the matched joint mode isolates pool scheduling.
                const auto reads = joint ? prepare_tables(layout, mapped, table_rows, hashes.data(), test.tokens, actual.data())
                                         : std::vector<row_read>{};
                const auto start = std::chrono::steady_clock::now();
                if (joint) run_tables(pool, reads, rows, index != 0);
                else layout.lookup(0, mapped, table_rows, row_bytes, hashes.data(), test.tokens, actual.data(), dequantize,
                    [&](size_t count, auto row) { pool.run(count, row); });
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                        "Cold/parallel/warm sparse lookup changed results");
                std::printf("{\"mode\":\"%s\",\"schedule\":\"%s\",\"tables\":%zu,\"tokens\":%zu,\"head_dim\":%u,\"row_bytes\":%zu,"
                            "\"dequantization\":\"synthetic_int16_div127\",\"sample\":%zu,\"order\":%zu,"
                            "\"threads\":%u,\"warm\":%s,\"bytes\":%zu,\"selected_rows\":%zu,"
                            "\"initial_resident_pages\":%zu,\"total_pages\":%zu,\"selected_pages\":%zu,"
                            "\"initial_selected_resident_pages\":%zu,\"cold_pages_confirmed\":%s,\"warm_seconds\":%.9f,"
                            "\"lookup_seconds\":%.9f,\"warm_checksum\":%llu,\"exact_output\":true}\n",
                            joint ? "scratch_joint" : "scratch_file", joint && index ? "joint" : "separate", tables,
                            test.tokens, layout.head_dim, row_bytes, sample, order, pool.threads(), warm ? "true" : "false",
                            bytes, hashes.size(), resident, bytes / page, selected_pages, selected_resident,
                            !warm && selected_resident == 0 ? "true" : "false", warm_seconds, seconds,
                            (unsigned long long) warm_checksum);
                std::fflush(stdout);
            }
        }
    }
#else
    std::fprintf(stderr, "Filesystem benchmark requires Linux; use --warm-memory on this platform.\n");
    return 2;
#endif
} catch (const std::exception & error) {
    std::fprintf(stderr, "Engram I/O benchmark failed: %s\n", error.what());
    return 1;
}
