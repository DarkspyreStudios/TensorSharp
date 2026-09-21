// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../dsv4_exact_read.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace tsg_dsv4;

struct injected_reader {
    std::vector<file_read_attempt> steps;
    std::vector<unsigned char> expected;
    std::vector<uint64_t> offsets;
    std::vector<size_t> lengths;
    std::vector<exact_read_retry> retries;
    std::vector<unsigned> delays;
    std::vector<unsigned char> output;
    size_t step = 0;

    explicit injected_reader(std::initializer_list<file_read_attempt> faults, size_t length = 32)
        : steps(faults), expected(length), output(length, 0xff) {
        for (size_t i = 0; i < length; ++i) expected[i] = static_cast<unsigned char>(3 * i + 7);
    }
    exact_read_result run() {
        return read_range_exact(output.data(), output.size(), 100,
            [&](uint64_t offset, void * target, size_t length) {
                offsets.push_back(offset);
                lengths.push_back(length);
                assert(offset >= 100 && offset - 100 <= output.size());
                assert(target == output.data() + (offset - 100));
                const auto attempt = step < steps.size() ? steps[step] : file_read_attempt{length};
                ++step;
                assert(attempt.bytes <= length);
                std::copy_n(expected.data() + (offset - 100), attempt.bytes,
                    static_cast<unsigned char *>(target));
                return attempt;
            }, [&](const exact_read_retry & retry) { retries.push_back(retry); },
            [&](unsigned ms) { delays.push_back(ms); });
    }
};

static void injection_tests() {
    injected_reader full({});
    auto result = full.run();
    assert(result.ok && result.bytes == 32 && result.retries == 0);
    assert(full.step == 1 && full.delays.empty() && full.output == full.expected);

    injected_reader partial({{7}, {11}});
    result = partial.run();
    assert(result.ok && result.retries == 0 && partial.output == partial.expected);
    assert(partial.offsets == std::vector<uint64_t>({100, 107, 118}));
    assert(partial.lengths == std::vector<size_t>({32, 25, 14}));

    // fread can deliver valid bytes while setting its error flag. Those bytes
    // must survive retries; pread can fail without progress at the next offset.
    injected_reader recovered({{7, ENOMEM}, {0, EAGAIN}, {11, EINTR}});
    result = recovered.run();
    assert(result.ok && result.bytes == 32 && result.retries == 3);
    assert(recovered.output == recovered.expected);
    assert(recovered.offsets == std::vector<uint64_t>({100, 107, 107, 118}));
    assert(recovered.delays == std::vector<unsigned>({10, 20, 40}));
    assert(recovered.retries[0].error == ENOMEM && recovered.retries[0].offset == 107);
    assert(recovered.retries[1].error == EAGAIN && recovered.retries[1].bytes == 7);
    assert(recovered.retries[2].error == EINTR && recovered.retries[2].offset == 118);

    // Retry budget does not reset when every failing call makes progress.
    injected_reader exhausted({{2, ENOMEM}, {2, EINTR}, {2, EAGAIN},
        {2, ENOMEM}, {2, EINTR}, {2, EAGAIN}});
    result = exhausted.run();
    assert(!result.ok && result.bytes == 12 && result.error == EAGAIN && result.retries == 5);
    assert(exhausted.step == 6 && exhausted.retries.size() == 5);
    assert(exhausted.delays == std::vector<unsigned>({10, 20, 40, 80, 160}));
    assert(std::equal(exhausted.output.begin(), exhausted.output.begin() + 12, exhausted.expected.begin()));
    assert(exhausted.output[12] == 0xff);
    const auto diagnostic = exact_read_failure(result, 32, 100);
    assert(diagnostic.find("offset 112") != std::string::npos);
    assert(diagnostic.find("12/32 bytes read") != std::string::npos);
    assert(diagnostic.find("retries=5") != std::string::npos);
    assert(diagnostic.find("errno=" + std::to_string(EAGAIN)) != std::string::npos);

    injected_reader eof({{7}, {0, 0, true}});
    result = eof.run();
    assert(!result.ok && result.eof && result.bytes == 7 && result.retries == 0);
    assert(eof.delays.empty() && eof.step == 2);
    injected_reader stale_errno_at_eof({{7, ENOMEM, true}});
    result = stale_errno_at_eof.run();
    assert(!result.ok && result.eof && result.retries == 0);

    injected_reader permanent({{7}, {0, EIO}});
    result = permanent.run();
    assert(!result.ok && result.error == EIO && result.bytes == 7 && permanent.delays.empty());
    injected_reader seek_failure({{0, ENOMEM, false, true}});
    result = seek_failure.run();
    assert(!result.ok && result.seek_error && result.bytes == 0 && seek_failure.step == 1);
    assert(seek_failure.delays.empty());
}

static void adapter_tests() {
    FILE * file = tmpfile();
    assert(file);
    const char expected[] = "abcdefghijklmnopqrstuvwxyz";
    assert(fwrite(expected, 1, 26, file) == 26);
    assert(fflush(file) == 0);
    std::vector<char> output(20, 0);
    auto observer = [](const exact_read_retry &) { assert(false); };
    auto reader = [&](uint64_t off, void * data, size_t size) { return fread_at(file, off, data, size); };
    auto result = read_range_exact(output.data(), 20, 3, reader, observer);
    assert(result.ok && std::equal(output.begin(), output.end(), expected + 3));
    result = read_range_exact(output.data(), 20, 20, reader, observer);
    assert(!result.ok && result.eof && result.bytes == 6 && result.error == 0);
    // The adapter clears the EOF/error state before a new checked seek.
    result = read_range_exact(output.data(), 20, 0, reader, observer);
    assert(result.ok && std::equal(output.begin(), output.end(), expected));
    result = read_range_exact(output.data(), 20, uint64_t(INT64_MAX), reader, observer);
    assert(!result.ok && result.seek_error && result.error == EOVERFLOW);
#if !defined(_WIN32)
    result = read_range_exact(output.data(), 20, 3,
        [&](uint64_t off, void * data, size_t size) { return pread_at(fileno(file), off, data, size); }, observer);
    assert(result.ok && std::equal(output.begin(), output.end(), expected + 3));
    int pipe_fds[2];
    assert(pipe(pipe_fds) == 0);
    FILE * unseekable = fdopen(pipe_fds[0], "rb");
    assert(unseekable);
    result = read_range_exact(output.data(), 1, 0,
        [&](uint64_t off, void * data, size_t size) { return fread_at(unseekable, off, data, size); }, observer);
    assert(!result.ok && result.seek_error && result.error == ESPIPE && result.bytes == 0);
    fclose(unseekable);
    close(pipe_fds[1]);
#endif
    fclose(file);
}

int main() {
    injection_tests();
    adapter_tests();
    std::cout << "DeepSeek exact reads: full/partial/transient/exhaustion/EOF/seek tests passed\n";
}
