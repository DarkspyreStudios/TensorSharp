// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace tsg_dsv4 {

struct file_read_attempt {
    size_t bytes = 0;
    int error = 0;
    bool eof = false;
    bool seek_error = false;
};

struct exact_read_result {
    bool ok = false;
    size_t bytes = 0;
    unsigned retries = 0;
    int error = 0;
    bool eof = false;
    bool seek_error = false;
};

struct exact_read_retry {
    unsigned number;
    int error;
    size_t bytes;
    size_t requested;
    uint64_t offset;
    unsigned delay_ms;
};

inline bool transient_file_read_error(int error) {
    return error == EINTR || error == EAGAIN || error == ENOMEM;
}

// One normal read is still one call. A partial read advances BOTH the file
// offset and the destination; only transient read errors consume retries.
// The retry budget is per requested range, never reset by partial progress.
// Reader(offset, destination, length) must report a checked seek separately.
// Injecting reader, observer and sleep lets tests cover failures without a
// flaky filesystem or actual delays.
template<typename Reader, typename Observer, typename Sleep>
exact_read_result read_range_exact(void * destination, size_t requested, uint64_t offset,
                                   Reader read, Observer on_retry, Sleep sleep) {
    exact_read_result result;
    if (requested > UINT64_MAX - offset) {
        result.error = EOVERFLOW;
        result.seek_error = true;
        return result;
    }
    while (result.bytes < requested) {
        const size_t remaining = requested - result.bytes;
        const file_read_attempt attempt = read(offset + result.bytes,
            static_cast<unsigned char *>(destination) + result.bytes, remaining);
        if (attempt.bytes > remaining) { result.error = EIO; return result; }
        result.bytes += attempt.bytes;
        result.error = attempt.error;
        result.eof = attempt.eof;
        result.seek_error = attempt.seek_error;
        // A failed seek must never be treated as a read or retried, even if
        // the filesystem happens to report one of the transient read errnos.
        if (result.seek_error) return result;
        if (result.bytes == requested) { result.ok = true; return result; }
        if (result.eof) return result;
        if (result.error) {
            if (!transient_file_read_error(result.error) || result.retries == 5) return result;
            const unsigned delay = 10u << result.retries;
            ++result.retries;
            on_retry(exact_read_retry{result.retries, result.error, result.bytes,
                requested, offset + result.bytes, delay});
            sleep(delay);
        } else if (attempt.bytes == 0) {
            // An error-free zero-byte pread is EOF; other adapters should
            // also identify it explicitly. Never spin on a broken reader.
            result.eof = true;
            return result;
        }
    }
    result.ok = true;
    return result;
}

template<typename Reader, typename Observer>
exact_read_result read_range_exact(void * destination, size_t requested, uint64_t offset,
                                   Reader read, Observer on_retry) {
    return read_range_exact(destination, requested, offset, read, on_retry,
        [](unsigned ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); });
}

// Every continuation explicitly seeks to the first unread byte and clears a
// previous stream error. A failed seek cannot leave fread using an old offset.
inline file_read_attempt fread_at(FILE * file, uint64_t offset, void * destination, size_t length) {
    clearerr(file);
    errno = 0;
#if defined(_WIN32)
    if (offset > uint64_t(INT64_MAX) || length > uint64_t(INT64_MAX) - offset)
        return {0, EOVERFLOW, false, true};
    const int seek = _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET);
#else
    if (offset > uint64_t(std::numeric_limits<off_t>::max()) ||
        length > uint64_t(std::numeric_limits<off_t>::max()) - offset)
        return {0, EOVERFLOW, false, true};
    const int seek = fseeko(file, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (seek != 0) return {0, errno ? errno : EIO, false, true};
    errno = 0;
    const size_t got = fread(destination, 1, length, file);
    const int error = errno;
    return {got, ferror(file) ? (error ? error : EIO) : 0, feof(file) != 0, false};
}

#if !defined(_WIN32)
inline file_read_attempt pread_at(int fd, uint64_t offset, void * destination, size_t length) {
    if (offset > uint64_t(std::numeric_limits<off_t>::max()) ||
        length > uint64_t(std::numeric_limits<off_t>::max()) - offset)
        return {0, EOVERFLOW, false, true};
    const ssize_t got = ::pread(fd, destination, length, static_cast<off_t>(offset));
    if (got < 0) return {0, errno, false, false};
    return {static_cast<size_t>(got), 0, got == 0, false};
}
#endif

inline void format_exact_read_failure(char * output, size_t capacity, const exact_read_result & result,
                                      size_t requested, uint64_t offset) {
    const char * kind = result.seek_error ? "seek error" : result.eof ? "unexpected EOF" : "read error";
    std::snprintf(output, capacity, "%s at offset %llu: %zu/%zu bytes read, retries=%u, errno=%d%s%s%s",
        kind, static_cast<unsigned long long>(offset + result.bytes), result.bytes, requested,
        result.retries, result.error, result.error ? " (" : "",
        result.error ? std::strerror(result.error) : "", result.error ? ")" : "");
}

inline std::string exact_read_failure(const exact_read_result & result, size_t requested, uint64_t offset) {
    char message[256];
    format_exact_read_failure(message, sizeof(message), result, requested, offset);
    return message;
}

} // namespace tsg_dsv4
