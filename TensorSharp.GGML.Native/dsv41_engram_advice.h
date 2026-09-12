// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tsg_dsv41 {

// The unset policy preserves the one-worker path while allowing sparse reads
// on the parallel path to avoid unnecessary filesystem readahead.
inline bool resolve_engram_random(const char * option, unsigned threads) {
    if (!option) return threads > 1;
    if (std::strcmp(option, "0") == 0) return false;
    if (std::strcmp(option, "1") == 0) return true;
    throw std::runtime_error("TS_DSV41_ENGRAM_RANDOM must be 0 or 1 (unset selects automatically)");
}

struct mapped_advice_range {
    const void * address = nullptr;
    size_t bytes = 0;
};

// mmap bases are page aligned, but a tensor can begin/end within a page. The
// start is rounded down inside the mapping; the byte length stops at the
// tensor end. The OS applies advice at page granularity, including boundary
// pages. Avoid rounding the end up in integer arithmetic or crossing the
// caller's mapping bound. This helper never dereferences either pointer.
inline bool plan_mapped_advice(const void * mapping, size_t mapping_bytes,
        const void * data, size_t bytes, size_t page_bytes, mapped_advice_range & result) noexcept {
    result = {};
    if (!mapping || !data || !mapping_bytes || !page_bytes) return false;
    const auto base = reinterpret_cast<uintptr_t>(mapping);
    const auto address = reinterpret_cast<uintptr_t>(data);
    if (base % page_bytes || mapping_bytes > std::numeric_limits<uintptr_t>::max() - base || address < base)
        return false;
    const uintptr_t offset = address - base;
    if (offset > mapping_bytes || bytes > mapping_bytes - offset) return false;
    if (!bytes) {
        result.address = data;
        return true;
    }
    const size_t aligned_offset = (size_t(offset) / page_bytes) * page_bytes;
    result.address = reinterpret_cast<const void *>(base + aligned_offset);
    result.bytes = size_t(offset) - aligned_offset + bytes;
    return true;
}

enum class mapped_advice_status { applied, empty, unsupported, invalid_range, system_error };

struct mapped_advice_result {
    mapped_advice_status status;
    size_t bytes;
    int error;
};

// This is an optional mapping hint, not a read, allocation, or residency
// guarantee. Unsupported platforms and OS failures remain nonfatal to callers.
inline mapped_advice_result advise_engram_random(const void * mapping, size_t mapping_bytes,
        const void * data, size_t bytes) noexcept {
#if defined(__linux__)
    const long system_page = sysconf(_SC_PAGESIZE);
    if (system_page <= 0) return {mapped_advice_status::system_error, 0, EINVAL};
    mapped_advice_range range;
    if (!plan_mapped_advice(mapping, mapping_bytes, data, bytes, size_t(system_page), range))
        return {mapped_advice_status::invalid_range, 0, EINVAL};
    if (!range.bytes) return {mapped_advice_status::empty, 0, 0};
    const int error = posix_madvise(const_cast<void *>(range.address), range.bytes, POSIX_MADV_RANDOM);
    return {error ? mapped_advice_status::system_error : mapped_advice_status::applied, range.bytes, error};
#else
    (void) mapping; (void) mapping_bytes; (void) data; (void) bytes;
    return {mapped_advice_status::unsupported, 0, 0};
#endif
}

} // namespace tsg_dsv41
