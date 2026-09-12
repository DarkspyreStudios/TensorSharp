// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Return the finite rows of a single-query raw SWA mask in ascending physical
// ring order. Preserving that order avoids an unnecessary attention reduction
// permutation when a chronological window straddles the ring boundary.
inline void dsv41_raw_window_ids(int64_t position, int64_t ring_rows, int64_t window,
                                 std::vector<int32_t> & ids)
{
    if (window <= 0 || ring_rows < window || ring_rows > std::numeric_limits<int32_t>::max() ||
        position < window - 1)
        throw std::invalid_argument("V4.1 raw gather requires a complete window within an I32-addressable ring");
    const int64_t end = position % ring_rows;
    const int64_t start = (position - (window - 1)) % ring_rows;
    ids.resize(static_cast<size_t>(window));
    for (int64_t i = 0; i < window; ++i)
        ids[static_cast<size_t>(i)] = static_cast<int32_t>(start <= end ? start + i
            : i <= end ? i : start + i - end - 1);
}
