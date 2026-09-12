// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cstdint>

// How far back a DeepSeek V4.1 sequence slot can move its head, and how.
//
// Two of the slot's caches are addressed MODULARLY, so a later position
// overwrites an earlier one's row and moving the head back does not bring the
// earlier value with it:
//
//   * the raw sliding-window ring, `ring_raw = pad64(n_swa + n_ubatch, 256)`
//     rows indexed `pos % ring_raw`;
//   * the compressor state ring, `ratio` rows indexed `pos % ratio`, holding
//     the per-token halves a block boundary compresses.
//
// Everything else the forward writes is addressed by ABSOLUTE index and only
// ever grows - the compressed K caches by block (`pos / ratio`, visible up to
// `(pos + 1) / ratio`), the Engram history by position - so dropping the tail
// makes it invisible rather than wrong.
//
// Hence two conditions, and this header is the one place that states them:
//
//   1. ALIGNMENT. No compression block may straddle the new head, or the next
//      boundary would read state-ring rows the abandoned tail overwrote.
//      `align` is the lcm of the per-layer compression ratios (2 for the
//      released checkpoint, whose ratios are 0, 1 and 2).
//   2. DEPTH. A query at the new head `t` reads raw rows for positions
//      (t - n_swa, t]. A ring that was last written at position `r` holds
//      [r - ring_raw, r - 1], so those rows survive exactly while
//      `r - t <= ring_raw - n_swa + 1`, which is `span`. The same bound makes
//      the converse safe: no row left over from an abandoned position can be
//      mis-read as a live one, because the mask attributes such a row to
//      position `q - ring_raw <= t - n_swa`, one short of visible.
//
// `r` is `n_past` for the live rings, or `cp_n_past` when the slot's
// checkpoint shadow rings are restored first. The checkpoint is what makes a
// conversational rewind possible at all: generating a turn's answer moves
// `n_past` thousands of positions past the prompt boundary the next turn wants
// to rewind to, far outside `span`, while the checkpoint sits one position away
// from it.
enum class dsv41_truncate_route
{
    refuse,      // nothing correct is available; reset and re-prefill instead
    none,        // the head is already there
    reset,       // the target is 0, which is a full reset
    live,        // the live rings still hold the window the new head reads
    checkpoint,  // restore the shadow rings first, then move the head
};

// `cp_n_past` is -1 when the slot has no checkpoint. Pure arithmetic: it never
// looks at a tensor, so the guards can be tested without a model.
inline dsv41_truncate_route dsv41_plan_truncate(int64_t target, int64_t n_past, int64_t cp_n_past,
                                                int64_t span, int64_t align)
{
    if (target < 0 || target > n_past || align <= 0) return dsv41_truncate_route::refuse;
    if (target == n_past) return dsv41_truncate_route::none;
    if (target == 0)      return dsv41_truncate_route::reset;
    if (target % align != 0) return dsv41_truncate_route::refuse;
    if (n_past - target <= span) return dsv41_truncate_route::live;
    if (cp_n_past >= target && cp_n_past - target <= span)
        return dsv41_truncate_route::checkpoint;
    return dsv41_truncate_route::refuse;
}
