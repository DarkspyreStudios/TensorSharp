// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The V4.1 truncation guards, checked against an independent simulation of the
// raw sliding-window ring rather than against the same arithmetic restated.
//
// The simulation models what the forward and the mask actually do:
//   * writing position p puts p in physical slot p % ring;
//   * a checkpoint copies the whole ring;
//   * every token of a ubatch is written before that ubatch attends, so after
//     the head moves to `target` the first ubatch's positions are resident;
//   * the mask (the raw SWA fill in ggml_ops_deepseek4.cpp) attributes slot s to
//     the newest causal position congruent to s, derived from `p_last` ALONE -
//     it has no idea the head moved - and marks it visible when that position
//     falls inside the query's window.
//
// So the property to prove is: for every query of the first ubatch after a
// truncation the planner allowed, every slot the mask marks visible physically
// holds the position the mask thinks it holds. That covers both failure modes at
// once - a row the window needs having been overwritten by the abandoned tail,
// and a row left over from an abandoned position being mistaken for a live one.
#include "dsv41_truncate.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

namespace
{

void require(bool value, const char * message)
{
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

// Ring contents after positions [0, filled) have been written in order.
std::vector<int64_t> ring_after(int64_t filled, int64_t ring)
{
    std::vector<int64_t> slots((size_t) ring, -1);
    const int64_t first = std::max<int64_t>(0, filled - ring);
    for (int64_t p = first; p < filled; ++p) slots[(size_t) (p % ring)] = p;
    return slots;
}

// True when a ubatch of `nt` tokens starting at `target` reads only rows that
// hold what the mask believes they hold.
//
// A slot is visible to at least one query of the ubatch exactly when its
// believed position is <= p_last and > target - window: for such a slot take
// p = max(target, believed); outside that range no query in [target, p_last]
// can reach it. So one pass over the ring settles the whole ubatch.
bool reads_are_sound(std::vector<int64_t> slots, int64_t target, int64_t nt,
                     int64_t ring, int64_t window)
{
    const int64_t p_last = target + nt - 1;
    for (int64_t p = target; p <= p_last; ++p) slots[(size_t) (p % ring)] = p;

    for (int64_t s = 0; s < ring; ++s)
    {
        const int64_t believed = p_last - ((p_last - s) % ring + ring) % ring;
        if (believed < 0 || believed > p_last || believed <= target - window) continue;
        if (slots[(size_t) s] != believed) return false;
    }
    return true;
}

// ring = pad64(n_swa + n_ubatch, 256), so the ubatch width is the slack.
struct geometry { int64_t ring, window; };

const std::vector<geometry> kGeometries = {
    { 512, 128 },    // the released DeepSeek-V4.1-Flash checkpoint (n_swa 128, ubatch 256)
    { 1280, 128 },   // TS_DSV4_UBATCH=1024
    { 256, 8 },      // the small numerical fixture
    { 512, 512 },    // a window that fills the ring: no slack at all
};

std::vector<int64_t> ubatch_widths(int64_t ring, int64_t window)
{
    const int64_t w = std::max<int64_t>(1, ring - window);
    std::set<int64_t> s = { 1, 2, w / 2 > 0 ? w / 2 : 1, w };
    return { s.begin(), s.end() };
}

// Boundary-dense plus coarse: every value where the guards could change, and a
// stride through the rest so nothing in between is assumed.
std::vector<int64_t> targets_for(int64_t n_past, int64_t cp, int64_t span, int64_t ring)
{
    std::set<int64_t> t;
    for (int64_t anchor : { (int64_t) 0, cp, n_past, n_past - span, cp - span, ring, 2 * ring })
        for (int64_t d = -2; d <= 2; ++d)
            if (anchor + d >= 0 && anchor + d <= n_past) t.insert(anchor + d);
    const int64_t stride = std::max<int64_t>(1, n_past / 24);
    for (int64_t p = 0; p <= n_past; p += stride) t.insert(p);
    return { t.begin(), t.end() };
}

} // namespace

int main()
{
    size_t allowed = 0, refused = 0;

    for (const geometry & g : kGeometries)
    {
        const int64_t span = g.ring - g.window + 1;
        const std::vector<int64_t> widths = ubatch_widths(g.ring, g.window);

        for (int64_t align : { (int64_t) 1, (int64_t) 2, (int64_t) 128 })
        {
            for (int64_t n_past : { span / 2 + 1, span, span + 1, 3 * g.ring + 7, 9 * g.ring })
            {
                if (n_past <= 0) continue;
                for (int64_t cp : { (int64_t) -1, (int64_t) 0, n_past / 3,
                                    n_past - span - 1, n_past - span, n_past - 1, n_past })
                {
                    if (cp > n_past) continue;
                    for (int64_t target : targets_for(n_past, cp, span, g.ring))
                    {
                        const dsv41_truncate_route route =
                            dsv41_plan_truncate(target, n_past, cp, span, align);
                        if (route == dsv41_truncate_route::refuse) { ++refused; continue; }
                        if (route == dsv41_truncate_route::none)
                        {
                            require(target == n_past, "route none for a target that is not the head");
                            continue;
                        }
                        if (route == dsv41_truncate_route::reset)
                        {
                            require(target == 0, "route reset for a non-zero target");
                            continue;
                        }

                        require(target % align == 0, "an allowed target straddles a compression block");
                        require(route != dsv41_truncate_route::checkpoint || cp >= target,
                                "the checkpoint route chose a checkpoint behind the target");
                        const std::vector<int64_t> slots = ring_after(
                            route == dsv41_truncate_route::checkpoint ? cp : n_past, g.ring);

                        for (int64_t nt : widths)
                        {
                            require(reads_are_sound(slots, target, nt, g.ring, g.window),
                                    "an allowed truncation reads a row that does not hold "
                                    "what the mask expects");
                        }
                        ++allowed;
                    }
                }
            }
        }
    }

    // Completeness in the direction that matters: the planner must not refuse a
    // rewind the geometry can serve. Sweep aligned targets and assert that every
    // one the simulation proves sound - from the live ring or from the
    // checkpoint - is allowed.
    size_t sound_and_allowed = 0;
    for (const geometry & g : kGeometries)
    {
        const int64_t span = g.ring - g.window + 1;
        const int64_t n_past = 9 * g.ring;
        const std::vector<int64_t> widths = ubatch_widths(g.ring, g.window);
        for (int64_t cp : { (int64_t) -1, n_past - 1, n_past / 2 })
        {
            for (int64_t target = 2; target < n_past; target += 2)
            {
                bool live_ok = true, cp_ok = cp >= target;
                for (int64_t nt : widths)
                {
                    if (!reads_are_sound(ring_after(n_past, g.ring), target, nt, g.ring, g.window))
                        live_ok = false;
                    if (cp_ok && !reads_are_sound(ring_after(cp, g.ring), target, nt, g.ring, g.window))
                        cp_ok = false;
                }
                if (!live_ok && !cp_ok) continue;
                require(dsv41_plan_truncate(target, n_past, cp, span, /*align*/ 2)
                            != dsv41_truncate_route::refuse,
                        "the planner refused a rewind the ring can serve");
                ++sound_and_allowed;
            }
        }
    }
    require(sound_and_allowed > 0, "the completeness sweep proved nothing sound");

    // The reported bug's shape: a 39-token prompt, a 7009-token answer, then a
    // rewind to 38 - far outside the live ring, one position inside the
    // checkpoint taken at the end of the prompt.
    require(dsv41_plan_truncate(38, 39 + 7009, -1, 385, 2) == dsv41_truncate_route::refuse,
            "a deep rewind with no checkpoint must be refused");
    require(dsv41_plan_truncate(38, 39 + 7009, 39, 385, 2) == dsv41_truncate_route::checkpoint,
            "a deep rewind to one position before the prompt boundary must use the checkpoint");
    require(dsv41_plan_truncate(39, 39 + 7009, 39, 385, 2) == dsv41_truncate_route::refuse,
            "an odd target must be refused so no compression block straddles the head");
    require(dsv41_plan_truncate(2430, 2431 + 8467, 2431, 385, 2) == dsv41_truncate_route::checkpoint,
            "turn 3's rewind to the whole of turn 2's prompt must use the checkpoint");
    require(dsv41_plan_truncate(10897, 10898, 2431, 385, 2) == dsv41_truncate_route::refuse,
            "dropping one token off the head is still refused when it is misaligned");
    require(dsv41_plan_truncate(10896, 10898, 2431, 385, 2) == dsv41_truncate_route::live,
            "a shallow aligned rewind needs no checkpoint");

    std::printf("dsv41 truncate guards: %zu allowed rewinds verified against the ring simulation, "
                "%zu refused, %zu sound rewinds confirmed allowed\n",
                allowed, refused, sound_and_allowed);
    return 0;
}
