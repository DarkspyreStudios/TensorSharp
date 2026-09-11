// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "dsv41_raw_gather.h"
#include "ggml_ops_dsv4_fused.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool value, const char * message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main()
{
    size_t position_checks = 0, gather_checks = 0;
    std::vector<int32_t> ids, expected;
    for (int64_t ring : {256, 512, 1280})
        for (int64_t window : {8, 128})
            for (int64_t p = window - 1; p < window - 1 + 4*ring; ++p)
            {
                // Reconstruct the original finite mask independently by
                // inspecting every physical slot's most recent causal token.
                expected.clear();
                for (int64_t row = 0; row < ring; ++row)
                {
                    int64_t token = p - ((p-row)%ring + ring)%ring;
                    if (token >= 0 && token <= p && token > p-window)
                        expected.push_back(static_cast<int32_t>(row));
                }
                dsv41_raw_window_ids(p, ring, window, ids);
                require(ids == expected, "Compact raw IDs changed mask membership or physical order");
                ++position_checks;
            }
    for (const auto & geometry : std::vector<std::vector<int64_t>>{
            {126,512,128}, {127,64,128}, {127,512,0}, {-1,512,128},
            {127,int64_t(std::numeric_limits<int32_t>::max())+1,128}})
    {
        bool rejected = false;
        try { dsv41_raw_window_ids(geometry[0], geometry[1], geometry[2], ids); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Invalid/incomplete raw window was accepted");
    }

    constexpr int head = 512, window = 128, raw_padded = 256, top_k = 512, compressed_rows = 1024;
    for (int ring_rows : {512,1280})
    {
        auto * ctx = ggml_init({16*1024*1024, nullptr, false});
        require(ctx != nullptr, "Could not allocate raw-gather test context");
        auto * raw = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, ring_rows);
        auto * comp = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, compressed_rows);
        auto * selected = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, top_k);
        auto * raw_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, raw_padded);
        // Arbitrary half bit patterns, including signs and NaN payloads,
        // ensure gather is a lossless copy rather than numeric conversion.
        for (int i = 0; i < head*ring_rows; ++i)
            static_cast<ggml_fp16_t *>(raw->data)[i] = static_cast<ggml_fp16_t>(i*17+3);
        for (int i = 0; i < head*compressed_rows; ++i)
            static_cast<ggml_fp16_t *>(comp->data)[i] = static_cast<ggml_fp16_t>(i*29+5);
        for (int i = 0; i < top_k; ++i)
            static_cast<int32_t *>(selected->data)[i] = (i*37+13)%compressed_rows;

        tsg_dsv4_fused_desc baseline_desc, raw_desc, compact_desc;
        baseline_desc.kind = raw_desc.kind = compact_desc.kind = TSG_DSV4_FUSED_KGATHER;
        baseline_desc.i0 = ring_rows;
        raw_desc.i0 = 0;
        compact_desc.i0 = raw_padded;
        ggml_tensor * baseline_inputs[] = {raw,comp,selected};
        auto * baseline = ggml_custom_4d(ctx, GGML_TYPE_F16, head,1,ring_rows+top_k,1,
            baseline_inputs,3,tsg_dsv4_fused_cpu,1,&baseline_desc);
        ggml_tensor * raw_inputs[] = {raw,raw,raw_ids};
        auto * compact_raw = ggml_custom_4d(ctx, GGML_TYPE_F16, head,1,raw_padded,1,
            raw_inputs,3,tsg_dsv4_fused_cpu,1,&raw_desc);
        ggml_tensor * compact_inputs[] = {compact_raw,comp,selected};
        auto * compact = ggml_custom_4d(ctx, GGML_TYPE_F16, head,1,raw_padded+top_k,1,
            compact_inputs,3,tsg_dsv4_fused_cpu,1,&compact_desc);
        tsg_dsv4_fused_cpu(baseline,0,1,&baseline_desc);
        for (int64_t p : {127,128,255,256,511,512,639,1023,1024,1025,1279,1280,1407,1408,2047,2048,2560})
        {
            dsv41_raw_window_ids(p,ring_rows,window,ids);
            ids.resize(raw_padded,ids.front());
            std::memcpy(raw_ids->data,ids.data(),ids.size()*sizeof(int32_t));
            tsg_dsv4_fused_cpu(compact_raw,0,1,&raw_desc);
            tsg_dsv4_fused_cpu(compact,0,1,&compact_desc);
            for (int r = 0; r < raw_padded; ++r)
                require(std::memcmp(static_cast<ggml_fp16_t *>(compact->data)+r*head,
                    static_cast<ggml_fp16_t *>(baseline->data)+ids[r]*head,head*sizeof(ggml_fp16_t)) == 0,
                    "Two-stage compact gather changed a raw row");
            require(std::memcmp(static_cast<ggml_fp16_t *>(compact->data)+raw_padded*head,
                static_cast<ggml_fp16_t *>(baseline->data)+ring_rows*head,
                top_k*head*sizeof(ggml_fp16_t)) == 0,"Compact gather changed selected compressed rows");
            ++gather_checks;
        }
        ggml_free(ctx);
    }
    std::printf("Passed %zu independent window/mask comparisons, 5 invalid-window guards, %zu bit-exact paired gathers\n",
        position_checks,gather_checks);
}
