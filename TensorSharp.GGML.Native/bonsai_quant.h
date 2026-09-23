// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <cstddef>
#include <cstdint>

// Prism's private GGUF types are decoded here, never registered in or patched
// into ggml. On load, retain their exact scales/codes in upstream Q2_0 blocks.
// Format reference: PrismML-Eng/llama.cpp, prism commit
// bdc23b56b4458b9f1655aec5287f3ab56ee8daaa, ggml-common.h / ggml-quants.c.
constexpr int TSG_BONSAI_PQ2_0 = 142;
constexpr int TSG_BONSAI_PTQ1_0 = 143;

bool tsg_bonsai_quant_type(int type);
size_t tsg_bonsai_row_size(int type, int64_t elements);
int tsg_bonsai_dequantize(int type, const void * src, int64_t elements, float * dst);
int tsg_bonsai_transcode_q2_0(int type, const void * src, int64_t elements, void * dst);
