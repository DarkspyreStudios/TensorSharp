// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"

namespace tsg {

// A run of whole token rows is already contiguous when its source is. Retain
// the view's source edge so gallocr keeps the backing activation alive; no
// caller may write this borrowed slice in place. Narrowed feature dimensions
// can have a larger row stride, in which case a real copy is still required.
inline ggml_tensor * h3_token_rows(ggml_context * ctx, ggml_tensor * source,
                                 int64_t start, int64_t count) {
    auto * rows = ggml_view_2d(ctx, source, source->ne[0], count, source->nb[1],
                             static_cast<size_t>(start) * source->nb[1]);
    return ggml_is_contiguous(rows) ? rows : ggml_cont(ctx, rows);
}

} // namespace tsg
