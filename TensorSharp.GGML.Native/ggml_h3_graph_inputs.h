// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

namespace tsg {
// ggml-cuda compares op_params before allowing CUDA graph capture/replay. Keep
// the changing audio slope in a tiny graph after the main DiT submission. Use
// the original SCALE kernel: MUL plus ADD +0 differs from its fused arithmetic
// for negative underflow to zero. The output is an allocated, plain tensor;
// traversing a computed tensor here would accidentally rerun its whole graph.
inline ggml_cgraph* h3_output_scale_graph(ggml_context* ctx, ggml_tensor* output, float scale) {
    GGML_ASSERT(output->op == GGML_OP_NONE && output->buffer != nullptr && output->data != nullptr);
    auto* scaled = ggml_scale_inplace(ctx, output, scale);
    ggml_backend_view_init(scaled);
    ggml_set_output(scaled);
    auto* graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, scaled);
    return graph;
}
} // namespace tsg
