// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <memory>

namespace tsg {
// The identity is the managed weight's stable host pointer, not a transient
// ggml tensor address or its backend-specific resident address.
ggml_tensor * bonsai_mul_mat(ggml_context *, ggml_tensor *, ggml_tensor *, const void *);
ggml_tensor * bonsai_get_rows(ggml_context *, ggml_tensor *, ggml_tensor *, const void *);
ggml_tensor * bonsai_transform(ggml_context *, ggml_tensor *, const void *, bool inverse);
bool bonsai_weight_has_transform(const void *);
void bonsai_clear_backend();

// Share transformed activations between Q/K/V and gate/up within one graph.
// A scoped cache cannot accidentally reuse tensors from a freed ggml context.
struct BonsaiGraphScope {
    struct Impl;
    std::unique_ptr<Impl> impl;
    BonsaiGraphScope();
    ~BonsaiGraphScope();
    BonsaiGraphScope(const BonsaiGraphScope &) = delete;
    BonsaiGraphScope & operator=(const BonsaiGraphScope &) = delete;
};
}
