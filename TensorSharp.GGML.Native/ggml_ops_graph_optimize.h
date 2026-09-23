// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

// Direct-compute equivalent of the scheduler's allocation-dependency sink.
// Call before allocation. Dependency views belong to ctx and remain valid for
// exactly the same lifetime as the graph. A tight metadata budget falls back
// to retaining the affected storage through the end of the graph.
void tsg_optimize_graph_with_alloc_dependencies(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * graph);
