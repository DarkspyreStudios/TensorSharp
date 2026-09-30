#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TENSORSHARP_GGML_NATIVE_BUILD_DIR selects a separate build tree, so a release
# build (eng/build-ggml-natives.sh) never reuses a development CMake cache.
BUILD_DIR="${TENSORSHARP_GGML_NATIVE_BUILD_DIR:-${SCRIPT_DIR}/build}"

# Ensure the ggml sources are present (cloned from ggml-org/ggml at build time).
bash "${SCRIPT_DIR}/../eng/fetch-ggml.sh"

# Extra arguments are passed to the CMake configure step unchanged.
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "${BUILD_DIR}" --config Release --target GgmlOps
