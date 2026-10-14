#!/usr/bin/env bash
# Builds TensorAgent.Maui for the arm64 iOS simulator.
#
# Two things that silently break a fresh clone are handled here:
#   1. Only the user-local ~/.dotnet SDK carries the maui-ios workload; the
#      system dotnet fails restore with NETSDK1147 (see README.md).
#   2. TensorSharp.GGML.Native/build-ios/GgmlOps.xcframework is gitignored, so it
#      is built on demand (~4 min the first time; ExternalProjects/ggml must
#      already be checked out - the fetch is skipped, never run from here).
#
# Env:
#   CONFIGURATION   Debug (default) | Release
#   TENSORAGENT_REBUILD_XCFRAMEWORK=1   force build-ios.sh even if the xcframework exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"

export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="${DOTNET_ROOT}:${PATH}"
export TENSORSHARP_GGML_NO_UPDATE=1
# The head links GgmlOps statically from the xcframework and never uses MLX, so
# the desktop CMake builds of the two backends must not run. The csproj forwards
# TensorSharpSkip{Ggml,Mlx}Native to its references, but the env-var spelling is
# what the backend csprojs honour regardless of how they are reached.
export TENSORSHARP_GGML_NATIVE_SKIP=true
export TENSORSHARP_MLX_NATIVE_SKIP=true

XCFRAMEWORK="${REPO_ROOT}/TensorSharp.GGML.Native/build-ios/GgmlOps.xcframework"
if [[ ! -d "${XCFRAMEWORK}" || "${TENSORAGENT_REBUILD_XCFRAMEWORK:-0}" == "1" ]]; then
    if [[ ! -d "${REPO_ROOT}/ExternalProjects/ggml" ]]; then
        echo "ExternalProjects/ggml is missing; run eng/fetch-ggml.sh once (network) before building the xcframework." >&2
        exit 1
    fi
    echo "==> Building ${XCFRAMEWORK}"
    bash "${REPO_ROOT}/TensorSharp.GGML.Native/build-ios.sh"
fi

echo "==> dotnet $(dotnet --version): building TensorAgent.Maui (${CONFIGURATION}, iossimulator-arm64)"
# TensorSharpIosTargets=true must be on the command line, not only in the app's own
# csproj: it decides whether TensorSharp.Models builds a net10.0-ios slice at all, and a
# csproj property does not flow to referenced projects. Without it, restore writes an
# assets file with no iOS target and the build fails with NETSDK1005.
dotnet build "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj" \
    -f net10.0-ios \
    -p:RuntimeIdentifier=iossimulator-arm64 \
    -c "${CONFIGURATION}" \
    -p:TensorSharpIosTargets=true \
    -nologo

APP="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-ios/iossimulator-arm64/TensorAgent.Maui.app"
if [[ ! -d "${APP}" ]]; then
    echo "Build finished but the bundle is not at ${APP}" >&2
    exit 1
fi
echo "Bundle: ${APP}"
