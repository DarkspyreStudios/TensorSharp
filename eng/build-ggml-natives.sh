#!/usr/bin/env bash
# Builds one release GgmlOps variant with the portable CPU profile and stages it
# for packaging (eng/pack-ggml-natives.py).
#
# Usage:
#   eng/build-ggml-natives.sh --rid <rid> --variant <variant> [--out <dir>]
#                             [--redist-dir <dir>] [-- <extra cmake args>]
#
#   --rid         osx-arm64, linux-x64 or linux-arm64. The script refuses a RID
#                 that is not the build machine's own OS and architecture.
#   --variant     osx-arm64: metal. Linux: cpu, vulkan or cuda13.
#   --out         staging root (default: artifacts/ggml-natives/<version>).
#                 Files land in <out>/runtimes/<rid>/native/<variant>/.
#   --redist-dir  a directory of third-party runtime libraries to ship beside the
#                 bridge, for example the CUDA runtime and cuBLAS. It must hold
#                 the libraries at its top level and their license and notice
#                 files under licenses/. The script copies both.
#
# Release natives come from a clean tree: the ggml commit in eng/ggml-revision,
# a separate build directory per RID and variant, and the TensorSharp version in
# Directory.Build.props. The build identity compiled into the binary records the
# TensorSharp version, source commit, ggml commit, variant, RID and CPU profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NATIVE_DIR="${REPO_ROOT}/TensorSharp.GGML.Native"

# macOS 14.0 is the lowest macOS the consuming application supports. Every
# Apple Silicon Mac, including M1, runs it.
MACOS_DEPLOYMENT_TARGET="${TENSORSHARP_MACOS_DEPLOYMENT_TARGET:-14.0}"

RID=""
VARIANT=""
OUT=""
REDIST_DIR=""
EXTRA_ARGS=()
while (($# > 0)); do
    case "$1" in
        --rid) RID="$2"; shift ;;
        --variant) VARIANT="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --redist-dir) REDIST_DIR="$2"; shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

[[ -n "${RID}" && -n "${VARIANT}" ]] || { echo "--rid and --variant are required" >&2; exit 2; }

VERSION="$(sed -n 's:.*<TensorSharpVersion>\(.*\)</TensorSharpVersion>.*:\1:p' "${REPO_ROOT}/Directory.Build.props" | head -n 1)"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain -- TensorSharp.GGML.Native eng/ggml-revision Directory.Build.props)" ]]; then
    echo "error: native sources have uncommitted changes; a release native must be built from a commit." >&2
    exit 1
fi
OUT="${OUT:-${REPO_ROOT}/artifacts/ggml-natives/${VERSION}}"

host_os="$(uname -s)"
host_arch="$(uname -m)"
case "${host_os}/${host_arch}" in
    Darwin/arm64) HOST_RID="osx-arm64" ;;
    Linux/x86_64) HOST_RID="linux-x64" ;;
    Linux/aarch64|Linux/arm64) HOST_RID="linux-arm64" ;;
    *) echo "error: unsupported build host ${host_os}/${host_arch}" >&2; exit 1 ;;
esac
if [[ "${RID}" != "${HOST_RID}" ]]; then
    echo "error: this machine builds ${HOST_RID}, not ${RID}." >&2
    exit 1
fi

BUILD_ARGS=()
CMAKE_ARGS=(
    -DTENSORSHARP_GGML_NATIVE_PORTABLE=ON
    "-DTENSORSHARP_NATIVE_VARIANT=${VARIANT}"
    "-DTENSORSHARP_NATIVE_RID=${RID}"
    "-DTENSORSHARP_SOURCE_COMMIT=${SOURCE_COMMIT}"
    "-DTENSORSHARP_BUILD_VERSION=${VERSION}"
)
case "${RID}/${VARIANT}" in
    osx-arm64/metal)
        SCRIPT="${NATIVE_DIR}/build-macos.sh"
        ENTRY="libGgmlOps.dylib"
        CMAKE_ARGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_DEPLOYMENT_TARGET}")
        ;;
    linux-*/cpu|linux-*/vulkan|linux-*/cuda13)
        SCRIPT="${NATIVE_DIR}/build-linux.sh"
        ENTRY="libGgmlOps.so"
        case "${VARIANT}" in
            cpu) BUILD_ARGS+=(--no-cuda --no-vulkan) ;;
            vulkan) BUILD_ARGS+=(--no-cuda --vulkan) ;;
            cuda13) BUILD_ARGS+=(--cuda --no-vulkan) ;;
        esac
        # Libraries shipped beside the bridge resolve from its own directory,
        # never from a path on the build machine.
        CMAKE_ARGS+=(-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN')
        ;;
    *)
        echo "error: no release variant ${VARIANT} for ${RID}" >&2
        exit 2
        ;;
esac

BUILD_DIR="${REPO_ROOT}/artifacts/native-build/${RID}-${VARIANT}"
rm -rf "${BUILD_DIR}"
echo "Building GgmlOps ${VERSION} ${RID}/${VARIANT} from ${SOURCE_COMMIT} (ggml $(tr -d '[:space:]' < "${SCRIPT_DIR}/ggml-revision"))"
TENSORSHARP_GGML_NATIVE_BUILD_DIR="${BUILD_DIR}" TENSORSHARP_GGML_NATIVE_BUILD_TESTS=OFF \
    bash "${SCRIPT}" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} "${CMAKE_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

GGML_HEAD="$(git -C "${REPO_ROOT}/ExternalProjects/ggml" rev-parse HEAD)"
if [[ "${GGML_HEAD}" != "$(tr -d '[:space:]' < "${SCRIPT_DIR}/ggml-revision")" ]]; then
    echo "error: ExternalProjects/ggml is at ${GGML_HEAD}, not the pinned revision." >&2
    exit 1
fi

STAGE="${OUT}/runtimes/${RID}/native/${VARIANT}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/licenses"
cp "${BUILD_DIR}/${ENTRY}" "${STAGE}/${ENTRY}"
cp "${REPO_ROOT}/LICENSE" "${STAGE}/licenses/TensorSharp-LICENSE.txt"
cp "${REPO_ROOT}/ExternalProjects/ggml/LICENSE" "${STAGE}/licenses/ggml-LICENSE.txt"

if [[ -n "${REDIST_DIR}" ]]; then
    compgen -G "${REDIST_DIR}/licenses/*" >/dev/null || {
        echo "error: ${REDIST_DIR}/licenses holds no license or notice files." >&2
        exit 1
    }
    find "${REDIST_DIR}" -maxdepth 1 -type f -exec cp {} "${STAGE}/" \;
    cp "${REDIST_DIR}/licenses/"* "${STAGE}/licenses/"
fi
for forbidden in libcuda.so libcuda.so.1 nvcuda.dll; do
    [[ ! -e "${STAGE}/${forbidden}" ]] || { echo "error: the NVIDIA driver library ${forbidden} must not ship." >&2; exit 1; }
done

# The build record sits outside the artifact directory; the packer copies it
# into the artifact manifest.
RECORD="${OUT}/build/${RID}-${VARIANT}"
rm -rf "${RECORD}"
mkdir -p "${RECORD}"
cat > "${RECORD}/build-identity.json" <<JSON
{
  "tensorSharpBuild": "${VERSION}",
  "sourceCommit": "${SOURCE_COMMIT}",
  "ggmlCommit": "${GGML_HEAD}",
  "nativeAbi": "$(sed -n 's/^TENSORSHARP_NATIVE_ABI:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt")",
  "rid": "${RID}",
  "variant": "${VARIANT}",
  "cpuProfile": "portable",
  "macosDeploymentTarget": $([[ "${RID}" == osx-* ]] && echo "\"${MACOS_DEPLOYMENT_TARGET}\"" || echo null),
  "host": "$(uname -srm)",
  "compiler": "$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)",
  "builtAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
grep -E '^(GGML_NATIVE|GGML_CPU_ARM_ARCH|GGML_SSE42|GGML_AVX|GGML_AVX2|GGML_FMA|GGML_F16C|GGML_BMI2|GGML_OPENMP|GGML_METAL|GGML_METAL_EMBED_LIBRARY|GGML_CUDA|GGML_VULKAN|CMAKE_OSX_DEPLOYMENT_TARGET|CMAKE_CUDA_ARCHITECTURES|CMAKE_CXX_COMPILER|CMAKE_INSTALL_RPATH|CMAKE_BUILD_WITH_INSTALL_RPATH)(:[A-Z]+)?=' \
    "${BUILD_DIR}/CMakeCache.txt" > "${RECORD}/cmake-settings.txt" || true
echo "Staged ${STAGE}"
ls -l "${STAGE}"
