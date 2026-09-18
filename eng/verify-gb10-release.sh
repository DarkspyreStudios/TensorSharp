#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
ulimit -c 0

ARCHIVES="$(realpath "${1:?Usage: verify-gb10-release.sh ARCHIVES OUTPUT_DIR [--require-driver|--discard-after-check]}")"
OUTPUT_DIR="${2:?Specify a fresh extraction directory}"
REQUIRE_DRIVER=0
DISCARD_AFTER_CHECK=0
case "${3:-}" in
    "") ;;
    --require-driver) REQUIRE_DRIVER=1 ;;
    --discard-after-check) DISCARD_AFTER_CHECK=1 ;;
    *) echo "Unknown verification option: $3" >&2; exit 1 ;;
esac
PROBE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Gb10ReleaseProbe.dll"
[[ -s "$PROBE" ]] || { echo "Missing runtime verification hook: $PROBE" >&2; exit 1; }
(cd "$ARCHIVES" && sha256sum -c SHA256SUMS)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
unset LD_LIBRARY_PATH DOTNET_ROOT DOTNET_ROOT_ARM64 DOTNET_ADDITIONAL_DEPS DOTNET_SHARED_STORE

for entry in cli:TensorSharp.Cli server:TensorSharp.Server.Host; do
    app="${entry%%:*}"
    project="${entry#*:}"
    archives=("$ARCHIVES"/tensorsharp-"$app"-*-linux-arm64-cuda13-GB10.tar.gz)
    [[ ${#archives[@]} == 1 ]] || { echo "Expected exactly one $app GB10 archive." >&2; exit 1; }
    target="$OUTPUT_DIR/$app"
    mkdir "$target"
    tar --no-same-owner -xzf "${archives[0]}" -C "$target"

    while IFS= read -r -d '' file; do
        [[ "$(head -c 4 "$file" | od -An -tx1 | tr -d '[:space:]')" == 7f454c46 ]] || continue
        readelf -h "$file" | grep -q 'Machine:.*AArch64' \
            || { echo "Non-ARM64 native binary: $file" >&2; exit 1; }
        dependencies="$(ldd "$file")"
        missing="$(awk '$2 == "=>" && $3 == "not" && $4 == "found" {print $1}' <<< "$dependencies")"
        # CoreCLR deliberately tolerates absent LTTng tracing dependencies.
        if [[ "${file##*/}" == libcoreclrtraceptprovider.so && "$missing" == *liblttng-ust.so.0* ]]; then
            echo "NOTE: optional .NET LTTng tracing is unavailable (liblttng-ust.so.0)."
            missing="$(sed '/^liblttng-ust\.so\.0$/d' <<< "$missing")"
        fi
        if [[ "$REQUIRE_DRIVER" == 0 ]]; then
            missing="$(sed '/^libcuda\.so\.1$/d' <<< "$missing")"
        fi
        [[ -z "$missing" ]] || { printf 'Missing dependencies for %s:\n%s\n' "$file" "$missing" >&2; exit 1; }
    done < <(find "$target" -type f -print0)

    # Run the real apphost from elsewhere so the current directory cannot hide loader bugs.
    if ! (cd /tmp && DOTNET_STARTUP_HOOKS="$PROBE" GB10_REQUIRE_DRIVER="$REQUIRE_DRIVER" \
        "$target/$project" --help) > "$OUTPUT_DIR/$app-startup.log" 2>&1; then
        cat "$OUTPUT_DIR/$app-startup.log" >&2
        exit 1
    fi
    grep -Fq 'GB10 native dependency check passed.' "$OUTPUT_DIR/$app-startup.log"
    echo "PASS: $app archive, ARM64 dependencies and self-contained apphost"
    if [[ "$DISCARD_AFTER_CHECK" == 1 ]]; then
        rm -rf "$target"
    fi
done
if [[ "$REQUIRE_DRIVER" == 0 ]]; then
    echo "Headless check: CUDA driver absence is allowed; GPU execution was not tested."
fi
