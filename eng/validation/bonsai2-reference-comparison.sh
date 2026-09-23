#!/usr/bin/env bash
# =============================================================================
# bonsai2-reference-comparison.sh   (run on the A5000 CUDA VM)
#
# Reference-engine comparison for the Bonsai2 / DiffusionGemma work. It prints
# one "RESULT <name> PASS|FAIL <detail>" line per check and writes one
# machine-readable summary.json.
#
#   PHASE 0  preflight: inputs, revisions, ggml cleanliness, exclusive GPU
#   PHASE 1  PROVES the negative claims instead of asserting them - reads the
#            GGUF headers, greps the upstream sources, and actually tries to
#            load Bonsai2 PQ2_0 / PTQ1_0 and DiffusionGemma Q4_K_M with the
#            already-built upstream llama.cpp, keeping the exact refusal text;
#            records whether vLLM / SGLang exist on this host at all
#   PHASE 2  builds PrismML-Eng/llama.cpp at the pinned commit in a SEPARATE
#            tree; TensorSharp's ExternalProjects/ggml is never touched and its
#            cleanliness is asserted before and after (AGENTS.md)
#   PHASE 3  model-only throughput: llama-bench vs ParityHarness --bench
#   PHASE 4  HTTP single (c=1) and parallel (c=4) requests on the SAME GGUF,
#            same prompts, greedy, same budget, 1 discarded warmup, 3 timed
#            repeats, engine order rotated between rounds; then an untimed
#            projector/vision smoke on separate servers
#   PHASE 5  one summary.json with prefill/decode tok/s per engine and
#            concurrency plus ratios, and a hashed inventory
#
# Nothing here is a model-quality score. Matching greedy text is a regression
# check; every throughput cell is this host, this model, these flags.
#
# Usage:  bash bonsai2-reference-comparison.sh
#         PHASES="0 1" bash bonsai2-reference-comparison.sh
#         ROUNDS=1 TOKENS=64 bash bonsai2-reference-comparison.sh
# =============================================================================
set -uo pipefail
export LC_ALL=C
export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
export TENSORSHARP_GGML_NO_UPDATE=1

TS_ROOT=${TS_ROOT:-/root/TensorSharp}
MODELS=${MODELS:-/root/models}
PQ2=$MODELS/Ternary-Bonsai-2-27B-PQ2_0.gguf
PTQ1=$MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf
MMPROJ=$MODELS/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
DG=$MODELS/diffusiongemma-26B-A4B-it-Q4_K_M.gguf

UPSTREAM=${UPSTREAM:-/root/llama.cpp}
UPSTREAM_MARKER=${UPSTREAM_MARKER:-/root/ts-logs/stage3-llamacpp.log}
PRISM_DIR=${PRISM_DIR:-/root/llama.cpp-prism}
PRISM_URL=${PRISM_URL:-https://github.com/PrismML-Eng/llama.cpp}
PRISM_COMMIT=${PRISM_COMMIT:-bdc23b56b4458b9f1655aec5287f3ab56ee8daaa}

PARITY_DLL=$TS_ROOT/benchmarks/ParityHarness/bin/Release/net10.0/ParityHarness.dll
SERVER_DLL=$TS_ROOT/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll
BENCH_PY=$TS_ROOT/eng/validation/bonsai2-bench.py
NATIVE_LIB=$TS_ROOT/TensorSharp.GGML.Native/build/libGgmlOps.so

STAMP=$(date -u +%Y%m%d-%H%M%SZ)
RUN=${RUN:-$TS_ROOT/artifacts/bonsai2-refbench-$STAMP}   # artifacts/ is gitignored
LOGS=$RUN/logs
EV=$RUN/evidence
RESULTS=$RUN/results.txt

PHASES=${PHASES:-"0 1 2 3 4 5"}
ROUNDS=${ROUNDS:-2}
TOKENS=${TOKENS:-64}
REPEATS=${REPEATS:-3}
CONCURRENCY=${CONCURRENCY:-1,4}
ON_PAR=${ON_PAR:-0.95}                 # ts/reference ratio still counted "on par"
MIN_FREE_KIB=${MIN_FREE_KIB:-7340032}  # 7 GiB required before the fork build
BUILD_JOBS=${BUILD_JOBS:-32}
LLAMA_THREADS=${LLAMA_THREADS:-8}
MAX_CTX=${MAX_CTX:-2048}               # per sequence / per llama.cpp slot
SLOTS=${SLOTS:-4}
PHYS_BATCH=${PHYS_BATCH:-512}
GPU_WAIT=${GPU_WAIT:-7200}
GPU_OK=1

mkdir -p "$LOGS" "$EV" || exit 1
exec 9>/root/.bonsai2-refbench.lock
if ! flock -w "${LOCK_WAIT:-10800}" 9; then
    echo "RESULT lock FAIL another bonsai2-refbench run held /root/.bonsai2-refbench.lock for ${LOCK_WAIT:-10800}s"
    exit 1
fi
: > "$RESULTS"
exec > >(tee -a "$LOGS/run.log") 2>&1

say()  { printf '\n========== %s  [%s] ==========\n' "$*" "$(date -u +%H:%M:%SZ)"; }
note() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# RESULT lines are this script's machine-readable contract.
result() {
    local name=$1 status=$2
    shift 2
    printf 'RESULT %s %s %s\n' "$name" "$status" "$*" | tee -a "$RESULTS"
}

have_phase() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }
gpu_pids()    { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$'; }
gpu_mem_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
free_kib()    { df -Pk "$1" | awk 'NR==2 {print $4}'; }
oneline()     { tr '\n' ';' | tr -s ' ' | head -c "${1:-240}"; }

# Wait for a sibling stage of this campaign to publish its completion marker.
wait_marker() {
    local file=$1 marker=$2 secs=$3 deadline
    if [ ! -f "$file" ]; then note "marker file $file absent; not waiting"; return 0; fi
    deadline=$(( $(date +%s) + secs ))
    while ! grep -q "$marker" "$file" 2>/dev/null; do
        [ "$(date +%s)" -ge "$deadline" ] && return 1
        sleep 30
    done
    return 0
}

# A competing GPU job invalidates every number below, so this never proceeds on
# a busy device - it fails the run instead.
wait_gpu_idle() {
    local deadline=$(( $(date +%s) + GPU_WAIT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if [ -z "$(gpu_pids)" ]; then
            sleep 15
            [ -z "$(gpu_pids)" ] && return 0
        fi
        note "GPU busy (pids: $(gpu_pids | tr '\n' ' ')); waiting"
        sleep 30
    done
    return 1
}

free_port() {
    local port
    for port in "$@"; do
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$port\$" || { echo "$port"; return 0; }
    done
    return 1
}

SERVER_PID=""
SERVER_LABEL=""
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    note "stopping $SERVER_LABEL (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null
    local i
    for i in $(seq 1 90); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""; SERVER_LABEL=""
    for i in $(seq 1 60); do [ -z "$(gpu_pids)" ] && break; sleep 2; done
    sleep 5
}
trap 'stop_server' EXIT INT TERM

# Ready = liveness answers AND one real 1-token completion comes back.
wait_ready() {
    local url=$1 secs=$2 label=$3 deadline
    deadline=$(( $(date +%s) + secs ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            note "$label exited before becoming ready"; return 2
        fi
        if curl -fsS -m 5 "$url/health" >/dev/null 2>&1 &&
           curl -fsS -m 180 -H 'Content-Type: application/json' \
                -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"stream":false}' \
                "$url/v1/chat/completions" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

# This VM is shared. Any process whose cwd or command line lives under a path is
# treated as a foreign owner of it: never delete or rebuild underneath one.
foreign_pids() {
    local path=$1 pid target
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        target=$(readlink "/proc/$pid/cwd" 2>/dev/null)
        case "$target" in "$path"|"$path"/*) echo "$pid"; continue;; esac
        tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" | grep -q -- "$path" && echo "$pid"
    done | sort -u
}

# A readable, bounded description: the first few owners and how many more.
describe_pids() {
    local pid count=0
    for pid in $1; do
        count=$(( count + 1 ))
        if [ "$count" -le 4 ]; then
            printf '%s[%s] ' "$pid" "$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" | cut -c1-48)"
        fi
    done
    [ "$count" -gt 4 ] && printf '+%d more' "$(( count - 4 ))"
    return 0
}

binary_stamp() {
    stat -c '%n:%i:%Y:%s' "$PRISM_DIR/build/bin/llama-server" \
        "$PRISM_DIR/build/bin/llama-bench" 2>/dev/null | oneline 300
}

# ---------------------------------------------------------------- phase 0 ----
phase0_preflight() {
    say "PHASE 0  preflight"
    local missing="" f
    for f in "$PQ2" "$PTQ1" "$MMPROJ" "$DG" "$PARITY_DLL" "$SERVER_DLL" "$BENCH_PY"; do
        [ -e "$f" ] || missing="$missing $f"
    done
    if [ -n "$missing" ]; then
        result preflight-inputs FAIL "missing:$missing"
        return 2
    fi
    result preflight-inputs PASS "models, ParityHarness, Server.Host and bonsai2-bench.py all present"

    nvidia-smi > "$LOGS/nvidia-smi.txt" 2>&1
    df -h / /dev/shm > "$LOGS/df-start.txt" 2>&1
    {
        echo "tensorsharp_commit=$(git -C "$TS_ROOT" rev-parse HEAD 2>/dev/null)"
        echo "tensorsharp_dirty_files=$(git -C "$TS_ROOT" status --porcelain 2>/dev/null | wc -l)"
        echo "ggml_commit=$(git -C "$TS_ROOT/ExternalProjects/ggml" rev-parse HEAD 2>/dev/null)"
        echo "upstream_llamacpp_commit=$(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null)"
        echo "prism_commit_pinned=$PRISM_COMMIT"
        echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
        echo "cpus=$(nproc)"
        echo "dotnet=$(dotnet --version 2>/dev/null)"
        echo "native_lib=$(ls -l "$NATIVE_LIB" 2>/dev/null)"
    } | tee "$EV/environment.txt"

    # AGENTS.md: a TensorSharp build must not depend on a modified ggml tree.
    local dirty
    dirty=$(git -C "$TS_ROOT/ExternalProjects/ggml" status --porcelain 2>&1 | head -20)
    if [ -z "$dirty" ]; then
        result ggml-dependency-unmodified PASS \
            "ExternalProjects/ggml clean at $(git -C "$TS_ROOT/ExternalProjects/ggml" rev-parse HEAD 2>/dev/null)"
    else
        result ggml-dependency-unmodified FAIL "ggml checkout dirty: $(printf '%s' "$dirty" | oneline)"
    fi

    wait_marker /root/ts-logs/stage5.log STAGE5-DONE 10800 \
        || note "stage5 marker not seen within 3h; falling back to the GPU check"
    wait_marker "$UPSTREAM_MARKER" STAGE3-LLAMACPP-DONE 5400 \
        || note "stage3 marker not seen within 90m; phase 1 checks the binaries directly"
    if wait_gpu_idle; then
        result preflight-gpu-exclusive PASS "no other CUDA compute process; $(gpu_mem_used) MiB in use"
        return 0
    fi
    result preflight-gpu-exclusive FAIL \
        "GPU still busy after ${GPU_WAIT}s: $(describe_pids "$(gpu_pids)") - timed phases are skipped rather than measured against a competing job on this shared VM"
    return 1
}

# ---------------------------------------------------------------- phase 1 ----
write_gguf_reader() {
    cat > "$RUN/gguf_types.py" <<'PY'
#!/usr/bin/env python3
"""Report each GGUF's architecture and the distinct tensor type ids it stores.

Header only - no tensor payload is read - so this evidence is independent of
any engine's loader and of the wording of any engine's error message."""
import json
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
          7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, stream):
        self.stream = stream

    def take(self, count):
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("GGUF header truncated")
        return data

    def scalar(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.scalar("<Q")).decode("utf-8", "replace")

    def value(self, kind):
        if kind in SCALAR:
            return self.scalar(SCALAR[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element = self.scalar("<I")
            count = self.scalar("<Q")
            return [self.value(element) for _ in range(count)]
        raise ValueError("unknown GGUF value type %d" % kind)


def describe(path):
    with open(path, "rb") as stream:
        reader = Reader(stream)
        if reader.take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = reader.scalar("<I")
        tensors = reader.scalar("<Q")
        pairs = reader.scalar("<Q")
        metadata = {}
        for _ in range(pairs):
            key = reader.string()
            value = reader.value(reader.scalar("<I"))
            if isinstance(value, list) and len(value) > 8:
                value = {"array_length": len(value), "first": value[:4]}
            metadata[key] = value
        counts = {}
        for _ in range(tensors):
            reader.string()
            for _ in range(reader.scalar("<I")):
                reader.scalar("<Q")
            kind = reader.scalar("<I")
            reader.scalar("<Q")
            counts[kind] = counts.get(kind, 0) + 1
    return {"path": path, "gguf_version": version, "tensor_count": tensors,
            "architecture": metadata.get("general.architecture"),
            "file_type": metadata.get("general.file_type"),
            "tensor_type_ids": {str(key): value for key, value in sorted(counts.items())}}


if __name__ == "__main__":
    report = []
    for path in sys.argv[1:]:
        try:
            report.append(describe(path))
        except Exception as error:          # a malformed header is evidence too
            report.append({"path": path, "error": repr(error)})
    print(json.dumps(report, indent=2))
PY
}

# Load one model with the upstream binary and classify what actually came back.
upstream_probe() {
    local name=$1 model=$2 log=$LOGS/upstream-$1.log rc extra="" text specific generic reason
    "$UPSTREAM/build/bin/llama-cli" --help 2>&1 | grep -q -- '-no-cnv'     && extra="$extra -no-cnv"
    "$UPSTREAM/build/bin/llama-cli" --help 2>&1 | grep -q -- '--no-warmup' && extra="$extra --no-warmup"
    note "upstream llama-cli load probe: $model"
    # shellcheck disable=SC2086
    timeout -k 30 420 "$UPSTREAM/build/bin/llama-cli" \
        -m "$model" -ngl 0 -c 256 -n 1 -p "hi" $extra > "$log" 2>&1
    rc=$?
    text=$(tr -d '\000' < "$log" | tr '\r' '\n' | tail -120)
    if printf '%s' "$text" | grep -qiE 'unknown (argument|option)|invalid argument|unrecognized'; then
        result "upstream-load-$name" FAIL \
            "this llama-cli rejected the probe flags, so no load was attempted: $(printf '%s' "$text" | grep -iE 'unknown (argument|option)|invalid argument|unrecognized' | head -1 | oneline 160)"
        return 1
    fi
    if [ $rc -eq 0 ]; then
        result "upstream-load-$name" FAIL \
            "upstream llama.cpp LOADED $(basename "$model") (rc=0) - the unsupported-format claim is wrong and upstream must be added to the comparison"
        return 1
    fi
    # Prefer the line that names the actual cause over the generic wrapper.
    specific=$(printf '%s' "$text" | grep -iE 'invalid ggml type|unknown model architecture|unknown tensor type|unsupported|not supported|GGML_ASSERT' | sed -E 's/^.* E //' | awk '!seen[$0]++' | head -1 | oneline 200)
    generic=$(printf '%s' "$text" | grep -iE 'error loading model|failed to load model|failed to read tensor info|llama_model_load' | sed -E 's/^.* E //' | awk '!seen[$0]++' | head -1 | oneline 140)
    reason=$(printf '%s' "${specific:+$specific -- }$generic" | oneline 320)
    if [ -n "$reason" ]; then
        result "upstream-load-$name" PASS "rc=$rc refusal: $reason"
        return 0
    fi
    result "upstream-load-$name" FAIL \
        "inconclusive rc=$rc, no recognizable load error; tail: $(printf '%s' "$text" | tail -2 | oneline 160)"
    return 1
}

phase1_negative_claims() {
    say "PHASE 1  empirical negative claims (upstream llama.cpp, vLLM, SGLang)"
    write_gguf_reader
    timeout -k 30 900 python3 "$RUN/gguf_types.py" "$PQ2" "$PTQ1" "$DG" > "$EV/gguf-headers.json" 2>&1
    cat "$EV/gguf-headers.json"

    local type_count pq2_types ptq1_types dg_arch arch_hits
    type_count=$(grep -oE 'GGML_TYPE_COUNT[[:space:]]*=[[:space:]]*[0-9]+' \
        "$UPSTREAM/ggml/include/ggml.h" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "upstream_GGML_TYPE_COUNT=$type_count" >> "$EV/environment.txt"
    pq2_types=$(python3  -c 'import json,sys;d=json.load(open(sys.argv[1]));print(",".join(d[0].get("tensor_type_ids",{})))' "$EV/gguf-headers.json" 2>/dev/null)
    ptq1_types=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(",".join(d[1].get("tensor_type_ids",{})))' "$EV/gguf-headers.json" 2>/dev/null)
    dg_arch=$(python3    -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d[2].get("architecture"))'                  "$EV/gguf-headers.json" 2>/dev/null)

    if [ -n "$type_count" ] && printf '%s' "$pq2_types" | grep -q '142' \
                            && printf '%s' "$ptq1_types" | grep -q '143'; then
        result gguf-tensor-types PASS \
            "PQ2_0 stores tensor type id 142 and PTQ1_0 stores 143; upstream ggml.h declares GGML_TYPE_COUNT=$type_count, so neither id exists upstream at all"
    else
        result gguf-tensor-types FAIL \
            "header read did not confirm the claim (pq2=[$pq2_types] ptq1=[$ptq1_types] upstream_count=$type_count)"
    fi

    arch_hits=$(grep -ril 'diffusion.gemma' "$UPSTREAM/src" "$UPSTREAM/gguf-py" 2>/dev/null | head -5 | oneline 200)
    if [ "$dg_arch" = "diffusion-gemma" ] && [ -z "$arch_hits" ]; then
        result gguf-arch-diffusiongemma PASS \
            "GGUF general.architecture='diffusion-gemma'; that architecture appears nowhere in upstream llama.cpp src/ or gguf-py/"
    else
        result gguf-arch-diffusiongemma FAIL "arch='$dg_arch'; upstream hits: ${arch_hits:-none}"
    fi

    if [ -x "$UPSTREAM/build/bin/llama-cli" ]; then
        result upstream-present PASS \
            "$UPSTREAM at $(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null), llama-cli built with CUDA"
        upstream_probe bonsai2-pq2 "$PQ2"
        upstream_probe bonsai2-ptq1 "$PTQ1"
        upstream_probe diffusiongemma-q4km "$DG"
    else
        result upstream-present FAIL \
            "$UPSTREAM/build/bin/llama-cli missing (stage3 build incomplete), so the upstream refusals could not be demonstrated on this host"
    fi

    # vLLM / SGLang: record what is installed rather than asserting absence.
    local vllm_state sglang_state engine state
    vllm_state=$(timeout -k 10 300 python3   -c 'import vllm;   print("present", getattr(vllm,   "__version__", "?"))' 2>&1 | tail -1 | oneline 160)
    sglang_state=$(timeout -k 10 300 python3 -c 'import sglang; print("present", getattr(sglang, "__version__", "?"))' 2>&1 | tail -1 | oneline 160)
    {
        echo "python: $(python3 --version 2>&1)"
        echo "import vllm:   $vllm_state"
        echo "import sglang: $sglang_state"
        echo "pip:   $(pip3 list 2>/dev/null | grep -Ei '^(vllm|sglang)[[:space:]]' | tr '\n' ' ')"
        echo "which: $(command -v vllm sglang 2>/dev/null | tr '\n' ' ')"
        echo "non-GGUF weights in $MODELS: $(ls "$MODELS" 2>/dev/null | grep -viE '\.gguf$' | tr '\n' ' ')"
        echo "(both engines load HF-format checkpoints or a narrow GGUF subset; only publisher GGUFs exist here)"
    } > "$EV/vllm-sglang-probe.txt"
    cat "$EV/vllm-sglang-probe.txt"
    for engine in vllm sglang; do
        case $engine in vllm) state=$vllm_state;; sglang) state=$sglang_state;; esac
        if printf '%s' "$state" | grep -q '^present'; then
            result "$engine-availability" FAIL \
                "$engine IS importable here ($state); the infeasibility claim must be re-examined against its actual format support"
        else
            result "$engine-availability" PASS "not installed on this host: $state"
        fi
    done
    return 0
}

# ---------------------------------------------------------------- phase 2 ----
phase2_build_prism() {
    say "PHASE 2  publisher fork at $PRISM_COMMIT (separate tree, shared host)"

    if [ -d "$UPSTREAM/build" ] && [ -z "$(foreign_pids "$UPSTREAM")" ]; then
        local before after
        before=$(free_kib /root)
        find "$UPSTREAM/build" -name '*.o' -delete 2>/dev/null
        after=$(free_kib /root)
        note "pruned upstream object files: $(( (after - before) / 1024 )) MiB reclaimed (binaries kept)"
    else
        note "leaving $UPSTREAM alone (another process is using it, or no build dir)"
    fi

    local owners head free_now
    owners=$(foreign_pids "$PRISM_DIR")

    # (a) already built at the pinned commit by anyone: reuse it.
    if prism_ready; then
        result prism-checkout PASS "reusing $PRISM_DIR at $PRISM_COMMIT (built by this host already)"
        prism_verify_binaries
        return $?
    fi

    # (b) somebody else is building it right now: wait, do not duplicate or delete.
    if [ -n "$owners" ]; then
        note "another process owns $PRISM_DIR: $(describe_pids "$owners")"
        local deadline=$(( $(date +%s) + ${FOREIGN_BUILD_WAIT:-7200} ))
        while [ "$(date +%s)" -lt "$deadline" ]; do
            prism_ready && break
            sleep 60
        done
        if prism_ready; then
            result prism-checkout PASS \
                "waited for a concurrent build by another session; $PRISM_DIR at $PRISM_COMMIT"
            prism_verify_binaries
            return $?
        fi
        result prism-build FAIL \
            "another session has been building $PRISM_DIR for ${FOREIGN_BUILD_WAIT:-7200}s without producing llama-server/llama-bench at $PRISM_COMMIT: $(describe_pids "$(foreign_pids "$PRISM_DIR")")"
        return 1
    fi

    # (c) nobody owns it: take the build lock and do it ourselves.
    exec 8>/root/.prism-build.lock
    if ! flock -w "${FOREIGN_BUILD_WAIT:-7200}" 8; then
        result prism-build FAIL "could not take /root/.prism-build.lock"
        return 1
    fi
    prism_ready && { result prism-checkout PASS "reusing $PRISM_DIR at $PRISM_COMMIT"; prism_verify_binaries; return $?; }

    free_now=$(free_kib /root)
    if [ "$free_now" -lt "$MIN_FREE_KIB" ]; then
        result disk-budget FAIL \
            "$(( free_now / 1024 )) MiB free on /root, need $(( MIN_FREE_KIB / 1024 )) MiB for the fork build"
        return 1
    fi
    result disk-budget PASS "$(( free_now / 1024 )) MiB free before the fork build"

    if [ "$(git -C "$PRISM_DIR" rev-parse HEAD 2>/dev/null)" != "$PRISM_COMMIT" ]; then
        if [ -n "$(foreign_pids "$PRISM_DIR")" ]; then
            result prism-checkout FAIL "refusing to replace $PRISM_DIR: another process appeared in it"
            return 1
        fi
        rm -rf "$PRISM_DIR"
        mkdir -p "$PRISM_DIR" || return 1
        git -C "$PRISM_DIR" init -q .
        git -C "$PRISM_DIR" remote add origin "$PRISM_URL"
        timeout -k 60 1800 git -C "$PRISM_DIR" fetch --depth 1 origin "$PRISM_COMMIT" \
            > "$LOGS/prism-fetch.log" 2>&1
        git -C "$PRISM_DIR" checkout -q FETCH_HEAD >> "$LOGS/prism-fetch.log" 2>&1
    fi
    head=$(git -C "$PRISM_DIR" rev-parse HEAD 2>/dev/null)
    if [ "$head" = "$PRISM_COMMIT" ]; then
        result prism-checkout PASS \
            "$PRISM_DIR at $head ($(git -C "$PRISM_DIR" log --oneline -1 2>/dev/null | oneline 120))"
    else
        result prism-checkout FAIL "HEAD=$head expected $PRISM_COMMIT; see $LOGS/prism-fetch.log"
        return 1
    fi

    note "configuring and building the fork (expect 30-60 minutes)"
    timeout -k 120 1800 cmake -S "$PRISM_DIR" -B "$PRISM_DIR/build" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release > "$LOGS/prism-cmake.log" 2>&1
    if [ $? -ne 0 ]; then
        result prism-build FAIL "cmake configure failed: $(tail -4 "$LOGS/prism-cmake.log" | oneline)"
        return 1
    fi
    timeout -k 300 7200 cmake --build "$PRISM_DIR/build" --config Release \
        -j "$BUILD_JOBS" --target llama-cli llama-server llama-bench \
        > "$LOGS/prism-make.log" 2>&1
    if [ $? -ne 0 ]; then
        result prism-build FAIL "compile failed: $(grep -iE ' error|Error [0-9]|Fatal error' "$LOGS/prism-make.log" | head -3 | oneline)"
        return 1
    fi
    find "$PRISM_DIR/build" -name '*.o' -delete 2>/dev/null
    prism_verify_binaries
    return $?
}

prism_ready() {
    [ -x "$PRISM_DIR/build/bin/llama-server" ] && [ -x "$PRISM_DIR/build/bin/llama-bench" ] &&
    [ -x "$PRISM_DIR/build/bin/llama-cli" ] &&
    [ "$(git -C "$PRISM_DIR" rev-parse HEAD 2>/dev/null)" = "$PRISM_COMMIT" ]
}

# The binaries must actually run: a half-linked tree is not a reference engine.
prism_verify_binaries() {
    if ! prism_ready; then
        result prism-build FAIL "expected binaries missing, or checkout is not $PRISM_COMMIT"
        return 1
    fi
    if ! timeout -k 15 120 "$PRISM_DIR/build/bin/llama-bench" --help > "$LOGS/prism-bench-help.log" 2>&1; then
        result prism-build FAIL "llama-bench does not run: $(tail -3 "$LOGS/prism-bench-help.log" | oneline)"
        return 1
    fi
    du -sh "$PRISM_DIR" "$UPSTREAM" "$TS_ROOT" "$MODELS" 2>/dev/null > "$EV/tree-sizes.txt"
    df -h / >> "$EV/tree-sizes.txt"
    cat "$EV/tree-sizes.txt"
    binary_stamp > "$EV/prism-binary-stamp-before.txt"
    result prism-build PASS \
        "llama-cli/llama-server/llama-bench present and runnable at $PRISM_COMMIT; $(head -3 "$EV/tree-sizes.txt" | oneline 180)"

    if [ -z "$(git -C "$TS_ROOT/ExternalProjects/ggml" status --porcelain 2>&1)" ]; then
        result ggml-dependency-unmodified-after-build PASS \
            "ExternalProjects/ggml still clean at $(git -C "$TS_ROOT/ExternalProjects/ggml" rev-parse HEAD 2>/dev/null); the fork lives only in $PRISM_DIR"
    else
        result ggml-dependency-unmodified-after-build FAIL "ExternalProjects/ggml became dirty during the fork build"
    fi
    return 0
}

# ---------------------------------------------------------------- phase 3 ----
phase3_model_only() {
    say "PHASE 3  model-only throughput (llama-bench vs ParityHarness --bench)"
    if [ "$GPU_OK" != 1 ]; then
        result modelonly-skipped FAIL "GPU was not exclusive; model-only benchmarks were not run"
        return 1
    fi
    if [ ! -x "$PRISM_DIR/build/bin/llama-bench" ]; then
        result modelonly-prism FAIL "no llama-bench in $PRISM_DIR (phase 2 did not complete)"
    else
        wait_gpu_idle || { result modelonly-prism FAIL "GPU busy"; return 1; }
        note "llama-bench on the publisher fork"
        timeout -k 120 5400 "$PRISM_DIR/build/bin/llama-bench" \
            -m "$PQ2" -p 128,512 -n "$TOKENS" -d 0,128 -r "$REPEATS" \
            -ngl 99 -fa on -ctk f16 -ctv f16 -b "$PHYS_BATCH" -ub "$PHYS_BATCH" \
            -t "$LLAMA_THREADS" -o json > "$EV/prism-bench.json" 2> "$LOGS/prism-bench.log"
        if [ $? -eq 0 ] && [ -s "$EV/prism-bench.json" ]; then
            result modelonly-prism PASS \
                "$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(len(d),"cells,",d[0]["build_commit"],"fa=%s t=%s ub=%s"%(d[0]["flash_attn"],d[0]["n_threads"],d[0]["n_ubatch"]))' "$EV/prism-bench.json" 2>/dev/null || echo 'json written')"
        else
            result modelonly-prism FAIL "llama-bench failed: $(tail -4 "$LOGS/prism-bench.log" | oneline)"
        fi
    fi

    local depth rc cells
    for depth in 0 128; do
        wait_gpu_idle || { result "modelonly-tensorsharp-d$depth" FAIL "GPU busy"; continue; }
        note "ParityHarness --bench ggml_cuda depth=$depth"
        MAX_CONTEXT=$MAX_CTX KV_CACHE_DTYPE=f16 TS_PREFILL_CHUNK=$PHYS_BATCH \
        timeout -k 120 7200 dotnet "$PARITY_DLL" "$PQ2" --bench ggml_cuda \
            128,512 "$TOKENS" "$REPEATS" "$depth" > "$EV/ts-bench-d$depth.log" 2>&1
        rc=$?
        cells=$(grep -c '\[bench-statistics\]' "$EV/ts-bench-d$depth.log")
        if [ "$rc" -eq 0 ] && [ "$cells" -eq 3 ]; then
            result "modelonly-tensorsharp-d$depth" PASS "3 statistics cells, $REPEATS repeats each, f16 KV, chunk $PHYS_BATCH"
        else
            result "modelonly-tensorsharp-d$depth" FAIL \
                "rc=$rc cells=$cells: $(tail -4 "$EV/ts-bench-d$depth.log" | oneline)"
        fi
    done
    return 0
}

# ---------------------------------------------------------------- phase 4 ----
start_prism_server() {
    local port=$1 log=$2
    shift 2
    "$PRISM_DIR/build/bin/llama-server" -m "$PQ2" -ngl 99 -c $(( MAX_CTX * SLOTS )) \
        -np "$SLOTS" -fa on -ctk f16 -ctv f16 -b "$PHYS_BATCH" -ub "$PHYS_BATCH" \
        -t "$LLAMA_THREADS" --host 127.0.0.1 --port "$port" --jinja "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    SERVER_LABEL="llama-server(prism):$port"
}

start_ts_server() {
    local port=$1 log=$2
    shift 2
    MAX_CONTEXT=$MAX_CTX KV_CACHE_DTYPE=f16 TS_PREFILL_CHUNK=$PHYS_BATCH \
    TS_SCHED_MAX_RUNNING_SEQS=$SLOTS \
    dotnet "$SERVER_DLL" --model "$PQ2" --backend ggml_cuda \
        --host 127.0.0.1 --port "$port" --no-webui --no-prefix-cache \
        --kv-cache-dtype f16 --prefill-chunk-size "$PHYS_BATCH" --max-tokens 256 \
        "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    SERVER_LABEL="TensorSharp.Server.Host:$port"
}

engine_label() { case $1 in prism) echo llama-prism;; ts) echo tensorsharp;; esac; }

# One engine, one round: start, warm, measure, stop.
http_round() {
    local engine=$1 round=$2 port log label detail rc
    port=$(free_port 8083 8085 8086 8087 8088) || { result "http-$engine-r$round" FAIL "no free port"; return 1; }
    log=$LOGS/$engine-server-r$round.log
    label=$(engine_label "$engine")
    wait_gpu_idle || { result "http-$engine-r$round" FAIL "GPU busy before start"; return 1; }

    case $engine in
        prism) start_prism_server "$port" "$log" ;;
        ts)    start_ts_server    "$port" "$log" ;;
    esac
    note "$SERVER_LABEL starting (round $round); log $log"
    if ! wait_ready "http://127.0.0.1:$port" 1800 "$SERVER_LABEL"; then
        result "http-$engine-r$round" FAIL "server never became ready: $(tail -4 "$log" | oneline)"
        stop_server; return 1
    fi
    note "ready; GPU memory in use: $(gpu_mem_used) MiB"
    # Thermal/clock state belongs next to every timed cell.
    echo "$engine round $round vram_mib=$(gpu_mem_used) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>/dev/null)" >> "$EV/vram.txt"

    if [ "$round" = "1" ]; then
        timeout -k 60 3600 python3 "$BENCH_PY" quality \
            --url "http://127.0.0.1:$port" --engine "$label" \
            --model Ternary-Bonsai-2-27B-PQ2_0 --tokens "$TOKENS" \
            --output "$EV/$engine-quality.json" > "$LOGS/$engine-quality.log" 2>&1
        if [ $? -eq 0 ]; then
            result "quality-$engine" PASS "arithmetic, constrained JSON and tool call all passed"
        else
            detail=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("status"),[(s["name"],s["status"]) for s in d.get("scenarios",[])])' "$EV/$engine-quality.json" 2>/dev/null | oneline 240)
            result "quality-$engine" FAIL "${detail:-see $LOGS/$engine-quality.log}"
        fi
    fi

    timeout -k 120 5400 python3 "$BENCH_PY" http \
        --url "http://127.0.0.1:$port" --engine "$label" \
        --model Ternary-Bonsai-2-27B-PQ2_0 --tokens "$TOKENS" --repeats "$REPEATS" \
        --concurrency "$CONCURRENCY" --output "$EV/$engine-http-r$round.json" \
        > "$LOGS/$engine-http-r$round.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        detail=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("; ".join("c%d agg=%.2f tok/s match=%d/%d budget=%d"%(s["concurrency"],s["aggregate_tps_median"],s["matching_requests"],s["total_requests"],s["full_budget_requests"]) for s in d.get("summaries",[])))' "$EV/$engine-http-r$round.json" 2>/dev/null | oneline 240)
        result "http-$engine-r$round" PASS "${detail:-completed}"
    else
        detail=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("status"),d.get("error"))' "$EV/$engine-http-r$round.json" 2>/dev/null | oneline 240)
        result "http-$engine-r$round" FAIL "rc=$rc ${detail:-see $LOGS/$engine-http-r$round.log}"
    fi
    stop_server
    return 0
}

# Vision needs the projector, which changes VRAM and load time, so it runs on
# its own servers and never inside the timed comparison.
vision_round() {
    local engine=$1 port log label detail
    port=$(free_port 8083 8085 8086 8087 8088) || { result "vision-$engine" FAIL "no free port"; return 1; }
    log=$LOGS/$engine-vision-server.log
    label=$(engine_label "$engine")
    wait_gpu_idle || { result "vision-$engine" FAIL "GPU busy"; return 1; }
    case $engine in
        prism) start_prism_server "$port" "$log" --mmproj "$MMPROJ" ;;
        ts)    start_ts_server    "$port" "$log" --mmproj "$MMPROJ" ;;
    esac
    if ! wait_ready "http://127.0.0.1:$port" 1800 "$SERVER_LABEL"; then
        result "vision-$engine" FAIL "server with --mmproj never became ready: $(tail -4 "$log" | oneline)"
        stop_server; return 1
    fi
    timeout -k 60 3600 python3 "$BENCH_PY" quality --vision \
        --url "http://127.0.0.1:$port" --engine "$label" \
        --model Ternary-Bonsai-2-27B-PQ2_0 --tokens "$TOKENS" \
        --output "$EV/$engine-quality-vision.json" > "$LOGS/$engine-vision.log" 2>&1
    if [ $? -eq 0 ]; then
        result "vision-$engine" PASS "generated red-square/blue-circle fixture read correctly with the Q8_0 projector"
    else
        detail=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("status"),[(s["name"],s["status"]) for s in d.get("scenarios",[])])' "$EV/$engine-quality-vision.json" 2>/dev/null | oneline 240)
        result "vision-$engine" FAIL "${detail:-see $LOGS/$engine-vision.log}"
    fi
    stop_server
    return 0
}

phase4_http() {
    say "PHASE 4  HTTP single and parallel requests, engine order rotated"
    if [ "$GPU_OK" != 1 ]; then
        result http-skipped FAIL "GPU was not exclusive; HTTP comparison was not run"
        return 1
    fi
    [ "$ROUNDS" -lt 2 ] && note "ROUNDS=$ROUNDS: engine order is NOT rotated, so order bias is unmitigated"
    local round
    for round in $(seq 1 "$ROUNDS"); do
        if [ $(( round % 2 )) -eq 1 ]; then
            http_round prism "$round"; http_round ts "$round"
        else
            http_round ts "$round"; http_round prism "$round"
        fi
    done
    say "PHASE 4b  projector / vision smoke (separate servers, untimed)"
    vision_round prism
    vision_round ts
    return 0
}

# ---------------------------------------------------------------- phase 5 ----
write_summarizer() {
    cat > "$RUN/summarize.py" <<'PY'
#!/usr/bin/env python3
"""Fold one run's evidence into a single machine-readable summary.

Prints RESULT lines for the cross-engine checks and writes summary.json with
prefill/decode tok/s per engine and concurrency plus the ratios. A missing
input is reported as missing; it is never treated as a pass."""
import json
import os
import re
import statistics
import sys

RUN = sys.argv[1]
ON_PAR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
EV = os.path.join(RUN, "evidence")
ENGINES = {"tensorsharp": "ts", "llama_prism": "prism"}


def read_json(path):
    try:
        with open(path, errors="replace") as stream:
            text = stream.read()
    except OSError as error:
        return {"__error__": repr(error)}
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"[\[{].*[\]}]", text, re.S)   # tolerate leading log noise
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError as error:
                return {"__error__": repr(error)}
        return {"__error__": "unparseable: " + text[:200]}


def fmt(value):
    return "%.3f" % value if isinstance(value, (int, float)) else "n/a"


def cell_key(prompt, gen, depth):
    return ("pp%d" % prompt if gen == 0 else "tg%d" % gen) + "@d%d" % depth


def prism_model_only():
    raw = read_json(os.path.join(EV, "prism-bench.json"))
    if not isinstance(raw, list):
        return {"__error__": (raw or {}).get("__error__", "unexpected shape")}
    cells = {}
    for row in raw:
        cells[cell_key(row.get("n_prompt", 0), row.get("n_gen", 0), row.get("n_depth", 0))] = {
            "avg_ts": row.get("avg_ts"), "stddev_ts": row.get("stddev_ts"),
            "samples_ts": row.get("samples_ts"), "build_commit": row.get("build_commit"),
            "flash_attn": row.get("flash_attn"), "n_threads": row.get("n_threads"),
            "n_batch": row.get("n_batch"), "n_ubatch": row.get("n_ubatch"),
            "type_k": row.get("type_k"), "type_v": row.get("type_v"),
            "n_gpu_layers": row.get("n_gpu_layers")}
    return cells


def tensorsharp_model_only():
    cells = {}
    for depth in (0, 128):
        path = os.path.join(EV, "ts-bench-d%d.log" % depth)
        if not os.path.exists(path):
            continue
        for line in open(path, errors="replace"):
            if not line.startswith("[bench-statistics] "):
                continue
            row = json.loads(line[len("[bench-statistics] "):])
            cells[cell_key(row["n_prompt"], row["n_gen"], row["n_depth"])] = {
                "avg_ts": row["avg_ts"], "stddev_ts": row["stddev_ts"],
                "median_ts": row["median_ts"], "samples_ts": row["samples_ts"]}
    return cells


def http_engine(prefix):
    """Pool every round of one engine while keeping per-round medians visible."""
    rounds, solo, problems = [], None, []
    for index in range(1, 9):
        path = os.path.join(EV, "%s-http-r%d.json" % (prefix, index))
        if not os.path.exists(path):
            continue
        report = read_json(path)
        if not isinstance(report, dict) or report.get("status") != "completed":
            problems.append("%s: status=%s error=%s" % (os.path.basename(path),
                            (report or {}).get("status"), (report or {}).get("error")))
            if not isinstance(report, dict):
                continue
        if solo is None:
            solo = [{"prompt": (item.get("request", {}).get("messages") or [{}])[0].get("content"),
                     "content": item.get("content"), "reasoning": item.get("reasoning"),
                     "completion_tokens": item.get("completion_tokens"),
                     "prompt_tokens": (item.get("usage") or {}).get("prompt_tokens"),
                     "finish_reason": item.get("finish_reason")}
                    for item in report.get("solo", [])]
        rounds.append({"file": os.path.basename(path), "report": report})
    if not rounds:
        return None, problems
    out = {"rounds": [item["file"] for item in rounds], "solo": solo,
           "by_concurrency": {}, "issues": problems}
    concurrencies = sorted({run["concurrency"] for item in rounds
                            for run in item["report"].get("runs", [])})
    for concurrency in concurrencies:
        aggregate, decode, ttft, prefill = [], [], [], []
        matched = budget = total = 0
        per_round = {}
        for item in rounds:
            local = []
            for run in item["report"].get("runs", []):
                if run["concurrency"] != concurrency:
                    continue
                aggregate.append(run["aggregate_tokens_per_second"])
                local.append(run["aggregate_tokens_per_second"])
                for request in run["requests"]:
                    total += 1
                    matched += 1 if request.get("matches_solo") else 0
                    budget += 1 if request.get("requested_token_budget_reached") else 0
                    first = request.get("first_delta_seconds")
                    wall = request.get("wall_seconds")
                    tokens = request.get("completion_tokens") or 0
                    if first and wall and wall > first and tokens > 1:
                        decode.append((tokens - 1) / (wall - first))
                    if first:
                        ttft.append(first)
                        prompt_tokens = (request.get("usage") or {}).get("prompt_tokens")
                        if prompt_tokens:
                            prefill.append(prompt_tokens / first)
            if local:
                per_round[item["file"]] = statistics.median(local)
        out["by_concurrency"][str(concurrency)] = {
            "aggregate_tps_median": statistics.median(aggregate) if aggregate else None,
            "aggregate_tps_samples": aggregate,
            "aggregate_tps_per_round": per_round,
            "decode_tps_per_request_median": statistics.median(decode) if decode else None,
            "ttft_seconds_median": statistics.median(ttft) if ttft else None,
            "prefill_tps_proxy_median": statistics.median(prefill) if prefill else None,
            "requests_total": total, "requests_matching_solo": matched,
            "requests_at_full_budget": budget}
    return out, problems


def ratio(numerator, denominator):
    if not numerator or not denominator:
        return None
    return numerator / denominator


def emit(name, ok, detail):
    print("RESULT %s %s %s" % (name, "PASS" if ok else "FAIL", detail))


summary = {
    "run": RUN, "on_par_threshold": ON_PAR,
    "model": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "engines": {"tensorsharp": {}, "llama_prism": {}}, "ratios": {},
    "limitations": [
        "One RTX A5000, one model, one quantization, these flags: not a general claim.",
        "HTTP throughput includes HTTP, queueing, templating, prefill and generation.",
        "prefill_tps_proxy is prompt_tokens/first-delta and includes queue and template time.",
        "decode_tps_per_request excludes the first token, so it is a post-TTFT rate.",
        "Model-only cells use synthetic token streams; the two engines generate their own.",
        "Matching greedy text is a regression check, never a quality score.",
        "vLLM and SGLang are not compared here; see the phase 1 availability probes.",
    ]}

summary["engines"]["llama_prism"]["model_only"] = prism_model_only()
summary["engines"]["tensorsharp"]["model_only"] = tensorsharp_model_only()
for engine, prefix in ENGINES.items():
    data, problems = http_engine(prefix)
    summary["engines"][engine]["http"] = data
    summary["engines"][engine]["http_issues"] = problems

ts_cells = summary["engines"]["tensorsharp"]["model_only"]
pr_cells = summary["engines"]["llama_prism"]["model_only"]
summary["ratios"]["model_only"] = {}
for key in sorted(set(ts_cells) | set(pr_cells)):
    if key.startswith("__"):
        continue
    ts_value = (ts_cells.get(key) or {}).get("avg_ts")
    pr_value = (pr_cells.get(key) or {}).get("avg_ts")
    value = ratio(ts_value, pr_value)
    summary["ratios"]["model_only"][key] = value
    name = "perf-modelonly-" + key.replace("@", "-")
    if value is None:
        emit(name, False, "missing cell (tensorsharp=%s prism=%s)" % (ts_value, pr_value))
    else:
        emit(name, value >= ON_PAR,
             "tensorsharp=%.2f tok/s prism=%.2f tok/s ratio=%.3f threshold=%.2f"
             % (ts_value, pr_value, value, ON_PAR))

summary["ratios"]["http"] = {}
ts_http = summary["engines"]["tensorsharp"].get("http") or {}
pr_http = summary["engines"]["llama_prism"].get("http") or {}
for concurrency in sorted(set(ts_http.get("by_concurrency", {})) | set(pr_http.get("by_concurrency", {})),
                          key=int):
    ts_cell = ts_http.get("by_concurrency", {}).get(concurrency, {})
    pr_cell = pr_http.get("by_concurrency", {}).get(concurrency, {})
    value = ratio(ts_cell.get("aggregate_tps_median"), pr_cell.get("aggregate_tps_median"))
    summary["ratios"]["http"]["c" + concurrency] = {
        "aggregate_tps": value,
        "decode_tps": ratio(ts_cell.get("decode_tps_per_request_median"),
                            pr_cell.get("decode_tps_per_request_median")),
        "prefill_tps_proxy": ratio(ts_cell.get("prefill_tps_proxy_median"),
                                   pr_cell.get("prefill_tps_proxy_median")),
        "ttft_prism_over_tensorsharp": ratio(pr_cell.get("ttft_seconds_median"),
                                             ts_cell.get("ttft_seconds_median"))}
    name = "perf-http-c" + concurrency
    if value is None:
        emit(name, False, "missing measurement for concurrency %s" % concurrency)
    else:
        emit(name, value >= ON_PAR,
             "aggregate tensorsharp=%.2f prism=%.2f tok/s ratio=%.3f; decode/req ts=%s prism=%s; "
             "prefill-proxy ts=%s prism=%s; ttft ts=%ss prism=%ss"
             % (ts_cell["aggregate_tps_median"], pr_cell["aggregate_tps_median"], value,
                fmt(ts_cell.get("decode_tps_per_request_median")),
                fmt(pr_cell.get("decode_tps_per_request_median")),
                fmt(ts_cell.get("prefill_tps_proxy_median")),
                fmt(pr_cell.get("prefill_tps_proxy_median")),
                fmt(ts_cell.get("ttft_seconds_median")),
                fmt(pr_cell.get("ttft_seconds_median"))))

ts_solo = ts_http.get("solo") or []
pr_solo = pr_http.get("solo") or []
if ts_solo and pr_solo and len(ts_solo) == len(pr_solo):
    same = [index for index in range(len(ts_solo))
            if ts_solo[index]["content"] == pr_solo[index]["content"]
            and ts_solo[index]["completion_tokens"] == pr_solo[index]["completion_tokens"]]
    summary["cross_engine_greedy"] = {
        "prompts": [item["prompt"] for item in ts_solo],
        "identical_indices": same, "total": len(ts_solo),
        "tensorsharp": [item["content"] for item in ts_solo],
        "llama_prism": [item["content"] for item in pr_solo]}
    emit("cross-engine-greedy-match", len(same) == len(ts_solo),
         "%d/%d solo greedy continuations byte-identical across engines at %s tokens"
         % (len(same), len(ts_solo), ts_solo[0]["completion_tokens"]))
else:
    summary["cross_engine_greedy"] = {"error": "solo responses unavailable on both engines"}
    emit("cross-engine-greedy-match", False, "solo responses unavailable on both engines")

for engine in ENGINES:
    data = summary["engines"][engine].get("http")
    if not data:
        emit("http-selfconsistency-" + engine, False, "no HTTP evidence")
        continue
    bad = [(key, cell["requests_matching_solo"], cell["requests_at_full_budget"], cell["requests_total"])
           for key, cell in data["by_concurrency"].items()
           if cell["requests_matching_solo"] != cell["requests_total"]
           or cell["requests_at_full_budget"] != cell["requests_total"]]
    emit("http-selfconsistency-" + engine, not bad and not data["issues"],
         "every concurrent output matches its solo run and every request reached the budget"
         if not bad and not data["issues"]
         else "mismatched cells (c,match,budget,total)=%s; issues=%s" % (bad, data["issues"]))

path = os.path.join(RUN, "summary.json")
with open(path, "w") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)
emit("summary-json", True, path)

print("\n%-22s %14s %14s %8s" % ("cell", "tensorsharp", "llama-prism", "ts/ref"))
for key, value in sorted(summary["ratios"]["model_only"].items()):
    print("%-22s %14s %14s %8s" % ("model-only " + key,
          fmt((ts_cells.get(key) or {}).get("avg_ts")),
          fmt((pr_cells.get(key) or {}).get("avg_ts")), fmt(value)))
for key, value in sorted(summary["ratios"]["http"].items()):
    ts_cell = ts_http.get("by_concurrency", {}).get(key[1:], {})
    pr_cell = pr_http.get("by_concurrency", {}).get(key[1:], {})
    print("%-22s %14s %14s %8s" % ("http aggregate " + key,
          fmt(ts_cell.get("aggregate_tps_median")),
          fmt(pr_cell.get("aggregate_tps_median")), fmt(value["aggregate_tps"])))
PY
}

phase5_summary() {
    say "PHASE 5  summary"
    # The fork tree is shared; prove the binaries did not change under us.
    if [ -s "$EV/prism-binary-stamp-before.txt" ]; then
        binary_stamp > "$EV/prism-binary-stamp-after.txt"
        if diff -q "$EV/prism-binary-stamp-before.txt" "$EV/prism-binary-stamp-after.txt" >/dev/null 2>&1; then
            result prism-binaries-stable PASS "llama-server/llama-bench unchanged (inode, mtime, size) across every measurement"
        else
            result prism-binaries-stable FAIL \
                "the reference binaries changed during the run (another session rebuilt $PRISM_DIR); the cells are not all from one engine build"
        fi
    fi
    write_summarizer
    timeout -k 60 900 python3 "$RUN/summarize.py" "$RUN" "$ON_PAR" | tee -a "$RESULTS"

    # Provenance last, so the fork binary is part of the hashed inventory.
    timeout -k 120 5400 python3 "$BENCH_PY" inventory \
        --repo "$TS_ROOT" --repo "$TS_ROOT/ExternalProjects/ggml" \
        --repo "$PRISM_DIR" --repo "$UPSTREAM" \
        --file "$PQ2" --file "$MMPROJ" --file "$NATIVE_LIB" \
        --file "$PRISM_DIR/build/bin/llama-server" --hash-files \
        --output "$EV/inventory.json" > "$LOGS/inventory.log" 2>&1
    if [ $? -eq 0 ]; then
        result inventory PASS "$EV/inventory.json (repo revisions plus model and binary sha256)"
    else
        result inventory FAIL "$(tail -3 "$LOGS/inventory.log" | oneline)"
    fi
    return 0
}

# ------------------------------------------------------------------ main ----
say "Bonsai2 reference-engine comparison; run directory $RUN"
if have_phase 0; then
    phase0_preflight
    case $? in
        0) ;;
        2) say "RESULTS"; cat "$RESULTS"
           result overall FAIL "required inputs missing; nothing was measured"
           echo "REFBENCH-DONE"; exit 1 ;;
        *) GPU_OK=0 ;;
    esac
fi
have_phase 1 && phase1_negative_claims
have_phase 2 && phase2_build_prism
have_phase 3 && phase3_model_only
have_phase 4 && phase4_http
have_phase 5 && phase5_summary

say "RESULTS"
cat "$RESULTS"
TOTAL=$(grep -c '^RESULT ' "$RESULTS")
FAILED=$(grep -c '^RESULT [^ ]* FAIL ' "$RESULTS")
if [ "$FAILED" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    result overall PASS "$TOTAL checks, 0 failed; evidence in $RUN"
    echo "REFBENCH-DONE"
    exit 0
fi
result overall FAIL "$TOTAL checks, $FAILED failed; evidence in $RUN"
echo "REFBENCH-DONE"
exit 1
