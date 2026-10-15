#!/usr/bin/env bash
# One cell of the study: one GPU tier x one model format. See docs/study-protocol.md.
#
# Runs on a GPU pod started from the pinned vLLM image (vllm/vllm-openai:v0.30.0). Every
# tier runs exactly this sequence, so cells can be filled in any order and compared.
#
#   TIER=t1-rtx3090 FORMAT=bf16 RATE=0.50 RATE_DATE=2026-10-15 scripts/run_cell.sh
#
# FORMAT: bf16 | awq | gptq4 | gptq8
# SHAKEDOWN=1  tiny runs, to check every step works before paying for the real ones
# SKIP_QUALITY=1  latency only (the reference tier)
# DRY_RUN=1    print the commands without running anything
set -euo pipefail

: "${TIER:?set TIER, e.g. t1-rtx3090}"
: "${FORMAT:?set FORMAT: bf16 | awq | gptq4 | gptq8}"
: "${RATE:?set RATE, the hourly price actually paid, in USD}"
: "${RATE_DATE:?set RATE_DATE, YYYY-MM-DD}"
PROVIDER="${PROVIDER:-runpod-secure}"
FL_VERSION="${FL_VERSION:-0.2.1}"
RESULTS="${RESULTS:-/workspace/results}/${TIER}"
PORT=8000
URL="http://127.0.0.1:${PORT}/v1"
NAME=qwen2.5-7b

# Pinned weights: the exact revisions whose provenance was checked (study protocol, section 2).
case "$FORMAT" in
  bf16)  REPO=Qwen/Qwen2.5-7B-Instruct;            REV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
  awq)   REPO=Qwen/Qwen2.5-7B-Instruct-AWQ;        REV=b25037543e9394b818fdfca67ab2a00ecc7dd641 ;;
  gptq4) REPO=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4;  REV=e9c932ac1893a49ae0fc497ad6e1e86e2e39af20 ;;
  gptq8) REPO=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8;  REV=6711f2b6fc95545efe6018c469eca6832e28cefd ;;
  *) echo "unknown FORMAT: $FORMAT" >&2; exit 2 ;;
esac

if [[ "${SHAKEDOWN:-0}" == 1 ]]; then
  BENCH_ARGS=(-c 1,16 -n 10,32 --repeats 1 --warmup 2)
  QUALITY_LIMIT=(--limit 20)
  LABEL_PREFIX="shakedown-"
else
  BENCH_ARGS=(-c 1,4,16,64 -n 50,100,200,400 --repeats 3)
  QUALITY_LIMIT=()
  LABEL_PREFIX=""
fi

run() { echo "+ $*"; [[ "${DRY_RUN:-0}" == 1 ]] || "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# frugal-llm in its own virtualenv (made with uv, which needs no python3-venv package), so
# it can never disturb vLLM's dependencies.
FL=/opt/frugal-llm/bin/frugal-llm
if [[ "${DRY_RUN:-0}" != 1 && ! -x "$FL" ]]; then
  python3 -m pip install --quiet uv
  python3 -m uv venv --quiet /opt/frugal-llm --python python3
  python3 -m uv pip install --quiet --python /opt/frugal-llm/bin/python "frugal-llm==${FL_VERSION}"
fi
run mkdir -p "$RESULTS"

# The host, once per pod: hosts differ in CPU, memory and GPU clocks, so record them.
if [[ "${DRY_RUN:-0}" != 1 && ! -f "$RESULTS/host-info.txt" ]]; then
  { date -u; echo; nvidia-smi -q; echo; lscpu; echo; free -g; echo; df -h /workspace; }     > "$RESULTS/host-info.txt" 2>&1 || true
fi

# --- serve -------------------------------------------------------------------------
log "starting vLLM: $REPO@$REV"
SERVER_LOG="$RESULTS/vllm-${FORMAT}.log"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "+ vllm serve $REPO --revision $REV --served-model-name $NAME --max-model-len 4096 --gpu-memory-utilization 0.90 --port $PORT > $SERVER_LOG &"
else
  vllm serve "$REPO" --revision "$REV" --served-model-name "$NAME" \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --port "$PORT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  # On exit, stop vLLM and wait until it has gone, so the next cell finds the GPU memory free.
  trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 360); do  # up to 30 min: first start downloads the weights
    curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM exited; see $SERVER_LOG" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || { echo "vLLM never became healthy" >&2; exit 1; }
fi

# --- what the API can't report, recorded with every result ---------------------------
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  GPU="dry-run-gpu"; DRIVER="dry-run"; VLLM_VERSION="dry-run"
else
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
fi
META=(--meta "tier=$TIER" --meta "format=$FORMAT" --meta "weights=$REPO@$REV"
      --meta "gpu=$GPU" --meta "driver=$DRIVER" --meta "vllm=$VLLM_VERSION"
      --meta "provider=$PROVIDER" --meta "rate_usd_per_hour=$RATE" --meta "rate_date=$RATE_DATE")

# --- latency -----------------------------------------------------------------------
log "bench"
run "$FL" bench --url "$URL" --model "$NAME" "${BENCH_ARGS[@]}" --max-tokens 128 \
  --cost-per-hour "$RATE" --currency USD --label "${LABEL_PREFIX}${TIER}-${FORMAT}" \
  --out "$RESULTS" "${META[@]}"

# --- quality (BF16 twice: the second run is the noise floor) ------------------------
# SKIP_QUALITY=1 for the reference tier, which measures latency only (study protocol).
if [[ "$FORMAT" == bf16 ]]; then RUNS=(a b); else RUNS=(""); fi
[[ "${SKIP_QUALITY:-0}" == 1 ]] && { log "quality skipped (SKIP_QUALITY=1)"; TASKS=(); } || TASKS=(gsm8k mmlu)
for task in "${TASKS[@]}"; do
  for r in "${RUNS[@]}"; do
    label="${LABEL_PREFIX}${TIER}-${FORMAT}${r:+-$r}"
    log "quality $task $label"
    run "$FL" quality --url "$URL" --model "$NAME" --task "$task" -c 16 \
      "${QUALITY_LIMIT[@]}" --label "$label" --out "$RESULTS" "${META[@]}"
  done
done

log "cell done: $TIER $FORMAT -> $RESULTS (copy it off the pod now)"
