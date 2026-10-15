#!/usr/bin/env bash
# Pod start command for the study: run several cells (formats) of one tier unattended.
#
# Results are served read-only on port 8888 (python's http.server), so they can be collected
# through RunPod's HTTPS proxy without SSH. They contain only public test data.
#
#   TIER=t2-l4 FORMATS="bf16 awq gptq4 gptq8" RATE=0.59 RATE_DATE=2026-10-15 [SHAKEDOWN=1]
set -uo pipefail
: "${TIER:?}" "${FORMATS:?}" "${RATE:?}" "${RATE_DATE:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${RESULTS:-/workspace/results}/${TIER}"
mkdir -p "$OUT"

(cd "${RESULTS:-/workspace/results}" && nohup python3 -m http.server 8888 > /dev/null 2>&1 &)

for format in $FORMATS; do
  echo "[$(date -u +%H:%M:%S)] CELL-START $format"
  if FORMAT="$format" bash "$HERE/run_cell.sh" > "$OUT/cell-$format.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] CELL-OK $format"
  else
    echo "[$(date -u +%H:%M:%S)] CELL-FAILED $format (see cell-$format.log)"
  fi
done
echo "[$(date -u +%H:%M:%S)] ALL-CELLS-DONE"
touch "$OUT/ALL-CELLS-DONE"
sleep infinity
