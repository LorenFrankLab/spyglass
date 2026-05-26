#!/usr/bin/env bash
# Capture v1 MountainSort4 baselines for the polymer 60s fixture.
#
# Runs 4 captures (shanks 0–3) sequentially inside one long-lived tmux
# session (``parity-cap-ms4-matrix``). Parallel execution is unsafe
# because v1 spikesorting schemas hardcode their schema names and do
# NOT honor ``dj.config["database.prefix"]`` (see
# capture_polymer_clusterless.sh for the full concurrency note).
#
# Wall time ~12-15 min total (~3 min per shank for MS4 + analyzer build).
#
# Usage:
#     export SPIKESORTING_V2_BASELINE_ROOT=/path/to/baseline/root
#     bash tests/spikesorting/v2/scripts/capture_polymer_ms4.sh
#
# Monitor:
#     tmux ls
#     tmux attach -t parity-cap-ms4-matrix  (detach: Ctrl-b d)

set -euo pipefail

ROOT=${SPIKESORTING_V2_BASELINE_ROOT:?SPIKESORTING_V2_BASELINE_ROOT must be set}
BASE_ROOT=/tmp/spyglass-v1-base

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

V1_PYTHON=${SPYGLASS_V1_PARITY_PYTHON:-/home/edeno/miniconda3/envs/spyglass-v1-parity/bin/python}
if [ ! -x "$V1_PYTHON" ]; then
  echo "ERROR: v1 python interpreter $V1_PYTHON not found or not executable." >&2
  exit 1
fi

FIX=mearec_polymer_128ch_60s
ABV=p60
SESSION="parity-cap-ms4-matrix"

SEQUENCE=""
for SHANK in 0 1 2 3; do
  BASE_DIR="${BASE_ROOT}/${FIX}/ms4/shank${SHANK}"
  OUT_DIR="${ROOT}/${FIX}/ms4/shank${SHANK}"
  PREFIX="test_${ABV}_ms4_s${SHANK}"
  mkdir -p "$BASE_DIR" "$OUT_DIR"
  SEQUENCE+="echo '=== ${FIX} ms4 shank ${SHANK} ===' && "
  SEQUENCE+="'$V1_PYTHON' tests/spikesorting/v2/baseline_capture.py "
  SEQUENCE+="--nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb "
  SEQUENCE+="--team-name v2_test_team "
  SEQUENCE+="--interval-list-name 'raw data valid times' "
  SEQUENCE+="--sort-group-id $SHANK "
  SEQUENCE+="--sorter mountainsort4 "
  SEQUENCE+="--sorter-param-name ms4_60s_polymer "
  SEQUENCE+="--artifact-param-name none "
  SEQUENCE+="--database-prefix '$PREFIX' "
  SEQUENCE+="--base-dir '$BASE_DIR' "
  SEQUENCE+="--output-dir '$OUT_DIR' "
  SEQUENCE+="2>&1 | tee '$OUT_DIR/capture.log'; "
done
SEQUENCE+="echo '=== ALL 4 MS4 CAPTURES COMPLETE ==='"

tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && $SEQUENCE"

echo "Launched serial MS4 capture sequence in tmux session: $SESSION"
echo "Monitor: tmux attach -t $SESSION  (detach: Ctrl-b d)"
echo "         tail -f $ROOT/${FIX}/ms4/shank<N>/capture.log"
echo "Output:  $ROOT/${FIX}/ms4/shank<N>/"
