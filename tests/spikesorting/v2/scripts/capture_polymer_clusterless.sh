#!/usr/bin/env bash
# Capture v1 clusterless_thresholder baselines for the polymer fixtures.
#
# Launches 8 parallel tmux sessions (2 fixtures × 4 shanks) under the
# spyglass-v1-parity conda env. Each session writes its baseline
# artifacts to $SPIKESORTING_V2_BASELINE_ROOT/<fixture_stem>/clusterless/shank<N>/
# for the v2 parity test to consume.
#
# Worktree pin: the plan's "cd /tmp/spyglass-master" recipe assumes the
# v1 worktree carries the v2 testing tree. In this repo the master
# branch does not (the v2 tree is spikesorting-v2-only), so the script
# cd's to the v2 checkout's repo root instead. The spyglass-v1-parity
# env is dev-installed against the same repo and pins SI 0.99, so the
# v1 imports inside baseline_capture.py still resolve to v1 tables;
# only the cwd differs from the plan. Database isolation between v1
# captures and v2 tests is enforced by --database-prefix (per-case
# `test_<fix>_cl_s<N>` schema name) and --base-dir (per-case temp
# directory), not by cwd separation.
#
# Usage:
#     export SPIKESORTING_V2_BASELINE_ROOT=/path/to/baseline/root
#     bash tests/spikesorting/v2/scripts/capture_polymer_clusterless.sh
#
# Monitor:
#     tmux ls          # see all running sessions
#     tmux attach -t parity-cap-clusterless-mearec_polymer_smoke-shank0
#     (detach: Ctrl-b d)

set -euo pipefail

ROOT=${SPIKESORTING_V2_BASELINE_ROOT:?SPIKESORTING_V2_BASELINE_ROOT must be set}
BASE_ROOT=/tmp/spyglass-v1-base

# Resolve repo root from this script's location (tests/spikesorting/v2/scripts/<script>.sh).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

declare -A ABBREV=(
  [mearec_polymer_smoke]=smk
  [mearec_polymer_128ch_60s]=p60
)

for FIX in mearec_polymer_smoke mearec_polymer_128ch_60s; do
  ABV=${ABBREV[$FIX]}
  for SHANK in 0 1 2 3; do
    SESSION="parity-cap-clusterless-${FIX}-shank${SHANK}"
    BASE_DIR="${BASE_ROOT}/${FIX}/clusterless/shank${SHANK}"
    OUT_DIR="${ROOT}/${FIX}/clusterless/shank${SHANK}"
    PREFIX="test_${ABV}_cl_s${SHANK}"
    mkdir -p "$BASE_DIR" "$OUT_DIR"
    tmux new-session -d -s "$SESSION" "
      cd '$REPO_ROOT' &&
      conda run -n spyglass-v1-parity python \
        tests/spikesorting/v2/baseline_capture.py \
        --nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb \
        --team-name v2_test_team \
        --interval-list-name 'raw data valid times' \
        --sort-group-id $SHANK \
        --sorter-param-name smoke_clusterless_5uv \
        --database-prefix '$PREFIX' \
        --base-dir '$BASE_DIR' \
        --output-dir '$OUT_DIR' \
        2>&1 | tee '$OUT_DIR/capture.log'
    "
  done
done

echo "Launched 8 tmux clusterless captures."
echo "Monitor: tmux ls"
echo "Output: $ROOT/<fixture_stem>/clusterless/shank<N>/"
