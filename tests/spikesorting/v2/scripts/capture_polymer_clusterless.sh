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
# only the cwd differs from the plan.
#
# Concurrency note: the plan calls for 8 parallel tmux sessions with
# per-case --database-prefix isolation. Empirically that does NOT work
# on this codebase: v1 schemas declare ``dj.schema("spikesorting_v1_*")``
# with a hardcoded name (NO ``database.prefix`` prepended), so all
# parallel sessions race on the SAME MySQL schema and the smoke-row
# delete↔insert + fetch1 chain blow up. The script runs the eight
# captures sequentially inside ONE long-lived tmux session; total wall
# time ~14 min (4 × 30 s smoke + 4 × 3 min for 60 s polymer) instead of
# the plan's ~25-min parallel estimate. Acceptable for Phase A.
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

# ``conda run -n spyglass-v1-parity python`` resolves to the wrong env
# on this host (CONDA_DEFAULT_ENV leakage from the parent shell makes
# conda run pick spyglass-dlc instead). Invoke the env's python
# directly. Resolution order:
#   1. $SPYGLASS_V1_PARITY_PYTHON (operator override)
#   2. ``conda env list`` lookup for ``spyglass-v1-parity`` envs/bin/python
#   3. Common defaults under $HOME/miniconda3 / $HOME/anaconda3 / /opt/conda
# Fails loudly with an instruction if all three fail.
_resolve_v1_python() {
  if [ -n "${SPYGLASS_V1_PARITY_PYTHON:-}" ] && [ -x "$SPYGLASS_V1_PARITY_PYTHON" ]; then
    echo "$SPYGLASS_V1_PARITY_PYTHON"; return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local from_conda
    from_conda=$(conda env list 2>/dev/null \
      | awk '$1 == "spyglass-v1-parity" {print $NF "/bin/python"}' | head -1)
    if [ -n "$from_conda" ] && [ -x "$from_conda" ]; then
      echo "$from_conda"; return 0
    fi
  fi
  for cand in \
    "$HOME/miniconda3/envs/spyglass-v1-parity/bin/python" \
    "$HOME/anaconda3/envs/spyglass-v1-parity/bin/python" \
    "/opt/conda/envs/spyglass-v1-parity/bin/python"; do
    if [ -x "$cand" ]; then echo "$cand"; return 0; fi
  done
  return 1
}
V1_PYTHON=$(_resolve_v1_python) || true
if [ -z "${V1_PYTHON:-}" ] || [ ! -x "$V1_PYTHON" ]; then
  echo "ERROR: cannot locate spyglass-v1-parity python interpreter." >&2
  echo "       Set SPYGLASS_V1_PARITY_PYTHON=/path/to/env/bin/python and re-run." >&2
  echo "       (Searched: env var; ``conda env list`` for spyglass-v1-parity;" >&2
  echo "       \$HOME/{miniconda3,anaconda3}/envs/spyglass-v1-parity; /opt/conda/...)" >&2
  exit 1
fi
echo "Using V1_PYTHON=$V1_PYTHON"

declare -A ABBREV=(
  [mearec_polymer_smoke]=smk
  [mearec_polymer_128ch_60s]=p60
)

# Build the sequential capture command as a single shell pipeline so it
# can run inside one long-lived tmux session.
SESSION="parity-cap-clusterless-matrix"
SEQUENCE=""
for FIX in mearec_polymer_smoke mearec_polymer_128ch_60s; do
  ABV=${ABBREV[$FIX]}
  for SHANK in 0 1 2 3; do
    BASE_DIR="${BASE_ROOT}/${FIX}/clusterless/shank${SHANK}"
    OUT_DIR="${ROOT}/${FIX}/clusterless/shank${SHANK}"
    PREFIX="test_${ABV}_cl_s${SHANK}"
    mkdir -p "$BASE_DIR" "$OUT_DIR"
    SEQUENCE+="echo '=== ${FIX} shank ${SHANK} ===' && "
    SEQUENCE+="'$V1_PYTHON' tests/spikesorting/v2/baseline_capture.py "
    SEQUENCE+="--nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb "
    SEQUENCE+="--team-name v2_test_team "
    SEQUENCE+="--interval-list-name 'raw data valid times' "
    SEQUENCE+="--sort-group-id $SHANK "
    SEQUENCE+="--sorter-param-name smoke_clusterless_5uv "
    SEQUENCE+="--artifact-param-name none "
    SEQUENCE+="--database-prefix '$PREFIX' "
    SEQUENCE+="--base-dir '$BASE_DIR' "
    SEQUENCE+="--output-dir '$OUT_DIR' "
    SEQUENCE+="2>&1 | tee '$OUT_DIR/capture.log'; "
  done
done
SEQUENCE+="echo '=== ALL 8 CAPTURES COMPLETE ==='"

tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && $SEQUENCE"

echo "Launched serial capture sequence in tmux session: $SESSION"
echo "Monitor: tmux attach -t $SESSION  (detach: Ctrl-b d)"
echo "         tail -f $ROOT/<fixture_stem>/clusterless/shank<N>/capture.log"
echo "Output:  $ROOT/<fixture_stem>/clusterless/shank<N>/"
