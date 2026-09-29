#!/usr/bin/env bash
# Test the tests: confirm that four regression tests actually catch the bugs
# they were written for.
#
# For each row below, this script (a) refuses to start if the target
# production file already has uncommitted changes, (b) applies a one-line
# revert of the fix straight to that file with a Python in-place edit that
# asserts its pattern matched exactly once (never a silent no-op), (c) runs
# the named test node in its own pytest session, (d) restores the file with
# ``git checkout --`` from a trap, so an interrupted run always restores, and
# (e) records whether the test FAILED in its call phase (the check bites, as
# it should), PASSED despite the revert (SURVIVED -- the test does not catch
# this regression), or ERRORed/SKIPped (INVALID -- inconclusive, not
# evidence either way). A pytest "failed" caused by a collection or setup
# error is not evidence the assertion bites, so each run's ``--junitxml`` is
# inspected for a <failure> in the call phase specifically, not just a
# non-zero pytest exit code.
#
#   1. src/spyglass/spikesorting/v2/_sorting_dispatch.py -- drop
#      ``seed=random_seed`` from ``pinned_whiten``'s ``sip.whiten`` call, so
#      SpikeInterface draws its random data-fitting chunks unseeded.
#   2. src/spyglass/spikesorting/v2/_concat_recording.py -- reverse the
#      member order passed into ``concatenate_recordings`` inside
#      ``build_concatenated_recording``.
#   3. src/spyglass/spikesorting/v2/_observed_time.py -- restore the interval
#      end formula ``observed_intervals`` used before commit 81b32b14
#      ("derive observed-interval ends from recorded timestamps"): the end
#      of a segment was the nominal sampling rate applied from its first
#      timestamp (``t + (end - first) / fs``), rather than the recording's
#      own timestamp at the last sample plus one nominal period.
#   4. src/spyglass/spikesorting/v2/_recording_nwb.py -- skip the
#      ``_persist_channel_geometry(...)`` call after the pre-motion NWB
#      write, so the reloaded recording's channel geometry is never
#      persisted.
#
# This script is run manually, not by CI, after touching any of the four
# files above or periodically as a suite-health check. It needs the v2 test
# environment and a database (regression #4 needs a real DataJoint
# connection and ``tests/_data/raw/minirec20230622.nwb``; it is slow).
#
# Usage (from the repo root, v2 env active):
#     source ~/spyglass_v2_env.sh
#     bash tests/spikesorting/v2/scripts/verify_regression_tests_bite.sh \
#       [--container-name NAME] [--container-port PORT]
#
# Defaults: --container-name spyglass-pytest-bite, --container-port 33099.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
cd "$REPO_ROOT"

CONTAINER_NAME="spyglass-pytest-bite"
CONTAINER_PORT="33099"

while [ $# -gt 0 ]; do
  case "$1" in
    --container-name)
      CONTAINER_NAME=$2
      shift 2
      ;;
    --container-port)
      CONTAINER_PORT=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

WORK_DIR=$(mktemp -d)

# Parallel arrays (indices 0-3), one entry per regression.
FILES=(
  "src/spyglass/spikesorting/v2/_sorting_dispatch.py"
  "src/spyglass/spikesorting/v2/_concat_recording.py"
  "src/spyglass/spikesorting/v2/_observed_time.py"
  "src/spyglass/spikesorting/v2/_recording_nwb.py"
)
TEST_NODES=(
  "tests/spikesorting/v2/test_sorting_dispatch.py::test_pinned_whiten_is_deterministic_across_calls"
  "tests/spikesorting/v2/test_concat_recording.py::test_build_concatenated_recording_returns_members_traces_unchanged"
  "tests/spikesorting/v2/test_observed_time.py::test_observed_intervals_end_from_timestamps"
  "tests/spikesorting/v2/single_session/test_recording.py::test_xz_geometry_reloads_with_four_positions"
)
DESCRIPTIONS=(
  "pinned_whiten: drop the seed=random_seed forwarding to sip.whiten"
  "build_concatenated_recording: reverse the member order"
  "observed_intervals: restore the pre-81b32b14 nominal-rate interval end"
  "_recording_nwb: skip the _persist_channel_geometry(...) call"
)

# ---------------------------------------------------------------------------
# (a) Refuse to start if any target file already has uncommitted changes.
# Scoped to exactly these four files -- never touches, checks, or reports on
# any other path in the working tree.
# ---------------------------------------------------------------------------
dirty=$(git status --porcelain -- "${FILES[@]}")
if [ -n "$dirty" ]; then
  echo "refusing to start: target files already have uncommitted changes:" >&2
  echo "$dirty" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# (b) One-line reverts. Each is a Python in-place edit that asserts its
# pattern matched exactly once; a match count of 0 or >1 raises and aborts
# the whole script (via set -e) rather than silently doing nothing or
# reverting the wrong thing.
# ---------------------------------------------------------------------------
apply_revert() { # index
  local idx=$1
  case "$idx" in
    0)
      python3 - "${FILES[0]}" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "sip.whiten(recording, dtype=np.float64, seed=random_seed)"
new = "sip.whiten(recording, dtype=np.float64)"
count = text.count(old)
assert count == 1, f"expected exactly 1 match of {old!r}, found {count}"
open(path, "w").write(text.replace(old, new, 1))
PY
      ;;
    1)
      python3 - "${FILES[1]}" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "concatenate_recordings(recordings, ignore_times=True)"
new = "concatenate_recordings(recordings[::-1], ignore_times=True)"
count = text.count(old)
assert count == 1, f"expected exactly 1 match of {old!r}, found {count}"
open(path, "w").write(text.replace(old, new, 1))
PY
      ;;
    2)
      python3 - "${FILES[2]}" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = (
    "                t1 = (\n"
    "                    float(_segment_times_at(recording, "
    "np.array([end - 1]))[0])\n"
    "                    + 1.0 / fs\n"
    "                )\n"
)
new = "                t1 = t0 + (end - first) / fs\n"
count = text.count(old)
assert count == 1, f"expected exactly 1 match of {old!r}, found {count}"
open(path, "w").write(text.replace(old, new, 1))
PY
      ;;
    3)
      python3 - "${FILES[3]}" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "        _persist_channel_geometry(\n"
new = "        (lambda *a, **k: None)(\n"
count = text.count(old)
assert count == 1, f"expected exactly 1 match of {old!r}, found {count}"
open(path, "w").write(text.replace(old, new, 1))
PY
      ;;
    *)
      echo "apply_revert: bad index $idx" >&2
      exit 2
      ;;
  esac
}

# ---------------------------------------------------------------------------
# (e) Always restore whichever file is currently reverted, even on Ctrl-C or
# an unexpected error.
# ---------------------------------------------------------------------------
CURRENT_FILE=""
cleanup() {
  if [ -n "$CURRENT_FILE" ]; then
    git checkout -- "$CURRENT_FILE"
    CURRENT_FILE=""
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT INT TERM

# Parse a junitxml report for the single testcase's outcome. Prints one of
# FAILED_IN_CALL / SURVIVED / ERROR / SKIPPED / NO_TESTCASE.
classify_junit() { # xml_path
  python3 - "$1" <<'PY'
import sys
import xml.etree.ElementTree as ET

try:
    root = ET.parse(sys.argv[1]).getroot()
except ET.ParseError:
    print("NO_TESTCASE")
    raise SystemExit

cases = root.findall(".//testcase")
if len(cases) != 1:
    print("NO_TESTCASE")
    raise SystemExit

case = cases[0]
if case.find("failure") is not None:
    print("FAILED_IN_CALL")
elif case.find("error") is not None:
    print("ERROR")
elif case.find("skipped") is not None:
    print("SKIPPED")
else:
    print("SURVIVED")
PY
}

RESULT_LABELS=()
RESULT_DETAILS=()
ANY_BAD=0

for i in 0 1 2 3; do
  n=$((i + 1))
  file="${FILES[$i]}"
  node="${TEST_NODES[$i]}"
  desc="${DESCRIPTIONS[$i]}"
  xml="$WORK_DIR/regression_${n}.xml"

  echo
  echo "=== regression $n/4: $desc ==="
  echo "target: $file"
  echo "test:   $node"

  apply_revert "$i"
  CURRENT_FILE="$file"
  echo "reverted (pattern matched exactly once)"

  set +e
  python -m pytest -p no:xvfb --no-dlc --no-cov \
    --base-dir=./tests/_data/ -q -x -rs \
    --container-name "$CONTAINER_NAME" --container-port "$CONTAINER_PORT" \
    --junitxml="$xml" \
    "$node"
  pytest_rc=$?
  set -e

  git checkout -- "$file"
  CURRENT_FILE=""
  echo "restored $file (pytest exit $pytest_rc)"

  outcome=$(classify_junit "$xml")
  case "$outcome" in
    FAILED_IN_CALL)
      label="PASS-OF-THE-CHECK"
      detail="test failed in its call phase, as required"
      ;;
    SURVIVED)
      label="SURVIVED"
      ANY_BAD=1
      detail="test PASSED despite the revert -- it does not catch this regression"
      ;;
    ERROR)
      label="INVALID"
      ANY_BAD=1
      detail="test errored (setup/collection), not a call-phase failure"
      ;;
    SKIPPED)
      label="INVALID"
      ANY_BAD=1
      detail="test was skipped -- a skip proves nothing"
      ;;
    NO_TESTCASE)
      label="INVALID"
      ANY_BAD=1
      detail="no single testcase found in the junitxml report (pytest exit $pytest_rc)"
      ;;
  esac
  echo "result: $label -- $detail"

  RESULT_LABELS+=("$label")
  RESULT_DETAILS+=("$detail")
done

echo
echo "=== summary ==="
printf '%-3s %-70s %-20s\n' "#" "regression" "result"
for i in 0 1 2 3; do
  n=$((i + 1))
  printf '%-3s %-70s %-20s\n' "$n" "${DESCRIPTIONS[$i]}" "${RESULT_LABELS[$i]}"
  echo "    ${RESULT_DETAILS[$i]}"
done

final_dirty=$(git status --porcelain -- "${FILES[@]}")
if [ -n "$final_dirty" ]; then
  echo
  echo "FAIL: target files are not clean after restore:" >&2
  echo "$final_dirty" >&2
  ANY_BAD=1
fi

echo
if [ "$ANY_BAD" -eq 0 ]; then
  echo "all four regression tests bite: each failed in its call phase" \
    "when its fix was reverted, and all target files are clean."
  exit 0
else
  echo "one or more regressions did not bite as expected -- see SURVIVED /" \
    "INVALID rows above." >&2
  exit 1
fi
