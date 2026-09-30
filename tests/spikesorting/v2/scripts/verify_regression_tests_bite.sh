#!/usr/bin/env bash
# Test the tests: confirm that four regression tests actually catch the bugs
# they were written for.
#
# For each row below, this script (a) refuses to start if the target
# production file already has uncommitted changes, (b) runs the named test
# node against the unmodified tree and requires it to PASS -- a test that
# already fails, errors or skips proves nothing about the revert, so that row
# is INVALID ("baseline did not pass") and its revert is never applied, (c)
# applies a one-line revert of the fix straight to that file with a Python
# in-place edit that asserts its pattern matched exactly once (never a silent
# no-op), (d) runs the same test node again in its own pytest session, (e)
# restores the file with ``git checkout --`` (also from a trap, so an
# interrupted run always restores and then stops), and (f) classifies the
# reverted run from its ``--junitxml`` report, not from pytest's exit code:
#
#   PASS-OF-THE-CHECK  the call phase failed with an AssertionError -- the
#                      test's assertion caught the regression, as it should.
#   SURVIVED           the call phase passed despite the revert -- the test
#                      does not catch this regression. A teardown error after
#                      a passing call is noted but does not change this: the
#                      verdict is about the call phase.
#   INVALID            inconclusive, not evidence either way: the baseline
#                      did not pass; the call phase raised something other
#                      than an AssertionError (ImportError, NameError,
#                      AttributeError, ...); setup or collection errored; the
#                      test was skipped; or the report held no single test.
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
# This script never creates, modifies or deletes the developer's DataJoint
# config. The test session's ``dj_config`` fixture writes
# ``dj_local_conf.json`` (deleting any existing one first) into pytest's
# current working directory, so every pytest session here runs from a
# private directory inside the script's temporary directory, with absolute
# paths for the test node, ``--rootdir``, ``-c <repo>/pyproject.toml`` (the
# repository's addopts and markers still apply) and ``--base-dir``. The
# temporary directory, and the config written into it, is removed on exit.
# ``<repo>/src`` is put first on PYTHONPATH so the session imports the
# ``spyglass`` package whose files this script reverts, even when an
# editable install of another checkout is present.
#
# This script is run manually, not by CI, after touching any of the four
# files above or periodically as a suite-health check. It needs the v2 test
# environment and a database (regression #4 needs a real DataJoint
# connection and ``tests/_data/raw/minirec20230622.nwb``; it is slow).
#
# Usage (from any directory, v2 env active):
#     source ~/spyglass_v2_env.sh
#     bash tests/spikesorting/v2/scripts/verify_regression_tests_bite.sh \
#       [--container-name NAME] [--container-port PORT]
#
# Defaults: --container-name spyglass-pytest-bite, --container-port 33099.
# Exit status: 0 only when every row is PASS-OF-THE-CHECK and all target
# files are clean afterwards; 1 otherwise (including a failed restore); 2 on
# bad arguments or a dirty target file at start; 130 / 143 when interrupted
# by SIGINT / SIGTERM, after restoring the reverted file and removing the
# temporary directory. bash handles those signals once the running command
# returns: Ctrl-C reaches the whole process group and so stops pytest too,
# but a SIGTERM sent to the script's process alone takes effect only after
# the running pytest session finishes.

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
# Always restore whichever file is currently reverted and remove the
# temporary directory, even on Ctrl-C or an unexpected error. Registered
# before anything is created so no exit path leaks the temporary directory.
# Cleanup runs only from the EXIT trap; SIGINT and SIGTERM just exit (130 /
# 143), which fires it. A trap that cleaned up and returned would let bash
# resume the loop and start the next revert (see the exit-status notes in
# the header for when bash acts on the signal). A failed restore is
# reported loudly, does not skip removing the temporary directory, and
# makes the script exit non-zero.
# ---------------------------------------------------------------------------
CURRENT_FILE=""
WORK_DIR=""
RESTORE_FAILED=0
# shellcheck disable=SC2329  # invoked by the trap below
cleanup() {
  local status=$?
  if [ -n "$CURRENT_FILE" ]; then
    restore_current_file
  fi
  if [ -n "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
    WORK_DIR=""
  fi
  if [ "$RESTORE_FAILED" -ne 0 ] && [ "$status" -eq 0 ]; then
    exit 1
  fi
}

# Restore CURRENT_FILE from git. Never aborts the script; a failure is
# printed and recorded in RESTORE_FAILED.
restore_current_file() {
  if ! git checkout -- "$CURRENT_FILE"; then
    echo "RESTORE FAILED: $CURRENT_FILE is still reverted; restore it" \
      "with: git checkout -- $CURRENT_FILE" >&2
    RESTORE_FAILED=1
  fi
  CURRENT_FILE=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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

WORK_DIR=$(mktemp -d)
# pytest's working directory. The session's dj_config fixture writes
# dj_local_conf.json here, never into the repository.
RUN_DIR="$WORK_DIR/pytest-cwd"
mkdir "$RUN_DIR"

# ---------------------------------------------------------------------------
# (c) One-line reverts. replace_once asserts its pattern matched exactly
# once; a match count of 0 or >1 raises and aborts the whole script (via
# set -e) rather than silently doing nothing or reverting the wrong thing.
# It writes a temporary file beside the target and os.replace()s it into
# place, so an interrupt leaves either the original or the complete revert,
# never a truncated file.
# ---------------------------------------------------------------------------
replace_once() { # path old new
  python3 - "$1" "$2" "$3" <<'PY'
import os
import sys
import tempfile

path, old, new = sys.argv[1:4]
with open(path) as fh:
    text = fh.read()
count = text.count(old)
assert count == 1, f"expected exactly 1 match of {old!r} in {path}, found {count}"
fd, tmp = tempfile.mkstemp(
    dir=os.path.dirname(os.path.abspath(path)), prefix=".bite-revert-"
)
try:
    with os.fdopen(fd, "w") as fh:
        fh.write(text.replace(old, new, 1))
    os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.replace(tmp, path)
except BaseException:
    os.unlink(tmp)
    raise
PY
}

apply_revert() { # index
  local idx=$1
  case "$idx" in
    0)
      replace_once "${FILES[0]}" \
        "sip.whiten(recording, dtype=np.float64, seed=random_seed)" \
        "sip.whiten(recording, dtype=np.float64)"
      ;;
    1)
      replace_once "${FILES[1]}" \
        "concatenate_recordings(recordings, ignore_times=True)" \
        "concatenate_recordings(recordings[::-1], ignore_times=True)"
      ;;
    2)
      local old_end
      old_end=$'                t1 = (\n'
      old_end+=$'                    float(_segment_times_at(recording, '
      old_end+=$'np.array([end - 1]))[0])\n'
      old_end+=$'                    + 1.0 / fs\n'
      old_end+=$'                )\n'
      replace_once "${FILES[2]}" "$old_end" \
        $'                t1 = t0 + (end - first) / fs\n'
      ;;
    3)
      replace_once "${FILES[3]}" \
        $'        _persist_channel_geometry(\n' \
        $'        (lambda *a, **k: None)(\n'
      ;;
    *)
      echo "apply_revert: bad index $idx" >&2
      exit 2
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Run one test node in its own pytest session from RUN_DIR (never the
# repository root) and write its junitxml report. Prints pytest's output;
# returns pytest's exit status.
# ---------------------------------------------------------------------------
run_node() { # node xml_path
  (
    cd "$RUN_DIR"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      python -m pytest \
      -c "$REPO_ROOT/pyproject.toml" --rootdir="$REPO_ROOT" \
      -p no:xvfb --no-dlc --no-cov \
      --base-dir="$REPO_ROOT/tests/_data/" -q -x -rs \
      --container-name "$CONTAINER_NAME" --container-port "$CONTAINER_PORT" \
      --junitxml="$2" \
      "$REPO_ROOT/$1"
  )
}

# Parse a junitxml report for one test node's outcome. Prints
# "<KIND><TAB><detail>", KIND being one of PASSED / PASSED_TEARDOWN_ERROR /
# FAILED_ASSERTION / ASSERTION_OUTSIDE_TEST / FAILED_OTHER / ERROR / SKIPPED /
# NO_TESTCASE.
#
# pytest's junitxml writer records a call-phase failure as <failure> whose
# message is the crash's exception text: "AssertionError: ..." for an
# assertion with a message or from numpy.testing, or the bare "assert ..."
# for a plain rewritten assert. Anything else in a <failure> is a different
# exception type. Setup and teardown errors are <error> elements whose
# message starts "failed on setup with" / "failed on teardown with";
# collection errors are "collection failure". A call failure followed by a
# teardown error is written as two <testcase> elements for the same test.
#
# An AssertionError counts only when it was raised in the test module
# itself: the <failure> text is pytest's traceback, whose final
# "<path>:<line>: <ExceptionType>" line names the frame that raised. An
# assert inside SpikeInterface or production code is not the test's check
# (numpy.testing's frames are hidden by __tracebackhide__, so its failures
# point at the calling test line). A relative path there is relative to
# pytest's working directory.
classify_junit() { # xml_path test_file pytest_cwd
  python3 - "$1" "$2" "$3" <<'PY'
import os
import re
import sys
import xml.etree.ElementTree as ET


def emit(kind, detail=""):
    detail = " ".join(detail.split())[:240]
    print(f"{kind}\t{detail}")
    raise SystemExit


def first_line(message):
    lines = (message or "").strip().splitlines()
    return lines[0] if lines else ""


try:
    root = ET.parse(sys.argv[1]).getroot()
except (ET.ParseError, OSError):
    emit("NO_TESTCASE", "no readable junitxml report")

cases = root.findall(".//testcase")
failures = [el for case in cases for el in case.findall("failure")]
errors = [el for case in cases for el in case.findall("error")]
skips = [el for case in cases for el in case.findall("skipped")]
teardown_errors = [
    el
    for el in errors
    if el.get("message", "").startswith("failed on teardown")
]
other_errors = [el for el in errors if el not in teardown_errors]

note = ""
if teardown_errors:
    note = (
        " (teardown also errored: "
        + first_line(teardown_errors[0].get("message"))
        + ")"
    )

# A collection error can be reported under more than one <testcase> name
# (with -x, pytest adds a session-level "collection failure" with an empty
# name beside the module's), so errors are classified before the one-test
# check.
if other_errors:
    emit("ERROR", first_line(other_errors[0].get("message")) + note)

tests = {(case.get("classname"), case.get("name")) for case in cases}
if len(tests) != 1:
    emit("NO_TESTCASE", f"expected one test in the report, found {len(tests)}")

if skips:
    emit("SKIPPED", first_line(skips[0].get("message")) + note)
if failures:
    message = failures[0].get("message", "")
    if not re.match(r"(AssertionError\b|assert )", message):
        emit("FAILED_OTHER", first_line(message) + note)
    locations = re.findall(
        r"^(.+):(\d+): ([A-Za-z_][\w.]*)$",
        failures[0].text or "",
        flags=re.MULTILINE,
    )
    if not locations:
        emit(
            "ASSERTION_OUTSIDE_TEST",
            "no crash location in the traceback: " + first_line(message) + note,
        )
    path, line, exc_type = locations[-1]
    where = f"{path}:{line}"
    crash = os.path.realpath(os.path.join(sys.argv[3], path))
    if exc_type != "AssertionError" or crash != os.path.realpath(sys.argv[2]):
        emit(
            "ASSERTION_OUTSIDE_TEST",
            f"{exc_type} raised at {where}, not in the test module: "
            + first_line(message)
            + note,
        )
    emit(
        "FAILED_ASSERTION",
        f"at {os.path.basename(path)}:{line}: " + first_line(message) + note,
    )
if teardown_errors:
    emit("PASSED_TEARDOWN_ERROR", note.strip())
emit("PASSED")
PY
}

BASELINE_LABELS=()
RESULT_LABELS=()
RESULT_DETAILS=()
ANY_BAD=0

for i in 0 1 2 3; do
  n=$((i + 1))
  file="${FILES[$i]}"
  node="${TEST_NODES[$i]}"
  test_file="$REPO_ROOT/${node%%::*}"
  desc="${DESCRIPTIONS[$i]}"
  baseline_xml="$WORK_DIR/regression_${n}_baseline.xml"
  xml="$WORK_DIR/regression_${n}.xml"

  echo
  echo "=== regression $n/4: $desc ==="
  echo "target: $file"
  echo "test:   $node"

  # (b) The same node must pass on the unmodified tree first.
  echo "--- baseline (unmodified tree) ---"
  set +e
  run_node "$node" "$baseline_xml"
  pytest_rc=$?
  set -e
  classified=$(classify_junit "$baseline_xml" "$test_file" "$RUN_DIR")
  kind=${classified%%$'\t'*}
  detail=${classified#*$'\t'}
  if [ "$kind" != "PASSED" ]; then
    BASELINE_LABELS+=("FAIL")
    RESULT_LABELS+=("INVALID")
    RESULT_DETAILS+=("baseline did not pass ($kind, pytest exit $pytest_rc${detail:+: $detail}) -- revert not applied")
    ANY_BAD=1
    echo "result: INVALID -- baseline did not pass ($kind); revert not applied"
    continue
  fi
  BASELINE_LABELS+=("pass")
  echo "baseline passed"

  # (c)-(e) Revert, rerun, restore.
  echo "--- reverted ---"
  # Marked current before the edit, so an interrupt during it still
  # restores the file (a checkout of the unmodified file is a no-op; the
  # file was verified clean at start).
  CURRENT_FILE="$file"
  echo "reverting $file"
  apply_revert "$i"
  echo "reverted (pattern matched exactly once)"

  set +e
  run_node "$node" "$xml"
  pytest_rc=$?
  set -e

  restore_current_file
  if [ "$RESTORE_FAILED" -ne 0 ]; then
    # Later rows would run against a still-reverted tree; stop here.
    exit 1
  fi
  echo "restored $file (pytest exit $pytest_rc)"

  # (f) Classify the reverted run.
  classified=$(classify_junit "$xml" "$test_file" "$RUN_DIR")
  kind=${classified%%$'\t'*}
  detail=${classified#*$'\t'}
  case "$kind" in
    FAILED_ASSERTION)
      label="PASS-OF-THE-CHECK"
      detail="assertion failed in the call phase, as required${detail:+: $detail}"
      ;;
    PASSED | PASSED_TEARDOWN_ERROR)
      label="SURVIVED"
      ANY_BAD=1
      detail="test PASSED despite the revert -- it does not catch this regression${detail:+ $detail}"
      ;;
    ASSERTION_OUTSIDE_TEST)
      label="INVALID"
      ANY_BAD=1
      detail="AssertionError not raised by the test's own check: $detail"
      ;;
    FAILED_OTHER)
      label="INVALID"
      ANY_BAD=1
      detail="call phase raised a non-assertion exception, not evidence the assertion bites: $detail"
      ;;
    ERROR)
      label="INVALID"
      ANY_BAD=1
      detail="test errored (setup/collection), not a call-phase failure: $detail"
      ;;
    SKIPPED)
      label="INVALID"
      ANY_BAD=1
      detail="test was skipped -- a skip proves nothing: $detail"
      ;;
    *)
      label="INVALID"
      ANY_BAD=1
      detail="$detail (pytest exit $pytest_rc)"
      ;;
  esac
  echo "result: $label -- $detail"

  RESULT_LABELS+=("$label")
  RESULT_DETAILS+=("$detail")
done

echo
echo "=== summary ==="
printf '%-3s %-70s %-9s %-20s\n' "#" "regression" "baseline" "result"
for i in 0 1 2 3; do
  n=$((i + 1))
  printf '%-3s %-70s %-9s %-20s\n' "$n" "${DESCRIPTIONS[$i]}" \
    "${BASELINE_LABELS[$i]}" "${RESULT_LABELS[$i]}"
  echo "    ${RESULT_DETAILS[$i]}"
done

final_dirty=$(git status --porcelain -- "${FILES[@]}")
if [ -n "$final_dirty" ]; then
  echo
  echo "FAIL: target files are not clean after restore:" >&2
  echo "$final_dirty" >&2
  ANY_BAD=1
fi
if [ "$RESTORE_FAILED" -ne 0 ]; then
  ANY_BAD=1
fi

echo
if [ "$ANY_BAD" -eq 0 ]; then
  echo "all four regression tests bite: each passed on the unmodified tree" \
    "and failed an assertion in its call phase when its fix was reverted," \
    "and all target files are clean."
  exit 0
else
  echo "one or more regressions did not bite as expected -- see SURVIVED /" \
    "INVALID rows above." >&2
  exit 1
fi
