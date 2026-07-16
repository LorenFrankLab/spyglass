#!/usr/bin/env bash
# Verify the v2 NWB fixtures used by the parity matrix are present
# locally and (if a v1 worktree exists with the fixture path) match
# across worktrees by sha256.
#
# A0.4 preflight: the v1 capture and the v2 test both fingerprint the
# NWB by sha256; if the two worktrees see different bytes the
# nwb_sha256 invariant FAILs in a misleading way. This script
# surfaces that mismatch BEFORE captures run.
#
# Worktree note: on this repo the master branch (where
# /tmp/spyglass-master typically points) does not carry the
# tests/spikesorting/v2/ tree, so the cross-worktree check usually
# reports "v1 worktree lacks fixture" -- that's expected and means the
# captures will read the v2 checkout's copy via the script-level cwd
# pin (see capture_polymer_clusterless.sh).
#
# Usage:
#     bash tests/spikesorting/v2/scripts/verify_fixture_sync.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V2_REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
V1_WORKTREE=${SPIKESORTING_V1_WORKTREE:-/tmp/spyglass-master}

FIXTURES=(
  mearec_polymer_smoke
  mearec_polymer_128ch_60s
)

exit_code=0
for FIX in "${FIXTURES[@]}"; do
  V2_PATH="$V2_REPO_ROOT/tests/spikesorting/v2/fixtures/${FIX}.nwb"
  V1_PATH="$V1_WORKTREE/tests/spikesorting/v2/fixtures/${FIX}.nwb"

  if [ ! -f "$V2_PATH" ]; then
    echo "$FIX: V2 MISSING ($V2_PATH)" >&2
    exit_code=1
    continue
  fi

  V2_HASH=$(sha256sum "$V2_PATH" | cut -d' ' -f1)
  if [ ! -f "$V1_PATH" ]; then
    echo "$FIX: V2=${V2_HASH:0:12}  V1=(absent: $V1_PATH)"
    continue
  fi
  V1_HASH=$(sha256sum "$V1_PATH" | cut -d' ' -f1)
  if [ "$V1_HASH" = "$V2_HASH" ]; then
    echo "$FIX: MATCH ${V1_HASH:0:12}"
  else
    echo "$FIX: DIFFER V1=${V1_HASH:0:12} V2=${V2_HASH:0:12}" >&2
    exit_code=1
  fi
done

exit "$exit_code"
