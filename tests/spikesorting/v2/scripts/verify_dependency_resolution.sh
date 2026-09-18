#!/usr/bin/env bash
# Verify that the dependency pins this repo declares actually resolve, and
# that each install lane lands on the stack it is supposed to.
#
# Two layers:
#
#   1. ``uv pip compile`` resolves the pip requirements once per install lane
#      (the base requirements, then the dlc / moseq-cpu / spikesorting-v2 /
#      spikesorting-v2-matching extras) and the resolved pins are asserted
#      against that lane's contract: one SpikeInterface for everybody, the
#      numpy<2 line for DeepLabCut and keypoint-moseq, the numpy 2 line for
#      the v2 spike-sorting extras. The dlc and moseq-cpu lanes also compile
#      mountainsort4 alongside pyproject.toml, because the environment files
#      pip-install it next to ``..[<extra>]`` and it belongs to no extra, so
#      compiling pyproject.toml alone would leave it unresolved.
#
#   2. ``conda create --dry-run`` solves the *conda* section of each
#      environment file. Between them the two layers cover a file completely:
#      layer 2 takes its conda section, layer 1 takes both entries of its pip
#      section.
#
# The conda solve uses each file's own channel list, read out of the file and
# passed with --override-channels. That keeps the franklab and edeno channels
# the files rely on (position_tools, non_local_detector, ripple_detection and
# track_linearization live there) while keeping whatever extra channels happen
# to sit in the developer's ~/.condarc out of the result.
#
# Both layers solve for Linux x86_64 by default -- conda subdir linux-64, pip
# platform x86_64-manylinux_2_28 -- not for the machine running the script, so
# the two layers describe one target and the answer does not change with the
# developer's laptop. These are Linux/GPU install recipes anyway:
# environment_dlc.yml asks for cudatoolkit=11.3, which has no macOS build at
# all.
#
# Solving linux-64 from a non-Linux host also needs CONDA_OVERRIDE_GLIBC. conda
# derives the ``__glibc`` virtual package from the running kernel, so off Linux
# it is absent and every package that declares a glibc floor -- which is every
# recent conda-forge build -- becomes uninstallable. The solver does not say
# so: it quietly backtracks onto ancient builds that predate the metadata, and
# takes many minutes to do it. The override states the target's glibc instead,
# matched to the manylinux tag above.
#
# SOLVE_SUBDIR, SOLVE_PYTHON_PLATFORM and SOLVE_GLIBC retarget all of this;
# keep them describing the same platform.
#
# Reaches the network and takes minutes, so it is a script rather than a test.
#
# Usage:
#     bash tests/spikesorting/v2/scripts/verify_dependency_resolution.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
cd "$REPO_ROOT"

PYTHON_VERSION=3.11
SOLVE_SUBDIR=${SOLVE_SUBDIR:-linux-64}
SOLVE_PYTHON_PLATFORM=${SOLVE_PYTHON_PLATFORM:-x86_64-manylinux_2_28}
SOLVE_GLIBC=${SOLVE_GLIBC:-2.28}
export CONDA_OVERRIDE_GLIBC="$SOLVE_GLIBC"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# The environment files' pip sections install this package plus mountainsort4,
# which no pyproject extra declares. Compiling this file alongside
# pyproject.toml makes the lane's resolution match what those files install.
MS4_REQUIREMENT="$WORK_DIR/mountainsort4.in"
echo "mountainsort4" >"$MS4_REQUIREMENT"

exit_code=0

fail() { # lane, message
  echo "FAIL [$1] $2" >&2
  exit_code=1
}

# Print one list field of a YAML environment file, one plain string per line
# (the nested ``pip:`` mapping is skipped).
yaml_field() { # yml, field
  python3 - "$1" "$2" <<'PY'
import sys

import yaml

document = yaml.safe_load(open(sys.argv[1]))
for item in document[sys.argv[2]]:
    if isinstance(item, str):
        print(item)
PY
}

# Assert the lane resolved a pin matching a regex, and echo what it matched.
assert_pin() { # lane, resolved file, regex, description
  local lane=$1 resolved=$2 regex=$3 description=$4 matched
  if matched=$(grep -E "$regex" "$resolved"); then
    echo "  $lane: $matched"
  else
    fail "$lane" "no resolved pin for $description (/$regex/)"
  fi
}

# Assert the lane resolved a package at or above a floor.
assert_min_version() { # lane, resolved file, package, floor
  local lane=$1 resolved=$2 package=$3 floor=$4 line version
  line=$(grep -E "^${package}==" "$resolved" | head -n 1) || true
  if [ -z "$line" ]; then
    fail "$lane" "$package is not in the resolved set"
    return
  fi
  version=${line#*==}
  if python3 - "$version" "$floor" <<'PY'
import re
import sys


def release(version):
    """The first three numeric components, as a comparable tuple."""
    return tuple(int(n) for n in re.findall(r"\d+", version.split("+")[0])[:3])


sys.exit(0 if release(sys.argv[1]) >= release(sys.argv[2]) else 1)
PY
  then
    echo "  $lane: $line (floor $package>=$floor)"
  else
    fail "$lane" "$line is below the declared floor $package>=$floor"
  fi
}

echo "host:           $(uname -sm)"
echo "python target:  $PYTHON_VERSION"
echo "uv:             $(uv --version)"
echo "conda:          $(conda --version)"
echo "solve target:   $SOLVE_PYTHON_PLATFORM (pip) / $SOLVE_SUBDIR glibc" \
  "$SOLVE_GLIBC (conda)"
echo "date (UTC):     $(date -u)"

echo
echo "== pyproject resolution (uv pip compile) =="
for lane in base dlc moseq-cpu spikesorting-v2 spikesorting-v2-matching; do
  resolved="$WORK_DIR/$lane.txt"
  compile_args=(
    --quiet
    --python-version "$PYTHON_VERSION"
    --python-platform "$SOLVE_PYTHON_PLATFORM"
  )
  if [ "$lane" != base ]; then
    compile_args+=(--extra "$lane")
  fi

  # The lanes whose environment file also pip-installs mountainsort4.
  inputs=(pyproject.toml)
  case "$lane" in
    dlc | moseq-cpu) inputs+=("$MS4_REQUIREMENT") ;;
  esac

  echo "$lane:"
  if ! uv pip compile "${compile_args[@]}" "${inputs[@]}" -o "$resolved"; then
    fail "$lane" "uv pip compile failed"
    continue
  fi

  assert_pin "$lane" "$resolved" '^spikeinterface==0\.104\.3$' \
    "the SpikeInterface hard pin"

  case "$lane" in
    dlc)
      assert_min_version "$lane" "$resolved" deeplabcut 3.0
      assert_pin "$lane" "$resolved" '^numpy==1\.' "the numpy 1.x line"
      assert_pin "$lane" "$resolved" '^mountainsort4==' "mountainsort4"
      ;;
    moseq-cpu)
      assert_min_version "$lane" "$resolved" keypoint-moseq 0.6
      assert_pin "$lane" "$resolved" '^numpy==1\.' "the numpy 1.x line"
      assert_pin "$lane" "$resolved" '^mountainsort4==' "mountainsort4"
      ;;
    spikesorting-v2 | spikesorting-v2-matching)
      assert_pin "$lane" "$resolved" '^numpy==2\.' "the numpy 2.x line"
      ;;
  esac
done

echo
echo "== conda section resolution (conda create --dry-run) =="
for env_file in environment.yml environment_dlc.yml environment_moseq_cpu.yml \
  environment_moseq_gpu.yml; do
  yml="environments/$env_file"
  lane="conda:$env_file"
  log="$WORK_DIR/${env_file%.yml}.conda.log"

  channel_args=()
  while IFS= read -r channel; do
    channel_args+=(-c "$channel")
  done < <(yaml_field "$yml" channels)

  specs=()
  while IFS= read -r spec; do
    specs+=("$spec")
  done < <(yaml_field "$yml" dependencies)

  if [ "${#channel_args[@]}" -eq 0 ] || [ "${#specs[@]}" -eq 0 ]; then
    fail "$lane" "could not read channels and dependencies out of $yml
       (is PyYAML installed for the python3 on PATH?)"
    continue
  fi

  echo "$lane: solving ${#specs[@]} specs for $SOLVE_SUBDIR"
  if conda create --dry-run --subdir "$SOLVE_SUBDIR" --override-channels \
    "${channel_args[@]}" -n "verify-deps-${env_file%.yml}" "${specs[@]}" \
    >"$log" 2>&1; then
    grep -E '^  (numpy|scipy|python|jax|non_local_detector|pytorch) +' "$log" |
      sed 's/^ */  /' || true
    echo "  $lane: conda section solves"
  else
    fail "$lane" "conda section does not solve"
    tail -n 20 "$log" >&2
  fi
done

echo
if [ "$exit_code" -eq 0 ]; then
  echo "all lanes resolve as declared"
else
  echo "one or more lanes failed; see the FAIL lines above" >&2
fi
exit "$exit_code"
