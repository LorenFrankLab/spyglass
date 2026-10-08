"""Route the v2 CI suite from one list of expensive integration modules.

The ``unit`` workload shard automatically gets every other test, including
light database tests. This is independent of the database-free ``unit`` test
tier (``pytest tests/spikesorting/v2 --v2-tier unit``). New integration modules
are listed once here, rather than repeated as inclusions and exclusions in CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

INTEGRATION_MODULES = (
    "test_unitmatch.py",
    "test_unit_match_generation.py",
    "test_unitmatch_concat.py",
    "test_session_group_concat.py",
    "test_recompute.py",
    "test_curation_evaluation.py",
    "test_analyzer_lifecycle.py",
    "test_pipeline_session.py",
    "test_downstream_consumers.py",
    "test_clusterless_waveform_features.py",
    "test_fixture_ingestion.py",
    "test_brain_region_attribution.py",
    "test_recording_nwb.py",
    "test_motion_correction.py",
    "test_motion_consumers.py",
)
SHARDS = ("single-session", "integration", "unit")
REPO_ROOT = Path(__file__).resolve().parents[4]
SUITE = Path("tests/spikesorting/v2")


def shard_args(shard: str, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return pytest selection arguments, rejecting stale integration paths."""
    if shard not in SHARDS:
        raise ValueError(f"Unknown v2 shard: {shard!r}")
    paths = [SUITE / name for name in INTEGRATION_MODULES]
    missing = [str(path) for path in paths if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"Missing v2 integration modules: {missing}")
    if shard == "single-session":
        return [str(SUITE / "single_session")]
    if shard == "integration":
        return list(map(str, paths))
    return [
        str(SUITE),
        f"--ignore={SUITE / 'single_session'}",
        *(f"--ignore={path}" for path in paths),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=SHARDS)
    args, pytest_args = parser.parse_known_args(argv)
    selection = shard_args(args.shard)
    return subprocess.call(
        [sys.executable, "-m", "pytest", *selection, *pytest_args],
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
