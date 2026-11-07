"""The CI shards partition the suite, including newly added test modules."""

from pathlib import Path, PurePosixPath

from tests._ci_lanes import (
    INTEGRATION_MODULES,
    LANES,
    V2,
    collects,
    missing_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARDS = ("v2-single-session", "v2-integration", "v2-unit")


def test_ci_shards_cover_every_test_once():
    files = [
        PurePosixPath(path.relative_to(REPO_ROOT).as_posix())
        for path in (REPO_ROOT / V2).rglob("test_*.py")
    ]
    assert files
    for file in files:
        owners = [name for name in SHARDS if collects(LANES[name], file)]
        assert len(owners) == 1, (file, owners)
    assert len(INTEGRATION_MODULES) == len(set(INTEGRATION_MODULES))


def test_new_test_defaults_to_unit_shard():
    added = PurePosixPath(V2) / "test_new_regression.py"
    owners = [name for name in SHARDS if collects(LANES[name], added)]
    assert owners == ["v2-unit"]


def test_stale_integration_path_fails_instead_of_losing_coverage(tmp_path):
    assert all(not missing_paths(lane, REPO_ROOT) for lane in LANES.values())
    missing = missing_paths(LANES["v2-integration"], tmp_path)
    assert missing == sorted(f"{V2}/{name}" for name in INTEGRATION_MODULES)
