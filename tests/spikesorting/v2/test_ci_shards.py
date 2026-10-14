"""The CI shards partition the suite, including newly added test modules."""

from pathlib import Path

import pytest

from tests.spikesorting.v2.scripts.run_ci_shard import (
    INTEGRATION_MODULES,
    REPO_ROOT,
    SHARDS,
    SUITE,
    shard_args,
)


def _selected_test_files(args, repo_root):
    ignored = {
        repo_root / arg.removeprefix("--ignore=")
        for arg in args
        if arg.startswith("--ignore=")
    }
    files = set()
    for arg in args:
        if arg.startswith("--"):
            continue
        path = repo_root / arg
        candidates = path.rglob("test_*.py") if path.is_dir() else [path]
        files.update(
            candidate
            for candidate in candidates
            if not any(
                candidate == skip or skip in candidate.parents
                for skip in ignored
            )
        )
    return files


def test_ci_shards_cover_every_test_once():
    shards = [
        _selected_test_files(shard_args(name), REPO_ROOT) for name in SHARDS
    ]
    expected = set((REPO_ROOT / SUITE).rglob("test_*.py"))
    assert set.union(*shards) == expected
    assert sum(map(len, shards)) == len(expected)
    assert len(INTEGRATION_MODULES) == len(set(INTEGRATION_MODULES))


def test_new_test_defaults_to_unit_shard(tmp_path):
    suite = tmp_path / SUITE
    suite.mkdir(parents=True)
    for name in INTEGRATION_MODULES:
        (suite / name).touch()
    added = suite / "test_new_regression.py"
    added.touch()
    assert added in _selected_test_files(
        shard_args("unit", repo_root=tmp_path), tmp_path
    )


def test_stale_integration_path_fails_instead_of_losing_coverage(tmp_path):
    with pytest.raises(ValueError, match="Missing v2 integration modules"):
        shard_args("integration", repo_root=tmp_path)
