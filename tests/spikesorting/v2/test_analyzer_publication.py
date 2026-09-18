"""A duplicate or interrupted compute cannot change the scientific winner."""

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest
import spikeinterface as si


@pytest.mark.parametrize("winner_already_committed", [True, False])
def test_duplicate_compute_keeps_winners_analyzer(
    planted_two_unit_sort, monkeypatch, winner_already_committed
):
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    table, key = Sorting(), planted_two_unit_sort
    clear_curations_for(key)
    original = table.get_sorting(key)
    fetched = table.make_fetch(key)
    if not winner_already_committed:
        (Sorting & key).super_delete(warn=False)

    def alternate(*args, **kwargs):
        return si.NumpySorting.from_unit_dict(
            {
                int(u): original.get_unit_spike_train(u) + 20
                for u in original.unit_ids
            },
            sampling_frequency=original.get_sampling_frequency(),
        )

    monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(alternate))
    # Exercise the real compute/NWB/insert path with the two possible orderings.
    # Only the sorter is substituted to make the nondeterministic difference
    # explicit; _allow_insert reproduces DataJoint's populate dispatch context.
    monkeypatch.setattr(Sorting, "_allow_insert", True)
    loser = table.make_compute(key, *fetched)
    try:
        if not winner_already_committed:
            monkeypatch.setattr(
                Sorting, "_run_sorter", staticmethod(lambda *a, **k: original)
            )
            winner = table.make_compute(key, *fetched)
            table.make_insert(key, *winner)
        with pytest.raises(dj.errors.DuplicateError):
            table.make_insert(key, *loser)
        analyzer = table.get_analyzer(key)
        stored = table.get_sorting(key)
        for unit_id in original.unit_ids:
            np.testing.assert_array_equal(
                stored.get_unit_spike_train(unit_id),
                original.get_unit_spike_train(unit_id),
            )
            np.testing.assert_array_equal(
                analyzer.sorting.get_unit_spike_train(unit_id),
                stored.get_unit_spike_train(unit_id),
            )
        assert not loser.staged_analyzer.folder.exists()
        assert not Path(loser.staged_analyzer._lock.lock_file).exists()
    finally:
        loser.staged_analyzer.close()


@pytest.mark.parametrize("kill_point", ["build", "swap"])
def test_killed_publisher_staging_is_reclaimable(
    tmp_path, restore_custom_config, kill_point
):
    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_path,
        cleanup_analyzer_staging,
        publish_analyzer_atomically,
        remove_analyzer_cache,
    )

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sorting_id = str(uuid.uuid4())
    canonical = analyzer_path(sorting_id, "test")

    def build(folder):
        folder.mkdir()
        (folder / "marker").write_text("complete")

    publish_analyzer_atomically(canonical, build)
    child = """
import os, signal, sys
from pathlib import Path
import datajoint as dj
from spyglass.spikesorting.v2 import _analyzer_cache as cache
dj.config.setdefault('custom', {})['spikesorting_v2_analyzer_dir'] = sys.argv[1]
canonical = cache.analyzer_path(sys.argv[2], 'test')
replace = os.replace
def interrupted_replace(src, dst):
    replace(src, dst)
    if Path(src) == canonical and sys.argv[3] == 'swap':
        os.kill(os.getpid(), signal.SIGKILL)
cache.os.replace = interrupted_replace
def build(folder):
    folder.mkdir()
    (folder / 'marker').write_text('interrupted')
    if sys.argv[3] == 'build':
        os.kill(os.getpid(), signal.SIGKILL)
cache.publish_analyzer_atomically(canonical, build)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), sorting_id, kill_point],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == -signal.SIGKILL, result.stderr
    abandoned = cleanup_analyzer_staging(sorting_id)
    assert len(abandoned) == (1 if kill_point == "build" else 2)
    assert all(Path(folder).exists() for folder in abandoned)
    publish_analyzer_atomically(canonical, build)
    assert (canonical / "marker").read_text() == "complete"
    assert remove_analyzer_cache(sorting_id)
    assert not list(tmp_path.glob(".*.analyzer"))
    assert not list(tmp_path.glob(".*.analyzer.lock"))


def test_cleanup_keeps_active_attempt(tmp_path, restore_custom_config):
    from spyglass.spikesorting.v2._analyzer_cache import (
        StagedAnalyzer,
        analyzer_path,
        cleanup_analyzer_staging,
        remove_analyzer_cache,
    )

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sorting_id = uuid.uuid4()
    with StagedAnalyzer(analyzer_path(sorting_id, "test")) as staged:
        staged.folder.mkdir()
        (staged.folder / "marker").write_text("active")
        assert cleanup_analyzer_staging(sorting_id) == []
        assert cleanup_analyzer_staging(sorting_id, dry_run=False) == []
        assert not remove_analyzer_cache(sorting_id)
        assert (staged.folder / "marker").read_text() == "active"
    assert not staged.folder.exists()


def test_orphan_audit_reclaims_only_abandoned_staging(
    dj_conn, tmp_path, restore_custom_config, monkeypatch
):
    from spyglass.spikesorting.v2._analyzer_cache import (
        StagedAnalyzer,
        analyzer_path,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sorting_id = uuid.uuid4()
    canonical = analyzer_path(sorting_id, "test")
    # An older PID-named build has no surviving ownership lock.
    abandoned = tmp_path / f".{canonical.stem}.build-123.analyzer"
    abandoned.mkdir()
    with StagedAnalyzer(canonical) as active:
        active.folder.mkdir()
        report = Sorting.find_orphaned_analyzer_folders(sorting_id=sorting_id)
        assert report["staging"] == [str(abandoned)]
        assert abandoned.exists()
        monkeypatch.setattr(dj.utils, "user_choice", lambda *a, **k: "yes")
        Sorting.find_orphaned_analyzer_folders(
            sorting_id=sorting_id, dry_run=False
        )
        assert not abandoned.exists()
        assert active.folder.exists()
