"""A duplicate or interrupted compute cannot change the scientific winner."""

import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest
import spikeinterface as si

from tests.spikesorting.v2._sorter_stub import plant_sorter


@pytest.mark.parametrize("winner_already_committed", [True, False])
def test_duplicate_compute_keeps_winners_analyzer(
    planted_two_unit_sort, monkeypatch, winner_already_committed
):
    from spyglass.spikesorting.v2.recording import _unlink_staged_analysis_file
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._tripart_helpers import (
        assert_no_staged_analysis_files,
    )

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

    plant_sorter(monkeypatch, alternate)
    # Exercise the real compute/NWB/insert path with the two possible orderings.
    # Only the sorter is substituted to make the nondeterministic difference
    # explicit; _allow_insert reproduces DataJoint's populate dispatch context.
    monkeypatch.setattr(Sorting, "_allow_insert", True)
    loser = table.make_compute(key, *fetched)
    try:
        if not winner_already_committed:
            plant_sorter(monkeypatch, lambda *a, **k: original)
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
        # A direct make_insert call (only ever done by this test) never runs
        # inside populate(), so StagedOutputCleanupMixin never records this
        # attempt's staged units NWB; unlink it ourselves.
        _unlink_staged_analysis_file(loser.analysis_file_name, context="test")
    assert_no_staged_analysis_files([loser.analysis_file_name])


@pytest.mark.parametrize("kill_point", ["build", "swap"])
def test_killed_publisher_staging_is_reclaimable(
    tmp_path, restore_custom_config, kill_point
):
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
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
from spyglass.spikesorting.v2._storage import analyzer_cache as cache
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
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
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


def test_cleanup_skips_staging_still_in_use(
    tmp_path, restore_custom_config, monkeypatch
):
    """A staging folder whose files are still open is left for later.

    On NFS, removing a file another handle still maps leaves a ``.nfs``
    placeholder that cannot be removed (``EBUSY``), so the folder cannot be
    removed either. Such a folder is in use, not abandoned: the cleanup
    leaves it, reports only what it removed, and does not raise.
    """
    import errno

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sorting_id = str(uuid.uuid4())
    canonical = cache.analyzer_path(sorting_id, "test")
    busy = cache._publish_sibling(canonical, "trash")
    free = cache._publish_sibling(canonical, "trash")
    for folder in (busy, free):
        folder.mkdir(parents=True)
        (folder / "marker").write_text("old copy")
    real_rmtree = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path) == busy:
            raise OSError(errno.EBUSY, "Device or resource busy", ".nfs0001")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cache.shutil, "rmtree", rmtree)
    assert sorted(cache.cleanup_analyzer_staging(sorting_id)) == sorted(
        [str(busy), str(free)]
    )
    assert cache.cleanup_analyzer_staging(sorting_id, dry_run=False) == [
        str(free)
    ]
    assert busy.exists() and not free.exists()


def test_orphan_audit_skips_orphan_folder_still_in_use(
    dj_conn, tmp_path, restore_custom_config, monkeypatch
):
    """A disk-side orphan whose files are still open is left for later.

    On NFS a folder that a process still maps cannot be removed (``EBUSY`` /
    ``ENOTEMPTY`` from its ``.nfs`` placeholder). The confirmed audit leaves
    it, removes the other orphans, and does not raise.
    """
    import errno

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache
    from spyglass.spikesorting.v2.sorting import Sorting

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sorting_id = uuid.uuid4()
    busy = cache.analyzer_path(sorting_id, "busy")
    free = cache.analyzer_path(sorting_id, "free")
    for folder in (busy, free):
        (folder / "extensions" / "waveforms").mkdir(parents=True)
        (folder / "extensions" / "waveforms" / "waveforms.npy").write_text("x")
    report = Sorting.find_orphaned_analyzer_folders(sorting_id=sorting_id)
    assert sorted(report["disk_side"]) == sorted([str(busy), str(free)])
    real_rmtree = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path) == busy:
            raise OSError(errno.ENOTEMPTY, "Directory not empty", "waveforms")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cache.shutil, "rmtree", rmtree)
    monkeypatch.setattr(dj.utils, "user_choice", lambda *a, **k: "yes")
    Sorting.find_orphaned_analyzer_folders(sorting_id=sorting_id, dry_run=False)
    assert busy.exists() and not free.exists()


def test_orphan_audit_reclaims_only_abandoned_staging(
    dj_conn, tmp_path, restore_custom_config, monkeypatch
):
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
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


def test_copy_keeps_the_analyzers_sampling_frequency(tmp_path):
    """A copied analyzer reports exactly its source's sampling frequency.

    v2 reads traces at a rate estimated from the timestamps (e.g.
    30000.000000000182 Hz) while the stored units say 30000.0; the analyzer
    keeps the recording's rate. A copy rebuilt from the units' provenance
    reverted to 30000.0, so durations and firing rates on the copy differed
    from the source in the last bits.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        copy_analyzer_folder,
        load_analyzer_folder,
    )

    rate = 30000.000000000182
    recording = si.generate_recording(
        num_channels=2, sampling_frequency=rate, durations=[2.0], seed=0
    )
    sorting = si.NumpySorting.from_unit_dict(
        {1: np.array([100, 5000, 40000])}, sampling_frequency=30000.0
    )
    source = si.create_sorting_analyzer(
        sorting,
        recording,
        format="binary_folder",
        folder=tmp_path / "source.analyzer",
        sparse=False,
    )
    source.compute("random_spikes", seed=0)
    source = load_analyzer_folder(tmp_path / "source.analyzer")
    copy = copy_analyzer_folder(source, tmp_path / "copy.analyzer")
    assert copy.sampling_frequency == source.sampling_frequency
    assert copy.get_total_duration() == source.get_total_duration()
