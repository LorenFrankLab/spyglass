import inspect
import os
from pathlib import Path

import pytest

_HOUR_NS = 60 * 60 * 1_000_000_000


class _FakeExternalTable:
    def __init__(self, external_paths=()):
        self.external_paths = list(external_paths)

    def fetch_external_paths(self):
        return self.external_paths


class _EmptyQuery:
    def proj(self):
        return self

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def delete_quick(self):
        raise AssertionError("empty query should not be deleted")


@pytest.fixture
def common_nwbfile(common):
    """Return the common NWBFile module."""
    return common.common_nwbfile


@pytest.fixture
def lockfile(base_dir, teardown):
    lockfile = base_dir / "temp.lock"
    lockfile.touch()
    os.environ["NWB_LOCK_FILE"] = str(lockfile)
    yield lockfile
    if teardown:
        lockfile.unlink(missing_ok=True)


def _table(common_nwbfile, analysis_dir, tracked=()):
    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable(
        [(str(index), path) for index, path in enumerate(tracked)]
    )
    return table


def _make_policy(*, min_file_age_hours=0, **kwargs):
    """Build a CleanupPolicy for tests (immediate cleanup by default).

    Imported lazily: importing ``spyglass.common`` at module scope needs a
    live database, so this runs only once the ``common`` fixtures have set the
    package up.
    """
    from spyglass.common._nwbfile_cleanup import CleanupPolicy

    return CleanupPolicy(min_file_age_hours=min_file_age_hours, **kwargs)


def _plan(table, custom_tables=(), *, policy=None, min_age=0, now_ns=None):
    if policy is None:
        policy = _make_policy(min_file_age_hours=min_age)
    return table._build_untracked_file_plan(
        list(custom_tables),
        policy=policy,
        now_ns=now_ns,
    )


def _destructive_plan(common_nwbfile, analysis_dir, tracked=()):
    """Build a deletion plan with one real tracked file as its safety anchor."""
    tracked_anchor = analysis_dir / "tracked_anchor.nwb"
    tracked_anchor.write_text("tracked")
    return _plan(
        _table(common_nwbfile, analysis_dir, [tracked_anchor, *tracked])
    )


def test_add_to_lock(common_nwbfile, lockfile, mini_copy_name):
    common_nwbfile.Nwbfile.add_to_lock(mini_copy_name)
    assert mini_copy_name in lockfile.read_text()

    with pytest.raises(FileNotFoundError):
        common_nwbfile.Nwbfile.add_to_lock("non-existent-file.nwb")


def test_nwbfile_cleanup(common_nwbfile):
    before = len(common_nwbfile.Nwbfile.fetch())
    common_nwbfile.Nwbfile.cleanup(delete_files=False)
    after = len(common_nwbfile.Nwbfile.fetch())
    assert before == after, "Nwbfile cleanup changed table entry count."


def test_cleanup_controls_are_keyword_only(common_nwbfile):
    parameters = inspect.signature(
        common_nwbfile.AnalysisNwbfile.cleanup
    ).parameters

    assert parameters["dry_run"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in (
        "max_delete_fraction",
        "max_delete_to_tracked_ratio",
        "min_file_age_hours",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_plan_classifies_tracked_untracked_and_empty(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    tracked = analysis_dir / "tracked.nwb"
    tracked.write_text("tracked")
    tracked_empty = analysis_dir / "tracked_empty.nwb"
    tracked_empty.touch()
    untracked = analysis_dir / "untracked.nwb"
    untracked.write_text("untracked")
    empty = analysis_dir / "empty.nwb"
    empty.touch()

    plan = _plan(_table(common_nwbfile, analysis_dir, [tracked, tracked_empty]))

    assert plan.scanned_files == {
        tracked.resolve(),
        tracked_empty.resolve(),
        untracked.resolve(),
        empty.resolve(),
    }
    assert plan.files_to_delete == {untracked.resolve(), empty.resolve()}
    assert plan.untracked_files == {untracked.resolve()}
    assert plan.empty_files == {empty.resolve()}


def test_custom_table_paths_are_included_in_snapshot(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    common_path = analysis_dir / "common.nwb"
    custom_path = analysis_dir / "custom.nwb"
    common_path.write_text("common")
    custom_path.write_text("custom")

    common_table = _table(common_nwbfile, analysis_dir, [common_path])
    custom_table = type(
        "CustomTable",
        (),
        {"_ext_tbl": _FakeExternalTable([("custom", custom_path)])},
    )()

    plan = _plan(common_table, [custom_table])

    assert plan.files_to_delete == set()
    assert plan.tracked_files == {common_path.resolve(), custom_path.resolve()}


def test_age_gate_uses_mtime_only(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    candidate = analysis_dir / "candidate.nwb"
    candidate.write_text("old")
    old_mtime_ns = candidate.stat().st_mtime_ns - 25 * _HOUR_NS
    os.utime(candidate, ns=(old_mtime_ns, old_mtime_ns))

    # Metadata maintenance leaves mtime old but makes ctime current. A
    # max(mtime, ctime) gate would defer this file; the mtime-only gate does not.
    candidate.chmod(0o600)
    now_ns = candidate.stat().st_ctime_ns
    plan = _plan(
        _table(common_nwbfile, analysis_dir),
        min_age=24,
        now_ns=now_ns,
    )

    assert plan.files_to_delete == {candidate.resolve()}
    assert plan.deferred_recent_files == set()


def test_recent_file_is_deferred_at_24_hour_gate(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    candidate = analysis_dir / "candidate.nwb"
    candidate.write_text("recent")
    mtime_ns = candidate.stat().st_mtime_ns
    table = _table(common_nwbfile, analysis_dir)

    recent = _plan(
        table,
        min_age=24,
        now_ns=mtime_ns + 24 * _HOUR_NS - 1,
    )
    boundary = _plan(
        table,
        min_age=24,
        now_ns=mtime_ns + 24 * _HOUR_NS,
    )

    assert recent.deferred_recent_files == {candidate.resolve()}
    assert recent.files_to_delete == set()
    assert boundary.files_to_delete == {candidate.resolve()}


def test_zero_age_gate_includes_future_mtime(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    candidate = analysis_dir / "future.nwb"
    candidate.write_text("future")
    future_ns = candidate.stat().st_mtime_ns + _HOUR_NS
    os.utime(candidate, ns=(future_ns, future_ns))

    plan = _plan(
        _table(common_nwbfile, analysis_dir),
        min_age=0,
        now_ns=future_ns - _HOUR_NS,
    )

    assert plan.files_to_delete == {candidate.resolve()}


def test_leaf_symlink_deletes_cross_volume_target_and_leaf(
    common_nwbfile, tmp_path
):
    volume = tmp_path / "volume_b"
    volume.mkdir()
    target = volume / "external.nwb"
    target.write_text("external")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    leaf = analysis_dir / "external.nwb"
    leaf.symlink_to(target)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    plan.execute(dry_run=False)

    assert not target.exists()
    assert not leaf.is_symlink()


def test_tracked_leaf_symlink_is_preserved(common_nwbfile, tmp_path):
    volume = tmp_path / "volume_b"
    volume.mkdir()
    target = volume / "external.nwb"
    target.write_text("external")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    leaf = analysis_dir / "external.nwb"
    leaf.symlink_to(target)

    plan = _plan(_table(common_nwbfile, analysis_dir, [leaf]))
    plan.execute(dry_run=False)

    assert target.exists()
    assert leaf.is_symlink()
    assert plan.files_to_delete == set()


def test_broken_leaf_removes_only_the_leaf(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    missing = tmp_path / "volume_b" / "missing.nwb"
    leaf = analysis_dir / "missing.nwb"
    leaf.symlink_to(missing)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    assert plan.broken_links == {missing.resolve()}

    plan.execute(dry_run=False)

    assert not leaf.is_symlink()
    assert not missing.exists()


def test_vanished_live_target_still_removes_leaf(common_nwbfile, tmp_path):
    target = tmp_path / "volume_b" / "external.nwb"
    target.parent.mkdir()
    target.write_text("external")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    leaf = analysis_dir / "external.nwb"
    leaf.symlink_to(target)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    target.unlink()
    plan.execute(dry_run=False)

    assert not leaf.is_symlink()


@pytest.mark.parametrize("cycle_length", [1, 2])
def test_cyclic_leaf_symlinks_are_removed(
    common_nwbfile, tmp_path, cycle_length
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    leaves = [
        analysis_dir / f"cycle_{index}.nwb" for index in range(cycle_length)
    ]
    for index, leaf in enumerate(leaves):
        leaf.symlink_to(leaves[(index + 1) % cycle_length].name)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    plan.execute(dry_run=False)

    assert not any(leaf.is_symlink() for leaf in leaves)


def test_symlinked_directories_are_not_traversed(common_nwbfile, tmp_path):
    # Cleanup deletes files, so it must not follow a directory symlink out of
    # analysis_dir and delete untracked files in an unrelated tree. Leaf
    # symlinks are the only cross-volume mechanism (tested separately).
    external_dir = tmp_path / "volume_b" / "analysis"
    external_dir.mkdir(parents=True)
    external_file = external_dir / "remote.nwb"
    external_file.write_text("remote")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    directory_link = analysis_dir / "remote_volume"
    directory_link.symlink_to(external_dir, target_is_directory=True)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    assert external_file.resolve() not in plan.scanned_files
    assert plan.files_to_delete == set()

    plan.execute(dry_run=False)

    assert external_file.exists()
    assert directory_link.is_symlink()


def test_file_below_symlinked_directory_is_not_scanned(
    common_nwbfile, tmp_path
):
    # Even a tracked file behind a directory symlink is simply never reached,
    # so nothing in that external tree is touched.
    external_dir = tmp_path / "volume_b" / "analysis"
    external_dir.mkdir(parents=True)
    external_file = external_dir / "tracked.nwb"
    external_file.write_text("tracked")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "remote_volume").symlink_to(
        external_dir, target_is_directory=True
    )

    plan = _plan(_table(common_nwbfile, analysis_dir, [external_file]))
    plan.execute(dry_run=False)

    assert external_file.exists()
    assert external_file.resolve() not in plan.scanned_files
    assert plan.files_to_delete == set()


def test_directory_symlink_cycle_does_not_hang_or_delete(
    common_nwbfile, tmp_path
):
    # With directory symlinks unfollowed, the walk cannot loop (no cycle
    # detection needed) and never reaches the external tree.
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_file = external_dir / "remote.nwb"
    external_file.write_text("remote")

    (analysis_dir / "external").symlink_to(
        external_dir, target_is_directory=True
    )
    (external_dir / "back").symlink_to(analysis_dir, target_is_directory=True)

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert plan.scanned_files == set()
    assert plan.files_to_delete == set()
    assert external_file.exists()


def test_duplicate_aliases_delete_target_once_and_remove_all_leaves(
    common_nwbfile, tmp_path
):
    target = tmp_path / "volume_b" / "shared.nwb"
    target.parent.mkdir()
    target.write_text("shared")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    first = analysis_dir / "first.nwb"
    second = analysis_dir / "second.nwb"
    first.symlink_to(target)
    second.symlink_to(target)

    plan = _destructive_plan(common_nwbfile, analysis_dir)
    assert plan.files_to_delete == {target.resolve()}

    plan.execute(dry_run=False)

    assert not target.exists()
    assert not first.is_symlink()
    assert not second.is_symlink()


def test_dry_run_reports_logical_bytes_without_deleting(
    common_nwbfile, tmp_path, caplog
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    candidate = analysis_dir / "candidate.nwb"
    candidate.write_bytes(b"12345")

    plan = _plan(_table(common_nwbfile, analysis_dir))
    with caplog.at_level("INFO"):
        plan.execute(dry_run=True)

    assert plan.files_to_delete == {candidate.resolve()}
    assert candidate.exists()
    assert plan.candidate_bytes == 5
    assert "Cleanup plan would be refused" in caplog.text
    assert "5 logical candidate bytes" in caplog.text


def test_nonregular_nwb_entry_is_skipped(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fifo = analysis_dir / "pipe.nwb"
    os.mkfifo(fifo)

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert plan.scanned_files == set()
    assert plan.files_to_delete == set()


def test_unlink_error_is_logged_and_other_candidates_continue(
    common_nwbfile, tmp_path, monkeypatch, caplog
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    blocked = analysis_dir / "blocked.nwb"
    removable = analysis_dir / "removable.nwb"
    blocked.write_text("blocked")
    removable.write_text("removable")
    plan = _destructive_plan(common_nwbfile, analysis_dir)

    real_unlink = Path.unlink

    def fail_one(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one)
    with caplog.at_level("ERROR"):
        plan.execute(dry_run=False)

    assert blocked.exists()
    assert not removable.exists()
    assert "Error deleting file" in caplog.text
    assert "denied" in caplog.text


def test_plan_validation_limits(common_nwbfile, tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    tracked = analysis_dir / "tracked.nwb"
    candidate = analysis_dir / "candidate.nwb"
    tracked.write_text("tracked")
    candidate.write_text("candidate")

    def plan_with(**limits):
        return _plan(
            _table(common_nwbfile, analysis_dir, [tracked]),
            policy=_make_policy(**limits),
        )

    assert plan_with(max_delete_fraction=0.5).validate() == (True, None)
    accepted, reason = plan_with(max_delete_fraction=0.49).validate()
    assert not accepted
    assert "above the safety limit" in reason

    assert plan_with(
        max_delete_fraction=1.0,
        max_delete_to_tracked_ratio=1.0,
    ).validate() == (True, None)
    accepted, reason = plan_with(
        max_delete_fraction=1.0,
        max_delete_to_tracked_ratio=0.99,
    ).validate()
    assert not accepted
    assert "1.0x" in reason


def test_plan_refuses_deletion_with_no_local_tracked_files(
    common_nwbfile, tmp_path
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    candidate = analysis_dir / "candidate.nwb"
    candidate.write_text("candidate")

    plan = _plan(_table(common_nwbfile, analysis_dir))
    accepted, reason = plan.validate()

    assert not accepted
    assert "no tracked analysis files" in reason
    with pytest.raises(RuntimeError, match="no tracked analysis files"):
        plan.execute(dry_run=False)
    assert candidate.exists()


def test_absent_tracked_path_does_not_dilute_deletion_fraction(
    common_nwbfile, tmp_path
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    tracked = analysis_dir / "tracked.nwb"
    tracked.write_text("tracked")
    missing_tracked = analysis_dir / "missing.nwb"
    for index in range(2):
        (analysis_dir / f"candidate_{index}.nwb").write_text("candidate")

    plan = _plan(
        _table(common_nwbfile, analysis_dir, [tracked, missing_tracked]),
        policy=_make_policy(max_delete_fraction=0.5),
    )
    accepted, reason = plan.validate()

    assert not accepted
    assert "2/3 eligible analysis files" in reason


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_delete_fraction", True),
        ("max_delete_fraction", float("nan")),
        ("max_delete_fraction", 1.1),
        ("max_delete_to_tracked_ratio", float("inf")),
        ("max_delete_to_tracked_ratio", -1),
        ("min_file_age_hours", float("nan")),
        ("min_file_age_hours", -1),
    ],
)
def test_policy_rejects_bad_limits(common_nwbfile, name, value):
    # The limits are validated once, when the immutable CleanupPolicy is
    # constructed -- before any insert-blocking trigger is acquired.
    with pytest.raises(ValueError, match=name):
        _make_policy(**{name: value})


def _configure_cleanup(
    common_nwbfile,
    monkeypatch,
    registry,
    events,
    validation=(True, None),
    plan_kwargs=None,
):
    class _Plan:
        def execute(self, *, dry_run):
            accepted, reason = validation
            if not accepted:
                if dry_run:
                    common_nwbfile.logger.warning(
                        f"Cleanup plan would be refused: {reason}"
                    )
                else:
                    raise RuntimeError(reason)
            events.append("files")

    def _fake_build_plan(self, custom_tables, **kwargs):
        events.append("plan")
        if plan_kwargs is not None:
            plan_kwargs.update(kwargs)
        return _Plan()

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: events.append("orphans") or _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        _fake_build_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, **kwargs: events.append("external") or [],
    )


def test_cleanup_coordinates_block_files_database_and_unblock(
    common_nwbfile, monkeypatch
):
    events = []
    plan_kwargs = {}

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            events.append("block")

        def unblock_new_inserts(self):
            events.append("unblock")

    _configure_cleanup(
        common_nwbfile,
        monkeypatch,
        _Registry(),
        events,
        plan_kwargs=plan_kwargs,
    )
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    table.cleanup(
        dry_run=False,
        max_delete_fraction=0.8,
        max_delete_to_tracked_ratio=2.0,
    )

    assert events == [
        "block",
        "orphans",
        "plan",
        "files",
        "external",
        "unblock",
    ]
    # The validated policy is threaded into the plan builder, carrying the
    # caller's limits without re-validating at the deletion boundary.
    policy = plan_kwargs["policy"]
    assert policy.max_delete_fraction == 0.8
    assert policy.max_delete_to_tracked_ratio == 2.0


def test_cleanup_rejects_nan_before_installing_triggers(
    common_nwbfile, monkeypatch
):
    events = []

    class _Registry:
        def block_new_inserts(self, dry_run):
            events.append("block")

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: _Registry())
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(ValueError, match="max_delete_fraction"):
        table.cleanup(dry_run=False, max_delete_fraction=float("nan"))

    assert events == []


def test_cleanup_refuses_destructive_invalid_plan(common_nwbfile, monkeypatch):
    events = []

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            events.append("block")

        def unblock_new_inserts(self):
            events.append("unblock")

    _configure_cleanup(
        common_nwbfile,
        monkeypatch,
        _Registry(),
        events,
        validation=(False, "unsafe plan"),
    )
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(RuntimeError, match="unsafe plan"):
        table.cleanup(dry_run=False)

    assert events == ["block", "orphans", "plan", "unblock"]


def test_cleanup_dry_run_warns_and_continues_invalid_plan(
    common_nwbfile, monkeypatch, caplog
):
    events = []

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            events.append("block")

        def unblock_new_inserts(self):
            events.append("unblock")

    _configure_cleanup(
        common_nwbfile,
        monkeypatch,
        _Registry(),
        events,
        validation=(False, "unsafe plan"),
    )
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with caplog.at_level("WARNING"):
        table.cleanup(dry_run=True)

    assert "Cleanup plan would be refused: unsafe plan" in caplog.text
    assert events == [
        "block",
        "orphans",
        "plan",
        "files",
        "external",
    ]


def test_cleanup_block_failure_does_not_unblock(common_nwbfile, monkeypatch):
    unblock_calls = []

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            raise RuntimeError("block failed")

        def unblock_new_inserts(self):
            unblock_calls.append(True)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: _Registry())
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(RuntimeError, match="block failed"):
        table.cleanup(dry_run=False)

    assert unblock_calls == []


def test_cleanup_unblock_failure_propagates_after_success(
    common_nwbfile, monkeypatch
):
    events = []

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            events.append("block")

        def unblock_new_inserts(self):
            raise RuntimeError("unblock failed")

    _configure_cleanup(common_nwbfile, monkeypatch, _Registry(), events)
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(RuntimeError, match="unblock failed"):
        table.cleanup(dry_run=False)


def test_cleanup_body_failure_still_unblocks(common_nwbfile, monkeypatch):
    events = []

    class _Registry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            events.append("block")

        def unblock_new_inserts(self):
            events.append("unblock")

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: _Registry())
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: (_ for _ in ()).throw(RuntimeError("body failed")),
    )
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(RuntimeError, match="body failed"):
        table.cleanup(dry_run=False)

    assert events == ["block", "unblock"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_block_new_inserts_refuses_preexisting_trigger(
    common_nwbfile, monkeypatch, dry_run
):
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *args, **kwargs: ["db.`t1`"],
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_exists",
        lambda self, table: True,
    )
    attempted = []
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_single_table",
        lambda self, table, dry_run=False: attempted.append(table),
    )

    with pytest.raises(RuntimeError, match="already exist"):
        registry.block_new_inserts(dry_run=dry_run)

    assert attempted == []


def test_block_new_inserts_stops_after_partial_acquisition_failure(
    common_nwbfile, monkeypatch
):
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    tables = ["db.`t2`", "db.`t1`", "db.`t3`"]
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *args, **kwargs: tables,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_exists",
        lambda self, table: False,
    )
    attempted = []

    def block(self, table, dry_run=False):
        attempted.append(table)
        return "trigger creation failed" if table == "db.`t2`" else None

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_single_table",
        block,
    )

    with pytest.raises(
        RuntimeError, match="Some analysis tables may remain blocked"
    ):
        registry.block_new_inserts(dry_run=False)

    assert attempted == ["db.`t1`", "db.`t2`"]


def test_unblock_new_inserts_skips_absent_and_aggregates_drop_errors(
    common_nwbfile, monkeypatch
):
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    tables = ["present", "absent", "broken_one", "broken_two", "after"]
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *args, **kwargs: tables,
    )
    checked = []

    def block_exists(self, table):
        checked.append(table)
        return table != "absent"

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_exists",
        block_exists,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_get_block_info",
        lambda self, table: ("db", f"{table}_trigger"),
    )

    drops = []

    class _Connection:
        def query(self, statement):
            drops.append(statement)
            if "broken_one" in statement:
                raise RuntimeError("first drop failed")
            if "broken_two" in statement:
                raise RuntimeError("second drop failed")

    monkeypatch.setattr(common_nwbfile.dj, "conn", lambda: _Connection())

    with pytest.raises(RuntimeError) as exc_info:
        registry.unblock_new_inserts()

    error = str(exc_info.value)
    assert "Failed to unblock 2 table(s)" in error
    assert "Failed to unblock broken_one: first drop failed" in error
    assert "Failed to unblock broken_two: second drop failed" in error
    assert checked == tables
    assert drops == [
        "DROP TRIGGER db.present_trigger;",
        "DROP TRIGGER db.broken_one_trigger;",
        "DROP TRIGGER db.broken_two_trigger;",
        "DROP TRIGGER db.after_trigger;",
    ]


# ------------------------------------------------------------------ #
# AnalysisRegistry._parse_table_name
# ------------------------------------------------------------------ #


def test_parse_table_name_basic(common_nwbfile):
    """Parses a standard full table name into components."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse("`user_nwbfile`.`analysis_nwbfile`")
    assert db == "user_nwbfile"
    assert table == "analysis_nwbfile"
    assert prefix == "user"
    assert suffix == "nwbfile"


def test_parse_table_name_no_backticks(common_nwbfile):
    """Backtick-free names are also parsed correctly."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse("myteam_nwbfile.analysis_nwbfile")
    assert db == "myteam_nwbfile"
    assert table == "analysis_nwbfile"
    assert prefix == "myteam"
    assert suffix == "nwbfile"


def test_parse_table_name_multi_underscore_prefix(common_nwbfile):
    """Prefix with multiple underscores splits at the last one."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse(
        "`first_second_nwbfile`.`analysis_nwbfile`"
    )
    assert db == "first_second_nwbfile"
    assert prefix == "first_second"
    assert suffix == "nwbfile"


# ------------------------------------------------------------------ #
# Nwbfile.get_abs_path error case
# ------------------------------------------------------------------ #


def test_get_abs_path_no_match_raises(common_nwbfile):
    """get_abs_path raises ValueError if entry not found in table."""
    with pytest.raises(ValueError):
        common_nwbfile.Nwbfile.get_abs_path("nonexistent_file.nwb")
