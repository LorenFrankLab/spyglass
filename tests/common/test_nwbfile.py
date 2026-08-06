import os
from pathlib import Path

import pytest


class _FakeExternalTable:
    """Stand-in for a DataJoint external table.

    Cleanup uses ``fetch_external_paths()`` both while planning and for the
    canonical per-candidate tracking refresh.
    """

    def __init__(self, external_paths):
        self._external_paths = external_paths

    def fetch_external_paths(self):
        return self._external_paths


class _EmptyQuery:
    def proj(self):
        return self

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def delete_quick(self):
        raise AssertionError("empty query should not be deleted")


class _FakeRegistry:
    def __init__(self):
        self.all_classes = []
        self.blocked = False
        self.unblocked = False

    def block_new_inserts(self, dry_run):
        assert dry_run is False
        self.blocked = True

    def unblock_new_inserts(self):
        self.unblocked = True


@pytest.fixture
def common_nwbfile(common):
    """Return a common NWBFile object."""
    return common.common_nwbfile


@pytest.fixture
def lockfile(base_dir, teardown):
    lockfile = base_dir / "temp.lock"
    lockfile.touch()
    os.environ["NWB_LOCK_FILE"] = str(lockfile)
    yield lockfile
    if teardown:
        os.remove(lockfile)


def test_add_to_lock(common_nwbfile, lockfile, mini_copy_name):
    common_nwbfile.Nwbfile.add_to_lock(mini_copy_name)
    with lockfile.open("r") as f:
        assert mini_copy_name in f.read()

    with pytest.raises(FileNotFoundError):
        common_nwbfile.Nwbfile.add_to_lock("non-existent-file.nwb")


def test_nwbfile_cleanup(common_nwbfile):
    before = len(common_nwbfile.Nwbfile.fetch())
    common_nwbfile.Nwbfile.cleanup(delete_files=False)
    after = len(common_nwbfile.Nwbfile.fetch())
    assert before == after, "Nwbfile cleanup changed table entry count."


def _cleanup_plan(common_nwbfile, *, scanned, tracked, delete):
    """Build a plan with the shape the real builder produces.

    Tracked entries are drawn from the scanned set, immediately after the
    deleted ones, so `scanned & tracked` is non-empty and the validator
    exercises the branch each test is aiming at. The real builder can never
    produce disjoint sets: a scanned file is kept because it is tracked.
    """
    scanned_files = [Path(f"scan_{i}.nwb") for i in range(scanned)]
    files_to_delete = set(scanned_files[:delete])
    tracked_files = set(scanned_files[delete : delete + tracked])
    return common_nwbfile.CleanupPlan(
        scanned_files=set(scanned_files),
        tracked_files=tracked_files,
        files_to_delete=files_to_delete,
        empty_files=set(),
        untracked_files=files_to_delete,
        candidates={},
        deferred_recent_files=set(),
        broken_links=set(),
    )


def test_build_untracked_file_plan_uses_resolved_external_paths(
    common_nwbfile, tmp_path
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    tracked = analysis_dir / "tracked.nwb"
    tracked.write_text("tracked")

    untracked = analysis_dir / "untracked.nwb"
    untracked.write_text("untracked")

    empty = analysis_dir / "empty.nwb"
    empty.touch()

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable(
        [("tracked", analysis_dir / "subdir" / ".." / "tracked.nwb")]
    )

    plan = table._build_untracked_file_plan(
        custom_tables=[], min_file_age_hours=0
    )

    assert plan.scanned_files == {
        tracked.resolve(),
        untracked.resolve(),
        empty.resolve(),
    }
    assert plan.tracked_files == {tracked.resolve()}
    assert plan.empty_files == {empty.resolve()}
    assert plan.untracked_files == {untracked.resolve()}
    assert plan.files_to_delete == {empty.resolve(), untracked.resolve()}


def test_analysis_cleanup_plan_rejects_delete_without_tracked_files(
    common_nwbfile,
):
    plan = _cleanup_plan(common_nwbfile, scanned=1, tracked=0, delete=1)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)
    assert ok is False
    assert "no tracked analysis files" in msg


def test_analysis_cleanup_plan_rejects_high_delete_fraction(common_nwbfile):
    plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan,
        max_delete_fraction=0.25,
        max_delete_to_tracked_ratio=10.0,
    )
    assert ok is False
    assert "above the safety limit" in msg


def test_ratio_guard_fires_when_fraction_is_raised(common_nwbfile):
    """The ratio guard is dead only at the default fraction, not always.

    At max_delete_fraction=0.9 it provably cannot fire (every scanned file
    that is kept is tracked, so the fraction caps the ratio at 9). Raising
    the fraction is how a caller actually reaches it.
    """
    plan = _cleanup_plan(common_nwbfile, scanned=12, tracked=1, delete=11)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=1.0
    )
    assert ok is False
    assert "11.0x" in msg


def test_analysis_cleanup_plan_accepts_plausible_delete_plan(common_nwbfile):
    plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    assert common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan) == (
        True,
        None,
    )


def test_analysis_cleanup_plan_accepts_empty_delete(common_nwbfile):
    """delete_count == 0 short-circuits before any other validation kicks in."""
    # tracked=0 would normally trigger "no tracked analysis files"; the empty
    # delete plan must return before that check.
    plan = _cleanup_plan(common_nwbfile, scanned=3, tracked=0, delete=0)

    assert common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan) == (
        True,
        None,
    )


def test_analysis_cleanup_plan_accepts_fraction_at_threshold(common_nwbfile):
    """Fraction exactly at max_delete_fraction must pass (strict > guard)."""
    plan = _cleanup_plan(common_nwbfile, scanned=10, tracked=10, delete=9)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=0.9, max_delete_to_tracked_ratio=10.0
    )
    assert (ok, msg) == (True, None)


def test_analysis_cleanup_plan_rejects_fraction_just_above_threshold(
    common_nwbfile,
):
    """Fraction just above max_delete_fraction must be refused."""
    plan = _cleanup_plan(common_nwbfile, scanned=11, tracked=10, delete=10)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=0.9, max_delete_to_tracked_ratio=100.0
    )
    assert ok is False
    assert "above the safety limit" in msg


def test_analysis_cleanup_plan_accepts_ratio_at_threshold(common_nwbfile):
    """Ratio exactly at max_delete_to_tracked_ratio must pass."""
    plan = _cleanup_plan(common_nwbfile, scanned=100, tracked=1, delete=10)

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=1.0, max_delete_to_tracked_ratio=10.0
    )
    assert (ok, msg) == (True, None)


def test_remove_untracked_files_refuses_path_outside_analysis_dir(
    common_nwbfile, tmp_path
):
    """Plans containing paths outside analysis_dir must not unlink anything.

    The candidate is fully populated on purpose: an empty `candidates`
    dict would make the deletion loop iterate nothing, so the test would
    pass without exercising the guard it is named for.
    """
    outside_target = tmp_path / "outside" / "shared.nwb"
    outside_target.parent.mkdir()
    outside_target.write_text("shared data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    real = outside_target.resolve()
    st = outside_target.stat()
    forged = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=common_nwbfile.TargetSnapshot(
            real_path=real,
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=real,
                is_link=False,  # no in-root link vouches for it
                raw_link_target=None,
                dev=st.st_dev,
                ino=st.st_ino,
                mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files={real},
        candidates={real: forged},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable([])

    # min_file_age_hours=0 so the age gate cannot be what makes this
    # pass -- the containment guard must be what refuses.
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert outside_target.exists()


def test_analysis_cleanup_dry_run_warns_on_refused_plan(
    common_nwbfile, monkeypatch, caplog
):
    """dry_run=True must surface a validation refusal as a warning, not raise."""
    registry = _FakeRegistry()
    # block_new_inserts is hard-coded to expect dry_run=False; relax for this test.
    monkeypatch.setattr(
        _FakeRegistry,
        "block_new_inserts",
        lambda self, dry_run: setattr(self, "blocked", True),
    )
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan, **kw: (set(), set()),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with caplog.at_level("WARNING"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=True, max_delete_fraction=0.25
        )

    assert any(
        "Cleanup plan would be refused" in record.message
        for record in caplog.records
    )


def test_analysis_cleanup_validates_plan_before_unlink(
    common_nwbfile, monkeypatch
):
    """cleanup() must abort destructive unlink when the plan exceeds limits."""
    registry = _FakeRegistry()
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    def _remove_untracked_files(
        self, custom_tables, dry_run=True, plan=None, **kw
    ):
        raise AssertionError("dangerous final cleanup plan should not unlink")

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        _remove_untracked_files,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.25
        )

    assert registry.blocked
    assert registry.unblocked


def test_cleanup_preserves_original_exception_when_unblock_raises(
    common_nwbfile, monkeypatch
):
    """If both the cleanup body AND unblock_new_inserts raise, the original
    cleanup-body exception must propagate (unblock failure becomes a log line,
    not the user-visible error)."""

    class _RaisingRegistry:
        all_classes = []
        blocked = False

        def block_new_inserts(self, dry_run):
            self.blocked = True

        def unblock_new_inserts(self):
            raise RuntimeError("unblock_kaboom")

    registry = _RaisingRegistry()
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    # The validator should raise "above the safety limit", not "unblock_kaboom".
    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.25
        )


def test_cleanup_propagates_unblock_failure_when_body_succeeds(
    common_nwbfile, monkeypatch
):
    """If the cleanup body succeeds but unblock_new_inserts raises, the unblock
    failure must propagate so the operator notices the stuck-blocked state."""

    class _RaisingRegistry:
        all_classes = []
        blocked = False

        def block_new_inserts(self, dry_run):
            self.blocked = True

        def unblock_new_inserts(self):
            raise RuntimeError("unblock_kaboom")

    registry = _RaisingRegistry()
    good_plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: good_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan, **kw: (set(), set()),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with pytest.raises(RuntimeError, match="unblock_kaboom"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.9
        )


def _plan(table, **kwargs):
    """Build a plan with the age gate disabled unless overridden."""
    kwargs.setdefault("min_file_age_hours", 0)
    return table._build_untracked_file_plan(custom_tables=[], **kwargs)


def _table(common_nwbfile, analysis_dir, tracked=()):
    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable([("h", p) for p in tracked])
    return table


def _patch_config_dir(common_nwbfile, monkeypatch, key, path):
    """Patch one directory only for this module, not shared settings state."""
    patched = dict(common_nwbfile.config)
    patched[key] = str(path)
    monkeypatch.setattr(common_nwbfile, "config", patched)


def test_scan_includes_untracked_leaf_symlink(common_nwbfile, tmp_path):
    """A leaf *.nwb symlink is a candidate, keyed by its real target."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "far.nwb"
    target.write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert plan.files_to_delete == {target.resolve()}
    candidate = plan.candidates[target.resolve()]
    assert candidate.broken is False
    assert len(candidate.accesses) == 1
    access = candidate.accesses[0]
    assert access.is_link is True
    assert access.access_path == link
    assert access.raw_link_target == str(target)


def test_scan_leaves_tracked_leaf_symlink_alone(common_nwbfile, tmp_path):
    """A tracked symlink must not become a candidate."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "far.nwb"
    target.write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    plan = _plan(_table(common_nwbfile, analysis_dir, tracked=[link]))

    assert plan.files_to_delete == set()


def test_scan_recognizes_tracked_physical_alias(common_nwbfile, tmp_path):
    """Tracking by filesystem identity survives a different path spelling.

    A hard link is a portable stand-in for case and bind-mount aliases: its
    resolved path differs, but its device/inode identifies the same file.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("registered data")
    tracked_alias = volume2 / "registered.nwb"
    os.link(target, tracked_alias)

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    plan = _plan(_table(common_nwbfile, analysis_dir, tracked=[tracked_alias]))

    assert tracked_alias.resolve() != target.resolve()
    assert tracked_alias.stat().st_ino == target.stat().st_ino
    assert plan.files_to_delete == set()
    assert target.resolve() in plan.tracked_files
    assert plan.scanned_files & plan.tracked_files == {target.resolve()}


def test_scan_preserves_tracked_broken_leaf(common_nwbfile, tmp_path):
    """A registered dangling link can be tracked only by leaf identity."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "registered_broken.nwb"
    link.symlink_to("missing.nwb")

    # A hard link to the symlink has the same leaf inode, while its relative
    # target resolves under a different parent. There is no live target inode,
    # so neither resolved-path nor target-identity matching can protect it.
    tracked_dir = tmp_path / "registered_elsewhere"
    tracked_dir.mkdir()
    tracked_alias = tracked_dir / "registered_broken.nwb"
    os.link(link, tracked_alias, follow_symlinks=False)

    assert link.lstat().st_ino == tracked_alias.lstat().st_ino
    assert Path(os.path.realpath(link)) != Path(os.path.realpath(tracked_alias))

    plan = _plan(_table(common_nwbfile, analysis_dir, tracked=[tracked_alias]))

    assert plan.files_to_delete == set()
    assert link.is_symlink()


def test_scan_does_not_descend_directory_symlink(common_nwbfile, tmp_path):
    """Directory symlinks stay undescended (unchanged from master)."""
    volume2 = tmp_path / "volume2" / "session"
    volume2.mkdir(parents=True)
    (volume2 / "far.nwb").write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "session").symlink_to(volume2, target_is_directory=True)
    (analysis_dir / "near.nwb").write_text("near")

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert plan.scanned_files == {(analysis_dir / "near.nwb").resolve()}


def test_scan_treats_tracked_empty_file_as_tracked(common_nwbfile, tmp_path):
    """Tracked wins over empty: a tracked 0-byte file is never deleted.

    Deleting it would leave a dangling DataJoint row.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    empty = analysis_dir / "empty.nwb"
    empty.touch()

    plan = _plan(_table(common_nwbfile, analysis_dir, tracked=[empty]))

    assert plan.files_to_delete == set()
    assert plan.empty_files == set()


def test_scan_flags_untracked_broken_symlink(common_nwbfile, tmp_path):
    """A dangling *.nwb symlink with no tracked entry is a candidate."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "dangling.nwb"
    link.symlink_to(tmp_path / "nope.nwb")

    plan = _plan(_table(common_nwbfile, analysis_dir))

    real = Path(os.path.realpath(link))
    assert plan.broken_links == {real}
    assert plan.candidates[real].broken is True


def test_scan_ignores_directory_named_nwb(common_nwbfile, tmp_path):
    """A directory named *.nwb is not a deletion candidate."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "dir.nwb").mkdir()
    (analysis_dir / "real.nwb").write_text("real")

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert plan.scanned_files == {(analysis_dir / "real.nwb").resolve()}


def test_scan_fails_closed_on_walk_error(common_nwbfile, tmp_path, monkeypatch):
    """A permission error must abort, not silently scan a partial tree."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "real.nwb").write_text("real")

    real_walk = os.walk

    def _boom(top, **kwargs):
        onerror = kwargs.get("onerror")
        if onerror is not None:
            onerror(PermissionError(13, "Permission denied", str(top)))
        return real_walk(top, **kwargs)

    monkeypatch.setattr(os, "walk", _boom)

    with pytest.raises(PermissionError):
        _plan(_table(common_nwbfile, analysis_dir))


def test_scan_defers_recent_files(common_nwbfile, tmp_path):
    """Files newer than min_file_age_hours are deferred, not deleted.

    The clock is injected because os.utime cannot backdate ctime, so a
    real file always reads as new under max(mtime_ns, ctime_ns).
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fresh = analysis_dir / "fresh.nwb"
    fresh.write_text("fresh")

    table = _table(common_nwbfile, analysis_dir)
    now_ns = fresh.stat().st_ctime_ns + 3600 * 10**9  # 1 hour later

    plan = table._build_untracked_file_plan(
        custom_tables=[], min_file_age_hours=24.0, now_ns=now_ns
    )

    assert plan.files_to_delete == set()
    assert plan.deferred_recent_files == {fresh.resolve()}


def test_scan_age_boundary_is_inclusive(common_nwbfile, tmp_path):
    """Exactly min_file_age_hours old is eligible; a nanosecond less is not."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "boundary.nwb"
    f.write_text("x")

    table = _table(common_nwbfile, analysis_dir)
    newest = max(f.stat().st_mtime_ns, f.stat().st_ctime_ns)
    day_ns = 24 * 3600 * 10**9

    exactly = table._build_untracked_file_plan(
        custom_tables=[], min_file_age_hours=24.0, now_ns=newest + day_ns
    )
    assert exactly.files_to_delete == {f.resolve()}

    just_under = table._build_untracked_file_plan(
        custom_tables=[], min_file_age_hours=24.0, now_ns=newest + day_ns - 1
    )
    assert just_under.files_to_delete == set()


def test_scan_defers_future_timestamps(common_nwbfile, tmp_path):
    """A file stamped in the future must be deferred, not treated as old."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "future.nwb"
    f.write_text("x")

    table = _table(common_nwbfile, analysis_dir)
    newest = max(f.stat().st_mtime_ns, f.stat().st_ctime_ns)

    plan = table._build_untracked_file_plan(
        custom_tables=[], min_file_age_hours=24.0, now_ns=newest - 10**9
    )
    assert plan.deferred_recent_files == {f.resolve()}


def test_delete_removes_symlink_target_and_link(common_nwbfile, tmp_path):
    """An in-root link authorizes its live non-protected regular target.

    Owner decision: cleanup reclaims across volumes in one pass, which is
    what makes symlinked multi-drive analysis storage usable.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "far.nwb"
    target.write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not target.exists(), "external target should be reclaimed"
    assert not link.is_symlink(), "the link is removed too"


def test_delete_skips_candidate_whose_identity_changed(
    common_nwbfile, tmp_path
):
    """A file replaced between planning and acting must not be deleted."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "swapped.nwb"
    f.write_text("original")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    f.unlink()
    f.write_text("replacement with a different inode")

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert f.exists(), "replaced file must not be deleted on a stale plan"


def test_delete_skips_candidate_that_became_tracked(common_nwbfile, tmp_path):
    """Tracking is re-fetched at act time and protects every candidate."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "registered.nwb"
    f.write_text("data")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert plan.files_to_delete == {f.resolve()}

    # Registered after the plan was built.
    table._ext_tbl = _FakeExternalTable([("h", f)])

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert f.exists(), "newly tracked file must not be deleted"


def test_delete_skips_late_tracked_physical_alias(common_nwbfile, tmp_path):
    """Act-time identity refresh protects a newly registered physical alias."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("registered data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert plan.files_to_delete == {target.resolve()}

    tracked_alias = volume2 / "registered_late.nwb"
    os.link(target, tracked_alias)
    table._ext_tbl = _FakeExternalTable([("h", tracked_alias)])

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert target.exists(), "late physical alias must protect the target"
    assert link.is_symlink(), "the planned link must survive"
    assert tracked_alias.exists()


def test_delete_aborts_on_tracked_identity_inspection_error(
    common_nwbfile, tmp_path, monkeypatch
):
    """An unreadable tracked alias must fail closed before target deletion."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("must survive")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    tracked_alias = volume2 / "tracked_alias.nwb"
    os.link(target, tracked_alias)
    table._ext_tbl = _FakeExternalTable([("h", tracked_alias)])

    real_stat = common_nwbfile.os.stat

    def _deny_tracked_target(path, *args, **kwargs):
        if Path(path) == tracked_alias:
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(common_nwbfile.os, "stat", _deny_tracked_target)

    with pytest.raises(RuntimeError, match="tracked analysis path"):
        table._remove_untracked_files(
            custom_tables=[],
            dry_run=False,
            plan=plan,
            min_file_age_hours=0,
        )

    assert target.exists()
    assert link.is_symlink()


def test_delete_removes_broken_link_only(common_nwbfile, tmp_path):
    """A dangling link is unlinked; nothing else is touched."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "dangling.nwb"
    link.symlink_to(tmp_path / "nope.nwb")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not link.is_symlink()


def test_delete_skips_relinked_symlink(common_nwbfile, tmp_path):
    """A link re-pointed after planning must not have its new target hit."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    original = volume2 / "original.nwb"
    original.write_text("original")
    other = volume2 / "other.nwb"
    other.write_text("other, must survive")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(original)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    # Re-point WITHOUT changing the leaf inode, so the dev/ino check
    # cannot be what refuses and the raw-link-target comparison is the
    # guard actually under test.
    for access in plan.candidates[original.resolve()].accesses:
        if access.access_path == link:
            object.__setattr__(access, "raw_link_target", str(other))

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert other.exists(), "re-pointed link's new target must survive"
    assert original.exists(), "a link whose raw target changed must refuse"


def test_delete_refuses_repointed_intermediate_symlink(
    common_nwbfile, tmp_path
):
    """The complete live alias must still resolve to the planned target.

    Re-pointing an intermediate directory leaves the leaf inode and raw link
    text unchanged. Target identity alone also cannot detect it when the old
    target remains in place, so cleanup must re-resolve the full access path
    before deleting that old target.
    """
    volume_a = tmp_path / "volume_a"
    volume_b = tmp_path / "volume_b"
    volume_a.mkdir()
    volume_b.mkdir()
    old_target = volume_a / "target.nwb"
    new_target = volume_b / "target.nwb"
    old_target.write_text("old target must survive")
    new_target.write_text("new target must survive")

    pivot = tmp_path / "current_volume"
    pivot.symlink_to(volume_a, target_is_directory=True)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(pivot / "target.nwb")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    pivot.unlink()
    pivot.symlink_to(volume_b, target_is_directory=True)

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert old_target.exists(), "stale plan must not delete the old target"
    assert new_target.exists(), "cleanup must not follow the re-pointed chain"
    assert link.is_symlink(), "a candidate with changed authority is preserved"


def test_delete_refuses_outside_symlink_as_voucher(common_nwbfile, tmp_path):
    """A voucher must live inside analysis_dir, not merely be a symlink.

    Without the containment check on the access path, an outside symlink
    could authorize deleting an outside target.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    victim = volume2 / "victim.nwb"
    victim.write_text("must survive")

    outside_link = tmp_path / "elsewhere.nwb"
    outside_link.symlink_to(victim)

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    real = victim.resolve()
    st = victim.stat()
    lst = outside_link.lstat()
    forged = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=common_nwbfile.TargetSnapshot(
            real_path=real,
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=outside_link,
                is_link=True,  # a link, but NOT inside analysis_dir
                raw_link_target=str(victim),
                dev=lst.st_dev,
                ino=lst.st_ino,
                mtime_ns=lst.st_mtime_ns,
                ctime_ns=lst.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files={real},
        candidates={real: forged},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = _table(common_nwbfile, analysis_dir)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert victim.exists(), "outside link must not vouch for an outside target"


def test_delete_refuses_unrelated_in_root_nwb_voucher(common_nwbfile, tmp_path):
    """An in-root *.nwb link must canonically reach the planned target.

    Suffix and containment preflight alone would accept this forged plan and
    let an unrelated link authorize deletion of an arbitrary external file.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    victim = volume2 / "victim.nwb"
    victim.write_text("must survive")
    decoy = volume2 / "decoy.nwb"
    decoy.write_text("also survives")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    voucher = analysis_dir / "voucher.nwb"
    voucher.symlink_to(decoy)

    real = victim.resolve()
    target_stat = victim.stat()
    access_stat = voucher.lstat()
    candidate = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=common_nwbfile.TargetSnapshot(
            real_path=real,
            dev=target_stat.st_dev,
            ino=target_stat.st_ino,
            size=target_stat.st_size,
            mtime_ns=target_stat.st_mtime_ns,
            ctime_ns=target_stat.st_ctime_ns,
            mode=target_stat.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=voucher,
                is_link=True,
                raw_link_target=str(decoy),
                dev=access_stat.st_dev,
                ino=access_stat.st_ino,
                mtime_ns=access_stat.st_mtime_ns,
                ctime_ns=access_stat.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files={real},
        candidates={real: candidate},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = _table(common_nwbfile, analysis_dir)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert victim.exists(), "unrelated voucher must not authorize deletion"
    assert decoy.exists()
    assert voucher.is_symlink()


def test_delete_refuses_plan_with_mismatched_key(common_nwbfile, tmp_path):
    """A plan key that disagrees with its candidate must be refused.

    Otherwise the loop could verify one path and unlink another.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    victim = analysis_dir / "victim.nwb"
    victim.write_text("must survive")
    decoy = analysis_dir / "decoy.nwb"
    decoy.write_text("decoy")

    st = decoy.stat()
    candidate = common_nwbfile.CleanupCandidate(
        real_path=decoy.resolve(),
        target=common_nwbfile.TargetSnapshot(
            real_path=decoy.resolve(),
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=decoy,
                is_link=False,
                raw_link_target=None,
                dev=st.st_dev,
                ino=st.st_ino,
                mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns,
            ),
        ),
    )
    # Key names the victim; candidate names the decoy.
    plan = common_nwbfile.CleanupPlan(
        scanned_files={victim.resolve()},
        tracked_files=set(),
        files_to_delete={victim.resolve()},
        empty_files=set(),
        untracked_files={victim.resolve()},
        candidates={victim.resolve(): candidate},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = _table(common_nwbfile, analysis_dir)
    with pytest.raises(RuntimeError, match="deletion failed"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert victim.exists()
    assert decoy.exists()


def test_delete_raises_on_unlink_failure(common_nwbfile, tmp_path, monkeypatch):
    """Unlink failures must propagate, not be logged and swallowed.

    cleanup() runs database cleanup after this, and the maintenance driver
    gates later analysis phases on a failure escaping here.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "stubborn.nwb"
    f.write_text("data")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    def _boom(self, *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", _boom)

    with pytest.raises(RuntimeError, match="deletion failed"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )


def test_delete_rechecks_age_at_act_time(common_nwbfile, tmp_path):
    """Age is re-checked before deletion, not only during planning."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    f = analysis_dir / "fresh.nwb"
    f.write_text("data")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)  # built with the gate disabled
    assert plan.files_to_delete == {f.resolve()}

    # Act with the gate enabled: the candidate is now too young.
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=24.0
    )

    assert f.exists(), "act-time age check must defer a fresh candidate"


def test_cleanup_deletes_files_before_db_cleanup(common_nwbfile, monkeypatch):
    """File deletion must precede DB cleanup, shortening validate-to-act."""
    order = []
    registry = _FakeRegistry()
    good_plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: good_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan, **kw: (
            order.append("files"),
            (set(), set()),
        )[1],
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: (order.append("orphans"), _EmptyQuery())[1],
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: (
            order.append("external"),
            [],
        )[1],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    common_nwbfile.AnalysisNwbfile.cleanup(table, dry_run=False)

    assert order.index("files") < order.index("external")


def test_validator_excludes_deferred_from_denominator(common_nwbfile):
    """Deferred files must not dilute the delete fraction.

    89 deleted of 100 scanned reads as 89% only because 10 were deferred;
    of the 90 files actually eligible it is 98.9%.
    """
    scanned = {Path(f"s{i}.nwb") for i in range(100)}
    deferred = {Path(f"s{i}.nwb") for i in range(90, 100)}
    to_delete = {Path(f"s{i}.nwb") for i in range(89)}
    plan = common_nwbfile.CleanupPlan(
        scanned_files=scanned,
        tracked_files={Path("s89.nwb")},
        files_to_delete=to_delete,
        empty_files=set(),
        untracked_files=to_delete,
        candidates={},
        deferred_recent_files=deferred,
        broken_links=set(),
    )

    ok, msg = common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)
    assert ok is False
    assert "above the safety limit" in msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_delete_fraction": float("nan")},
        {"max_delete_fraction": float("inf")},
        {"max_delete_fraction": 1.5},
        {"max_delete_fraction": -0.1},
        {"max_delete_fraction": True},
        {"max_delete_to_tracked_ratio": float("nan")},
        {"max_delete_to_tracked_ratio": float("inf")},
        {"max_delete_to_tracked_ratio": -1},
        {"max_delete_to_tracked_ratio": True},
        {"min_file_age_hours": float("nan")},
        {"min_file_age_hours": float("inf")},
        {"min_file_age_hours": -1},
        {"min_file_age_hours": True},
    ],
)
def test_cleanup_rejects_invalid_safety_inputs(common_nwbfile, kwargs):
    """Bad limits must fail loudly, not silently disable the guard."""
    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with pytest.raises(ValueError, match="must be"):
        common_nwbfile.AnalysisNwbfile.cleanup(table, **kwargs)


def test_delete_refuses_link_touched_after_planning(common_nwbfile, tmp_path):
    """Touching a symlink after planning must prevent deletion.

    The act-time age check reads the frozen scan snapshot, so without a
    live alias-timestamp comparison `os.utime(..., follow_symlinks=False)`
    would make the link fresh while dev/ino stayed identical.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "far.nwb"
    target.write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    # Touch the LINK itself, not its target.
    later = link.lstat().st_mtime + 3600
    os.utime(link, (later, later), follow_symlinks=False)

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert target.exists(), "target of a touched link must survive"
    assert link.is_symlink(), "touched link must survive"


def test_delete_rechecks_tracking_per_candidate(common_nwbfile, tmp_path):
    """A file registered while an earlier candidate is deleted must survive.

    Tracking is re-read for every candidate, so B cannot be unlinked on a
    snapshot taken before A was processed.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    first = analysis_dir / "aaa.nwb"
    first.write_text("first")
    second = analysis_dir / "zzz.nwb"
    second.write_text("second")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert plan.files_to_delete == {first.resolve(), second.resolve()}

    # Register whichever candidate is still on disk once the first one has
    # been handled. Keyed off actual state rather than a fixed name, since
    # os.walk does not promise an order.
    class _LateRegistration:
        def __init__(self):
            self.calls = 0
            self.registered = None

        def fetch_external_paths(self):
            self.calls += 1
            if self.calls == 1:
                return []
            survivors = [p for p in (first, second) if p.exists()]
            self.registered = survivors[0] if survivors else None
            return [("h", self.registered)] if self.registered else []

        def __and__(self, restriction):
            self.calls += 1
            if self.calls == 1:
                return []
            survivors = [p for p in (first, second) if p.exists()]
            self.registered = survivors[0] if survivors else None
            rel = restriction["filepath"]
            if self.registered and str(self.registered).endswith(rel):
                return [("h", self.registered)]
            return []

    ext = _LateRegistration()
    table._ext_tbl = ext

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert ext.calls == 2, "tracking must be re-read once per candidate"
    survivors = [p for p in (first, second) if p.exists()]
    assert survivors == [
        ext.registered
    ], "exactly the late-registered candidate must survive"


def test_delete_rechecks_tracking_through_new_alias(
    common_nwbfile, tmp_path, monkeypatch
):
    """A late registration through an unscanned alias protects its target.

    Candidates are keyed by their resolved target, but DataJoint stores the
    registered access path. Restricting the act-time query to scan-time alias
    names misses a new alias and deletes the target out from under it.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("registered data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    scanned_alias = analysis_dir / "scanned.nwb"
    scanned_alias.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert plan.files_to_delete == {target.resolve()}

    # Created and registered only after the plan, so this path is absent from
    # candidate.accesses and can be found only by resolving live externals.
    new_alias = analysis_dir / "registered_late.nwb"
    new_alias.symlink_to(target)

    class _LateTable:
        full_table_name = "`late_nwbfile`.`analysis_nwbfile`"
        _ext_tbl = _FakeExternalTable([("h", new_alias)])

    class _Registry:
        all_classes = [_LateTable()]

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", _Registry)

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert target.exists(), "late registration must protect the real target"
    assert scanned_alias.is_symlink(), "the planned alias must be preserved"
    assert new_alias.is_symlink(), "the newly registered alias must survive"


def test_delete_refuses_forged_broken_plan_naming_outside_link(
    common_nwbfile, tmp_path
):
    """A forged broken candidate must not unlink an out-of-root symlink."""
    outside_link = tmp_path / "elsewhere.nwb"
    outside_link.symlink_to(tmp_path / "nope.nwb")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    real = Path(os.path.realpath(outside_link))
    lst = outside_link.lstat()
    forged = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=None,  # broken
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=outside_link,
                is_link=True,
                raw_link_target=str(tmp_path / "nope.nwb"),
                dev=lst.st_dev,
                ino=lst.st_ino,
                mtime_ns=lst.st_mtime_ns,
                ctime_ns=lst.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files=set(),
        candidates={real: forged},
        deferred_recent_files=set(),
        broken_links={real},
    )

    table = _table(common_nwbfile, analysis_dir)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert outside_link.is_symlink(), "out-of-root broken link must survive"


def test_delete_aborts_whole_plan_on_structural_mismatch(
    common_nwbfile, tmp_path
):
    """A malformed plan must delete NOTHING, not just skip the bad entry."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    valid = analysis_dir / "valid.nwb"
    valid.write_text("valid")
    victim = analysis_dir / "victim.nwb"
    victim.write_text("victim")
    decoy = analysis_dir / "decoy.nwb"
    decoy.write_text("decoy")

    table = _table(common_nwbfile, analysis_dir)
    good = _plan(table)

    def _snap(path):
        st = path.stat()
        return common_nwbfile.CleanupCandidate(
            real_path=path.resolve(),
            target=common_nwbfile.TargetSnapshot(
                real_path=path.resolve(),
                dev=st.st_dev,
                ino=st.st_ino,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns,
                mode=st.st_mode,
            ),
            accesses=(
                common_nwbfile.AccessSnapshot(
                    access_path=path,
                    is_link=False,
                    raw_link_target=None,
                    dev=st.st_dev,
                    ino=st.st_ino,
                    mtime_ns=st.st_mtime_ns,
                    ctime_ns=st.st_ctime_ns,
                ),
            ),
        )

    # One sound entry, one whose key names a different file.
    candidates = {
        valid.resolve(): good.candidates[valid.resolve()],
        victim.resolve(): _snap(decoy),
    }
    plan = common_nwbfile.CleanupPlan(
        scanned_files=set(candidates),
        tracked_files=set(),
        files_to_delete=set(candidates),
        empty_files=set(),
        untracked_files=set(candidates),
        candidates=candidates,
        deferred_recent_files=set(),
        broken_links=set(),
    )

    with pytest.raises(RuntimeError, match="malformed"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert valid.exists(), "sound entry must not be deleted on a bad plan"
    assert victim.exists()
    assert decoy.exists()


def test_registry_refresh_failure_aborts_cleanup(
    common_nwbfile, tmp_path, monkeypatch
):
    """A registry read failure must abort, not fall back to the snapshot.

    Falling back is safe only if the snapshot is a superset of live
    membership -- exactly false in the case the refresh exists to catch, so
    a table registered after the snapshot would have its files deleted.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    tracked_by_new_table = analysis_dir / "belongs_to_new_table.nwb"
    tracked_by_new_table.write_text("data")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert plan.files_to_delete == {tracked_by_new_table.resolve()}

    def _boom():
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(
        common_nwbfile,
        "AnalysisRegistry",
        lambda: type("R", (), {"all_classes": property(lambda s: _boom())})(),
    )

    with pytest.raises(RuntimeError, match="registry unreachable"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert tracked_by_new_table.exists(), "abort must leave files in place"


def test_registry_refresh_unions_snapshot_and_live(common_nwbfile, tmp_path):
    """Refresh must union, never drop, tables from the initial snapshot."""

    class _Tbl:
        def __init__(self, name):
            self.full_table_name = name
            self._ext_tbl = _FakeExternalTable([])

    snapshot = [_Tbl("`a`.`t`")]
    merged = common_nwbfile.AnalysisNwbfile._current_custom_tables(snapshot)
    names = {t.full_table_name for t in merged}

    assert "`a`.`t`" in names, "snapshot table must never be dropped"


def test_registry_knowledge_persists_across_candidates(
    common_nwbfile, tmp_path, monkeypatch
):
    """A table seen once must stay known even if it later disappears.

    Table U appears while candidate A is handled, tracks candidate B, then
    leaves the registry before B is reached. Re-unioning against the
    original snapshot each time would forget U and delete B's file.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    first = analysis_dir / "aaa.nwb"
    first.write_text("first")
    second = analysis_dir / "zzz.nwb"
    second.write_text("second")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert len(plan.candidates) == 2
    # Force order: `first` must be processed before `second`, otherwise the
    # old implementation could pass by accident on some filesystems.
    plan.candidates.clear()
    for path in (first, second):
        rebuilt = _plan(_table(common_nwbfile, analysis_dir))
        plan.candidates[path.resolve()] = rebuilt.candidates[path.resolve()]

    class _Ephemeral:
        """Registry table tracking whichever file is still on disk."""

        full_table_name = "`ephemeral`.`analysis_nwbfile`"

        def __init__(self):
            self._ext_tbl = self

        def fetch_external_paths(self):
            survivors = [p for p in (first, second) if p.exists()]
            return [("h", survivors[-1])] if survivors else []

        def __and__(self, restriction):
            survivors = [p for p in (first, second) if p.exists()]
            if not survivors:
                return []
            rel = restriction["filepath"]
            tracked = survivors[-1]
            return [("h", tracked)] if str(tracked).endswith(rel) else []

    ephemeral = _Ephemeral()
    state = {"calls": 0}

    class _Registry:
        @property
        def all_classes(self):
            state["calls"] += 1
            # Present on the first refresh only, then gone.
            return [ephemeral] if state["calls"] == 1 else []

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", _Registry)

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    survivors = [p for p in (first, second) if p.exists()]
    assert len(survivors) == 1, "exactly the tracked file should survive"
    assert survivors == [
        second
    ], "the ephemeral table's tracked file must survive its removal"


def test_delete_handles_chained_aliases_regardless_of_order(
    common_nwbfile, tmp_path
):
    """a.nwb -> b.nwb -> target: both links go, in either access order.

    Unlinking b first would leave a unable to resolve to the target, so a
    validate-then-unlink pass is required for the result to be
    order-independent.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    middle = analysis_dir / "b.nwb"
    middle.symlink_to(target)
    outer = analysis_dir / "a.nwb"
    outer.symlink_to(middle)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    candidate = plan.candidates[target.resolve()]
    assert len(candidate.accesses) == 2, "both links alias one candidate"

    # Force the order that removes the middle link first.
    ordered = sorted(
        candidate.accesses, key=lambda a: a.access_path.name, reverse=True
    )
    plan.candidates[target.resolve()] = common_nwbfile.CleanupCandidate(
        real_path=candidate.real_path,
        target=candidate.target,
        accesses=tuple(ordered),
    )

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not target.exists(), "external target should be reclaimed"
    assert not middle.is_symlink(), "middle link must be removed"
    assert not outer.is_symlink(), "outer link must be removed, not stranded"


def test_delete_refuses_non_nwb_voucher(common_nwbfile, tmp_path):
    """A forged plan cannot use a non-*.nwb in-root link as authority.

    The scanner only ever yields *.nwb entries, so any other suffix proves
    the plan did not come from it.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    victim = volume2 / "victim.nwb"
    victim.write_text("must survive")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    voucher = analysis_dir / "voucher.txt"
    voucher.symlink_to(victim)

    real = victim.resolve()
    vst = victim.stat()
    lst = voucher.lstat()
    forged = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=common_nwbfile.TargetSnapshot(
            real_path=real,
            dev=vst.st_dev,
            ino=vst.st_ino,
            size=vst.st_size,
            mtime_ns=vst.st_mtime_ns,
            ctime_ns=vst.st_ctime_ns,
            mode=vst.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=voucher,
                is_link=True,
                raw_link_target=str(victim),
                dev=lst.st_dev,
                ino=lst.st_ino,
                mtime_ns=lst.st_mtime_ns,
                ctime_ns=lst.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files={real},
        candidates={real: forged},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = _table(common_nwbfile, analysis_dir)
    with pytest.raises(RuntimeError, match="not a \\*.nwb entry"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert victim.exists(), "non-.nwb voucher must not authorize deletion"
    assert voucher.is_symlink()


def test_delete_rechecks_each_alias_immediately_before_unlink(
    common_nwbfile, tmp_path, monkeypatch
):
    """An approved link swapped mid-pass must not be unlinked blindly.

    Validating a group of links and then unlinking the group leaves a
    window: an earlier-approved link can be replaced by a fresh regular
    file while later ones are checked. Each link is therefore re-checked
    immediately before its own unlink.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link_a = analysis_dir / "a.nwb"
    link_a.symlink_to(target)
    link_b = analysis_dir / "b.nwb"
    link_b.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    original = common_nwbfile.AnalysisNwbfile._access_still_matches
    swapped = {"done": False}

    def _swap_then_check(access, **kwargs):
        # After the first link is approved, replace the OTHER one with a
        # regular file, exactly as a racing writer would.
        result = original(access, **kwargs)
        if result and not swapped["done"]:
            swapped["done"] = True
            other = link_b if access.access_path == link_a else link_a
            if other.is_symlink():
                other.unlink()
                other.write_text("innocent replacement")
        return result

    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_access_still_matches",
        staticmethod(_swap_then_check),
    )

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    replacements = [
        p for p in (link_a, link_b) if p.exists() and not p.is_symlink()
    ]
    assert replacements, "test did not exercise the swap"
    assert (
        replacements[0].read_text() == "innocent replacement"
    ), "a regular file that replaced an approved link must not be unlinked"


def test_delete_refuses_uppercase_nwb_voucher(common_nwbfile, tmp_path):
    """The suffix check must match the scanner's case-sensitive predicate.

    The scan uses `fname.endswith(".nwb")`, so `voucher.NWB` can never come
    from it; accepting it would let a forged plan authorize deletion.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    victim = volume2 / "victim.nwb"
    victim.write_text("must survive")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    voucher = analysis_dir / "voucher.NWB"
    voucher.symlink_to(victim)

    real = victim.resolve()
    vst = victim.stat()
    lst = voucher.lstat()
    forged = common_nwbfile.CleanupCandidate(
        real_path=real,
        target=common_nwbfile.TargetSnapshot(
            real_path=real,
            dev=vst.st_dev,
            ino=vst.st_ino,
            size=vst.st_size,
            mtime_ns=vst.st_mtime_ns,
            ctime_ns=vst.st_ctime_ns,
            mode=vst.st_mode,
        ),
        accesses=(
            common_nwbfile.AccessSnapshot(
                access_path=voucher,
                is_link=True,
                raw_link_target=str(victim),
                dev=lst.st_dev,
                ino=lst.st_ino,
                mtime_ns=lst.st_mtime_ns,
                ctime_ns=lst.st_ctime_ns,
            ),
        ),
    )
    plan = common_nwbfile.CleanupPlan(
        scanned_files={real},
        tracked_files=set(),
        files_to_delete={real},
        empty_files=set(),
        untracked_files={real},
        candidates={real: forged},
        deferred_recent_files=set(),
        broken_links=set(),
    )

    table = _table(common_nwbfile, analysis_dir)
    with pytest.raises(RuntimeError, match="not a \\*.nwb entry"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert victim.exists(), "uppercase-suffix voucher must not authorize"


def test_delete_handles_relative_chained_alias_across_subdir(
    common_nwbfile, tmp_path
):
    """A `../b.nwb` hop must be recognized as a dependency.

    os.walk yields root-level b.nwb before descending into sub/, so the
    unsafe [b, a] order is the natural one. Comparing paths lexically would
    score both links depth zero -- `analysis/sub/../b.nwb` is not equal to
    `analysis/b.nwb` as a Path -- keep that order, and strand a.nwb.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("data")

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sub").mkdir(parents=True)
    middle = analysis_dir / "b.nwb"
    middle.symlink_to(target)
    outer = analysis_dir / "sub" / "a.nwb"
    outer.symlink_to(Path("..") / "b.nwb")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    candidate = plan.candidates[target.resolve()]
    assert len(candidate.accesses) == 2, "both links alias one candidate"

    # Force the unsafe order explicitly: b (the traversed link) first.
    by_name = {a.access_path.name: a for a in candidate.accesses}
    plan.candidates[target.resolve()] = common_nwbfile.CleanupCandidate(
        real_path=candidate.real_path,
        target=candidate.target,
        accesses=(by_name["b.nwb"], by_name["a.nwb"]),
    )

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not target.exists(), "external target should be reclaimed"
    assert not middle.is_symlink(), "traversed link must be removed"
    assert not outer.is_symlink(), "relative outer link must not be stranded"


def test_delete_reclaims_out_of_root_target(common_nwbfile, tmp_path):
    """An untracked external target is reclaimed along with its link."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("external data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)
    local = analysis_dir / "local.nwb"
    local.write_text("in-root data")

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not target.exists(), "external target should be reclaimed"
    assert not link.is_symlink(), "its in-root link is removed"
    assert not local.exists(), "in-root regular files are still deleted"


def test_external_deletions_are_reported(common_nwbfile, tmp_path, caplog):
    """Cross-volume deletions must be visible in the weekly log."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("0123456789")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "link.nwb").symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    with caplog.at_level("WARNING"):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert any(
        "deleted OUTSIDE" in r.message and "10 bytes" in r.message
        for r in caplog.records
    ), "external deletions must be reported with a byte total"


def test_external_audit_does_not_claim_failed_unlink(
    common_nwbfile, tmp_path, monkeypatch, caplog
):
    """A failed target unlink must not be reported as reclaimed storage."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    path_type = type(target)
    real_unlink = path_type.unlink

    def _fail_target_unlink(path, *args, **kwargs):
        if path == target:
            raise OSError("simulated target unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "unlink", _fail_target_unlink)

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="simulated target unlink"):
            table._remove_untracked_files(
                custom_tables=[],
                dry_run=False,
                plan=plan,
                min_file_age_hours=0,
            )

    assert target.exists()
    assert link.is_symlink()
    messages = [record.message for record in caplog.records]
    assert not any(
        str(target) in message and "Deleted external" in message
        for message in messages
    )
    assert not any("deleted OUTSIDE" in message for message in messages)


def test_external_audit_precedes_later_candidate_failure(
    common_nwbfile, tmp_path, monkeypatch, caplog
):
    """A later fatal refresh must not erase an earlier deletion's audit."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    first = volume2 / "first.nwb"
    second = volume2 / "second.nwb"
    first.write_text("first")
    second.write_text("second")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    first_link = analysis_dir / "first.nwb"
    second_link = analysis_dir / "second.nwb"
    first_link.symlink_to(first)
    second_link.symlink_to(second)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    candidates = dict(plan.candidates)
    plan.candidates.clear()
    for path in (first, second):
        plan.candidates[path.resolve()] = candidates[path.resolve()]

    calls = {"count": 0}

    def _fail_second_refresh(snapshot):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("later registry failure")
        return list(snapshot)

    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_current_custom_tables",
        staticmethod(_fail_second_refresh),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="later registry failure"):
            table._remove_untracked_files(
                custom_tables=[],
                dry_run=False,
                plan=plan,
                min_file_age_hours=0,
            )

    assert not first.exists(), "first target must have been deleted"
    assert not first_link.is_symlink()
    assert second.exists(), "later failure occurs before the second deletion"
    assert second_link.is_symlink()
    assert any(
        "Deleted external analysis target" in record.message
        and str(first) in record.message
        for record in caplog.records
    ), "the successful deletion needs an immediate audit record"


def test_tracked_external_link_is_preserved(common_nwbfile, tmp_path):
    """A tracked link is left alone, link and target both."""
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "target.nwb"
    target.write_text("data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir, tracked=[link])
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert link.is_symlink(), "tracked link must be preserved"
    assert target.exists()


@pytest.mark.parametrize(
    "setting_key",
    [
        "SPYGLASS_RAW_DIR",
        "SPYGLASS_RECORDING_DIR",
        "SPYGLASS_SORTING_DIR",
        "SPYGLASS_VIDEO_DIR",
        "SPYGLASS_WAVEFORMS_DIR",
        "SPYGLASS_TEMP_DIR",
        "SPYGLASS_EXPORT_DIR",
        "KACHERY_CLOUD_DIR",
        "KACHERY_STORAGE_DIR",
        "KACHERY_TEMP_DIR",
        "DLC_PROJECT_DIR",
        "DLC_VIDEO_DIR",
        "DLC_OUTPUT_DIR",
        "MOSEQ_PROJECT_DIR",
        "MOSEQ_VIDEO_DIR",
        "FUTURE_PIPELINE_DIR",
    ],
)
def test_delete_refuses_target_in_another_spyglass_store(
    common_nwbfile, tmp_path, monkeypatch, setting_key
):
    """A symlink into any other managed store must not delete its target.

    `tracked` is built from the analysis external store only, so a file in
    another store reads as untracked. Those files may not be recomputable,
    and the age gate cannot help when they are already old.
    """
    managed_root = tmp_path / setting_key.lower()
    managed_root.mkdir()
    acquisition = managed_root / "sub-x_ses-y.nwb"
    acquisition.write_text("irreplaceable acquisition data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    convenience_link = analysis_dir / "session_copy.nwb"
    convenience_link.symlink_to(acquisition)

    _patch_config_dir(common_nwbfile, monkeypatch, setting_key, managed_root)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert (
        acquisition.exists()
    ), "a protected-store target must never be deleted"
    assert convenience_link.is_symlink(), "the link is left with its target"


def test_protected_roots_exclude_base_and_analysis(
    common_nwbfile, tmp_path, monkeypatch
):
    """Owned paths and non-directory settings are not protected roots."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    monkeypatch.setattr(
        common_nwbfile,
        "config",
        {
            "SPYGLASS_BASE_DIR": str(tmp_path),
            "SPYGLASS_ANALYSIS_DIR": str(analysis_dir),
            "KACHERY_ZONE": "test.zone",
            "KACHERY_CLOUD_EPHEMERAL": "TRUE",
        },
    )

    assert common_nwbfile.AnalysisNwbfile._other_managed_roots() == []


def test_dry_run_reports_target_candidate_not_leaf_unlink(
    common_nwbfile, tmp_path, monkeypatch
):
    """Dry-run reports raw target candidates, not an exact unlink manifest.

    Protected-store and act-time checks deliberately run only during the real
    pass. The preview can therefore include a target that will be refused, and
    it does not separately list the in-analysis leaf that an accepted
    candidate would also remove.
    """
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    target = raw_root / "protected.nwb"
    target.write_text("raw")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)
    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    preview, _ = table._remove_untracked_files(
        custom_tables=[], dry_run=True, plan=plan, min_file_age_hours=0
    )

    assert preview == {target.resolve()}
    assert link not in preview, "leaf access paths are not target candidates"
    assert target.exists() and link.is_symlink(), "dry-run must not unlink"

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )
    assert target.exists(), "the real pass must refuse the protected target"
    assert link.is_symlink(), "refusing a target must preserve its leaf"


def test_delete_refuses_physical_alias_of_protected_store(
    common_nwbfile, tmp_path, monkeypatch
):
    """Physical ancestry protects a store when lexical containment misses.

    The conditional monkeypatch reproduces the false negative seen with case
    variants on case-insensitive filesystems and with whole-root bind mounts,
    while leaving every other containment check untouched.
    """
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    acquisition = raw_root / "sub-x_ses-y.nwb"
    acquisition.write_text("irreplaceable acquisition data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    convenience_link = analysis_dir / "session_copy.nwb"
    convenience_link.symlink_to(acquisition)

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)
    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    real_is_relative_to = common_nwbfile.Path.is_relative_to
    target_path = acquisition.resolve()
    protected_path = raw_root.resolve()

    def _miss_lexical_alias(path, other, *args):
        if path == target_path and Path(other) == protected_path:
            return False
        return real_is_relative_to(path, other, *args)

    monkeypatch.setattr(
        common_nwbfile.Path, "is_relative_to", _miss_lexical_alias
    )

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert acquisition.exists(), "physical protected-root alias must survive"
    assert convenience_link.is_symlink()


def test_delete_aborts_when_protected_store_root_cannot_be_resolved(
    common_nwbfile, tmp_path, monkeypatch
):
    """A protected-store resolution failure must not disable protection."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    acquisition = raw_root / "sub-x_ses-y.nwb"
    acquisition.write_text("irreplaceable acquisition data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    convenience_link = analysis_dir / "session_copy.nwb"
    convenience_link.symlink_to(acquisition)

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    original_resolve = common_nwbfile.Path.resolve

    def _fail_for_raw_root(path, *args, **kwargs):
        if path == raw_root:
            raise OSError("protected store is temporarily unavailable")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(common_nwbfile.Path, "resolve", _fail_for_raw_root)

    with pytest.raises(
        RuntimeError,
        match="Cannot resolve protected Spyglass raw store root",
    ):
        table._remove_untracked_files(
            custom_tables=[],
            dry_run=False,
            plan=plan,
            min_file_age_hours=0,
        )

    assert acquisition.exists(), "resolution failure must preserve the target"
    assert convenience_link.is_symlink(), "the authorizing link must survive"


def test_delete_aborts_when_protected_store_root_cannot_be_statted(
    common_nwbfile, tmp_path, monkeypatch
):
    """Protected-root inspection fails before any candidate is unlinked."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    first = analysis_dir / "first.nwb"
    second = analysis_dir / "second.nwb"
    first.write_text("first")
    second.write_text("second")

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)
    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    real_stat = common_nwbfile.os.stat
    raw_resolved = raw_root.resolve()

    def _fail_for_raw_root(path, *args, **kwargs):
        if Path(path) == raw_resolved:
            raise PermissionError("protected store is temporarily unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(common_nwbfile.os, "stat", _fail_for_raw_root)

    with pytest.raises(
        RuntimeError,
        match="Cannot resolve protected Spyglass raw store root",
    ):
        table._remove_untracked_files(
            custom_tables=[],
            dry_run=False,
            plan=plan,
            min_file_age_hours=0,
        )

    assert first.exists(), "root inspection must precede the first unlink"
    assert second.exists()


def test_managed_root_containment_does_not_match_sibling_prefix(
    common_nwbfile, tmp_path, monkeypatch
):
    """A sibling named like a protected root is still eligible storage."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    sibling = tmp_path / "raw_archive"
    sibling.mkdir()
    target = sibling / "recomputable.nwb"
    target.write_text("analysis result")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert not target.exists(), "path-prefix siblings are not inside raw_dir"
    assert not link.is_symlink()


def test_candidate_rejects_empty_accesses(common_nwbfile, tmp_path):
    """A candidate with no accesses is unrepresentable.

    Such a candidate passes the structural preflight (its *.nwb loop is
    vacuous) and then raises from max() partway through the deletion loop,
    after earlier candidates have already been unlinked.
    """
    with pytest.raises(ValueError, match="no access paths"):
        common_nwbfile.CleanupCandidate(
            real_path=tmp_path / "x.nwb", target=None, accesses=()
        )


def test_cleanup_unblock_failure_raises_when_called_from_except(
    common_nwbfile, monkeypatch
):
    """An unblock failure must raise even inside a caller's except block.

    sys.exc_info() returns the exception being handled anywhere up the
    stack, so using it here silently downgraded the failure -- leaving
    inserts blocked database-wide with only a log line.
    """

    class _RaisingRegistry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            pass

        def unblock_new_inserts(self):
            raise RuntimeError("unblock_kaboom")

    good_plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)
    monkeypatch.setattr(
        common_nwbfile, "AnalysisRegistry", lambda: _RaisingRegistry()
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables, **kw: good_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan, **kw: (set(), set()),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    # The body succeeds; the CALLER is mid-except. This is the case that
    # sys.exc_info() got wrong.
    try:
        raise ValueError("unrelated caller error")
    except ValueError:
        with pytest.raises(RuntimeError, match="unblock_kaboom"):
            common_nwbfile.AnalysisNwbfile.cleanup(table, dry_run=False)


def test_scan_skips_symlink_loop_without_aborting(common_nwbfile, tmp_path):
    """An ELOOP entry is skipped, not fatal.

    One accidental loop must not wedge the weekly sweep -- and with it
    every later maintenance phase -- since an entry that cannot be stat'd
    never becomes a deletion candidate anyway.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    good = analysis_dir / "good.nwb"
    good.write_text("data")

    a = analysis_dir / "loop_a.nwb"
    b = analysis_dir / "loop_b.nwb"
    a.symlink_to(b)
    b.symlink_to(a)

    plan = _plan(_table(common_nwbfile, analysis_dir))

    assert good.resolve() in plan.files_to_delete, "scan must still complete"


def test_delete_aborts_when_protected_store_root_is_missing(
    common_nwbfile, tmp_path, monkeypatch
):
    """An unavailable configured store root fails closed before deletion.

    A missing spelling may be an unavailable mount whose contents remain
    reachable through another alias. Without the root identity cleanup cannot
    prove an apparently external candidate is outside that store.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    untracked = analysis_dir / "untracked.nwb"
    untracked.write_text("stale")

    missing_raw = tmp_path / "unavailable_raw_mount"
    _patch_config_dir(
        common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", missing_raw
    )

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    with pytest.raises(
        RuntimeError, match="Cannot resolve protected Spyglass raw store root"
    ):
        table._remove_untracked_files(
            custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
        )

    assert untracked.exists(), "root uncertainty must preserve every candidate"


def test_delete_aborts_when_managed_root_ancestry_cannot_be_statted(
    common_nwbfile, tmp_path, monkeypatch
):
    """A stat failure while walking a target's ancestry fails closed.

    When lexical containment misses (case variant, bind mount) the guard
    walks the target's parents. A parent that cannot be inspected cannot
    rule out a protected store, so cleanup must abort before any unlink.
    """
    raw_root = tmp_path / "raw"
    intermediate = raw_root / "sub"
    intermediate.mkdir(parents=True)
    acquisition = intermediate / "sub-x.nwb"
    acquisition.write_text("irreplaceable")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "copy.nwb"
    link.symlink_to(acquisition)

    _patch_config_dir(common_nwbfile, monkeypatch, "SPYGLASS_RAW_DIR", raw_root)
    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    # Force the physical-ancestry walk by making lexical containment miss,
    # so the guard falls through to stat'ing the target's parents.
    real_is_relative_to = common_nwbfile.Path.is_relative_to
    target_path = acquisition.resolve()
    protected_path = raw_root.resolve()

    def _miss_lexical(path, other, *args):
        if path == target_path and Path(other) == protected_path:
            return False
        return real_is_relative_to(path, other, *args)

    monkeypatch.setattr(common_nwbfile.Path, "is_relative_to", _miss_lexical)

    # _other_managed_roots stats raw_root itself and must succeed; only the
    # ancestry walk's stat of the intermediate directory fails.
    real_stat = common_nwbfile.os.stat
    intermediate_resolved = intermediate.resolve()

    def _fail_intermediate(path, *args, **kwargs):
        if Path(path) == intermediate_resolved:
            raise OSError("ancestry stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(common_nwbfile.os, "stat", _fail_intermediate)

    with pytest.raises(RuntimeError, match="Cannot inspect ancestry"):
        table._remove_untracked_files(
            custom_tables=[],
            dry_run=False,
            plan=plan,
            min_file_age_hours=0,
        )

    assert acquisition.exists(), "ancestry failure must preserve the target"
    assert link.is_symlink(), "the authorizing link must survive"


def test_delete_refuses_target_that_became_non_regular_at_act_time(
    common_nwbfile, tmp_path
):
    """A target planned as a regular file but now a directory is refused.

    The scan-time snapshot recorded a regular file; if the path is a
    directory by deletion time, the act-time re-check must refuse it rather
    than unlink an unexpected target that materialized in the window.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "far.nwb"
    target.write_text("far")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)

    # Between planning and acting, the target becomes a directory.
    target.unlink()
    target.mkdir()

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert target.is_dir(), "a now-directory target must not be unlinked"
    assert link.is_symlink(), "its authorizing link must survive too"


def test_delete_refuses_broken_link_whose_target_now_resolves(
    common_nwbfile, tmp_path
):
    """A dangling link planned as broken but now resolving is refused.

    A broken candidate authorizes only removing the leaf. If its target
    exists by act time, the leaf now points at a real file, so removing it
    would silently orphan that file; the re-check refuses instead.
    """
    volume2 = tmp_path / "volume2"
    volume2.mkdir()
    target = volume2 / "appeared.nwb"

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    link = analysis_dir / "link.nwb"
    link.symlink_to(target)  # target does not exist yet -> broken

    table = _table(common_nwbfile, analysis_dir)
    plan = _plan(table)
    assert target.resolve() in plan.broken_links

    # The target materializes between planning and acting.
    target.write_text("appeared")

    table._remove_untracked_files(
        custom_tables=[], dry_run=False, plan=plan, min_file_age_hours=0
    )

    assert link.is_symlink(), "a now-resolving link must not be removed"
    assert target.exists(), "the newly appeared target must survive"


def test_cleanup_block_failure_propagates_without_unblock(
    common_nwbfile, monkeypatch
):
    """A block failure surfaces and must not trigger the unblock path.

    block_new_inserts() runs outside cleanup()'s try/finally, so a failure
    there must propagate without the finally-block unblock running -- which
    would drop a concurrent run's triggers.
    """
    unblock_calls = []

    class _BlockFailRegistry:
        all_classes = []

        def block_new_inserts(self, dry_run):
            raise RuntimeError("block_kaboom")

        def unblock_new_inserts(self):
            unblock_calls.append(True)

    monkeypatch.setattr(
        common_nwbfile, "AnalysisRegistry", lambda: _BlockFailRegistry()
    )
    table = object.__new__(common_nwbfile.AnalysisNwbfile)

    with pytest.raises(RuntimeError, match="block_kaboom"):
        common_nwbfile.AnalysisNwbfile.cleanup(table, dry_run=False)

    assert not unblock_calls, "unblock must not run when block itself failed"


def test_block_new_inserts_partial_failure_requires_safe_recovery(
    common_nwbfile, monkeypatch
):
    """A partial failure must not prescribe an unconditional global unblock.

    Acquisition uses one sorted snapshot and stops at the first error. Earlier
    triggers may still belong to this call, so recovery must exclude an active
    cleanup before removing anything.
    """
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *a, **k: ["db.`t3`", "db.`t1`", "db.`t2`"],
    )
    inspected = []
    attempted = []

    def _exists(self, table):
        inspected.append(table)
        return False

    def _block(self, table, dry_run=False):
        attempted.append(table)
        return "trigger DDL error" if table == "db.`t2`" else None

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry, "_block_exists", _exists
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry, "_block_single_table", _block
    )

    with pytest.raises(RuntimeError) as exc:
        registry.block_new_inserts(dry_run=False)

    msg = str(exc.value)
    assert "trigger DDL error" in msg
    assert "may remain blocked" in msg
    assert "unblock_new_inserts()" in msg
    assert "Confirm that no cleanup is active" in msg
    assert "inspect the blocking triggers" in msg
    assert "only after confirming every such trigger is stale" in msg
    assert inspected == ["db.`t1`", "db.`t2`", "db.`t3`"]
    assert attempted == ["db.`t1`", "db.`t2`"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_block_new_inserts_refuses_preexisting_trigger_before_acquisition(
    common_nwbfile, monkeypatch, dry_run
):
    """An active-or-stale trigger is never adopted, even by a preview."""
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *a, **k: ["db.`t3`", "db.`t1`", "db.`t2`"],
    )
    inspected = []
    attempted = []

    def _exists(self, table):
        inspected.append(table)
        return table == "db.`t2`"

    def _block(self, table, dry_run=False):
        attempted.append(table)

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry, "_block_exists", _exists
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry, "_block_single_table", _block
    )

    with pytest.raises(RuntimeError) as exc:
        registry.block_new_inserts(dry_run=dry_run)

    msg = str(exc.value)
    assert "already exist" in msg
    assert "Another cleanup may be active" in msg
    assert "triggers may be stale" in msg
    assert "Confirm that no cleanup is active" in msg
    assert "unblock_new_inserts()" in msg
    assert inspected == ["db.`t1`", "db.`t2`", "db.`t3`"]
    assert attempted == []


def test_block_new_inserts_refuses_trigger_that_appears_after_preflight(
    common_nwbfile, monkeypatch
):
    """A racing trigger creation is an error rather than adopted ownership."""
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *a, **k: ["db.`t1`"],
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_get_block_info",
        lambda self, table: ("db", "t1_block_inserts"),
    )
    checks = iter([False, True])
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_exists",
        lambda self, table: next(checks),
    )

    with pytest.raises(RuntimeError) as exc:
        registry.block_new_inserts(dry_run=False)

    msg = str(exc.value)
    assert "blocking trigger already exists" in msg
    assert "another cleanup may be active" in msg
    assert "trigger may be stale" in msg


def test_block_new_inserts_fails_closed_when_preflight_cannot_inspect(
    common_nwbfile, monkeypatch
):
    """An unreadable blocker state aborts before trigger acquisition."""
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *a, **k: ["db.`t1`"],
    )
    attempted = []

    def _cannot_inspect(self, table):
        raise PermissionError("denied")

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_exists",
        _cannot_inspect,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_block_single_table",
        lambda self, table, dry_run=False: attempted.append(table),
    )

    with pytest.raises(RuntimeError, match="Failed to inspect insert blocker"):
        registry.block_new_inserts(dry_run=False)

    assert attempted == []


def test_block_new_inserts_dry_run_is_sorted_and_has_no_ddl(
    common_nwbfile, monkeypatch, caplog
):
    """A successful preview inspects/logs deterministically without CREATE."""
    registry = object.__new__(common_nwbfile.AnalysisRegistry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "fetch",
        lambda self, *a, **k: ["db.`t2`", "db.`t1`"],
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry,
        "_get_block_info",
        lambda self, table: ("db", f"{table[-3:-1]}_block_inserts"),
    )
    inspected = []

    def _exists(self, table):
        inspected.append(table)
        return False

    monkeypatch.setattr(
        common_nwbfile.AnalysisRegistry, "_block_exists", _exists
    )

    with caplog.at_level("INFO"):
        registry.block_new_inserts(dry_run=True)

    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("Dry run: would block")
    ]
    assert inspected == ["db.`t1`", "db.`t2`", "db.`t1`", "db.`t2`"]
    assert messages == [
        "Dry run: would block inserts into db.`t1`",
        "Dry run: would block inserts into db.`t2`",
    ]


def test_access_snapshot_rejects_link_flag_target_mismatch(common_nwbfile):
    """is_link and raw_link_target must agree at construction."""
    with pytest.raises(ValueError, match="disagrees with raw_link_target"):
        common_nwbfile.AccessSnapshot(
            access_path=Path("/x.nwb"),
            is_link=True,
            raw_link_target=None,
            dev=1,
            ino=1,
            mtime_ns=0,
            ctime_ns=0,
        )
    with pytest.raises(ValueError, match="disagrees with raw_link_target"):
        common_nwbfile.AccessSnapshot(
            access_path=Path("/x.nwb"),
            is_link=False,
            raw_link_target="/y.nwb",
            dev=1,
            ino=1,
            mtime_ns=0,
            ctime_ns=0,
        )


def test_access_snapshot_from_path_regular_file(common_nwbfile, tmp_path):
    """The factory records a regular leaf's lstat identity."""
    path = tmp_path / "regular.nwb"
    path.write_text("data")
    lst = os.lstat(path)

    snapshot = common_nwbfile.AccessSnapshot.from_path(path)

    assert snapshot.access_path == path
    assert snapshot.is_link is False
    assert snapshot.raw_link_target is None
    assert (snapshot.dev, snapshot.ino) == (lst.st_dev, lst.st_ino)
    assert (snapshot.mtime_ns, snapshot.ctime_ns) == (
        lst.st_mtime_ns,
        lst.st_ctime_ns,
    )


def test_access_snapshot_from_path_symlink(common_nwbfile, tmp_path):
    """The factory records the link itself without following its target."""
    link = tmp_path / "link.nwb"
    link.symlink_to("missing.nwb")
    lst = os.lstat(link)

    snapshot = common_nwbfile.AccessSnapshot.from_path(link)

    assert snapshot.access_path == link
    assert snapshot.is_link is True
    assert snapshot.raw_link_target == "missing.nwb"
    assert (snapshot.dev, snapshot.ino) == (lst.st_dev, lst.st_ino)
    assert (snapshot.mtime_ns, snapshot.ctime_ns) == (
        lst.st_mtime_ns,
        lst.st_ctime_ns,
    )


def test_cleanup_candidate_rejects_mismatched_target(common_nwbfile, tmp_path):
    """A candidate's target snapshot must describe its own real_path."""
    a = tmp_path / "a.nwb"
    a.write_text("a")
    b = tmp_path / "b.nwb"
    b.write_text("b")
    st_a = os.stat(a)
    st_b = os.stat(b)
    lst_a = os.lstat(a)
    access = common_nwbfile.AccessSnapshot(
        access_path=a,
        is_link=False,
        raw_link_target=None,
        dev=lst_a.st_dev,
        ino=lst_a.st_ino,
        mtime_ns=lst_a.st_mtime_ns,
        ctime_ns=lst_a.st_ctime_ns,
    )

    def _snap(path, st):
        return common_nwbfile.TargetSnapshot(
            real_path=path,
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        )

    # Matching target is accepted.
    common_nwbfile.CleanupCandidate(
        real_path=a, target=_snap(a, st_a), accesses=(access,)
    )
    # A target snapshot naming a different path is refused at construction.
    with pytest.raises(ValueError, match="does not match its target"):
        common_nwbfile.CleanupCandidate(
            real_path=a, target=_snap(b, st_b), accesses=(access,)
        )
