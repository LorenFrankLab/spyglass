"""Plan reuse against real edits on disk, rather than a simulated digest.

`test_plan_reuse.py` shifts a digest to stand in for an edit, which checks the
reuse decision but not the chain that produces it. These tests edit the file
with h5py and let the hasher notice, so an object-index blind spot shows up as
a table that fails to re-plan.

The workflow being protected: ingest, read the report, fix the file, try
again. Only what the fix touched should cost anything the second time.
"""

from contextlib import contextmanager

import h5py
import pytest

# An attribute the hasher folds into the object's digest -- not in
# IGNORED_KEYS, which exists so a version bump does not invalidate a file.
PROBE_ATTR = "ffi_reuse_probe"


@contextmanager
def _edit(path):
    """Open the file for writing, dropping Spyglass's cached handle first.

    Parsing leaves the file open and cached, which costs twice here: h5py
    cannot open it for writing, and a later plan would read the pre-edit file
    out of that cache and see none of this.
    """
    from spyglass.utils.nwb_helper_fn import close_nwb_files

    close_nwb_files()
    with h5py.File(path, "a") as file:
        yield file
    close_nwb_files()


def _perturb(path, h5_path, value=1):
    """Change one object's digest by adding an attribute to it.

    Type-agnostic on purpose: rewriting a dataset means knowing its dtype and
    shape, while an attribute edits the same way whatever the object is, and
    the hasher folds attributes into exactly the one object's digest.
    """
    with _edit(path) as file:
        assert h5_path in file, f"Premise: {h5_path} is in the file"
        file[h5_path].attrs[PROBE_ATTR] = value


def _stage(name):
    """Plan the file from scratch and stage it, as a first attempt would."""
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    log.clear(name)
    plan = plan_nwbfile(name, force_replan=True)
    log.stage(plan)
    return log, plan


def _productive(plan):
    """Tables that planned at least one entry.

    The claims below are about these. A table that found no source read
    nothing, so its digest is a digest of nothing and it must always
    re-parse -- see `test_plan_reuse.py` for why that is not a shortfall.
    """
    return {tp.table_name for tp in plan.table_plans if tp.entry_count}


def _readers_of(plan, hasher, h5_path):
    """Tables whose read-set covers the object at `h5_path`, or a container.

    Invalidation is scoped to the changed object *and the containers holding
    it*, since a container's digest folds in everything beneath it. A test
    that only looked for direct readers would call a legitimate parent
    re-parse collateral damage.

    Mirrors the index's ownership rule: a path no object claims belongs to
    the root, and one that is claimed does not -- so an edit inside a
    container is not also an edit to the file's metadata.
    """
    changed = {
        object_id
        for object_id, (path, _) in hasher.obj_ids.items()
        if path != "/" and (path == h5_path or h5_path.startswith(f"{path}/"))
    }
    if not changed:
        changed = {
            object_id
            for object_id, (path, _) in hasher.obj_ids.items()
            if path == "/"
        }

    return {
        tp.table_name
        for tp in plan.table_plans
        if changed & set(tp.reads or ())
    }


@pytest.fixture
def hasher_for(common):
    """Build an object-id index for a file, as the planner does."""
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.utils.nwb_hash import NwbfileHasher

    def build(name):
        return NwbfileHasher(
            Nwbfile.get_abs_path(name), keep_obj_hash=True, object_ids=True
        )

    return build


def test_editing_one_object_re_parses_its_readers_and_nothing_else(
    common, editable_copy, parse_counter, hasher_for
):
    """A targeted edit invalidates its readers, and leaves the rest alone.

    `SampleCount` reads one object, so this is the narrowest case the file
    offers: everything else must come back from storage.
    """
    from spyglass.data_import.planner import plan_nwbfile

    name, path = editable_copy
    _, first = _stage(name)
    productive = _productive(first)
    assert productive, "Premise: the mini file fills some tables"

    target = "processing/sample_count/sample_count"
    readers = _readers_of(first, hasher_for(name), target)
    assert common.SampleCount().full_table_name in readers, (
        "Premise: SampleCount is recorded as reading " + target
    )

    _perturb(path, target)
    parse_counter.clear()
    plan_nwbfile(name)

    parsed = set(parse_counter)
    assert readers <= parsed, (
        "A table that read the edited object must re-parse; missed: "
        + f"{sorted(readers - parsed)}"
    )
    collateral = (parsed & productive) - readers
    assert not collateral, (
        "An edit must not invalidate tables that did not read it; also "
        + f"re-parsed: {sorted(collateral)}"
    )


def test_editing_file_metadata_re_parses_the_tables_that_read_it(
    common, editable_copy, parse_counter, hasher_for
):
    """Editing `general/lab` must re-plan `Lab`.

    File-level metadata belongs to the root NWBFile object, which the
    traversal visits the members of but never itself. While the root carried
    no digest, every table reading session or lab metadata hashed to the
    constant an unknown id scores, so no edit here could ever invalidate
    them and reuse served the previous attempt's rows.
    """
    from spyglass.data_import.planner import plan_nwbfile

    name, path = editable_copy
    _, first = _stage(name)

    lab = common.Lab().full_table_name
    assert lab in _productive(first), "Premise: Lab plans an entry"
    readers = _readers_of(first, hasher_for(name), "general/lab")

    _perturb(path, "general/lab")
    parse_counter.clear()
    plan_nwbfile(name)
    parsed = set(parse_counter)

    # Asserted on the table directly, not only through `_readers_of`: an
    # unindexed root makes that lookup come back empty, and a claim about an
    # empty set is satisfied by reusing everything.
    assert lab in parsed, "Editing general/lab must re-plan Lab"

    missed = readers - parsed
    assert not missed, (
        "Editing file metadata must re-plan the tables that read it; "
        + f"reused instead: {sorted(missed)}"
    )


def test_an_unrelated_edit_re_parses_nothing(
    common, editable_copy, parse_counter, hasher_for
):
    """The invariant Phase 6 exists for: an unrelated fix costs nothing.

    `camera_frame_counts` is indexed, so an edit to it is visible to the
    hasher, and no ingestion table reads it -- which is what makes it the
    honest unrelated case rather than one the index simply cannot see.
    """
    from spyglass.data_import.planner import plan_nwbfile

    name, path = editable_copy
    _, first = _stage(name)
    productive = _productive(first)

    unrelated = "processing/camera_sample_frame_counts/camera_frame_counts"
    hasher = hasher_for(name)
    assert any(
        path_ == unrelated for path_, _ in hasher.obj_ids.values()
    ), f"Premise: {unrelated} is indexed, so an edit to it is visible"
    assert not _readers_of(
        first, hasher, unrelated
    ), f"Premise: nothing reads {unrelated}"

    _perturb(path, unrelated)
    parse_counter.clear()
    plan_nwbfile(name)

    re_planned = set(parse_counter) & productive
    assert not re_planned, (
        "An unrelated edit must not re-plan anything that found something; "
        + f"re-parsed: {sorted(re_planned)}"
    )


def test_fixing_a_defect_on_disk_re_plans_only_the_affected_table(
    common, editable_copy, parse_counter, hasher_for
):
    """The primary workflow: a failed attempt, a fix on disk, a retry.

    Breaking `camera_id` is reversible, so the file can be fixed back toward
    a working state the way a user would, and the retry must charge only for
    what the fix touched while every other table's entries survive.
    """
    from spyglass.data_import.planner import plan_nwbfile

    name, path = editable_copy

    with _edit(path) as file:
        camera = "processing/tasks/task_0/camera_id"
        assert camera in file, f"Premise: {camera} is in the file"
        was = file[camera][()]
        file[camera][...] = 99  # a camera no CameraDevice provides

    _, broken = _stage(name)
    assert broken.blocking, "Premise: a dangling camera_id blocks the plan"

    # Read off the table plans rather than `problem.table`, which a
    # file-level problem leaves unset.
    offenders = {tp.table_name for tp in broken.table_plans if not tp.is_clean}
    assert offenders, "Premise: some table owns the blocking problem"

    survivors = {
        tp.table_name: tp.entry_count
        for tp in broken.table_plans
        if tp.entry_count and tp.table_name not in offenders
    }

    with _edit(path) as file:
        file[camera][...] = was  # the fix a user would make

    parse_counter.clear()
    fixed = plan_nwbfile(name)

    parsed = set(parse_counter)
    assert offenders & parsed, (
        "The table whose defect was fixed must re-plan; parsed: "
        + f"{sorted(parsed)}"
    )

    # The fix is an edit like any other, so its blast radius is bounded the
    # same way: the retry must not re-parse the tables it did not touch.
    touched = _readers_of(broken, hasher_for(name), "processing/tasks/task_0")
    collateral = (parsed & set(survivors)) - touched - offenders
    assert not collateral, (
        "Fixing one defect must not re-plan unrelated tables; also "
        + f"re-parsed: {sorted(collateral)}"
    )

    still_blocking = {
        tp.table_name for tp in fixed.table_plans if not tp.is_clean
    }
    assert not (offenders & still_blocking), (
        "The fixed defect must clear: still blocking for "
        + f"{sorted(offenders & still_blocking)}"
    )

    after = {tp.table_name: tp.entry_count for tp in fixed.table_plans}
    lost = {
        table: (count, after.get(table))
        for table, count in survivors.items()
        if after.get(table) != count
    }
    assert not lost, (
        "Entries staged before the fix must survive it; changed: " + f"{lost}"
    )
