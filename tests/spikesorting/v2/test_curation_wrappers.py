"""Tests for the friendly curation writers + summarize_curation.

The wrappers pre-fill ``insert_curation`` arguments, inherit its validation,
and branch merges off the parent. ``CurationV2`` is imported inside each test
so collection does not connect before the MySQL container is ready.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import clear_curations_for

_TEAM = "curation_wrappers_team"
# The smoke sort yields one unit; merge tests need the nightly 60s fixture
# and skip without it.
_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


@pytest.fixture(scope="module")
def polymer_60s_sort(dj_conn):
    """A MountainSort5 sort that yields several well-isolated single units.

    Module-scoped; ingested under a distinct session name so other modules
    using the 60s fixture do not collide.
    """
    if not _POLYMER_60S_PATH.exists():
        pytest.skip(f"Fixture {_POLYMER_60S_PATH.name} not found.")

    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from tests.spikesorting.v2._ingest_helpers import (
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )

    nwb_file_name = copy_and_insert_nwb(
        _POLYMER_60S_PATH, dest_name="mearec_curwrap_60s.nwb"
    )
    inputs = configure_v2_run_inputs(
        nwb_file_name, _TEAM, team_description="curation wrapper tests"
    )
    run_summary = run_v2_pipeline(
        **inputs,
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    )
    return {"sorting_id": run_summary["sorting_id"]}


def _unit_ids(sort_pk) -> list[int]:
    """Sorted real unit ids of the populated sort."""
    from spyglass.spikesorting.v2.sorting import Sorting

    return sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))


def _two_unit_ids(sort_pk) -> tuple[int, int]:
    """First two real unit ids, or skip if the sort has fewer than two."""
    ids = _unit_ids(sort_pk)
    if len(ids) < 2:
        pytest.skip(f"need >=2 sort units for merge tests; got {len(ids)}")
    return ids[0], ids[1]


@pytest.mark.database
@pytest.mark.slow
def test_create_initial_curation_equiv(populated_sorting):
    """``create_initial_curation`` == ``insert_curation(parent=-1)``."""
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(populated_sorting)
    wrapper_key = CurationV2.create_initial_curation(populated_sorting)
    # Idempotent default root: same row.
    expert_key = CurationV2.insert_curation(
        sorting_key=populated_sorting, parent_curation_id=-1
    )
    assert wrapper_key == expert_key
    assert bool((CurationV2 & wrapper_key).fetch1("merges_applied")) is False
    assert len(CurationV2.Unit & wrapper_key) >= 1


@pytest.mark.database
@pytest.mark.slow
def test_preview_merges_records_not_applies(polymer_60s_sort):
    """``preview_merges`` records merges without applying them."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import create_initial_curation

    a, b = _two_unit_ids(polymer_60s_sort)
    clear_curations_for(polymer_60s_sort)
    root = create_initial_curation(polymer_60s_sort)
    key = root.preview_merges([[a, b]]).as_key()
    assert key["curation_id"] != root.curation_id
    assert (
        int((CurationV2 & key).fetch1("parent_curation_id")) == root.curation_id
    )
    assert bool((CurationV2 & key).fetch1("merges_applied")) is False
    units_after = {int(u) for u in (CurationV2.Unit & key).fetch("unit_id")}
    assert {a, b} <= units_after
    groups = CurationV2.get_unit_contributor_groups(key)
    assert any(len(c) > 1 for c in groups.values())
    assert CurationV2.summarize_curation(key)["is_merge_preview"] is True


@pytest.mark.database
@pytest.mark.slow
def test_commit_merges_commits(polymer_60s_sort):
    """``commit_merges`` commits the merged unit set."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import create_initial_curation

    a, b = _two_unit_ids(polymer_60s_sort)
    n_sort_units = len(_unit_ids(polymer_60s_sort))
    clear_curations_for(polymer_60s_sort)
    root = create_initial_curation(polymer_60s_sort)
    key = root.commit_merges([[a, b]]).as_key()
    assert (
        int((CurationV2 & key).fetch1("parent_curation_id")) == root.curation_id
    )
    assert bool((CurationV2 & key).fetch1("merges_applied")) is True
    assert CurationV2.summarize_curation(key)["is_merge_preview"] is False
    assert len(CurationV2.Unit & key) == n_sort_units - 1


@pytest.mark.database
@pytest.mark.slow
def test_commit_merges_reuses_matching_child(polymer_60s_sort):
    """Reuse by default makes a manual-merge notebook cell rerunnable."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2._core.enums import CurationLabel
    from spyglass.spikesorting.v2.curation_api import create_initial_curation

    a, b = _two_unit_ids(polymer_60s_sort)
    all_unit_ids = _unit_ids(polymer_60s_sort)
    merged_unit_id = max(all_unit_ids) + 1
    clear_curations_for(polymer_60s_sort)
    root = create_initial_curation(polymer_60s_sort)

    first = root.commit_merges(
        [[a, b]],
        labels={merged_unit_id: [CurationLabel.accept]},
        description="manual burst-pair merge",
    )
    child_restriction = CurationV2 & {
        "sorting_id": first.sorting_id,
        "parent_curation_id": root.curation_id,
        "description": "manual burst-pair merge",
        "merges_applied": True,
    }
    n_children = len(child_restriction)

    # Same merge in a different order, with enum vs string labels, reuses
    # the existing child.
    second = root.commit_merges(
        [[b, a]],
        labels={merged_unit_id: ["accept"]},
        description="manual burst-pair merge",
    )
    assert second == first
    assert len(child_restriction) == n_children


@pytest.mark.database
def test_merge_facade_forwards_to_insert_curation(planted_two_unit_sort):
    """``preview_merges`` / ``commit_merges`` forward apply_merge and parent.

    Runs on the planted two-unit sort, so it needs no 60s fixture.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        commit_merges,
        create_initial_curation,
        preview_merges,
    )

    sorting_key = dict(planted_two_unit_sort)
    a, b = _two_unit_ids(sorting_key)
    clear_curations_for(sorting_key)
    try:
        root = create_initial_curation(sorting_key)
        for merge, applied in ((preview_merges, False), (commit_merges, True)):
            child = merge(
                parent_curation=root,
                groups=[[a, b]],
                description=f"{merge.__name__} child",
            ).as_key()
            row = (CurationV2 & child).fetch1()
            assert row["sorting_id"] == root.sorting_id
            assert int(row["parent_curation_id"]) == root.curation_id
            assert bool(row["merges_applied"]) is applied
            assert row["description"] == f"{merge.__name__} child"
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.database
def test_curation_rejects_lossy_unit_ids(planted_two_unit_sort):
    """Expert and manual writes reject lossy IDs before creating a child."""
    import numpy as np

    from spyglass.spikesorting.v2.curation import CurationV2

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    a, b = _unit_ids(sorting_key)
    try:
        root = CurationV2.create_initial_curation(
            sorting_key, labels={np.int64(a): ["mua"]}
        )
        for writer in (
            CurationV2.insert_curation,
            CurationV2.save_manual_curation,
        ):
            for bad_id in (a + 0.9, float(a), True, False):
                for edits in (
                    {"labels": {bad_id: ["accept"]}},
                    {"merge_groups": [[bad_id, b]]},
                ):
                    with pytest.raises(
                        ValueError, match="unit_id must be an integer"
                    ):
                        writer(
                            sorting_key,
                            parent_curation_id=root["curation_id"],
                            **edits,
                        )
        assert len(CurationV2 & sorting_key) == 1

        child = CurationV2.save_manual_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={a: ["accept"]},
        )
        assert CurationV2._labels_by_unit(child) == {a: ["accept"]}
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.database
def test_save_manual_curation_labels_child(populated_sorting):
    """``save_manual_curation`` stores native labels as a child."""
    from spyglass.spikesorting.v2.curation import CurationV2

    ids = _unit_ids(populated_sorting)
    if not ids:
        pytest.skip("need >=1 sort unit")
    a = ids[0]
    clear_curations_for(populated_sorting)
    root = CurationV2.create_initial_curation(populated_sorting)

    child = CurationV2.save_manual_curation(
        populated_sorting,
        parent_curation_id=root["curation_id"],
        labels={a: ["mua"]},
        curation_source="figpack",
    )

    assert child["curation_id"] != root["curation_id"]
    assert CurationV2.is_committed_curation(child)
    assert (CurationV2 & child).fetch1("curation_source") == "figpack"
    labels = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
    }
    assert labels == {(a, "mua")}


@pytest.mark.database
@pytest.mark.slow
def test_save_manual_curation_preview_merge(polymer_60s_sort):
    """``merge_action='preview'`` records a draft merge without applying it."""
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    clear_curations_for(polymer_60s_sort)
    root = CurationV2.create_initial_curation(polymer_60s_sort)

    preview = CurationV2.save_manual_curation(
        polymer_60s_sort,
        parent_curation_id=root["curation_id"],
        merge_groups=[[a, b]],
        merge_action="preview",
        curation_source="figpack",
    )

    assert CurationV2.is_committed_curation(preview) is False
    assert CurationV2.has_unapplied_proposed_merges(preview) is True
    assert bool((CurationV2 & preview).fetch1("merges_applied")) is False
    assert (CurationV2 & preview).fetch1("curation_source") == "figpack"
    units_after = {int(u) for u in (CurationV2.Unit & preview).fetch("unit_id")}
    assert {a, b} <= units_after


@pytest.mark.database
@pytest.mark.slow
def test_save_manual_curation_commit_merge_inherits_parent_labels(
    polymer_60s_sort,
):
    """Committed manual/FigPack merges inherit labels from contributors."""
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    unit_ids = _unit_ids(polymer_60s_sort)
    merged_id = max(unit_ids) + 1
    clear_curations_for(polymer_60s_sort)
    root = CurationV2.create_initial_curation(
        polymer_60s_sort, labels={a: ["mua"], b: ["noise"]}
    )

    child = CurationV2.save_manual_curation(
        polymer_60s_sort,
        parent_curation_id=root["curation_id"],
        merge_groups=[[a, b]],
        merge_action="commit",
        curation_source="figpack",
    )

    assert CurationV2.is_committed_curation(child)
    assert bool((CurationV2 & child).fetch1("merges_applied")) is True
    assert (CurationV2 & child).fetch1("curation_source") == "figpack"
    assert len(CurationV2.Unit & child) == len(unit_ids) - 1
    labels = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
    }
    assert (merged_id, "mua") in labels
    assert (merged_id, "noise") in labels


@pytest.mark.database
@pytest.mark.slow
def test_save_manual_curation_commit_merge_keeps_edited_contributor_labels(
    polymer_60s_sort,
):
    """A committed merge labels the merged unit from the edited labels.

    A FigPack import supplies labels on absorbed contributors; the merged
    unit must use them, not the parent's.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    merged_id = max(_unit_ids(polymer_60s_sort)) + 1
    clear_curations_for(polymer_60s_sort)
    root = CurationV2.create_initial_curation(
        polymer_60s_sort, labels={a: ["noise"], b: ["accept"]}
    )

    child = CurationV2.save_manual_curation(
        polymer_60s_sort,
        parent_curation_id=root["curation_id"],
        labels={a: ["accept"], b: ["accept"]},
        merge_groups=[[a, b]],
        merge_action="commit",
        curation_source="figpack",
    )

    merged = (CurationV2.UnitLabel & child & {"unit_id": merged_id}).fetch(
        "curation_label"
    )
    assert sorted(merged) == ["accept"]


@pytest.mark.database
def test_save_manual_curation_rejects_root_reuse(
    populated_sorting,
):
    """Root reuse would ignore manual edits, so require an explicit parent."""
    from spyglass.spikesorting.v2.curation import CurationV2

    ids = _unit_ids(populated_sorting)
    if not ids:
        pytest.skip("need >=1 sort unit")
    clear_curations_for(populated_sorting)
    CurationV2.create_initial_curation(populated_sorting)

    with pytest.raises(ValueError, match="reuse_existing=True"):
        CurationV2.save_manual_curation(
            populated_sorting,
            labels={ids[0]: ["mua"]},
            curation_source="figpack",
            reuse_existing=True,
        )


@pytest.mark.database
@pytest.mark.slow
def test_summarize_curation_fields(populated_sorting):
    """``summarize_curation`` reports the curation's fields from the parts.

    Also asserts a minimal curation key and a full pipeline run summary normalize
    to the same summary.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    ids = _unit_ids(populated_sorting)
    if not ids:
        pytest.skip("need >=1 sort unit")
    a = ids[0]
    clear_curations_for(populated_sorting)
    key = CurationV2.create_initial_curation(
        populated_sorting, labels={a: ["mua"]}, description="summary test"
    )

    summary = CurationV2.summarize_curation(key)
    assert summary["sorting_id"] == populated_sorting["sorting_id"]
    assert summary["curation_id"] == key["curation_id"]
    assert summary["n_units"] == len(CurationV2.Unit & key)
    assert summary["labels"].get(a) == ["mua"]
    contributor_groups = CurationV2.get_unit_contributor_groups(key)
    assert summary["unit_contributor_groups"] == contributor_groups
    assert summary["merge_groups"] == {
        unit_id: contributors
        for unit_id, contributors in contributor_groups.items()
        if len(contributors) > 1
    }
    assert summary["merges_applied"] is False
    assert summary["is_merge_preview"] is False
    assert summary["description"] == "summary test"
    expected_merge = (SpikeSortingOutput.CurationV2 & key).fetch1("merge_id")
    assert summary["merge_id"] == expected_merge

    # Extra non-PK keys are ignored.
    key_with_extras = {
        **key,
        "pipeline_preset": "irrelevant",
        "recording_id": "irrelevant",
        "artifact_detection_id": "irrelevant",
        "n_units": 999,
        "root_merge_id": "irrelevant",
    }
    assert CurationV2.summarize_curation(key_with_extras) == summary


@pytest.mark.database
def test_summarize_unregistered_merge_id_none(populated_sorting):
    """An unregistered curation summarizes with ``merge_id`` None (no raise)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(populated_sorting)
    key = CurationV2.create_initial_curation(populated_sorting)
    merge_id = (SpikeSortingOutput.CurationV2 & key).fetch1("merge_id")
    (SpikeSortingOutput & {"merge_id": merge_id}).super_delete(warn=False)

    assert CurationV2.summarize_curation(key)["merge_id"] is None


_PARTIAL_CURATION_KEYS = [{"curation_id": 0}, {"sorting_id": "s"}]


@pytest.mark.database
@pytest.mark.parametrize("partial_key", _PARTIAL_CURATION_KEYS)
def test_summarize_curation_requires_full_pk(dj_conn, partial_key):
    """summarize_curation rejects a key lacking sorting_id or curation_id.

    Raises before any DB access.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    with pytest.raises(ValueError, match="sorting_id"):
        CurationV2.summarize_curation(partial_key)


@pytest.mark.database
@pytest.mark.parametrize("partial_key", _PARTIAL_CURATION_KEYS)
def test_summarize_curation_pk_guard_runs_before_schema_access(
    dj_conn, monkeypatch, partial_key
):
    """The full-PK guard raises before importing ``SpikeSortingOutput``.

    The merge import activates its ``dj.schema``. A sentinel module that
    fails on any attribute access catches a guard that runs too late.
    """
    import sys

    from spyglass.spikesorting.v2.curation import CurationV2

    class _Boom:
        def __getattr__(self, name):
            raise AssertionError(
                f"summarize_curation accessed merge module .{name} before "
                "the full-PK guard"
            )

    monkeypatch.setitem(
        sys.modules, "spyglass.spikesorting.spikesorting_merge", _Boom()
    )
    with pytest.raises(ValueError, match="sorting_id"):
        CurationV2.summarize_curation(partial_key)


@pytest.mark.database
@pytest.mark.slow
def test_commit_merges_forwards_allow_custom_labels(planted_two_unit_sort):
    """``commit_merges`` forwards allow_custom_labels, so a child inheriting a
    custom (non-canonical) parent label does not fail the child insert.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import create_initial_curation
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        # A root carrying a CUSTOM label on unit 0 (needs the flag here too).
        root = create_initial_curation(
            sorting_key,
            labels={unit_ids[0]: ["my_custom"]},
            allow_custom_labels=True,
        )
        # The merged unit inherits unit 0's custom label.
        with pytest.raises(ValueError, match="not in CurationLabel"):
            root.commit_merges([[unit_ids[0], unit_ids[1]]])
        merged = root.commit_merges(
            [[unit_ids[0], unit_ids[1]]], allow_custom_labels=True
        )
        merged_labels = {
            r["curation_label"]
            for r in (CurationV2.UnitLabel & merged.as_key()).fetch(
                as_dict=True
            )
        }
        assert "my_custom" in merged_labels
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.database
@pytest.mark.slow
def test_insert_curation_dedupes_repeated_labels(planted_two_unit_sort):
    """A label repeated in the payload yields a single UnitLabel row, not a
    duplicate (unit_id, curation_label) primary-key insert error.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(
            sorting_key=sorting_key,
            labels={unit_ids[0]: ["mua", "mua", "noise"]},
        )
        rows = sorted(
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & root).fetch(as_dict=True)
        )
        assert rows == [(unit_ids[0], "mua"), (unit_ids[0], "noise")]
    finally:
        clear_curations_for(planted_two_unit_sort)
