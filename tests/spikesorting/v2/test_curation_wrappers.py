"""Tests for the friendly CurationV2 wrappers + summarize_curation.

The three write wrappers (create_initial_curation / propose_merge_curation /
create_merged_curation) are thin intent-first sugar over the expert
``insert_curation``; these tests pin that they pre-fill the right arguments,
inherit its validation (not re-implement it), and thread ``parent_curation_id``
so merges branch off an initial curation. ``summarize_curation`` is a pure read
accessor whose fields are checked against the underlying parts/registration.

All DB-tier; the single-unit checks reuse the shared package-scoped
``populated_sorting`` fixture, while the merge checks use a module-scoped sort
that yields several well-isolated units (``polymer_60s_sort``). Each test clears
curations first so it is order-independent. ``CurationV2`` (a ``@schema`` table)
is imported INSIDE each test, not at module top level, so pytest collection does
not open a DB connection before the conftest's MySQL container is ready.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import clear_curations_for

_TEAM = "curation_wrappers_team"
# The smoke sort yields a single unit -- too few to form a merge. The 60s
# polymer shank reliably yields several (a nightly-tier fixture, like the
# waveform tests use); the merge tests skip when it is absent (per-PR CI).
_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


@pytest.fixture(scope="module")
def polymer_60s_sort(dj_conn):
    """A MountainSort5 sort that yields several well-isolated single units.

    The merge wrappers operate on distinct single units (e.g. merging an
    oversplit cluster back together), so they need a sort with at least two
    units. The smoke sort yields only one; the 60s polymer shank reliably
    yields several. Module-scoped so the one 60s sort is shared across the
    merge tests; each clears curations first, so they stay order-independent.
    Ingested under a distinct session name to avoid colliding with other
    modules that use the same 60s polymer fixture.
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
        **inputs, pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06"
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
    # insert_curation is idempotent on a default root, so the expert call
    # returns the very row the wrapper created.
    expert_key = CurationV2.insert_curation(
        sorting_key=populated_sorting, parent_curation_id=-1
    )
    assert wrapper_key == expert_key
    assert bool((CurationV2 & wrapper_key).fetch1("merges_applied")) is False
    assert len(CurationV2.Unit & wrapper_key) >= 1


@pytest.mark.database
@pytest.mark.slow
def test_propose_merge_records_not_applies(polymer_60s_sort):
    """``propose_merge_curation`` records merges without applying them."""
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    clear_curations_for(polymer_60s_sort)
    key = CurationV2.propose_merge_curation(
        polymer_60s_sort, merge_groups=[[a, b]]
    )
    assert bool((CurationV2 & key).fetch1("merges_applied")) is False
    # Preview keeps every original unit (no contributors absorbed).
    units_after = {int(u) for u in (CurationV2.Unit & key).fetch("unit_id")}
    assert {a, b} <= units_after
    # The proposed merge is recorded as a >1-contributor group.
    groups = CurationV2.get_merge_groups(key)
    assert any(len(c) > 1 for c in groups.values())
    assert CurationV2.summarize_curation(key)["is_merge_preview"] is True


@pytest.mark.database
@pytest.mark.slow
def test_create_merged_curation_commits(polymer_60s_sort):
    """``create_merged_curation`` commits the merged unit set."""
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    n_sort_units = len(_unit_ids(polymer_60s_sort))
    clear_curations_for(polymer_60s_sort)
    key = CurationV2.create_merged_curation(
        polymer_60s_sort, merge_groups=[[a, b]]
    )
    assert bool((CurationV2 & key).fetch1("merges_applied")) is True
    assert CurationV2.summarize_curation(key)["is_merge_preview"] is False
    # Two contributors collapse into one merged unit: count drops by one.
    assert len(CurationV2.Unit & key) == n_sort_units - 1


@pytest.mark.database
@pytest.mark.slow
def test_create_merged_curation_reuses_matching_child(polymer_60s_sort):
    """``reuse_existing=True`` makes a manual-merge notebook cell rerunnable."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.utils import CurationLabel

    a, b = _two_unit_ids(polymer_60s_sort)
    all_unit_ids = _unit_ids(polymer_60s_sort)
    merged_unit_id = max(all_unit_ids) + 1
    clear_curations_for(polymer_60s_sort)
    root = CurationV2.create_initial_curation(polymer_60s_sort)
    kwargs = {
        "sorting_key": polymer_60s_sort,
        "parent_curation_id": root["curation_id"],
        "description": "manual burst-pair merge",
        "reuse_existing": True,
    }

    first = CurationV2.create_merged_curation(
        merge_groups=[[a, b]],
        labels={merged_unit_id: [CurationLabel.accept]},
        **kwargs,
    )
    child_restriction = CurationV2 & {
        "sorting_id": first["sorting_id"],
        "parent_curation_id": root["curation_id"],
        "description": "manual burst-pair merge",
        "merges_applied": True,
    }
    n_children = len(child_restriction)

    # Same scientific merge, different list order: reuse the existing child
    # instead of creating a new curation / merge_id on notebook re-run.
    # Enum and string labels are semantically equivalent at insert time, so
    # reuse_existing compares them through the same canonicalization.
    second = CurationV2.create_merged_curation(
        merge_groups=[[b, a]],
        labels={merged_unit_id: ["accept"]},
        **kwargs,
    )
    assert second == first
    assert len(child_restriction) == n_children


@pytest.mark.database
@pytest.mark.parametrize(
    "wrapper_name", ["propose_merge_curation", "create_merged_curation"]
)
def test_merge_wrappers_reject_root_reuse_existing(
    populated_sorting, wrapper_name
):
    """Merge-wrapper reuse must branch from an explicit parent curation."""
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(populated_sorting)
    CurationV2.create_initial_curation(populated_sorting)
    wrapper = getattr(CurationV2, wrapper_name)

    with pytest.raises(
        ValueError, match="reuse_existing=True.*parent_curation_id"
    ):
        wrapper(populated_sorting, merge_groups=[], reuse_existing=True)


@pytest.mark.database
def test_wrappers_inherit_singleton_rejection(populated_sorting):
    """A singleton merge group raises the SAME error ``insert_curation`` does.

    Proves the wrapper does not bypass or weaken the >=2-member validation.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    ids = _unit_ids(populated_sorting)
    if not ids:
        pytest.skip("need >=1 sort unit")
    a = ids[0]
    clear_curations_for(populated_sorting)

    with pytest.raises(ValueError) as wrap_exc:
        CurationV2.propose_merge_curation(populated_sorting, merge_groups=[[a]])
    with pytest.raises(ValueError) as expert_exc:
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            merge_groups=[[a]],
            apply_merge=False,
        )
    assert str(wrap_exc.value) == str(expert_exc.value)


@pytest.mark.database
def test_merge_wrappers_forward_args(populated_sorting):
    """The merge wrappers forward apply_merge and parent_curation_id.

    Pins the forwarding contract -- the only thing that distinguishes
    ``propose_merge_curation`` from ``create_merged_curation``, plus the
    DAG-branch threading -- on the always-present smoke fixture. An empty
    ``merge_groups`` (no merges) isolates the forwarding without needing >=2
    units, so this runs on every PR; the apply-vs-preview merge BEHAVIOR
    (which needs multiple units) is covered by the 60s-fixture tests.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    # apply_merge forwarding: propose -> not applied, create_merged -> applied.
    clear_curations_for(populated_sorting)
    preview = CurationV2.propose_merge_curation(
        populated_sorting, merge_groups=[]
    )
    assert bool((CurationV2 & preview).fetch1("merges_applied")) is False

    clear_curations_for(populated_sorting)
    applied = CurationV2.create_merged_curation(
        populated_sorting, merge_groups=[]
    )
    assert bool((CurationV2 & applied).fetch1("merges_applied")) is True

    # parent_curation_id forwarding: a proposal branches off an initial root.
    clear_curations_for(populated_sorting)
    root = CurationV2.create_initial_curation(populated_sorting)
    child = CurationV2.propose_merge_curation(
        populated_sorting,
        merge_groups=[],
        parent_curation_id=root["curation_id"],
    )
    assert child["curation_id"] != root["curation_id"]
    assert (
        int((CurationV2 & child).fetch1("parent_curation_id"))
        == root["curation_id"]
    )


@pytest.mark.database
@pytest.mark.slow
def test_propose_merge_off_existing_initial_curation(polymer_60s_sort):
    """A merge proposal branches off an initial curation (DAG child)."""
    from spyglass.spikesorting.v2.curation import CurationV2

    a, b = _two_unit_ids(polymer_60s_sort)
    clear_curations_for(polymer_60s_sort)
    root = CurationV2.create_initial_curation(polymer_60s_sort)
    child = CurationV2.propose_merge_curation(
        polymer_60s_sort,
        merge_groups=[[a, b]],
        parent_curation_id=root["curation_id"],
    )
    assert child["curation_id"] != root["curation_id"]
    assert (
        int((CurationV2 & child).fetch1("parent_curation_id"))
        == root["curation_id"]
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
    assert summary["merge_groups"] == CurationV2.get_merge_groups(key)
    assert summary["merges_applied"] is False
    assert summary["is_merge_preview"] is False
    assert summary["description"] == "summary test"
    expected_merge = (SpikeSortingOutput.CurationV2 & key).fetch1("merge_id")
    assert summary["merge_id"] == expected_merge

    # A full run_v2_pipeline run summary (extra, non-PK keys) normalizes to the
    # same summary as the minimal curation key.
    run_summary_like = {
        **key,
        "pipeline_preset": "irrelevant",
        "recording_id": "irrelevant",
        "artifact_detection_id": "irrelevant",
        "n_units": 999,
        "merge_id": "irrelevant",
    }
    assert CurationV2.summarize_curation(run_summary_like) == summary


@pytest.mark.database
def test_summarize_unregistered_merge_id_none(populated_sorting):
    """An unregistered curation summarizes with ``merge_id`` None (no raise)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(populated_sorting)
    key = CurationV2.create_initial_curation(populated_sorting)
    merge_id = (SpikeSortingOutput.CurationV2 & key).fetch1("merge_id")
    # Drop the merge registration but leave the CurationV2 row in place.
    (SpikeSortingOutput & {"merge_id": merge_id}).super_delete(warn=False)

    assert CurationV2.summarize_curation(key)["merge_id"] is None


@pytest.mark.database
def test_summarize_curation_requires_full_pk(dj_conn):
    """summarize_curation rejects a key lacking sorting_id or curation_id.

    The guard prevents a ``curation_id``-only key (not globally unique) from
    silently restricting to the wrong rows; it raises before any DB access.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    with pytest.raises(ValueError, match="sorting_id"):
        CurationV2.summarize_curation({"curation_id": 0})
