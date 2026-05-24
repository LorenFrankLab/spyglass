"""Integrity tests for v2 spike-sorting tables.

Phase 1 plan-line 182 names this file in the suite-runner contract
(``pytest tests/spikesorting/v2/test_integrity.py -q``); this is
its canonical home. Tests verify cross-table invariants and
transactional atomicity that the per-table tests in
``test_single_session_pipeline.py`` do not exercise as a focused
gate:

- **Tri-part dispatch active**: ``Recording`` / ``ArtifactDetection``
  / ``Sorting`` use DataJoint's tri-part ``make_fetch`` /
  ``make_compute`` / ``make_insert`` rather than a monolithic
  ``make``. Phase 1b's primary motivation was moving the
  long-running compute step OUTSIDE the framework transaction so
  it does not hold row locks.
- **Source-part FK consistency**: every ``ArtifactSelection`` and
  ``SortingSelection`` master row has EXACTLY one source-part row
  (the Layer-1 invariant from shared-contracts.md "Source Part
  Pattern").
- **Merge-table CurationV2 part is correctly wired**:
  ``SpikeSortingOutput.CurationV2`` is the only v2 routing entry;
  v1's ``CurationV1`` and v0's ``CuratedSpikeSorting`` keep their
  separate parts. ``source_class_dict["CurationV2"] = CurationV2``
  so ``get_recording`` / ``get_sorting`` / ``get_sort_group_info``
  / ``get_spike_times`` all dispatch correctly.
- **No FK orphans**: no ``CurationV2.Unit`` row without a parent
  ``CurationV2`` master, no ``Sorting.Unit`` row without a parent
  ``Sorting``, no ``SharedArtifactGroup.Member`` row without a
  parent ``SharedArtifactGroup``.

These are static / AST / DB-tier checks; no populate is needed
for most of them, so the file runs in seconds.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.usefixtures("dj_conn")


def test_tripart_dispatch_active_on_all_v2_computed_tables():
    """``Recording`` / ``ArtifactDetection`` / ``Sorting`` route
    through DataJoint's tri-part dispatch.

    DataJoint fires tri-part dispatch only when
    ``inspect.isgeneratorfunction(self.make)`` is True (the
    inherited generator-based ``make`` from ``AutoPopulate``); if
    a subclass overrides ``make`` with a regular function,
    DataJoint falls back to monolithic and the tri-part methods
    become dead code. Without this gate a refactor that silently
    re-introduces a monolithic ``make`` would silently turn off
    tri-part for every Phase 1b motivation (long-transaction
    avoidance, parallel-populate).
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    for cls in (Recording, ArtifactDetection, Sorting):
        assert inspect.isgeneratorfunction(cls.make), (
            f"{cls.__name__}.make is not a generator -- DataJoint's "
            "tri-part dispatch fires only on generator make. A "
            "silent refactor to a monolithic make would turn off "
            "long-transaction avoidance + parallel populate."
        )
        for method in ("make_fetch", "make_compute", "make_insert"):
            assert hasattr(cls, method), (
                f"{cls.__name__} missing {method!r}; tri-part contract "
                "broken."
            )
        assert cls._parallel_make is True, (
            f"{cls.__name__}._parallel_make is not True; the "
            "non-daemon parallel-populate flag from Spyglass's "
            "PopulateMixin is off."
        )


def test_v2_dispatch_classes_wired_into_merge_table():
    """``source_class_dict["CurationV2"]`` resolves to the v2
    ``CurationV2`` class (not v1's), so the merge dispatcher's
    ``get_recording`` / ``get_sorting`` / etc. routes correctly.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )
    from spyglass.spikesorting.v2.curation import CurationV2 as V2Curation

    assert "CurationV2" in source_class_dict
    assert source_class_dict["CurationV2"] is V2Curation, (
        "source_class_dict['CurationV2'] does not resolve to the v2 "
        "CurationV2 class; merge-dispatch routing is broken."
    )
    # The merge-table part exists with the canonical name + FK.
    assert hasattr(SpikeSortingOutput, "CurationV2"), (
        "SpikeSortingOutput.CurationV2 part missing -- v2 entries "
        "cannot be registered into the merge table."
    )


def test_source_part_pattern_holds_for_artifact_and_sorting_selection():
    """Every ``ArtifactSelection`` master has exactly one source-
    part row; every ``SortingSelection`` master has exactly one.

    Layer-1 invariant from shared-contracts.md "Source Part
    Pattern". A master with zero parts is an orphan (cascade
    didn't fire); a master with multiple parts is a logical
    contradiction (a selection cannot be both recording-source
    and shared-group-source). Both are integrity bugs.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    art_masters = ArtifactSelection.fetch("artifact_id")
    for aid in art_masters:
        rec_parts = (
            ArtifactSelection.RecordingSource & {"artifact_id": aid}
        )
        shared_parts = (
            ArtifactSelection.SharedArtifactGroupSource
            & {"artifact_id": aid}
        )
        total = len(rec_parts) + len(shared_parts)
        assert total == 1, (
            f"ArtifactSelection artifact_id={aid!r} has {total} source-"
            f"part rows (expected 1: {len(rec_parts)} RecordingSource + "
            f"{len(shared_parts)} SharedArtifactGroupSource). Layer-1 "
            "source-part invariant violated."
        )

    sort_masters = SortingSelection.fetch("sorting_id")
    for sid in sort_masters:
        rec_parts = (
            SortingSelection.RecordingSource & {"sorting_id": sid}
        )
        concat_parts = (
            SortingSelection.ConcatenatedRecordingSource
            & {"sorting_id": sid}
        )
        total = len(rec_parts) + len(concat_parts)
        assert total == 1, (
            f"SortingSelection sorting_id={sid!r} has {total} source-"
            f"part rows (expected 1). Layer-1 source-part invariant "
            "violated."
        )


def test_no_orphan_part_rows_in_v2_tables():
    """No ``CurationV2.Unit`` / ``Sorting.Unit`` /
    ``SharedArtifactGroup.Member`` row without a parent master.

    DataJoint's FK cascade should prevent this; the test is a
    cheap belt-and-suspenders guard against an accidental
    ``super_delete`` ordering bug that leaks orphans.
    """
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    # CurationV2.Unit -> CurationV2 master
    unit_keys = (CurationV2.Unit).fetch("KEY", as_dict=True)
    master_pks = {(r["sorting_id"], r["curation_id"]) for r in unit_keys}
    for sid, cid in master_pks:
        assert len(
            CurationV2 & {"sorting_id": sid, "curation_id": cid}
        ) == 1, (
            f"CurationV2.Unit references missing master "
            f"sorting_id={sid!r}, curation_id={cid!r}"
        )

    # Sorting.Unit -> Sorting master
    sorting_unit_keys = (Sorting.Unit).fetch("KEY", as_dict=True)
    sorting_master_pks = {r["sorting_id"] for r in sorting_unit_keys}
    for sid in sorting_master_pks:
        assert len(Sorting & {"sorting_id": sid}) == 1, (
            f"Sorting.Unit references missing master sorting_id={sid!r}"
        )

    # SharedArtifactGroup.Member -> SharedArtifactGroup master
    member_keys = (SharedArtifactGroup.Member).fetch("KEY", as_dict=True)
    group_master_pks = {
        r["shared_artifact_group_name"] for r in member_keys
    }
    for name in group_master_pks:
        assert len(
            SharedArtifactGroup & {"shared_artifact_group_name": name}
        ) == 1, (
            f"SharedArtifactGroup.Member references missing master "
            f"shared_artifact_group_name={name!r}"
        )


def test_v2_lookup_tables_validate_via_pydantic():
    """``PreprocessingParameters`` / ``ArtifactDetectionParameters``
    / ``SorterParameters`` all validate ``params`` through Pydantic
    on ``insert1``. The Phase 1b ``_validate_params`` +
    ``_assert_schema_version_matches`` wiring is the
    contract; a refactor that silently dropped either check would
    let bogus params blobs land in the Lookup tables and crash
    only at populate time.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    bogus_pydantic_row = {
        "preproc_params_name": "v2_integrity_test_bogus",
        "params": {"bandpass_filter": {"freq_min": "not_a_float"}},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError, dj.errors.DataJointError)):
        PreprocessingParameters.insert1(bogus_pydantic_row)

    bogus_artifact_row = {
        "artifact_params_name": "v2_integrity_test_bogus",
        "params": {"amplitude_thresh_uV": "not_a_float"},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError, dj.errors.DataJointError)):
        ArtifactDetectionParameters.insert1(bogus_artifact_row)

    bogus_sorter_row = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "v2_integrity_test_bogus",
        "params": {"detect_threshold": "not_a_float"},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError, dj.errors.DataJointError)):
        SorterParameters.insert1(bogus_sorter_row)
