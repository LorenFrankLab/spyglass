"""Cross-table integrity gate for v2 spike-sorting tables.

Tests verify cross-table invariants and transactional atomicity that
the per-table tests in the ``single_session/`` suite do not
exercise as a focused gate:

- **Tri-part dispatch active**: ``Recording`` / ``ArtifactDetection``
  / ``Sorting`` / ``UnitMatch`` use DataJoint's tri-part ``make_fetch`` /
  ``make_compute`` / ``make_insert`` rather than a monolithic
  ``make``. The reason is to move the long-running compute step
  OUTSIDE the framework transaction so it does not hold row locks.
- **Source-part FK consistency**: every ``ArtifactDetectionSelection`` and
  ``SortingSelection`` master row has EXACTLY one source-part row (the
  source of the selection is recorded in exactly one source-part table,
  never zero and never more than one).
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

The source-part and no-orphan tests share the package-scoped
``populated_sorting`` fixture from ``conftest.py`` so they have at
least one master row to exercise; without the fixture an empty test DB
would let the iteration loops pass vacuously. The fixture is resolved
from conftest rather than imported from a sibling test module so a CI
shard split that collects this module alone still gets the populated
state (a cross-module import would silently leave the loops iterating
over an empty DB).
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.usefixtures("dj_conn")


def test_tripart_dispatch_active_on_all_v2_computed_tables():
    """``Recording`` / ``ArtifactDetection`` / ``Sorting`` / ``UnitMatch``
    route through DataJoint's tri-part dispatch.

    DataJoint fires tri-part dispatch only when
    ``inspect.isgeneratorfunction(self.make)`` is True (the
    inherited generator-based ``make`` from ``AutoPopulate``); if
    a subclass overrides ``make`` with a regular function,
    DataJoint falls back to monolithic and the tri-part methods
    become dead code. Without this gate a refactor that silently
    re-introduces a monolithic ``make`` would turn off tri-part
    dispatch (long-transaction avoidance + parallel-populate).

    Every heavy v2 Computed table is included -- anything that opens an NWB,
    loads/hashes an analyzer, or runs SI compute must keep that work outside the
    framework transaction: the sort/recording/artifact stages, the cross-session
    matcher, the concat cache, the recompute QC tables, the curation-evaluation
    metrics, the drift estimate, AND the ``*Versions`` inventory tables (which
    open the NWB / load + hash the analyzer -- not "pure bookkeeping" as once
    assumed). ``TrackedUnit`` is the only heavy-ish table intentionally excluded:
    it does DB reads + a bounded pure-Python clique partition (no SI/NWB I/O), so
    a monolithic make is acceptable there.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactVersions,
        SortingAnalyzerRecompute,
        SortingAnalyzerVersions,
    )
    from spyglass.spikesorting.v2.recording import DriftEstimate, Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_matching import UnitMatch

    for cls in (
        Recording,
        ArtifactDetection,
        Sorting,
        UnitMatch,
        ConcatenatedRecording,
        RecordingArtifactRecompute,
        SortingAnalyzerRecompute,
        RecordingArtifactVersions,
        SortingAnalyzerVersions,
        CurationEvaluation,
        DriftEstimate,
    ):
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
    ``CurationV2`` class and the ``SpikeSortingOutput.CurationV2``
    part exists.

    This is a registration / wiring check only -- behavioral
    coverage that the dispatch methods (``get_recording`` /
    ``get_sorting`` / etc.) actually return correct results on a
    v2 merge_id lives in ``test_downstream_consumers.py``.
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
    assert hasattr(SpikeSortingOutput, "CurationV2"), (
        "SpikeSortingOutput.CurationV2 part missing -- v2 entries "
        "cannot be registered into the merge table."
    )


def test_source_part_pattern_holds_for_artifact_and_sorting_selection(
    populated_sorting,
):
    """Every ``ArtifactDetectionSelection`` / ``SortingSelection`` master
    reachable from the populated fixture has EXACTLY one source-
    part row.

    Each selection master records its source in exactly one source-part
    table -- never zero, never more than one. The
    iteration is scoped to masters reachable from ``populated_
    sorting`` so other tests that intentionally inject orphans
    (e.g. ``test_prune_orphaned_selections_finds_master_without_part``
    exercising the prune helper) do not register as integrity
    violations here. A master with zero parts in this scoped
    iteration is still a real bug (the insert_selection path
    failed atomicity); a master with multiple parts is always a
    logical contradiction.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # Sorting master reachable from the populated_sorting PK.
    sort_master_pk = populated_sorting
    sort_master_rows = SortingSelection & sort_master_pk
    assert len(sort_master_rows) == 1, (
        f"populated_sorting fixture did not produce a single "
        f"SortingSelection master row matching {sort_master_pk}; "
        "fixture setup is broken."
    )
    sid = sort_master_pk["sorting_id"]
    rec_parts = SortingSelection.RecordingSource & {"sorting_id": sid}
    concat_parts = SortingSelection.ConcatenatedRecordingSource & {
        "sorting_id": sid
    }
    total = len(rec_parts) + len(concat_parts)
    assert total == 1, (
        f"SortingSelection sorting_id={sid!r} has {total} source-"
        f"part rows (expected 1). Source-part invariant violated."
    )

    # Artifact master reachable via SortingSelection -> artifact_detection_id.
    # The artifact pass is recorded on the zero-or-one ArtifactDetectionSource
    # part now (the master no longer carries a nullable artifact_detection_id FK).
    art_id = SortingSelection.resolve_artifact_detection(sort_master_pk)
    art_master_rows = ArtifactDetectionSelection & {
        "artifact_detection_id": art_id
    }
    assert len(art_master_rows) == 1, (
        f"populated_sorting points at artifact_detection_id={art_id!r} but no "
        "ArtifactDetectionSelection master row exists for it; FK chain broken."
    )
    rec_parts = ArtifactDetectionSelection.RecordingSource & {
        "artifact_detection_id": art_id
    }
    shared_parts = ArtifactDetectionSelection.SharedGroupSource & {
        "artifact_detection_id": art_id
    }
    total = len(rec_parts) + len(shared_parts)
    assert total == 1, (
        f"ArtifactDetectionSelection artifact_detection_id={art_id!r} has {total} source-"
        f"part rows (expected 1: {len(rec_parts)} RecordingSource + "
        f"{len(shared_parts)} SharedGroupSource). "
        "Source-part invariant violated."
    )


def test_no_orphan_part_rows_in_v2_tables(populated_sorting):
    """No ``CurationV2.Unit`` / ``Sorting.Unit`` /
    ``SharedArtifactGroup.Member`` row without a parent master.

    DataJoint's FK cascade should prevent this; the test is a
    cheap belt-and-suspenders guard against an accidental
    ``super_delete`` ordering bug that leaks orphans.

    The fixture guarantees at least one ``Sorting.Unit`` row;
    ``CurationV2.Unit`` and ``SharedArtifactGroup.Member`` may
    legitimately be empty in this fixture state, in which case the
    no-orphan check is trivially satisfied for those tables. The
    ``Sorting.Unit`` non-empty guard is the load-bearing one.
    """
    _ = populated_sorting  # ensure population
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    # Sorting.Unit -> Sorting master. This relation is guaranteed
    # non-empty by the populated_sorting fixture; the assertion
    # protects against vacuous pass if a regression empties it.
    sorting_unit_keys = (Sorting.Unit).fetch("KEY", as_dict=True)
    assert len(sorting_unit_keys) >= 1, (
        "Sorting.Unit is empty; orphan check would pass vacuously. "
        "Fixture failed to populate."
    )
    sorting_master_pks = {r["sorting_id"] for r in sorting_unit_keys}
    for sid in sorting_master_pks:
        assert (
            len(Sorting & {"sorting_id": sid}) == 1
        ), f"Sorting.Unit references missing master sorting_id={sid!r}"

    # CurationV2.Unit -> CurationV2 master. May be empty if no
    # curation has been inserted; the FK check still runs on any
    # rows that exist.
    unit_keys = (CurationV2.Unit).fetch("KEY", as_dict=True)
    master_pks = {(r["sorting_id"], r["curation_id"]) for r in unit_keys}
    for sid, cid in master_pks:
        assert len(CurationV2 & {"sorting_id": sid, "curation_id": cid}) == 1, (
            f"CurationV2.Unit references missing master "
            f"sorting_id={sid!r}, curation_id={cid!r}"
        )

    # SharedArtifactGroup.Member -> SharedArtifactGroup master.
    # Same: may be empty in fixtures that don't exercise the
    # shared-artifact path.
    member_keys = (SharedArtifactGroup.Member).fetch("KEY", as_dict=True)
    group_master_pks = {r["shared_artifact_group_name"] for r in member_keys}
    for name in group_master_pks:
        assert (
            len(SharedArtifactGroup & {"shared_artifact_group_name": name}) == 1
        ), (
            f"SharedArtifactGroup.Member references missing master "
            f"shared_artifact_group_name={name!r}"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_audit_source_part_integrity(populated_sorting):
    """``audit_source_part_integrity`` flags masters whose recording-source
    part count is not exactly one (0 = orphan, >=2 = ambiguous), counting ONLY
    the XOR recording-source pair so a valid artifact-bearing sorting is not
    falsely flagged.

    Four scenarios:
    * the real populated sorting (``RecordingSource`` + ``ArtifactDetectionSource``)
      -> NOT flagged (the artifact part is excluded from the count);
    * a clean single-recording-source master -> NOT flagged;
    * an orphan master (no source part) -> flagged with count 0;
    * an ambiguous master (``RecordingSource`` + ``ConcatenatedRecordingSource``)
      -> flagged with count 2.
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import audit_source_part_integrity

    recording_parts = [
        SortingSelection.RecordingSource,
        SortingSelection.ConcatenatedRecordingSource,
    ]

    sid_real = populated_sorting["sorting_id"]
    base = (SortingSelection & populated_sorting).fetch1()
    # The real sorting carries an artifact pass: RecordingSource(1) +
    # ArtifactDetectionSource(1). The audit must NOT flag it -- the artifact part
    # is excluded, leaving the recording-source count at 1.
    assert (
        len(SortingSelection.ArtifactDetectionSource & {"sorting_id": sid_real})
        == 1
    ), "fixture sorting expected to carry an artifact-detection pass"
    rec_id = (
        SortingSelection.RecordingSource & {"sorting_id": sid_real}
    ).fetch1("recording_id")
    sorter, spn = base["sorter"], base["sorter_params_name"]

    orphan, single, ambiguous = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    conn = SortingSelection.connection
    try:
        # Orphan: master with no source part.
        SortingSelection().insert1(
            {"sorting_id": orphan, "sorter": sorter, "sorter_params_name": spn},
            allow_direct_insert=True,
        )
        # Clean single recording source.
        SortingSelection().insert1(
            {"sorting_id": single, "sorter": sorter, "sorter_params_name": spn},
            allow_direct_insert=True,
        )
        SortingSelection.RecordingSource.insert1(
            {"sorting_id": single, "recording_id": rec_id}
        )
        # Ambiguous: BOTH a recording source AND a concat source. The concat FK
        # target need not exist for the count to be ambiguous, so inject the
        # second source under FOREIGN_KEY_CHECKS=0 to avoid building a real concat.
        SortingSelection().insert1(
            {
                "sorting_id": ambiguous,
                "sorter": sorter,
                "sorter_params_name": spn,
            },
            allow_direct_insert=True,
        )
        SortingSelection.RecordingSource.insert1(
            {"sorting_id": ambiguous, "recording_id": rec_id}
        )
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            SortingSelection.ConcatenatedRecordingSource.insert1(
                {"sorting_id": ambiguous, "concat_recording_id": uuid.uuid4()}
            )
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")

        flagged = audit_source_part_integrity(SortingSelection, recording_parts)
        counts = {row["sorting_id"]: row["source_part_count"] for row in flagged}

        assert counts.get(orphan) == 0, "zero-source master not flagged"
        assert counts.get(ambiguous) == 2, "dual-recording-source master not flagged"
        assert single not in counts, "clean single-source master wrongly flagged"
        assert sid_real not in counts, (
            "valid Recording+Artifact sorting wrongly flagged -- the artifact "
            "part must be excluded from the recording-source count"
        )

        # The artifact-selection side uses the [RecordingSource, SharedGroupSource]
        # XOR pair; the real artifact selection has exactly one and is not flagged.
        from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection

        art_id = SortingSelection.resolve_artifact_detection(populated_sorting)
        art_flagged = audit_source_part_integrity(
            ArtifactDetectionSelection,
            [
                ArtifactDetectionSelection.RecordingSource,
                ArtifactDetectionSelection.SharedGroupSource,
            ],
        )
        assert art_id not in {row["artifact_detection_id"] for row in art_flagged}
    finally:
        for sid in (orphan, single, ambiguous):
            (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_v2_lookup_tables_validate_via_pydantic():
    """``PreprocessingParameters`` / ``ArtifactDetectionParameters``
    / ``SorterParameters`` all validate ``params`` through Pydantic
    on ``insert1``. The ``_validate_params`` +
    ``_assert_schema_version_matches`` wiring is the contract; a
    refactor that silently dropped either check would let bogus
    params blobs land in the Lookup tables and crash only at
    populate time.

    The ``match=`` strings pin the assertion to the validator's
    own message (the offending field name), so an unrelated
    ``DataJointError`` (e.g. a PK collision) cannot satisfy the
    raises-context.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    bogus_pydantic_row = {
        "preprocessing_params_name": "v2_integrity_test_bogus",
        "params": {"bandpass_filter": {"freq_min": "not_a_float"}},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError), match="freq_min"):
        PreprocessingParameters.insert1(bogus_pydantic_row)

    bogus_artifact_row = {
        "artifact_detection_params_name": "v2_integrity_test_bogus",
        "params": {"amplitude_threshold_uv": "not_a_float"},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError), match="amplitude_threshold_uv"):
        ArtifactDetectionParameters.insert1(bogus_artifact_row)

    bogus_sorter_row = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "v2_integrity_test_bogus",
        "params": {"detect_threshold": "not_a_float"},
        "params_schema_version": 2,
        "job_kwargs": None,
    }
    with pytest.raises((ValueError, TypeError), match="detect_threshold"):
        SorterParameters.insert1(bogus_sorter_row)
