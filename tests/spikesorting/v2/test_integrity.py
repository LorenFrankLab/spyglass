"""Cross-table integrity gate for v2 spike-sorting tables.

Tests verify cross-table invariants and transactional atomicity that
the per-table tests in the ``single_session/`` suite do not
exercise as a focused gate:

- **Tri-part dispatch active**: every v2 ``AutoPopulate`` table (discovered
  from the package, minus a short documented exclusion set) uses DataJoint's
  tri-part ``make_fetch`` / ``make_compute`` / ``make_insert`` rather than a
  monolithic ``make``. The reason is to move the long-running compute step
  OUTSIDE the framework transaction so it does not hold row locks.
- **Selection FK consistency**: every ``SortingSelection`` master row has
  EXACTLY one source-part row (recording XOR concatenated), and an
  artifact-backed sort's ``artifact_detection_id`` maps back to exactly one
  ``RecordingArtifactSelection`` master. (The split artifact selection is
  structural single-source -- the recording source is a required master FK on
  ``RecordingArtifactSelection``, not an XOR source part.)
- **Merge-table v2 parts are correctly wired**:
  ``SpikeSortingOutput.CurationV2`` and
  ``SpikeSortingOutput.ConcatMemberCuration`` are the v2 routing entries;
  v1's ``CurationV1`` and v0's ``CuratedSpikeSorting`` keep their
  separate parts. ``source_class_dict["CurationV2"] = CurationV2``
  so ``get_recording`` / ``get_sorting`` / ``get_sort_group_info``
  / ``get_spike_times`` all dispatch correctly.
- **No FK orphans**: no ``CurationV2.Unit`` row without a parent
  ``CurationV2`` master, no ``Sorting.Unit`` row without a parent
  ``Sorting``, no ``SharedArtifactGroup.Member`` row without a
  parent ``SharedArtifactGroup``.

The selection-FK and no-orphan tests share the package-scoped
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


#: v2 ``AutoPopulate`` tables that intentionally keep a monolithic ``make``,
#: each with the reason it does no work worth moving out of the framework
#: transaction. Anything not listed here must be tri-part.
_MONOLITHIC_MAKE_EXCLUSIONS = {
    # DB reads plus a bounded pure-Python clique partition (``max_strict_nodes``
    # budget enforced); no SpikeInterface, NWB, or file I/O.
    "spyglass.spikesorting.v2.unit_matching.TrackedUnit",
}


def _autopopulate_tables_in(package) -> dict[str, type]:
    """Return the ``AutoPopulate`` tables declared under ``package``.

    Discovery imports every module in the package, subpackages included,
    instead of trusting a hand-written list, so a new ``dj.Computed`` /
    ``dj.Imported`` table cannot escape the gate. Only classes defined in the
    module being scanned count (re-exports are skipped). Tables are keyed by
    ``f"{obj.__module__}.{obj.__qualname__}"`` so two same-named classes in
    different modules stay apart; two distinct classes with one key (for
    example two classes built by one factory function) fail here instead of
    one silently replacing the other.
    """
    import importlib
    import pkgutil

    from datajoint.autopopulate import AutoPopulate

    tables: dict[str, type] = {}
    clashes: list[str] = []
    for info in pkgutil.walk_packages(
        package.__path__, prefix=f"{package.__name__}."
    ):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if not (
                inspect.isclass(obj)
                and issubclass(obj, AutoPopulate)
                and obj.__module__ == module.__name__
            ):
                continue
            key = f"{obj.__module__}.{obj.__qualname__}"
            if tables.setdefault(key, obj) is not obj:
                clashes.append(key)
    assert not clashes, (
        f"Distinct AutoPopulate tables share the discovery key(s) {clashes}; "
        "give each table its own module-level class."
    )
    return tables


def _v2_autopopulate_tables() -> dict[str, type]:
    """Return every ``AutoPopulate`` table declared in the v2 package."""
    import spyglass.spikesorting.v2 as v2

    return _autopopulate_tables_in(v2)


def test_tripart_dispatch_active_on_all_v2_computed_tables():
    """Every v2 ``AutoPopulate`` table routes through tri-part dispatch.

    DataJoint fires tri-part dispatch only when
    ``inspect.isgeneratorfunction(self.make)`` is True (the inherited
    generator-based ``make`` from ``AutoPopulate``, which exists only when the
    class defines ``make_fetch`` / ``make_compute`` / ``make_insert``); a
    subclass that overrides ``make`` with a regular function runs monolithic,
    with the whole ``make`` inside the populate transaction, and its tri-part
    methods become dead code. ``_parallel_make`` routes
    ``populate(processes>1)`` through Spyglass's non-daemon pool, which a
    ``make_compute`` running SpikeInterface ``n_jobs>1`` needs.

    The tables are discovered from the v2 package, not listed by hand. The
    only exceptions are :data:`_MONOLITHIC_MAKE_EXCLUSIONS`, and those must
    really be monolithic (a regular-function ``make``): an exclusion that was
    converted, or that no longer exists, fails here so the set stays honest.
    """
    tables = _v2_autopopulate_tables()

    stale = sorted(_MONOLITHIC_MAKE_EXCLUSIONS - set(tables))
    assert not stale, (
        f"Monolithic-make exclusions {stale} are not v2 AutoPopulate tables; "
        "remove them from _MONOLITHIC_MAKE_EXCLUSIONS."
    )
    # A discovery that silently found nothing would pass every loop below.
    assert {
        "spyglass.spikesorting.v2.recording.Recording",
        "spyglass.spikesorting.v2.sorting.Sorting",
        "spyglass.spikesorting.v2.metric_curation.CurationEvaluation",
    } <= set(tables)

    not_tripart = []
    for name, cls in sorted(tables.items()):
        if name in _MONOLITHIC_MAKE_EXCLUSIONS:
            assert not inspect.isgeneratorfunction(cls.make), (
                f"{name} is listed as a monolithic-make exclusion but its "
                "make is a generator (tri-part); remove it from "
                "_MONOLITHIC_MAKE_EXCLUSIONS."
            )
            continue
        missing = [
            method
            for method in ("make_fetch", "make_compute", "make_insert")
            if not callable(getattr(cls, method, None))
        ]
        if (
            missing
            or not inspect.isgeneratorfunction(cls.make)
            or getattr(cls, "_parallel_make", False) is not True
        ):
            not_tripart.append(
                f"{name} (generator make="
                f"{inspect.isgeneratorfunction(cls.make)}, missing={missing}, "
                f"_parallel_make={getattr(cls, '_parallel_make', None)!r})"
            )
    assert not not_tripart, (
        "v2 AutoPopulate tables not on DataJoint's tri-part dispatch with "
        "_parallel_make=True: "
        + "; ".join(not_tripart)
        + ". Split make into make_fetch / make_compute / make_insert (keeping "
        "the inherited generator make) or, for a table that does only bounded "
        "DB bookkeeping, add it to _MONOLITHIC_MAKE_EXCLUSIONS with a reason."
    )


#: v2 tri-part tables whose ``make_compute`` stages a file or folder that only
#: ``make_insert`` registers.
_STAGING_TABLES = {
    "spyglass.spikesorting.v2.recording.Recording",
    "spyglass.spikesorting.v2.session_group.ConcatenatedRecording",
    "spyglass.spikesorting.v2.sorting.Sorting",
    "spyglass.spikesorting.v2.motion.MotionCorrectedRecording",
    "spyglass.spikesorting.v2.metric_curation.CurationEvaluation",
    "spyglass.spikesorting.v2.concat_member_curation.ConcatMemberCuration",
    "spyglass.spikesorting.v2.figpack_curation.FigPackCuration",
    "spyglass.spikesorting.v2.unit_matching.UnitMatch",
}


def test_staging_tables_remove_outputs_of_failed_populates():
    """Every staging table cleans up after a populate that stops short.

    DataJoint re-runs ``make_fetch`` inside the insert transaction and can
    refuse the insert after ``make_compute`` has staged its output; only
    ``StagedOutputCleanupMixin`` removes that output. Its ``_populate1``
    must run around DataJoint's, so the mixin must precede the DataJoint
    bases.
    """
    from datajoint.autopopulate import AutoPopulate

    from spyglass.spikesorting.v2._staged_outputs import (
        StagedOutputCleanupMixin,
    )

    tables = _v2_autopopulate_tables()
    assert _STAGING_TABLES <= set(tables)
    wrong = [
        name
        for name in sorted(_STAGING_TABLES)
        if not issubclass(tables[name], StagedOutputCleanupMixin)
        or tables[name].__mro__.index(StagedOutputCleanupMixin)
        > tables[name].__mro__.index(AutoPopulate)
    ]
    assert (
        not wrong
    ), f"{wrong} do not use StagedOutputCleanupMixin ahead of AutoPopulate"


def _write_package(root, name: str, files: dict[str, str]):
    """Write a throwaway package under ``root`` and import it."""
    import importlib
    import textwrap

    for rel, body in {"__init__.py": "", **files}.items():
        path = root / name / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return importlib.import_module(name)


def test_table_discovery_walks_subpackages_and_keys_by_module_path(
    tmp_path, monkeypatch
):
    """Discovery reaches subpackage modules and keeps same-named tables apart.

    Two tables share the class name ``Table``: keyed by name, one would
    silently replace the other. A re-export and a same-module alias of an
    already-found table are not new tables.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    package = _write_package(
        tmp_path,
        "gate_walk_pkg",
        {
            "a.py": """
                from datajoint.autopopulate import AutoPopulate

                class Table(AutoPopulate):
                    pass

                Alias = Table
            """,
            "sub/__init__.py": "",
            "sub/b.py": """
                from datajoint.autopopulate import AutoPopulate

                from gate_walk_pkg.a import Table as Reexported

                class Table(AutoPopulate):
                    pass
            """,
        },
    )

    tables = _autopopulate_tables_in(package)

    assert set(tables) == {"gate_walk_pkg.a.Table", "gate_walk_pkg.sub.b.Table"}
    assert (
        tables["gate_walk_pkg.a.Table"]
        is not tables["gate_walk_pkg.sub.b.Table"]
    )


def test_table_discovery_rejects_two_tables_with_one_key(tmp_path, monkeypatch):
    """Two distinct tables with one ``module.qualname`` key fail discovery."""
    monkeypatch.syspath_prepend(str(tmp_path))
    package = _write_package(
        tmp_path,
        "gate_clash_pkg",
        {
            "factory.py": """
                from datajoint.autopopulate import AutoPopulate

                def make():
                    class Table(AutoPopulate):
                        pass

                    return Table

                First = make()
                Second = make()
            """,
        },
    )

    with pytest.raises(AssertionError, match=r"make\.<locals>\.Table"):
        _autopopulate_tables_in(package)


def test_v2_dispatch_classes_wired_into_merge_table():
    """Both v2 source classes and merge parts are registered.

    This is a registration / wiring check only -- behavioral
    coverage that the dispatch methods (``get_recording`` /
    ``get_sorting`` / etc.) actually return correct results on a
    v2 merge_id lives in ``test_downstream_consumers.py``.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
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
    assert source_class_dict["ConcatMemberCuration"] is ConcatMemberCuration
    assert hasattr(SpikeSortingOutput, "ConcatMemberCuration"), (
        "SpikeSortingOutput.ConcatMemberCuration part missing -- member "
        "outputs cannot be registered into the merge table."
    )


def test_selection_fk_chain_holds_for_artifact_and_sorting_selection(
    populated_sorting,
):
    """A ``SortingSelection`` master reachable from the populated fixture has
    EXACTLY one source-part row (recording XOR concatenated) and, when
    artifact-backed, its ``artifact_detection_id`` maps back to exactly one
    ``RecordingArtifactSelection`` master.

    The sort source is recorded in exactly one of the two XOR source parts --
    never zero (the insert_selection path failed atomicity) and never more than
    one (a logical contradiction). The artifact selection is structural
    single-source: the recording is a required FK on the
    ``RecordingArtifactSelection`` master, so on that side the FK chain is what
    remains to guard -- an artifact-backed sort's ``artifact_detection_id``
    points at exactly one master (never zero: insert_selection would have failed
    atomicity).
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactSelection,
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
    # A sort's source is recorded in exactly one of the two XOR source parts
    # (recording XOR concatenated): never zero, never more than one.
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
    # The recording-backed fixture's detection lives in the split
    # RecordingArtifactSelection master (recording is a required FK on it).
    art_id = SortingSelection.resolve_artifact_detection(sort_master_pk)
    art_master_rows = RecordingArtifactSelection & {
        "artifact_detection_id": art_id
    }
    assert len(art_master_rows) == 1, (
        f"populated_sorting points at artifact_detection_id={art_id!r} but no "
        "RecordingArtifactSelection master row exists for it; FK chain broken."
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
        counts = {
            row["sorting_id"]: row["source_part_count"] for row in flagged
        }

        assert counts.get(orphan) == 0, "zero-source master not flagged"
        assert (
            counts.get(ambiguous) == 2
        ), "dual-recording-source master not flagged"
        assert (
            single not in counts
        ), "clean single-source master wrongly flagged"
        assert sid_real not in counts, (
            "valid Recording+Artifact sorting wrongly flagged -- the artifact "
            "part must be excluded from the recording-source count"
        )
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
