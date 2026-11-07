"""Cross-table integrity tests for v2 spike-sorting tables.

- Every v2 ``AutoPopulate`` table uses tri-part ``make`` (compute runs outside
  the transaction), except a documented exclusion set.
- Each ``SortingSelection`` has exactly one source part, and an artifact-backed
  sort maps to exactly one ``RecordingArtifactSelection``.
- The v2 merge parts and ``source_class_dict`` entries are registered.
- No part rows without a master.

The DB tests use the ``populated_sorting`` fixture so their loops are not
vacuous.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.usefixtures("dj_conn")


#: v2 ``AutoPopulate`` tables that keep a monolithic ``make``, each with why.
#: Anything not listed here must be tri-part.
_MONOLITHIC_MAKE_EXCLUSIONS = {
    # DB reads plus a bounded pure-Python clique partition (``max_strict_nodes``
    # budget enforced); no SpikeInterface, NWB, or file I/O.
    "spyglass.spikesorting.v2.unit_matching.TrackedUnit",
}


def _autopopulate_tables_in(package) -> dict[str, type]:
    """Return the ``AutoPopulate`` tables declared under ``package``.

    Imports every module, subpackages included; re-exports are skipped.
    Tables are keyed by ``module.qualname``; two distinct classes with one key
    raise.
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

    - DataJoint uses tri-part only when ``make`` is a generator function; a
      regular ``make`` override runs monolithic.
    - ``_parallel_make`` is required so ``populate(processes>1)`` uses the
      non-daemon pool that SI ``n_jobs>1`` needs.
    - Each :data:`_MONOLITHIC_MAKE_EXCLUSIONS` entry must exist and be
      monolithic.
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
    ``PopulateMixin``'s cleanup removes that output, and only for a table
    whose ``_staged_outputs`` hook ``StagedOutputCleanupMixin`` supplies.
    The mixin must precede ``PopulateMixin``, whose hook is ``None``.
    """
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        StagedOutputCleanupMixin,
    )
    from spyglass.utils.mixins.populate import PopulateMixin

    tables = _v2_autopopulate_tables()
    assert _STAGING_TABLES <= set(tables)
    wrong = [
        name
        for name in sorted(_STAGING_TABLES)
        if not issubclass(tables[name], StagedOutputCleanupMixin)
        or tables[name].__mro__.index(StagedOutputCleanupMixin)
        > tables[name].__mro__.index(PopulateMixin)
    ]
    assert (
        not wrong
    ), f"{wrong} do not use StagedOutputCleanupMixin ahead of PopulateMixin"

    # Beyond the list: the mixin and a carrier that names its staged outputs
    # must go together on every tri-part table.
    carriers = {
        name: _compute_carrier(cls)
        for name, cls in tables.items()
        if name not in _MONOLITHIC_MAKE_EXCLUSIONS
    }
    unresolved = sorted(name for name, c in carriers.items() if c is None)
    assert (
        not unresolved
    ), f"{unresolved}: annotate make_compute's return with its carrier class"
    mismatched = sorted(
        name
        for name, carrier in carriers.items()
        if issubclass(tables[name], StagedOutputCleanupMixin)
        != callable(getattr(carrier, "staged_outputs", None))
    )
    assert not mismatched, (
        f"{mismatched}: a table uses StagedOutputCleanupMixin exactly when "
        "its make_compute carrier defines staged_outputs()"
    )


def _compute_carrier(cls) -> type | None:
    """Return the class ``cls.make_compute`` is annotated to return."""
    import sys

    annotation = inspect.signature(cls.make_compute).return_annotation
    if isinstance(annotation, str):
        annotation = vars(sys.modules[cls.__module__]).get(annotation)
    return annotation if inspect.isclass(annotation) else None


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

    Dispatch behavior is tested in ``test_downstream_consumers.py``.
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
    """A ``SortingSelection`` has exactly one source part (recording XOR
    concatenated) and, when artifact-backed, its ``artifact_detection_id`` maps
    to exactly one ``RecordingArtifactSelection``.
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

    Only ``Sorting.Unit`` is guaranteed non-empty by the fixture.
    """
    _ = populated_sorting  # ensure population
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    # Sorting.Unit -> Sorting master; must be non-empty.
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

    # CurationV2.Unit -> CurationV2 master; may be empty.
    unit_keys = (CurationV2.Unit).fetch("KEY", as_dict=True)
    master_pks = {(r["sorting_id"], r["curation_id"]) for r in unit_keys}
    for sid, cid in master_pks:
        assert len(CurationV2 & {"sorting_id": sid, "curation_id": cid}) == 1, (
            f"CurationV2.Unit references missing master "
            f"sorting_id={sid!r}, curation_id={cid!r}"
        )

    # SharedArtifactGroup.Member -> SharedArtifactGroup master; may be empty.
    member_keys = (SharedArtifactGroup.Member).fetch("KEY", as_dict=True)
    group_master_pks = {r["shared_artifact_group_name"] for r in member_keys}
    for name in group_master_pks:
        assert (
            len(SharedArtifactGroup & {"shared_artifact_group_name": name}) == 1
        ), (
            f"SharedArtifactGroup.Member references missing master "
            f"shared_artifact_group_name={name!r}"
        )


def test_v2_lookup_tables_validate_via_pydantic():
    """``PreprocessingParameters`` / ``ArtifactDetectionParameters`` /
    ``SorterParameters`` validate ``params`` through Pydantic on ``insert1``.

    ``match=`` names the offending field so an unrelated ``DataJointError``
    cannot satisfy the check.
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
