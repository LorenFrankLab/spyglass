"""Database contracts for typed, curation-scoped unit annotations."""

from __future__ import annotations

from unittest.mock import patch

import datajoint as dj
import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture
def versioned_annotation_sets(planted_two_unit_sort):
    """Two versions with different value types in the same curation."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        unit_ids = (CurationV2.Unit & root.as_key()).fetch(
            "unit_id", order_by="unit_id"
        )
        refs = []
        for version, value_type, values in (
            (np.int64(1), "float", [0.5, 1.5]),
            (2, "int", [1, 2]),
        ):
            definition = UnitAnnotationDefinition.insert_definition(
                "versioned_annotation_score", version, value_type
            )
            refs.append(
                CurationUnitAnnotationSet.from_dataframe(
                    root,
                    definition,
                    pd.DataFrame({"score": values}, index=unit_ids),
                    producer="annotation-regression-test",
                )
            )
        yield root, refs
    finally:
        clear_curations_for(sorting_key)


def _select_count(query, table):
    return sum(
        call.args[0].lstrip().upper().startswith("SELECT")
        and table.full_table_name in call.args[0]
        for call in query.call_args_list
    )


def test_annotation_versions_preserve_integer_identity(
    versioned_annotation_sets,
):
    """Factories and reference resolution reject versions that change identity."""
    from spyglass.spikesorting.v2.unit_annotation import (
        AnnotationDefinitionRef,
        AnnotationSetRef,
        UnitAnnotationDefinition,
    )

    _, (score, _) = versioned_annotation_sets
    key = score.as_key()
    definition = AnnotationDefinitionRef.from_key(key)
    numpy_key = {**key, "annotation_version": np.int64(1)}
    assert AnnotationDefinitionRef.from_key(numpy_key) == definition
    assert AnnotationSetRef.from_key(numpy_key) == score

    for version in (1.9, 1.0, True, "1"):
        invalid_key = {**key, "annotation_version": version}
        with pytest.raises(
            ValueError, match="annotation_version must be an integer"
        ):
            UnitAnnotationDefinition.insert_definition(
                score.annotation_name, version, "float"
            )
        with pytest.raises(
            ValueError, match="annotation_version must be an integer"
        ):
            AnnotationDefinitionRef.from_key(invalid_key)
        with pytest.raises(
            ValueError, match="annotation_version must be an integer"
        ):
            AnnotationSetRef.from_key(invalid_key)


def test_annotation_batch_reads_each_definition_once(versioned_annotation_sets):
    """Mixed-version batches validate types with one lookup per definition."""
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
    )

    _, refs = versioned_annotation_sets
    rows = [
        row
        for ref in refs
        for row in (CurationUnitAnnotationSet.Value & ref.as_key()).fetch(
            as_dict=True
        )
    ]
    connection = CurationUnitAnnotationSet.connection
    # Each call must resolve its definitions anew; the cache is batch-local.
    for _ in range(2):
        with patch.object(connection, "query", wraps=connection.query) as query:
            CurationUnitAnnotationSet.Value.insert(
                iter(rows), skip_duplicates=True, allow_direct_insert=True
            )
        assert _select_count(query, UnitAnnotationDefinition) == 2

    invalid = {**rows[-1], "value_float": 0.5}
    with pytest.raises(TypeError, match="only value_int may be populated"):
        CurationUnitAnnotationSet.Value.insert(
            [*rows, invalid], skip_duplicates=True, allow_direct_insert=True
        )


def test_annotation_reads_validate_once_and_reject_stale_refs(
    versioned_annotation_sets,
):
    """Composed reads avoid repeated checks but revalidate on every operation."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.exceptions import CurationNotFoundError
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
        read_unit_properties,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    root, (score, _) = versioned_annotation_sets
    connection = CurationUnitAnnotationSet.connection
    for supplied in (score, score.snapshot()):
        with patch.object(connection, "query", wraps=connection.query) as query:
            frame = CurationUnitAnnotationSet.to_dataframe(supplied)
        assert frame.iloc[:, 0].tolist() == [0.5, 1.5]
        assert _select_count(query, CurationV2) == 1
        assert _select_count(query, CurationUnitAnnotationSet) == 1
        assert _select_count(query, UnitAnnotationDefinition) == 1

        with patch.object(connection, "query", wraps=connection.query) as query:
            properties = read_unit_properties(
                root, evaluation=None, annotation_sets=[supplied]
            )
        assert properties[score.column_name].tolist() == [0.5, 1.5]
        assert _select_count(query, CurationUnitAnnotationSet) == 1
        assert _select_count(query, UnitAnnotationDefinition) == 1

    with pytest.raises(ValueError, match="repeats set_hash"):
        read_unit_properties(
            root, evaluation=None, annotation_sets=[score, score.snapshot()]
        )

    sorting_key = {"sorting_id": root.sorting_id}
    clear_curations_for(sorting_key)
    replacement = CurationRef.from_key(
        CurationV2.create_initial_curation(sorting_key)
    )
    assert replacement.curation_id == root.curation_id
    assert replacement.curation_uuid != root.curation_uuid
    with pytest.raises(CurationNotFoundError):
        score.to_dataframe()
    with pytest.raises(CurationNotFoundError):
        read_unit_properties(
            replacement, evaluation=None, annotation_sets=[score]
        )


def test_typed_annotation_sets_and_explicit_common_reader(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Sets are typed, immutable, exact-namespace, and explicitly selected."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_annotation import (
        AnnotationSetRef,
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
        read_unit_properties,
    )

    sorting_key = dict(planted_two_unit_sort)
    from spyglass.spikesorting.v2 import (
        AnnotationSetRef as RootAnnotationSetRef,
    )
    from spyglass.spikesorting.v2.pipeline import (
        AnnotationSetRef as PipelineAnnotationSetRef,
    )

    assert RootAnnotationSetRef is AnnotationSetRef
    assert PipelineAnnotationSetRef is AnnotationSetRef
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        unit_ids = sorted(
            map(int, (Sorting.Unit & sorting_key).fetch("unit_id"))
        )
        master_row = (CurationV2 & root.as_key()).fetch1()
        assert master_row["created_at"] is not None
        assert master_row["created_by"] == dj.config["database.user"]
        assert root.created_at == master_row["created_at"]
        assert root.created_by == dj.config["database.user"]

        score_definition = UnitAnnotationDefinition.insert_definition(
            "phase4_custom_score",
            1,
            "float",
            physical_unit="a.u.",
            description="Phase 4 integration score",
        )
        with pytest.raises(ValueError, match="immutable"):
            UnitAnnotationDefinition.insert_definition(
                "phase4_custom_score",
                1,
                "float",
                physical_unit="changed",
            )

        score_frame = pd.DataFrame(
            {"phase4_custom_score": [1.0000000000000002, 2.25]},
            index=pd.Index(unit_ids, name="unit_id"),
        )
        score = CurationUnitAnnotationSet.from_dataframe(
            root,
            score_definition,
            score_frame,
            producer="phase4-test",
            producer_version="1.0",
            producer_parameters={"window": 4},
        )
        repeated = CurationUnitAnnotationSet.from_dataframe(
            root,
            score_definition,
            score_frame,
            producer="phase4-test",
            producer_version="1.0",
            producer_parameters={"window": 4},
        )
        assert repeated == score
        assert len(CurationUnitAnnotationSet & root.as_key()) == 1
        stale_snapshot = score.snapshot()
        stale_snapshot["curation_uuid"] = "00000000-0000-0000-0000-000000000001"
        from spyglass.spikesorting.v2.exceptions import CurationNotFoundError

        with pytest.raises(CurationNotFoundError, match="stale curation_uuid"):
            AnnotationSetRef.from_key(stale_snapshot)
        round_trip = score.to_dataframe()
        assert round_trip.index.tolist() == unit_ids
        assert round_trip.dtypes.iloc[0] == "float64"
        assert round_trip.iloc[0, 0] == 1.0000000000000002
        assert (
            "double"
            in str(
                CurationUnitAnnotationSet.Value.heading.attributes[
                    "value_float"
                ].type
            ).lower()
        )

        changed_frame = score_frame.copy()
        changed_frame.iloc[0, 0] = 1.0
        with CurationUnitAnnotationSet.connection.transaction:
            with pytest.raises(RuntimeError, match="outside an open"):
                CurationUnitAnnotationSet.from_dataframe(
                    root,
                    score_definition,
                    changed_frame,
                    producer="phase4-test",
                    producer_version="1.0",
                    producer_parameters={"window": 4},
                )
        changed = CurationUnitAnnotationSet.from_dataframe(
            root,
            score_definition,
            changed_frame,
            producer="phase4-test",
            producer_version="1.0",
            producer_parameters={"window": 4},
        )
        assert changed.set_hash != score.set_hash
        assert len(CurationUnitAnnotationSet & root.as_key()) == 2

        for name, value_type, values, expected_dtype in (
            ("phase4_custom_count", "int", [1, 2], "int64"),
            ("phase4_custom_flag", "bool", [True, False], "bool"),
            ("phase4_custom_note", "text", ["", "stable"], "object"),
        ):
            definition = UnitAnnotationDefinition.insert_definition(
                name, 1, value_type
            )
            ref = CurationUnitAnnotationSet.from_dataframe(
                root,
                definition,
                pd.DataFrame(
                    {name: values}, index=pd.Index(unit_ids, name="unit_id")
                ),
                producer="phase4-test",
            )
            assert str(ref.to_dataframe().dtypes.iloc[0]) == expected_dtype

        special_definition = UnitAnnotationDefinition.insert_definition(
            "phase4_special_float", 1, "float"
        )
        special_frame = pd.DataFrame(
            {
                "phase4_special_float": pd.Series(
                    [None, float("nan")], dtype="object"
                ).to_numpy()
            },
            index=pd.Index(unit_ids, name="unit_id"),
        )
        special = CurationUnitAnnotationSet.from_dataframe(
            root,
            special_definition,
            special_frame,
            producer="phase4-test",
        )
        special_round_trip = special.to_dataframe()
        assert special_round_trip.iloc[0, 0] is None
        assert pd.isna(special_round_trip.iloc[1, 0])
        assert (
            CurationUnitAnnotationSet.from_dataframe(
                root,
                special_definition,
                special_round_trip,
                producer="phase4-test",
            )
            == special
        )

        with pytest.raises(TypeError, match="float annotations"):
            CurationUnitAnnotationSet.from_dataframe(
                root,
                score_definition,
                pd.DataFrame(
                    {"phase4_custom_score": ["bad", "type"]},
                    index=pd.Index(unit_ids, name="unit_id"),
                ),
            )
        with pytest.raises(ValueError, match="outside the exact curation"):
            CurationUnitAnnotationSet.from_dataframe(
                root,
                score_definition,
                pd.DataFrame(
                    {"phase4_custom_score": [0.5]},
                    index=pd.Index([max(unit_ids) + 100], name="unit_id"),
                ),
            )

        master = (CurationUnitAnnotationSet & score.as_key()).fetch1()
        master["producer"] = "mutated"
        with pytest.raises(dj.errors.DataJointError, match="In-place update1"):
            CurationUnitAnnotationSet.update1(master)

        with pytest.raises(dj.errors.IntegrityError):
            CurationUnitAnnotationSet.Value.insert1(
                {
                    **score.as_key(),
                    "unit_id": max(unit_ids) + 100,
                    "value_float": 0.5,
                },
                allow_direct_insert=True,
            )

        merge_id_before = root.merge_id
        evaluation = root.evaluate(
            metric_params_name="minimal",
            auto_curation_rules_name="none",
        )
        with pytest.raises(TypeError):
            read_unit_properties(root)
        properties = read_unit_properties(
            root,
            evaluation=evaluation,
            annotation_sets=[score, changed],
        )
        assert "snr" in properties.columns
        assert score.column_name in properties.columns
        assert changed.column_name in properties.columns
        assert score.column_name != changed.column_name
        assert properties.index.tolist() == unit_ids

        summary = root.summarize(evaluation=evaluation, annotation_sets=[score])
        assert summary["created_by"] == dj.config["database.user"]
        assert score.column_name in summary["unit_properties"].columns
        assert root.merge_id == merge_id_before
        assert root.curation_uuid == CurationRef.from_key(root).curation_uuid
    finally:
        clear_curations_for(sorting_key)


def test_from_dataframe_adopts_duplicate_winner_and_propagates_other_failures(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """Only a duplicate-key race is recovered by adopting the winner.

    A concurrent caller landing the same content-addressed set between the
    reuse check and the insert surfaces as ``DuplicateError``; the factory
    then returns the winner (after the provenance / stored-value checks). Any
    other insert failure propagates unchanged, rolls the transaction back
    (no master row is left behind) and triggers no extra recovery query.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
    )

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        unit_ids = sorted(
            map(int, (Sorting.Unit & sorting_key).fetch("unit_id"))
        )
        definition = UnitAnnotationDefinition.insert_definition(
            "recovery_score", 1, "float"
        )
        frame = pd.DataFrame(
            {"recovery_score": [0.5, 1.5]},
            index=pd.Index(unit_ids, name="unit_id"),
        )
        kwargs = dict(producer="race-test", producer_version="1")
        winner = CurationUnitAnnotationSet.from_dataframe(
            root, definition, frame, **kwargs
        )

        # Simulate the race: the pre-insert reuse check sees nothing (as if
        # the winner landed a moment later), so the insert collides on the
        # content-addressed PK and the factory adopts the winner.
        real_reuse = CurationUnitAnnotationSet._reuse_existing.__func__
        calls: list[str] = []

        def _racing_reuse(cls, key, **kw):
            calls.append("reuse")
            if len(calls) == 1:
                return None
            return real_reuse(cls, key, **kw)

        monkeypatch.setattr(
            CurationUnitAnnotationSet,
            "_reuse_existing",
            classmethod(_racing_reuse),
        )
        adopted = CurationUnitAnnotationSet.from_dataframe(
            root, definition, frame, **kwargs
        )
        assert adopted == winner
        assert calls == ["reuse", "reuse"]
        assert len(CurationUnitAnnotationSet & root.as_key()) == 1
        monkeypatch.undo()

        # An unrelated failure inside the transaction propagates as-is, leaves
        # no master row, and does not run the recovery query.
        calls.clear()
        monkeypatch.setattr(
            CurationUnitAnnotationSet,
            "_reuse_existing",
            classmethod(
                lambda cls, key, **kw: (
                    calls.append("reuse"),
                    real_reuse(cls, key, **kw),
                )[1]
            ),
        )

        def _failing_value_insert(self, rows, **kw):
            raise ValueError("simulated value-row failure")

        monkeypatch.setattr(
            CurationUnitAnnotationSet.Value, "insert", _failing_value_insert
        )
        other_frame = frame.copy()
        other_frame.iloc[0, 0] = 9.5
        with pytest.raises(ValueError, match="simulated value-row failure"):
            CurationUnitAnnotationSet.from_dataframe(
                root, definition, other_frame, **kwargs
            )
        assert calls == ["reuse"], "no recovery query after a non-duplicate"
        assert len(CurationUnitAnnotationSet & root.as_key()) == 1
        monkeypatch.undo()
        assert (
            CurationUnitAnnotationSet.from_dataframe(
                root, definition, other_frame, **kwargs
            ).set_hash
            != winner.set_hash
        )
        assert len(CurationUnitAnnotationSet & root.as_key()) == 2
    finally:
        clear_curations_for(sorting_key)
