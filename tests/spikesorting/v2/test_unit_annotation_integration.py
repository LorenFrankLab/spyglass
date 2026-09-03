"""Database contracts for typed, curation-scoped unit annotations."""

from __future__ import annotations

import datajoint as dj
import pandas as pd
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


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
        with pytest.raises(LookupError, match="stale curation_uuid"):
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
