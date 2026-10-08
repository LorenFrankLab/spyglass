"""Grouping configuration is shared across matcher algorithms."""

import pytest
from pydantic import Field, ValidationError

from spyglass.spikesorting.v2._params.tracking import (
    TrackingParamsSchema,
    resolve_tracking_params,
)


def test_backend_without_tracking_fields_uses_shared_defaults():
    tracking = resolve_tracking_params({"model_name": "future-test-backend"})
    assert tracking.tracked_unit_threshold == 0.5
    assert tracking.max_strict_nodes == 2000


def test_tracking_overrides_are_independent_of_backend_parameters():
    tracking = resolve_tracking_params(
        {
            "tracked_unit_threshold": 0.8,
            "max_strict_nodes": 50,
            "model_seed": 17,
        }
    )
    assert tracking.model_dump() == {
        "tracked_unit_threshold": 0.8,
        "max_strict_nodes": 50,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"tracked_unit_threshold": -0.1},
        {"tracked_unit_threshold": 1.1},
        {"tracked_unit_threshold": float("nan")},
        {"max_strict_nodes": 0},
    ],
)
def test_invalid_stored_grouping_controls_fail_before_graph_search(params):
    with pytest.raises(ValidationError):
        resolve_tracking_params(params)


def test_custom_matcher_can_inherit_shared_grouping_schema():
    class FeatureMatcherParams(TrackingParamsSchema):
        model_name: str
        model_seed: int = Field(default=0, ge=0)

    recipe = FeatureMatcherParams(model_name="test", tracked_unit_threshold=0.7)
    assert (
        resolve_tracking_params(recipe.model_dump()).tracked_unit_threshold
        == 0.7
    )
    with pytest.raises(ValidationError):
        FeatureMatcherParams(model_name="test", max_strict_nodes=0)


def test_unitmatch_recipe_content_and_schema_version_are_preserved():
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    assert UnitMatchParamsSchema().model_dump() == {
        "ms_before": 1.5,
        "ms_after": 1.5,
        "max_spikes_per_unit": 100,
        "seed": 0,
        "match_threshold": 0.5,
        "tracked_unit_threshold": 0.5,
        "max_strict_nodes": 2000,
        "schema_version": 1,
    }
