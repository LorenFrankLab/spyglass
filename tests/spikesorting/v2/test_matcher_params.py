"""Hermetic (no-DB) tests for the matcher parameter schema."""

from __future__ import annotations

import pytest


def test_unitmatch_schema_defaults_round_trip():
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    dumped = UnitMatchParamsSchema().model_dump()
    assert dumped["match_threshold"] == 0.5
    assert dumped["tracked_unit_threshold"] == 0.5
    assert dumped["max_strict_nodes"] == 2000
    assert dumped["schema_version"] == 1


def test_unitmatch_schema_forbids_unknown_field():
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError):
        UnitMatchParamsSchema(typo_field=1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("match_threshold", 1.5),
        ("tracked_unit_threshold", -0.1),
        ("max_strict_nodes", 0),
    ],
)
def test_unitmatch_schema_rejects_out_of_range(field, value):
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError):
        UnitMatchParamsSchema(**{field: value})
