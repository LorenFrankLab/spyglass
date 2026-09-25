"""DB tests for the v2 motion-estimation tables."""

from __future__ import annotations

import pytest


@pytest.fixture
def motion_params(dj_conn):
    """``MotionEstimationParameters`` with the shipped rows installed."""
    from spyglass.spikesorting.v2.motion import MotionEstimationParameters

    MotionEstimationParameters.insert_default()
    return MotionEstimationParameters


def test_default_estimation_rows_install_idempotently(motion_params):
    motion_params.insert_default()

    names = set(motion_params.fetch("motion_estimation_params_name"))
    assert {"dredge_v1", "dredge_fast_v1"} <= names
    assert not (motion_params & {"motion_estimation_params_name": "rigid_fast"})
    presets = {
        name: params["preset"]
        for name, params in zip(
            *(
                motion_params
                & [
                    {"motion_estimation_params_name": "dredge_v1"},
                    {"motion_estimation_params_name": "dredge_fast_v1"},
                ]
            ).fetch("motion_estimation_params_name", "params")
        )
    }
    assert presets == {"dredge_v1": "dredge", "dredge_fast_v1": "dredge_fast"}


def test_estimation_row_with_unknown_override_is_rejected(motion_params):
    with pytest.raises(ValueError, match="detect_treshold"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "typo_row",
                "params": {
                    "preset": "dredge",
                    "detect_kwargs": {"detect_treshold": 6.0},
                },
            }
        )
    assert not (motion_params & {"motion_estimation_params_name": "typo_row"})


def test_estimation_row_rejects_job_kwargs_seed(motion_params):
    with pytest.raises(ValueError, match="noise_levels_seed"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "seeded_job_kwargs",
                "params": {"preset": "rigid_fast"},
                "job_kwargs": {"random_seed": 3},
            }
        )


def test_estimation_row_duplicating_a_default_is_rejected(motion_params):
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    with pytest.raises(DuplicateParameterContentError, match="dredge_v1"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "dredge_copy",
                "params": MotionEstimationParamsSchema(
                    preset="dredge"
                ).model_dump(),
            }
        )
