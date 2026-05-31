"""Scaffold checks for the modern spike sorting module.

These tests run only in the SpikeInterface 0.104 test environment. They
exercise the package skeleton, the preprocessing parameter schema, and the
shared helpers -- none of them require a database connection.
"""

import datajoint as dj
import pytest


def test_preprocessing_params_schema_default():
    """The preprocessing schema has the expected default shape and guards.

    ``schema_version`` is 2 (the v2 shipping schema): added
    ``min_segment_length`` and removed the dead
    ``common_reference.reference`` field. Defaults below reflect
    the schema as shipped.
    """
    import pydantic

    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    assert PreprocessingParamsSchema().model_dump() == {
        "schema_version": 3,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "whiten": None,
        "min_segment_length": 1.0,
    }

    # extra="forbid": an unknown field is rejected.
    with pytest.raises(pydantic.ValidationError):
        PreprocessingParamsSchema.model_validate({"unknown_field": 1})

    # Field bounds are enforced: freq_min must be non-negative.
    with pytest.raises(pydantic.ValidationError):
        PreprocessingParamsSchema.model_validate(
            {"bandpass_filter": {"freq_min": -1}}
        )

    # Cross-field rule: freq_min must be below freq_max.
    with pytest.raises(pydantic.ValidationError):
        PreprocessingParamsSchema.model_validate(
            {"bandpass_filter": {"freq_min": 6000, "freq_max": 300}}
        )


def test_preprocessing_params_stage_split():
    """The schema splits filtering/referencing from whitening."""
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    params = PreprocessingParamsSchema()
    assert params.to_pre_motion_dict() == {
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
    }
    # Whitening defaults to None (deferred to the sorter), so the
    # post-motion stage is empty by default.
    assert params.to_post_motion_dict() == {}

    # Explicitly enabling whitening populates the post-motion stage.
    whitened = PreprocessingParamsSchema(whiten={"dtype": "float32"})
    assert whitened.to_post_motion_dict() == {"whiten": {"dtype": "float32"}}


def test_resolved_job_kwargs_merge(restore_custom_config):
    """Job kwargs merge SI-global, config, and per-row blobs by precedence."""
    import spikeinterface as si

    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"n_jobs": 4}

    base = _resolved_job_kwargs()
    # DataJoint config overrides the SpikeInterface global default.
    assert base["n_jobs"] == 4
    # SpikeInterface global defaults pass through for keys config does not
    # override (asserted against SI's own values, not hard-coded ones).
    for key, value in si.get_global_job_kwargs().items():
        if key != "n_jobs":
            assert base[key] == value

    # A per-row blob wins over the DataJoint config.
    assert _resolved_job_kwargs({"n_jobs": 8})["n_jobs"] == 8
    # A later per-row argument wins over an earlier one.
    assert _resolved_job_kwargs({"n_jobs": 8}, {"n_jobs": 2})["n_jobs"] == 2
    # A None argument is skipped, leaving the config value in place.
    assert _resolved_job_kwargs(None)["n_jobs"] == 4
    # An empty-dict argument is skipped the same way.
    assert _resolved_job_kwargs({})["n_jobs"] == 4


