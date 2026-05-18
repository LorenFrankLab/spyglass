"""Scaffold checks for the modern spike sorting module.

These tests run only in the SpikeInterface 0.104 test environment. They
exercise the package skeleton, the preprocessing parameter schema, and the
shared helpers -- none of them require a database connection.
"""

import copy
from pathlib import Path

import datajoint as dj
import pytest


@pytest.fixture
def restore_custom_config():
    """Snapshot and restore ``dj.config['custom']`` around a test.

    Tests that mutate the custom config block must not leak changes into
    other tests in the session.
    """
    original = copy.deepcopy(dict(dj.config.get("custom", {}) or {}))
    yield
    dj.config["custom"] = copy.deepcopy(original)


def test_module_imports():
    """The package skeleton imports without optional dependencies."""
    from spyglass.spikesorting import v2

    assert v2.__all__ == []


def test_si_version_min():
    """The dedicated test environment runs the modern SpikeInterface API."""
    import spikeinterface as si
    from packaging.version import Version

    assert Version(si.__version__) >= Version("0.104")


def test_preprocessing_params_schema_default():
    """The preprocessing schema has the expected default shape and guards."""
    import pydantic

    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    assert PreprocessingParamsSchema().model_dump() == {
        "schema_version": 1,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"reference": "global", "operator": "median"},
        "whiten": {"dtype": "float32"},
    }

    # extra="forbid": an unknown field is rejected.
    with pytest.raises(pydantic.ValidationError):
        PreprocessingParamsSchema.model_validate({"unknown_field": 1})

    # Field bounds are enforced: freq_min must be non-negative.
    with pytest.raises(pydantic.ValidationError):
        PreprocessingParamsSchema.model_validate(
            {"bandpass_filter": {"freq_min": -1}}
        )


def test_preprocessing_params_stage_split():
    """The schema splits filtering/referencing from whitening."""
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    params = PreprocessingParamsSchema()
    assert params.to_pre_motion_dict() == {
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"reference": "global", "operator": "median"},
    }
    assert params.to_post_motion_dict() == {"whiten": {"dtype": "float32"}}

    # Whitening disabled -> the post-motion stage is empty.
    no_whiten = PreprocessingParamsSchema(whiten=None)
    assert no_whiten.to_post_motion_dict() == {}


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


def test_analyzer_path_format():
    """The analyzer path resolves under Spyglass's configured temp dir."""
    from spyglass.settings import temp_dir
    from spyglass.spikesorting.v2.utils import _analyzer_path

    path = _analyzer_path({"sorting_id": "abc123"})

    assert path == (
        Path(temp_dir) / "spikesorting_v2" / "analyzers" / "abc123.analyzer"
    )
