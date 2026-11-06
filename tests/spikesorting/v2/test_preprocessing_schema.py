"""Preprocessing parameter schema and shared helper tests (no DB)."""

import datajoint as dj
import pytest


def test_preprocessing_params_schema_default():
    """The preprocessing schema (``schema_version`` 4) has the expected
    defaults and guards.
    """
    import pydantic

    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    assert PreprocessingParamsSchema().model_dump() == {
        "schema_version": 4,
        "phase_shift": None,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "min_segment_length": 1.0,
        "bad_channel_handling": "remove",
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
    """The schema owns filtering/referencing; whitening is not accepted."""
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    params = PreprocessingParamsSchema()
    assert params.model_dump(
        include={
            "phase_shift",
            "bandpass_filter",
            "common_reference",
            "bad_channel_handling",
        }
    ) == {
        "phase_shift": None,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "bad_channel_handling": "remove",
    }
    # Sorter and analyzer rows own whitening; the recording stage rejects it.
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="whiten"):
        PreprocessingParamsSchema(whiten={"dtype": "float32"})
    assert not hasattr(params, "to_post_motion_dict")


def test_resolved_job_kwargs_merge(restore_custom_config):
    """Job kwargs merge SI-global, config, and per-row blobs by precedence."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

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


# ---------------------------------------------------------------------------
# Recording-timestamp branches (DB-free)
# ---------------------------------------------------------------------------


def test_get_recording_timestamps_concatenates_multi_segment():
    """``_get_recording_timestamps`` stitches every segment into one
    whole-session timeline.

    SI's ``get_times()`` returns one segment only. Unequal segment lengths
    expose seam off-by-ones.
    """
    import numpy as np
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2._core.signal_math import (
        _get_recording_timestamps,
    )

    seg0 = np.zeros((100, 4), dtype="float32")
    seg1 = np.zeros((150, 4), dtype="float32")
    rec = sc.NumpyRecording([seg0, seg1], sampling_frequency=1000.0)
    assert rec.get_num_segments() == 2  # precondition: the branch fires

    ts = _get_recording_timestamps(rec)

    frames = [rec.get_num_frames(segment_index=i) for i in range(2)]
    assert ts.shape == (sum(frames),)
    np.testing.assert_array_equal(
        ts[: frames[0]], rec.get_times(segment_index=0)
    )
    np.testing.assert_array_equal(
        ts[frames[0] :], rec.get_times(segment_index=1)
    )


def test_get_recording_timestamps_override_returned_verbatim():
    """A caller-supplied persisted-timestamps vector is returned untouched.

    ``Recording.make`` passes the persisted wall-clock times, which SI's
    ``frame_slice`` drops.
    """
    import numpy as np
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2._core.signal_math import (
        _get_recording_timestamps,
    )

    rec = sc.NumpyRecording(
        [np.zeros((50, 4), dtype="float32")], sampling_frequency=1000.0
    )
    persisted = np.linspace(10.0, 20.0, 50)
    out = _get_recording_timestamps(rec, override=persisted)
    np.testing.assert_array_equal(out, persisted)
    # Not the native times (which start at 0.0), confirming override wins.
    assert not np.array_equal(out, rec.get_times())
