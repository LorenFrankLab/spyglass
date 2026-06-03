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


def test_assert_v2_db_safe_rejects_nonlocal(monkeypatch):
    """``_assert_v2_db_safe`` raises when the DB host is not local.

    This guard is the last line of defense against registering or
    writing v2 schemas on a production server; the host-allowlist
    branch was previously untested. Hermetic -- monkeypatches
    ``dj.config`` and the environment, touches no real DB.
    """
    from spyglass.spikesorting.v2.utils import (
        _OVERRIDE_ENV,
        _assert_v2_db_safe,
    )

    # Ensure the override escape hatch is not set, so the host check runs.
    monkeypatch.delenv(_OVERRIDE_ENV, raising=False)
    monkeypatch.setitem(dj.config, "database.host", "prod.example.org")

    with pytest.raises(RuntimeError, match="refuses to register schemas"):
        _assert_v2_db_safe()


def test_assert_v2_db_safe_override_permits_nonlocal(monkeypatch):
    """Setting the override env var to ``"1"`` bypasses the host guard.

    The override returns before the host check, so a non-local host is
    permitted only when the caller has deliberately opted in.
    """
    from spyglass.spikesorting.v2.utils import (
        _OVERRIDE_ENV,
        _assert_v2_db_safe,
    )

    monkeypatch.setitem(dj.config, "database.host", "prod.example.org")
    monkeypatch.setenv(_OVERRIDE_ENV, "1")

    # Returns None (does not raise) despite the non-local host.
    assert _assert_v2_db_safe() is None


# ---------------------------------------------------------------------------
# A24: the two CRITICAL untested branches (utils.py). Hermetic -- they touch
# only in-memory SI objects and monkeypatched config, so they live here (no
# DB fixture) rather than in the dj_conn-marked integrity module, ensuring the
# CRITICAL coverage runs even in a no-database test slice.
# ---------------------------------------------------------------------------


def test_get_recording_timestamps_concatenates_multi_segment():
    """``_get_recording_timestamps`` stitches every segment into one
    whole-session timeline.

    SpikeInterface's ``recording.get_times()`` returns only the active
    segment's times, so a multi-epoch (multi-segment) NWB would silently
    report just segment 0. The helper must concatenate all segments. We build
    a two-segment ``NumpyRecording`` with *unequal* frame counts so an
    off-by-one in the cumulative-sum indexing would corrupt the seam, then
    assert (a) total length equals the summed frame counts, (b) each segment's
    slice maps back exactly to ``get_times(segment_index=i)``.
    """
    import numpy as np
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.utils import _get_recording_timestamps

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

    ``Recording.make`` threads the persisted wall-clock timestamps through
    ``override`` (the real per-interval times SI's ``frame_slice`` drops);
    the helper must return it verbatim rather than re-deriving from the
    recording. The override differs from the recording's native single-
    segment times, so a regression that ignored ``override`` would be caught.
    """
    import numpy as np
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.utils import _get_recording_timestamps

    rec = sc.NumpyRecording(
        [np.zeros((50, 4), dtype="float32")], sampling_frequency=1000.0
    )
    persisted = np.linspace(10.0, 20.0, 50)
    out = _get_recording_timestamps(rec, override=persisted)
    np.testing.assert_array_equal(out, persisted)
    # Not the native times (which start at 0.0), confirming override wins.
    assert not np.array_equal(out, rec.get_times())


def test_assert_v2_db_safe_localhost_passes(monkeypatch):
    """A24: a safe (local) host returns silently.

    The ``host in _SAFE_DB_HOSTS: return`` pass-through branch was the untested
    half of the guard (the existing scaffold tests cover the raise and the
    override). With the override env var cleared, every safe-host allowlist
    member must return ``None`` without raising.
    """
    from spyglass.spikesorting.v2.utils import (
        _OVERRIDE_ENV,
        _SAFE_DB_HOSTS,
        _assert_v2_db_safe,
    )

    monkeypatch.delenv(_OVERRIDE_ENV, raising=False)
    for safe_host in sorted(_SAFE_DB_HOSTS):
        monkeypatch.setitem(dj.config, "database.host", safe_host)
        assert (
            _assert_v2_db_safe() is None
        ), f"safe host {safe_host!r} should pass the DB-safety guard"


def test_assert_v2_db_safe_override_non_one_value_still_raises(monkeypatch):
    """A24: the override only bypasses when set to exactly ``"1"``.

    Existing coverage pins ``override == "1"`` permitting a non-local host.
    The audit found the *strictness* of that check untested: a value other
    than ``"1"`` (``"0"``, ``""``, ``"true"``) must NOT bypass. We also assert
    the raised message names both the offending host and the override env var,
    so an operator sees how to deliberately bypass.
    """
    from spyglass.spikesorting.v2.utils import (
        _OVERRIDE_ENV,
        _assert_v2_db_safe,
    )

    bad_host = "prod.example.org"
    monkeypatch.setitem(dj.config, "database.host", bad_host)
    for non_bypass in ("0", "", "true", "yes"):
        monkeypatch.setenv(_OVERRIDE_ENV, non_bypass)
        with pytest.raises(RuntimeError) as excinfo:
            _assert_v2_db_safe()
        message = str(excinfo.value)
        assert (
            bad_host in message
        ), f"raised message must name the offending host {bad_host!r}"
        assert _OVERRIDE_ENV in message, (
            "raised message must name the override env var so an operator "
            "knows how to bypass deliberately"
        )
