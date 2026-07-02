"""MotionCorrectionParameters presets, schema-version drift, and validation.

Covers ``initialize_v2_defaults()`` shipping the motion presets, the
outer/inner ``schema_version`` drift guard at insert time, and table-level
Pydantic validation of the ``params`` blob.
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("dj_conn")
def test_initialize_v2_defaults_installs_motion_correction_presets():
    """``initialize_v2_defaults()`` installs the motion presets too.

    The helper previously called ``insert_default`` on only three Lookups
    and skipped ``MotionCorrectionParameters``. The motion presets then
    shipped missing, so when the concat consumer lands every run would
    hit a missing-row FK violation unless the user remembered the call
    themselves. ``initialize_v2_defaults()`` must seed all four.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    initialize_v2_defaults()
    for name in ("none", "auto_default", "rigid_fast_default"):
        assert (
            len(
                MotionCorrectionParameters
                & {"motion_correction_params_name": name}
            )
            == 1
        ), f"motion preset {name!r} missing after initialize_v2_defaults()"


@pytest.mark.usefixtures("dj_conn")
def test_motion_correction_parameters_rejects_schema_version_drift(request):
    """Outer/inner schema-version drift is caught at insert time.

    ``MotionCorrectionParameters.insert1`` validated the ``params`` blob
    but never compared the outer ``params_schema_version`` column to the
    blob's inner ``schema_version`` (unlike ``SorterParameters.insert1``).
    A row whose outer says 1 and inner says 2 would land with the two
    disagreeing. ``_assert_schema_version_matches`` now closes the gap.
    """
    from spyglass.spikesorting.v2._params.motion_correction import (
        MotionCorrectionParamsSchema,
    )
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    drifted_params = MotionCorrectionParamsSchema(schema_version=2).model_dump()
    drift_row = {
        "motion_correction_params_name": "audit_a14_drift",
        "params": drifted_params,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    with pytest.raises(ValueError, match="schema_version"):
        MotionCorrectionParameters().insert1(drift_row, skip_duplicates=True)

    # A consistent row still inserts cleanly. The bare-default blob matches
    # the shipped ``"none"`` row, so opt out of the duplicate-content guard
    # (this test checks the schema-version drift guard, not duplicate content).
    ok_params = MotionCorrectionParamsSchema().model_dump()
    ok_row = {
        "motion_correction_params_name": "audit_a14_ok",
        "params": ok_params,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    MotionCorrectionParameters().insert1(
        ok_row, skip_duplicates=True, allow_duplicate_params=True
    )
    # Delete this default-content fork after the test: left behind it shadows
    # the shipped ``none`` row and a later ``insert_default()`` would raise
    # DuplicateParameterContentError.
    request.addfinalizer(
        lambda: (
            MotionCorrectionParameters
            & {"motion_correction_params_name": "audit_a14_ok"}
        ).delete(safemode=False)
    )
    assert MotionCorrectionParameters & {
        "motion_correction_params_name": "audit_a14_ok"
    }


@pytest.mark.usefixtures("dj_conn")
def test_motion_correction_parameters_validates_params_blob():
    """``MotionCorrectionParameters.insert1`` Pydantic-validates ``params``.

    The table-level wiring (not just the standalone schema tests) must reject a
    bogus blob -- here an unknown ``preset`` against the ``MotionPreset``
    Literal.
    """
    import pydantic

    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )

    with pytest.raises(pydantic.ValidationError):
        MotionCorrectionParameters().insert1(
            {
                "motion_correction_params_name": "a31_bogus_preset",
                "params": {"preset": "not_a_real_preset"},
                "params_schema_version": 1,
                "job_kwargs": None,
            }
        )
