"""Schema-level validation tests for v2 Pydantic parameter models.

These tests exercise the Pydantic models in
``spyglass.spikesorting.v2._params`` directly (no DataJoint server required)
so schema-regression catches happen before the slower
``test_single_session_pipeline`` integration suite ever spins up. The
DataJoint-level "reject bogus values at insert" assertion lives in the
integration suite once the Lookup tables exist; this file owns the cheap,
fast schema round-trip and rejection cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_correction import (
    MotionCorrectionParamsSchema,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import (
    ClusterlessThresholderSchema,
    GenericSorterParamsSchema,
    Kilosort4Schema,
    MountainSort4Schema,
    MountainSort5Schema,
    SpykingCircus2Schema,
    Tridesclous2Schema,
    _get_sorter_schema,
)


# ---------- preprocessing ---------------------------------------------------


def test_preprocessing_defaults_round_trip():
    """Default-constructed preprocessing params survive ``model_dump``."""
    blob = PreprocessingParamsSchema().model_dump()
    rebuilt = PreprocessingParamsSchema.model_validate(blob).model_dump()
    assert rebuilt == blob


def test_preprocessing_inverted_bandpass_rejected():
    """``freq_min`` above ``freq_max`` is a validation error."""
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(
            bandpass_filter={"freq_min": 6000.0, "freq_max": 300.0}
        )


def test_preprocessing_whiten_disabled_dumps_empty_post_motion():
    """KS4-style ``whiten=None`` produces an empty post-motion dict."""
    schema = PreprocessingParamsSchema(whiten=None)
    assert schema.to_post_motion_dict() == {}
    assert "bandpass_filter" in schema.to_pre_motion_dict()
    assert "common_reference" in schema.to_pre_motion_dict()


# ---------- artifact detection ---------------------------------------------


def test_artifact_default_keeps_v1_field_names():
    """Defaults match v1 field names plus the v2 additions.

    Shipping v2 default values:
    * ``amplitude_thresh_uV == 500.0`` -- v2's bug-fix value
      (matches v1's effective Intan-probe behavior; v1's
      nominal 3000 was a unit-conversion bug, see the
      CHANGELOG entry).
    * ``proportion_above_thresh == 1.0`` -- v1 parity revert
      from an earlier silently-changed 0.5.
    """
    blob = ArtifactDetectionParamsSchema().model_dump()
    assert blob["detect"] is True
    assert blob["amplitude_thresh_uV"] == 500.0
    assert blob["zscore_thresh"] is None
    assert blob["proportion_above_thresh"] == 1.0
    assert blob["removal_window_ms"] == 1.0
    assert blob["join_window_ms"] == 1.0


def test_artifact_none_preset_disables_detection():
    """``detect=False`` lets all threshold fields be None."""
    blob = ArtifactDetectionParamsSchema(
        detect=False, amplitude_thresh_uV=None
    ).model_dump()
    assert blob["detect"] is False


def test_artifact_detect_true_requires_a_threshold():
    """``detect=True`` with both thresholds None is a validation error."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(
            detect=True, amplitude_thresh_uV=None, zscore_thresh=None
        )


def test_artifact_rejects_unknown_field():
    """``extra='forbid'`` catches typos like ``aplitude_thresh_uV``."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema.model_validate(
            {"detect": True, "aplitude_thresh_uV": 500.0}
        )


def test_artifact_proportion_above_thresh_bounds():
    """``proportion_above_thresh`` must lie in (0, 1]."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(proportion_above_thresh=0.0)
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(proportion_above_thresh=1.5)


def test_artifact_zscore_description_documents_common_mode():
    """The ``zscore_thresh`` field documents that the cross-channel
    z-score is blind to pure common-mode events and that
    ``amplitude_thresh_uV`` is the way to catch them.

    Guards against the prior (false) claim that the cross-channel
    z-score *detects* common-mode artifacts.
    """
    desc = ArtifactDetectionParamsSchema.model_fields[
        "zscore_thresh"
    ].description
    assert desc is not None
    lowered = desc.lower()
    assert "common-mode" in lowered
    assert "amplitude_thresh_uV" in desc
    assert "not detected" in lowered or "blind" in lowered


# ---------- motion correction ----------------------------------------------


def test_motion_default_is_no_op():
    """The default preset is ``"none"`` with empty kwargs."""
    blob = MotionCorrectionParamsSchema().model_dump()
    assert blob["preset"] == "none"
    assert blob["preset_kwargs"] == {}


def test_motion_none_preset_rejects_kwargs():
    """``preset='none'`` with non-empty kwargs is a configuration error."""
    with pytest.raises(ValidationError):
        MotionCorrectionParamsSchema(
            preset="none", preset_kwargs={"detect_kwargs": {}}
        )


def test_motion_accepts_si_native_preset():
    """All SI 0.104 native presets are accepted."""
    for preset in (
        "dredge",
        "medicine",
        "dredge_fast",
        "nonrigid_accurate",
        "nonrigid_fast_and_accurate",
        "rigid_fast",
        "kilosort_like",
        "auto",
    ):
        blob = MotionCorrectionParamsSchema(preset=preset).model_dump()
        assert blob["preset"] == preset


def test_motion_rejects_forbidden_kwargs():
    """Forbidden kwargs change return type or write untracked artifacts."""
    for forbidden in ("output_motion", "output_motion_info", "folder", "overwrite"):
        with pytest.raises(ValidationError):
            MotionCorrectionParamsSchema(
                preset="rigid_fast", preset_kwargs={forbidden: True}
            )


def test_motion_rejects_unknown_preset():
    """An unknown preset string is a Pydantic ``Literal`` rejection."""
    with pytest.raises(ValidationError):
        MotionCorrectionParamsSchema(preset="rigid_unknown")


# ---------- sorter dispatch ------------------------------------------------


@pytest.mark.parametrize(
    "sorter_name,expected",
    [
        ("mountainsort4", MountainSort4Schema),
        ("mountainsort5", MountainSort5Schema),
        ("kilosort4", Kilosort4Schema),
        ("spykingcircus2", SpykingCircus2Schema),
        ("tridesclous2", Tridesclous2Schema),
        ("clusterless_thresholder", ClusterlessThresholderSchema),
        ("hdsort", GenericSorterParamsSchema),
        ("herding_spikes", GenericSorterParamsSchema),
    ],
)
def test_sorter_dispatch(sorter_name, expected):
    """The dispatcher returns dedicated schemas for v2 sorters and the
    generic fallback for everything else."""
    assert _get_sorter_schema(sorter_name) is expected


def test_ms4_default_matches_v1_mountain_default():
    """MS4 defaults mirror v1's ``mountain_default`` block."""
    blob = MountainSort4Schema().model_dump()
    assert blob["detect_sign"] == -1
    assert blob["adjacency_radius"] == 100.0
    assert blob["filter"] is False
    assert blob["whiten"] is True
    assert blob["num_workers"] == 1
    assert blob["clip_size"] == 40
    assert blob["detect_threshold"] == 3.0
    assert blob["detect_interval"] == 10


def test_ms4_detect_sign_rejects_invalid_value():
    """``detect_sign`` is a ``Literal[-1, 0, 1]``; other values reject."""
    with pytest.raises(ValidationError):
        MountainSort4Schema(detect_sign=2)


def test_ms5_default_matches_appendix():
    """MS5 defaults mirror the appendix's empirically-validated row."""
    blob = MountainSort5Schema().model_dump()
    assert blob["scheme"] == "2"
    assert blob["detect_threshold"] == 5.5
    assert blob["snippet_T1"] == 20
    assert blob["snippet_T2"] == 20
    assert blob["scheme2_detect_channel_radius"] == 50.0
    assert blob["detect_time_radius_msec"] == 0.5
    assert blob["scheme2_phase1_detect_channel_radius"] == 200.0


def test_ms5_scheme_literal_enforced():
    """MS5 ``scheme`` is a ``Literal['1','2','3']``."""
    with pytest.raises(ValidationError):
        MountainSort5Schema(scheme="4")


def test_ks4_defaults_match_appendix():
    """KS4 defaults mirror the appendix's documented kwargs."""
    blob = Kilosort4Schema().model_dump()
    assert blob["Th_universal"] == 9.0
    assert blob["Th_learned"] == 8.0
    assert blob["nblocks"] == 1
    assert blob["max_cluster_subset"] == 25_000
    assert blob["do_CAR"] is True


def test_clusterless_default_matches_v1():
    """Clusterless-thresholder defaults mirror v1's row.

    ``ClusterlessThresholderSchema`` drops ``outputs`` and
    ``random_chunk_kwargs`` (both stripped at runtime in
    ``Sorting._run_sorter``); ``noise_levels`` is now OPTIONAL --
    callers who want raw-uV threshold semantics pass
    ``noise_levels=[1.0]`` explicitly (mirroring v1's
    ``default_clusterless``); callers who want MAD-multiplier
    semantics omit it so SI computes per-channel noise internally.
    The shipped v2 ``default`` row passes ``[1.0]`` for v1 parity.
    """
    blob = ClusterlessThresholderSchema().model_dump()
    assert blob["detect_threshold"] == 100.0
    assert blob["method"] == "locally_exclusive"
    assert blob["peak_sign"] == "neg"
    # Regression guard: the dropped fields must not appear in the
    # validated blob. Re-adding them to the schema without
    # justification would silently break the runtime-strip
    # invariants.
    assert "outputs" not in blob
    assert "random_chunk_kwargs" not in blob
    # ``noise_levels`` is now optional; schema default is None
    # (SI computes per-channel MAD). Explicit ``[1.0]`` opt-in is
    # the path the v2 production ``default`` row takes.
    assert blob["noise_levels"] is None
    explicit = ClusterlessThresholderSchema(noise_levels=[1.0]).model_dump()
    assert explicit["noise_levels"] == [1.0]


def test_clusterless_default_row_ships_noise_levels_one(dj_conn):
    """The shipped ``clusterless_thresholder`` / ``default`` row has
    ``noise_levels=[1.0]`` baked into ``params``.

    Regression guard for the 1,400x noise_levels divergence: a
    future refactor that silently drops ``{"noise_levels": [1.0]}``
    from ``SorterParameters._DEFAULT_CONTENTS`` would let the v2
    production default fall back to MAD-multiplier semantics, so
    a user setting ``detect_threshold=100`` would silently get
    ~100xMAD on noisy channels (a ~5x detection shift) instead of
    the 100 uV raw-amplitude threshold the v1 ``default_clusterless``
    row ships.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters().insert_default()
    row = (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    ).fetch1()
    assert row["params"]["detect_threshold"] == 100.0
    assert row["params"]["noise_levels"] == [1.0], (
        "Shipped clusterless `default` row dropped the explicit "
        "noise_levels=[1.0] opt-in; production behavior has silently "
        "regressed from raw-uV semantics to MAD-multiplier semantics."
    )


def test_generic_schema_accepts_arbitrary_kwargs():
    """The generic fallback accepts any payload (``extra='allow'``)."""
    blob = GenericSorterParamsSchema.model_validate(
        {"some_random_kwarg": 42, "another": "value"}
    ).model_dump()
    assert blob["some_random_kwarg"] == 42
    assert blob["another"] == "value"


# ---------- dedicated sorter schemas reject typos --------------------------


@pytest.mark.parametrize(
    "schema_cls,typo_field",
    [
        (MountainSort4Schema, "detect_signe"),
        (MountainSort5Schema, "snippet_T_1"),
        (ClusterlessThresholderSchema, "detect_threshhold"),
    ],
)
def test_dedicated_sorter_schemas_reject_typos(schema_cls, typo_field):
    """The strict dedicated schemas catch misspelled field names.

    This is the value-add over the generic ``extra='allow'`` fallback:
    typos like ``detect_signe`` (extra 'e') silently pass on the generic
    schema but raise on MS4/MS5/Clusterless. SC2, TDC2 and Kilosort4
    use ``extra='allow'`` (KS4 by design, to mirror v1's escape
    hatch at ``v1/sorting.py:184-189`` that lets users pass any
    SI-recognized kwarg through without an upstream schema PR), so
    their typos still pass here -- documented in the module docstring.
    """
    with pytest.raises(ValidationError):
        schema_cls.model_validate({typo_field: 1})


def test_kilosort4_schema_accepts_extra_kwargs():
    """KS4 schema accepts undocumented SI kwargs.

    Mirrors v1's escape hatch: a user wanting to set ``batch_size``
    or ``nearest_chans`` on KS4 should not need a Spyglass PR to
    add the field to the schema. ``extra='allow'`` lets the field
    pass through to SI's KS4 runtime, which validates at sort time.
    """
    blob = Kilosort4Schema.model_validate(
        {"Th_universal": 9.0, "batch_size": 60000, "nearest_chans": 10}
    ).model_dump()
    assert blob["Th_universal"] == 9.0
    assert blob["batch_size"] == 60000
    assert blob["nearest_chans"] == 10


@pytest.mark.parametrize(
    "schema_cls",
    [SpykingCircus2Schema, Tridesclous2Schema],
)
def test_uncurated_sorter_schemas_accept_arbitrary_kwargs(schema_cls):
    """SC2 and TDC2 keep ``extra='allow'`` -- arbitrary kwargs pass.

    Documented in the module docstring as deliberate; SI validates these
    at sort time. This test pins the behavior so a future tighten-up has
    to actively decide to change it.
    """
    blob = schema_cls.model_validate(
        {"random_sc2_kwarg": 99, "another": True}
    ).model_dump()
    assert blob["random_sc2_kwarg"] == 99
    assert blob["another"] is True


# ---------- schema_version invariants --------------------------------------


@pytest.mark.parametrize(
    "schema_cls",
    [
        PreprocessingParamsSchema,
        ArtifactDetectionParamsSchema,
        MotionCorrectionParamsSchema,
        MountainSort4Schema,
        MountainSort5Schema,
        Kilosort4Schema,
        ClusterlessThresholderSchema,
        SpykingCircus2Schema,
        Tridesclous2Schema,
        GenericSorterParamsSchema,
    ],
)
def test_schema_version_present_and_positive(schema_cls):
    """Every v2 Pydantic schema carries a positive ``schema_version`` int.

    Pydantic Parameter Schema Convention requires this field so the
    Lookup row can store the schema generation number alongside the
    blob; a model-breaking change bumps the version and adds a
    LegacyParams shim rather than silently overwriting old rows.
    The exact value differs per schema (preprocessing + artifact +
    clusterless were bumped to 2 when their field sets changed).
    """
    blob = schema_cls().model_dump()
    assert isinstance(blob["schema_version"], int)
    assert blob["schema_version"] >= 1
