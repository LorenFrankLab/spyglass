"""Schema-level validation tests for v2 Pydantic parameter models.

These tests exercise the Pydantic models in
``spyglass.spikesorting.v2._params`` directly (no DataJoint server required)
so schema-regression catches happen before the slower
``single_session/`` integration suite ever spins up. The
DataJoint-level "reject bogus values at insert" assertion lives in the
integration suite once the Lookup tables exist; this file owns the cheap,
fast schema round-trip and rejection cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spyglass.spikesorting.v2._params.artifact_detection import (
    ARTIFACT_DETECTION_SCHEMA_VERSION,
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_correction import (
    MOTION_CORRECTION_SCHEMA_VERSION,
    MotionCorrectionParamsSchema,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PREPROCESSING_SCHEMA_VERSION,
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


def test_preprocessing_non_default_blob_round_trips():
    """A fully NON-default blob survives ``dump -> validate -> dump`` unchanged.

    A default-constructed model dumped then re-validated is equal essentially
    by construction (near-tautological). Round-tripping a blob whose every
    field diverges from the schema defaults gives the test teeth: a validator
    or normalizer that mangles a real configured params blob is caught here,
    cheaply, instead of at populate time.
    """
    schema = PreprocessingParamsSchema(
        phase_shift={"margin_ms": 50.0},
        bandpass_filter={"freq_min": 250.0, "freq_max": 5000.0},
        common_reference={"operator": "average"},
        whiten={"dtype": "float64"},
        min_segment_length=0.0015,
        bad_channel_handling="interpolate",
    )
    blob = schema.model_dump()
    # Sanity: the blob really IS non-default, else the round-trip proves little.
    assert blob != PreprocessingParamsSchema().model_dump()

    rebuilt = PreprocessingParamsSchema.model_validate(blob).model_dump()
    assert rebuilt == blob
    # Each non-default value is preserved verbatim through the round-trip.
    assert rebuilt["phase_shift"] == {"margin_ms": 50.0}
    assert rebuilt["bandpass_filter"] == {"freq_min": 250.0, "freq_max": 5000.0}
    assert rebuilt["common_reference"]["operator"] == "average"
    assert rebuilt["whiten"] == {"dtype": "float64"}
    assert rebuilt["min_segment_length"] == 0.0015
    assert rebuilt["bad_channel_handling"] == "interpolate"


@pytest.mark.parametrize("bad_min_segment", [-0.5, -1e-6])
def test_preprocessing_rejects_negative_min_segment_length(bad_min_segment):
    """``min_segment_length`` is non-negative (``ge=0``); negatives reject.

    The field drops disjoint-interval slivers shorter than this many seconds
    before the sorter; a negative value is meaningless and silently disabling
    the filter via a bad row must fail loudly at insert, not at sort time.
    """
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(min_segment_length=bad_min_segment)
    # Zero is the documented "drop nothing" boundary and is accepted.
    zero = PreprocessingParamsSchema(min_segment_length=0.0)
    assert zero.min_segment_length == 0.0


def test_preprocessing_rejects_unknown_bad_channel_handling():
    """``bad_channel_handling`` is ``Literal['remove','interpolate']``.

    A typo (e.g. ``'drop'``) must reject rather than silently fall through to
    the default handling, which would change which channels reach the sorter.
    """
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(bad_channel_handling="drop")
    assert (
        PreprocessingParamsSchema(
            bad_channel_handling="interpolate"
        ).bad_channel_handling
        == "interpolate"
    )


def test_phase_shift_off_by_default():
    """``phase_shift`` defaults to ``None`` (the franklab default is a no-op).

    Existing rows that predate the field still validate because the new
    optional field defaults to ``None`` (no ``schema_version`` bump).
    """
    schema = PreprocessingParamsSchema()
    assert schema.phase_shift is None
    assert schema.to_pre_motion_dict()["phase_shift"] is None
    # A pre-field blob (no ``phase_shift`` key) validates unchanged.
    legacy = {
        "schema_version": 3,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "whiten": None,
        "min_segment_length": 1.0,
    }
    assert PreprocessingParamsSchema.model_validate(legacy).phase_shift is None


def test_phase_shift_enabled_dumps_margin():
    """Enabling ``phase_shift`` carries ``margin_ms`` into the pre-motion dict."""
    schema = PreprocessingParamsSchema(phase_shift={"margin_ms": 100.0})
    assert schema.phase_shift is not None
    assert schema.phase_shift.margin_ms == 100.0
    assert schema.to_pre_motion_dict()["phase_shift"] == {"margin_ms": 100.0}
    # ``margin_ms`` defaults to 100.0 when omitted.
    assert (
        PreprocessingParamsSchema(phase_shift={}).phase_shift.margin_ms == 100.0
    )


def test_phase_shift_rejects_negative_margin_and_extra_keys():
    """``margin_ms`` is non-negative and the sub-model forbids extra keys."""
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(phase_shift={"margin_ms": -1.0})
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(phase_shift={"bogus": 1.0})


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


def test_preprocessing_no_filter_is_none():
    """``bandpass_filter=None`` disables filtering; the default keeps it.

    A real "no filter" config is ``bandpass_filter=None`` (the
    pre-motion dict carries ``None`` so the runtime skips the step),
    not a wide-band filter that silently still filters.
    """
    disabled = PreprocessingParamsSchema(bandpass_filter=None)
    assert disabled.bandpass_filter is None
    assert disabled.to_pre_motion_dict()["bandpass_filter"] is None
    # The default still ships a real bandpass.
    assert PreprocessingParamsSchema().bandpass_filter is not None
    assert (
        PreprocessingParamsSchema().to_pre_motion_dict()["bandpass_filter"]
        is not None
    )


def test_whiten_default_is_none():
    """The default ``whiten`` is ``None`` to match the runtime.

    Whitening is deferred to the sorter (the runtime applies it lazily
    after motion correction), so a default-constructed schema must NOT
    claim whitening is configured. ``to_post_motion_dict`` is empty by
    default.
    """
    schema = PreprocessingParamsSchema()
    assert schema.whiten is None
    assert schema.to_post_motion_dict() == {}


# ---------- artifact detection ---------------------------------------------


def test_artifact_default_field_values():
    """Defaults use the v2 field names and ship the expected values.

    Shipping v2 default values:
    * ``amplitude_threshold_uv == 500.0`` -- v2's bug-fix value
      (matches v1's effective Intan-probe behavior; v1's
      nominal 3000 was a unit-conversion bug, see the
      CHANGELOG entry).
    * ``proportion_above_threshold == 1.0`` -- v1 parity revert
      from an earlier silently-changed 0.5.
    """
    blob = ArtifactDetectionParamsSchema().model_dump()
    assert blob["detect"] is True
    assert blob["amplitude_threshold_uv"] == 500.0
    assert blob["zscore_threshold"] is None
    assert blob["proportion_above_threshold"] == 1.0
    assert blob["removal_window_ms"] == 1.0
    assert blob["join_window_ms"] == 1.0


def test_artifact_none_preset_disables_detection():
    """``detect=False`` lets all threshold fields be None and they round-trip.

    Turning detection off relaxes the ``detect=True`` requirement of at least
    one threshold: both thresholds may be None and persist as None. Asserting
    only ``detect is False`` would merely echo the constructor argument.
    """
    blob = ArtifactDetectionParamsSchema(
        detect=False, amplitude_threshold_uv=None, zscore_threshold=None
    ).model_dump()
    assert blob["detect"] is False
    assert blob["amplitude_threshold_uv"] is None
    assert blob["zscore_threshold"] is None


def test_artifact_detect_true_requires_a_threshold():
    """``detect=True`` with both thresholds None is a validation error."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(
            detect=True, amplitude_threshold_uv=None, zscore_threshold=None
        )


def test_artifact_rejects_unknown_field():
    """``extra='forbid'`` catches typos like ``aplitude_thresh_uV``."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema.model_validate(
            {"detect": True, "aplitude_thresh_uV": 500.0}
        )


def test_artifact_proportion_above_threshold_bounds():
    """``proportion_above_threshold`` must lie in (0, 1]."""
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(proportion_above_threshold=0.0)
    with pytest.raises(ValidationError):
        ArtifactDetectionParamsSchema(proportion_above_threshold=1.5)


def test_artifact_thresholds_or_semantics():
    """The two thresholds are an intentional OR, not mutually exclusive.

    Amplitude-only, z-score-only, and both-at-once all validate (the
    detector ORs amplitude and z-score). ``detect=False`` ignores both
    thresholds, so leaving stale thresholds set is not an error.
    """
    # amplitude-only
    ArtifactDetectionParamsSchema(
        amplitude_threshold_uv=500.0, zscore_threshold=None
    )
    # z-score-only
    ArtifactDetectionParamsSchema(
        amplitude_threshold_uv=None, zscore_threshold=5.0
    )
    # both thresholds at once -- the OR mode
    both = ArtifactDetectionParamsSchema(
        amplitude_threshold_uv=500.0, zscore_threshold=5.0
    ).model_dump()
    assert both["amplitude_threshold_uv"] == 500.0
    assert both["zscore_threshold"] == 5.0
    # detect=False ignores both thresholds even when they are still set
    disabled = ArtifactDetectionParamsSchema(
        detect=False, amplitude_threshold_uv=500.0, zscore_threshold=5.0
    ).model_dump()
    assert disabled["detect"] is False


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


@pytest.mark.parametrize(
    "preset",
    [
        "dredge",
        "medicine",
        "dredge_fast",
        "nonrigid_accurate",
        "nonrigid_fast_and_accurate",
        "rigid_fast",
        "kilosort_like",
        "auto",
    ],
)
def test_motion_accepts_si_native_preset(preset):
    """All SI 0.104 native presets are accepted."""
    blob = MotionCorrectionParamsSchema(preset=preset).model_dump()
    assert blob["preset"] == preset


@pytest.mark.parametrize(
    "forbidden",
    [
        "output_motion",
        "output_motion_info",
        "folder",
        "overwrite",
    ],
)
def test_motion_rejects_forbidden_kwargs(forbidden):
    """Forbidden kwargs change return type or write untracked artifacts."""
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


@pytest.mark.parametrize("bad_radius", [-0.5, -0.001, -0.999])
def test_ms4_rejects_open_interval_adjacency_radius(bad_radius):
    """``adjacency_radius`` in the open interval ``(-1, 0)`` is rejected.

    ``-1`` is SpikeInterface's "use all channels" sentinel and ``>= 0`` is an
    explicit radius; anything strictly between is meaningless. These values
    pass the ``ge=-1.0`` Field bound, so the rejection here exercises the
    custom ``_reject_open_interval_radius`` validator specifically (matched on
    its message), not the Field constraint.
    """
    with pytest.raises(ValidationError, match="open interval"):
        MountainSort4Schema(adjacency_radius=bad_radius)
    # The sentinel and the lower explicit-radius boundary are both accepted.
    assert MountainSort4Schema(adjacency_radius=-1.0).adjacency_radius == -1.0
    assert MountainSort4Schema(adjacency_radius=0.0).adjacency_radius == 0.0


def test_ms5_default_values():
    """MS5 defaults mirror the empirically-validated reference row."""
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


def test_ms5_schema_accepts_and_defaults_filter_whiten():
    """MS5 exposes ``filter`` / ``whiten`` with behavior-preserving defaults.

    ``filter`` defaults to ``False`` (the v2 recording stage already
    bandpass-filters the input, so MS5's internal filter must stay off to
    avoid double-filtering) and ``whiten`` defaults to ``True`` (matching
    the SI 0.104 ``Mountainsort5Sorter._default_params`` ``whiten=True``).
    Both toggle, and ``extra="forbid"`` still rejects an unknown kwarg.
    """
    blob = MountainSort5Schema().model_dump()
    assert blob["filter"] is False
    assert blob["whiten"] is True
    # Both toggles round-trip when set explicitly.
    flipped = MountainSort5Schema(filter=True, whiten=False).model_dump()
    assert flipped["filter"] is True
    assert flipped["whiten"] is False
    # Adding the toggles did not loosen the strict schema.
    with pytest.raises(ValidationError):
        MountainSort5Schema(filterr=True)


def test_ks4_default_values():
    """KS4 defaults mirror the documented reference kwargs."""
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
    semantics set ``threshold_unit='mad'`` (noise_levels omitted so SI
    computes per-channel noise internally). The shipped v2 ``default``
    row passes ``[1.0]`` for v1 parity.
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
    # ``noise_levels`` is now optional; the schema field default is None.
    # With the default ``threshold_unit='uv'`` the runtime derives [1.0] (a
    # microvolt/native threshold); MAD-multiplier mode is opt-in via
    # ``threshold_unit='mad'``. The v2 production ``default`` row passes
    # ``[1.0]`` explicitly.
    assert blob["noise_levels"] is None
    explicit = ClusterlessThresholderSchema(noise_levels=[1.0]).model_dump()
    assert explicit["noise_levels"] == [1.0]


def test_clusterless_threshold_unit_explicit():
    """``threshold_unit`` makes the noise_levels unit a first-class knob.

    The default is ``"uv"`` (the production/real-data 100 uV threshold; at
    runtime 'uv' derives noise_levels=[1.0]). ``"mad"`` is the MAD-multiplier
    mode (the simulation fixture's regime). Typos are rejected
    (``extra="forbid"`` + Literal), and the schema is at version 4.
    """
    default = ClusterlessThresholderSchema()
    assert default.threshold_unit == "uv"
    assert default.noise_levels is None
    assert default.schema_version == 4

    uv = ClusterlessThresholderSchema(threshold_unit="uv")
    assert uv.threshold_unit == "uv"

    # An explicit noise_levels override coexists with threshold_unit.
    override = ClusterlessThresholderSchema(
        threshold_unit="mad", noise_levels=[2.0]
    )
    assert override.noise_levels == [2.0]

    # Invalid unit is rejected by the Literal.
    with pytest.raises(ValidationError):
        ClusterlessThresholderSchema(threshold_unit="microvolts")


def test_clusterless_rejects_implausible_mad_threshold():
    """A microvolt threshold left in MAD units is rejected at insert.

    ``threshold_unit='mad'`` with no explicit ``noise_levels`` makes SI treat
    ``detect_threshold`` as a MAD multiplier (real ones are ~3-15). A value
    like 100 (a microvolt threshold copied into a MAD-mode row) makes the
    effective threshold absurd and silently detects almost nothing -- a
    zero-unit sort. The ``_guard_implausible_mad_threshold`` validator must
    reject it, accept a plausible multiplier, and stay out of the way of the
    explicit-``noise_levels`` advanced override.
    """
    # Reject: > _MAX_PLAUSIBLE_MAD_MULTIPLIER (50) in mad mode, no noise_levels.
    with pytest.raises(ValidationError, match="implausibly large MAD"):
        ClusterlessThresholderSchema(
            detect_threshold=100.0, threshold_unit="mad"
        )

    # Accept: a plausible MAD multiplier in the same mode.
    ok = ClusterlessThresholderSchema(
        detect_threshold=5.0, threshold_unit="mad"
    )
    assert ok.detect_threshold == 5.0 and ok.threshold_unit == "mad"

    # Bypass: an explicit noise_levels override is used verbatim, so a large
    # value is intentional and the guard must NOT fire.
    override = ClusterlessThresholderSchema(
        detect_threshold=100.0, threshold_unit="mad", noise_levels=[1.0]
    )
    assert override.detect_threshold == 100.0

    # Bypass: 'uv' mode never triggers the MAD guard, even at large thresholds.
    uv = ClusterlessThresholderSchema(
        detect_threshold=100.0, threshold_unit="uv"
    )
    assert uv.detect_threshold == 100.0


def test_insert_row_to_dict_normalizes_and_rejects_bad_shapes():
    """``_insert_row_to_dict`` handles the supported insert row shapes.

    Mapping rows pass through (shallow-copied); positional sequences zip
    against the heading order (the ``_DEFAULT_CONTENTS`` tuple form). A
    bare ``str``/``bytes`` row -- the shape that leaks through a
    ``DataFrame`` / CSV-path insert (iterating those yields column-name
    strings / characters) -- is rejected loudly rather than silently
    zipped into a malformed row.
    """
    from spyglass.spikesorting.v2.utils import _insert_row_to_dict

    names = ("sorter", "sorter_params_name", "params")
    # Mapping passes through as a shallow copy.
    src = {"sorter": "mountainsort4", "params": {"x": 1}}
    out = _insert_row_to_dict(src, names)
    assert out == src and out is not src
    # Positional tuple zips against the heading order.
    assert _insert_row_to_dict(("mountainsort4", "name", {"x": 1}), names) == {
        "sorter": "mountainsort4",
        "sorter_params_name": "name",
        "params": {"x": 1},
    }
    # A str/bytes row fails loudly (the DataFrame / CSV-path footgun).
    for bad in ("params", b"params"):
        with pytest.raises(TypeError, match="mapping or"):
            _insert_row_to_dict(bad, names)
    # Positional rows must match the heading length exactly. ``zip`` would
    # otherwise truncate extras silently or let missing attrs fail later with
    # an opaque KeyError.
    for bad in (
        ("mountainsort4", "name"),
        ("mountainsort4", "name", {"x": 1}, "ignored_extra"),
    ):
        with pytest.raises(ValueError, match="wrong length"):
            _insert_row_to_dict(bad, names)


def test_clusterless_noise_levels_length_guard():
    """The clusterless noise_levels length guard rejects bad explicit arrays.

    ``_assert_noise_levels_length`` lives in the import-pure ``utils``
    module (no ``dj.schema``), so this is a pure unit test of the guard
    the runtime applies at ``Sorting._run_clusterless_thresholder``
    before broadcasting / indexing ``noise_levels`` per channel. An
    explicit array must be length 1 (broadcast) or ``n_channels``; any
    other explicit length is a configuration error; ``None`` is always
    valid (SI estimates per-channel MAD).
    """
    from spyglass.spikesorting.v2.utils import _assert_noise_levels_length

    n_channels = 4
    # None and the two valid explicit lengths are accepted (no raise).
    _assert_noise_levels_length(None, n_channels)
    _assert_noise_levels_length([1.0], n_channels)  # broadcast
    _assert_noise_levels_length([1.0, 2.0, 3.0, 4.0], n_channels)  # per-chan
    # A wrong explicit length raises and names the expected lengths.
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0], [1.0] * 5):
        with pytest.raises(ValueError, match="length 1.*or.*n_channels=4"):
            _assert_noise_levels_length(bad, n_channels)
    # Raw update1 / legacy-row bypasses can carry non-list shapes. Reject them
    # here with the curated message rather than letting len()/np.asarray()
    # produce misleading behavior downstream.
    for bad in (1.0, "1234", b"1234"):
        with pytest.raises(ValueError, match="numeric sequence"):
            _assert_noise_levels_length(bad, n_channels)


def test_clusterless_noise_levels_derivation(dj_conn):
    """The runtime derives noise_levels from threshold_unit correctly.

    Precedence: an explicit noise_levels wins; otherwise ``"uv"`` derives
    ``[1.0]`` (raw-uV threshold) and ``"mad"`` derives ``None`` (SI
    estimates per-channel MAD).
    """
    from spyglass.spikesorting.v2.sorting import _clusterless_noise_levels

    # explicit override wins regardless of unit
    assert _clusterless_noise_levels([3.0], "uv") == [3.0]
    assert _clusterless_noise_levels([3.0], "mad") == [3.0]
    # derive from unit when unset
    assert _clusterless_noise_levels(None, "uv") == [1.0]
    assert _clusterless_noise_levels(None, "mad") is None


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


@pytest.mark.parametrize("bad", [True, False, 1])
def test_kilosort4_schema_rejects_whiten(bad):
    """A ``whiten`` key on a KS4 row is rejected at validation.

    KS4 has no ``whiten`` kwarg -- it whitens internally -- and the v2 runtime
    runs its external float64 whitening whenever the sorter params carry a
    truthy ``whiten`` (``_sorting_dispatch.py``). A ``whiten`` key, which
    ``extra="allow"`` would otherwise pass through silently, would therefore
    double-whiten the signal. The schema rejects it up front rather than
    letting it surface as silently-wrong spike times at sort time. Any value
    (even ``False``) is rejected, since KS4 has no such parameter at all.
    """
    with pytest.raises(ValidationError, match="whiten"):
        Kilosort4Schema.model_validate({"whiten": bad})


@pytest.mark.parametrize("sorter", ["kilosort2_5", "kilosort3", "ironclust"])
def test_internal_whiten_sorters_reject_truthy_whiten(sorter):
    """kilosort2_5 / kilosort3 / ironclust whiten internally and have no
    ``whiten`` kwarg, so a truthy ``whiten`` would trigger the v2 runtime's
    external float64 whitening on top of the sorter's own (double-whiten).
    Unlike KS4 they fall through to the ``extra='allow'`` generic schema, so
    the guard runs at ``SorterParameters`` insert via ``reject_internal_whiten``.
    """
    from spyglass.spikesorting.v2._params.sorter import reject_internal_whiten

    with pytest.raises(ValueError, match="whitens internally"):
        reject_internal_whiten(sorter, {"whiten": True})


def test_internal_whiten_guard_allows_absent_or_falsy_whiten():
    """An absent or falsy ``whiten`` is fine for the internal-whiten sorters."""
    from spyglass.spikesorting.v2._params.sorter import reject_internal_whiten

    reject_internal_whiten("kilosort2_5", {})
    reject_internal_whiten("kilosort3", {"whiten": False})
    reject_internal_whiten("ironclust", {"whiten": 0})


def test_internal_whiten_guard_ignores_mountainsort():
    """MS4/MS5 intentionally use ``whiten=True`` (routed through the external
    float64 whitening), so the guard must NOT reject them."""
    from spyglass.spikesorting.v2._params.sorter import reject_internal_whiten

    reject_internal_whiten("mountainsort4", {"whiten": True})
    reject_internal_whiten("mountainsort5", {"whiten": True})


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

    Every params schema must carry this field so the Lookup row can store
    the schema generation number alongside the blob; a model-breaking
    change bumps the version and adds a LegacyParams shim rather than
    silently overwriting old rows. The exact value differs per schema
    (preprocessing + artifact + clusterless were bumped to 2 when their
    field sets changed).
    """
    blob = schema_cls().model_dump()
    assert isinstance(blob["schema_version"], int)
    assert blob["schema_version"] >= 1


def test_schema_version_constants_match_schema_defaults():
    """Named constants are the single source for table-definition defaults."""
    assert (
        PreprocessingParamsSchema().schema_version
        == PREPROCESSING_SCHEMA_VERSION
    )
    assert (
        ArtifactDetectionParamsSchema().schema_version
        == ARTIFACT_DETECTION_SCHEMA_VERSION
    )
    assert (
        MotionCorrectionParamsSchema().schema_version
        == MOTION_CORRECTION_SCHEMA_VERSION
    )


def test_recipe_catalog_rows_copy_inner_schema_version():
    """Recipe rows derive the outer version from the validated params blob."""
    from spyglass.spikesorting.v2._recipe_catalog import (
        artifact_default_contents,
        preprocessing_default_contents,
        sorter_default_contents,
    )

    for name, params, version, _job_kwargs in preprocessing_default_contents():
        assert version == params["schema_version"], name

    for name, params, version, _job_kwargs in artifact_default_contents():
        assert version == params["schema_version"], name

    # Sorter rows are 7-tuples: (sorter, name, params, params_schema_version,
    # job_kwargs, execution_params, execution_params_schema_version). Both outer
    # versions must derive from their validated inner blob's schema_version.
    for (
        sorter,
        name,
        params,
        version,
        _job_kwargs,
        execution_params,
        execution_version,
    ) in sorter_default_contents():
        assert version == params["schema_version"], (sorter, name)
        assert execution_version == execution_params["schema_version"], (
            sorter,
            name,
        )


# ---------- SortGroupV2 reference-mode validation ---------------------------


def test_reference_fields_validation():
    """``_validate_reference_fields`` enforces the reference-mode invariants.

    The ``reference_mode`` varchar is typo-guarded against the
    ``ReferenceMode`` Literal, and ``reference_electrode_id`` is non-null
    iff the mode is ``"specific"``. (Validates the helper directly; the
    same helper runs inside ``SortGroupV2.insert1`` / ``insert``.)
    """
    from spyglass.spikesorting.v2.utils import _validate_reference_fields

    # Valid rows.
    _validate_reference_fields({"reference_mode": "none"})
    _validate_reference_fields({"reference_mode": "global_median"})
    _validate_reference_fields(
        {"reference_mode": "specific", "reference_electrode_id": 7}
    )
    # Absent mode defaults to "none".
    _validate_reference_fields({})

    # Typo'd mode is rejected.
    with pytest.raises(ValueError, match="reference_mode"):
        _validate_reference_fields({"reference_mode": "globalmedian"})

    # specific requires an electrode id.
    with pytest.raises(ValueError, match="requires a non-null"):
        _validate_reference_fields({"reference_mode": "specific"})

    # non-specific modes must NOT carry an electrode id.
    with pytest.raises(ValueError, match="must leave"):
        _validate_reference_fields(
            {"reference_mode": "none", "reference_electrode_id": 3}
        )
    with pytest.raises(ValueError, match="must leave"):
        _validate_reference_fields(
            {"reference_mode": "global_median", "reference_electrode_id": 3}
        )


# ---------- schema-version bumps -------------------------------------------


def test_schema_versions_bumped():
    """The schema-version markers reflect the current field sets.

    ``ClusterlessThresholderSchema`` gained ``threshold_unit`` (3 -> 4)
    and ``PreprocessingParamsSchema`` made ``bandpass_filter`` optional +
    flipped the ``whiten`` default (2 -> 3). An un-bumped marker would let
    a stale row validate against the new field set.
    """
    assert ClusterlessThresholderSchema().schema_version == 4
    assert PreprocessingParamsSchema().schema_version == 3


def test_shipped_rows_carry_current_params_schema_version(dj_conn):
    """No shipped default row is left on a stale ``params_schema_version``.

    Each shipped ``SorterParameters`` / ``PreprocessingParameters`` row's
    ``params_schema_version`` must equal the inner schema's
    ``schema_version`` so ``_assert_schema_version_matches`` (run on every
    insert1) stays green. Pins the version-bump bookkeeping for the
    clusterless and preprocessing default rows.
    """
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    PreprocessingParameters.insert_default()

    clusterless = (
        SorterParameters
        & {"sorter": "clusterless_thresholder", "sorter_params_name": "default"}
    ).fetch1()
    assert clusterless["params_schema_version"] == 4
    assert clusterless["params"]["schema_version"] == 4

    for name in (
        "default",
        "franklab_hippocampus_2026_06",
        "franklab_cortex_2026_06",
        "default_neuropixels",
        "no_filter",
    ):
        row = (
            PreprocessingParameters & {"preprocessing_params_name": name}
        ).fetch1()
        assert row["params_schema_version"] == 3, name
        assert row["params"]["schema_version"] == 3, name


# ---------- bulk-insert validation -----------------------------------------

# (module, class, valid_row_a, valid_row_b) for each param Lookup that
# overrides ``insert``. The two valid rows differ only in their primary
# key so a clean bulk insert lands two distinct rows. Table classes are
# imported lazily inside the test because their modules evaluate
# ``dj.schema(...)`` at import time (needs a DB connection).
_BULK_INSERT_CASES = [
    pytest.param(
        "spyglass.spikesorting.v2.sorting",
        "SorterParameters",
        {
            "sorter": "mountainsort4",
            "sorter_params_name": "_pytest_bulk_a",
            "params": MountainSort4Schema().model_dump(),
            "params_schema_version": 1,
        },
        {
            "sorter": "mountainsort4",
            "sorter_params_name": "_pytest_bulk_b",
            "params": MountainSort4Schema().model_dump(),
            "params_schema_version": 1,
        },
        id="SorterParameters",
    ),
    pytest.param(
        "spyglass.spikesorting.v2.artifact",
        "ArtifactDetectionParameters",
        {
            "artifact_detection_params_name": "_pytest_bulk_a",
            "params": ArtifactDetectionParamsSchema().model_dump(),
            "params_schema_version": 2,
        },
        {
            "artifact_detection_params_name": "_pytest_bulk_b",
            "params": ArtifactDetectionParamsSchema().model_dump(),
            "params_schema_version": 2,
        },
        id="ArtifactDetectionParameters",
    ),
    pytest.param(
        "spyglass.spikesorting.v2.session_group",
        "MotionCorrectionParameters",
        {
            "motion_correction_params_name": "_pytest_bulk_a",
            "params": MotionCorrectionParamsSchema().model_dump(),
            "params_schema_version": 1,
        },
        {
            "motion_correction_params_name": "_pytest_bulk_b",
            "params": MotionCorrectionParamsSchema().model_dump(),
            "params_schema_version": 1,
        },
        id="MotionCorrectionParameters",
    ),
    pytest.param(
        "spyglass.spikesorting.v2.recording",
        "PreprocessingParameters",
        {
            "preprocessing_params_name": "_pytest_bulk_a",
            "params": PreprocessingParamsSchema().model_dump(),
            "params_schema_version": 3,
        },
        {
            "preprocessing_params_name": "_pytest_bulk_b",
            "params": PreprocessingParamsSchema().model_dump(),
            "params_schema_version": 3,
        },
        id="PreprocessingParameters",
    ),
]


@pytest.mark.parametrize(
    "module_path,cls_name,valid_a,valid_b", _BULK_INSERT_CASES
)
def test_param_lookup_bulk_insert_validates(
    dj_conn, module_path, cls_name, valid_a, valid_b
):
    """Bulk ``insert`` validates every row, mirroring ``insert1``.

    Before this fix a ``Table.insert([{bad params}, ...])`` bypassed the
    per-table Pydantic validation that ``insert1`` runs, so an invalid
    ``params`` blob reached the DB unchecked. Each of the four param
    Lookups now overrides ``insert`` to map the same ``_validate_params``
    over the batch. Asserts: (1) a batch containing an invalid blob
    raises the same ``ValidationError`` and lands NO row (validation runs
    ahead of the single ``super().insert``); (2) a fully-valid batch
    lands every row. Parametrized across all four Lookups.
    """
    import copy
    import importlib

    Table = getattr(importlib.import_module(module_path), cls_name)
    table = Table()
    pk = table.primary_key

    def restr(row):
        return {k: row[k] for k in pk}

    # Clear any residue from a prior aborted run so the assertions start
    # from a known-empty state.
    (table & [restr(valid_a), restr(valid_b)]).delete_quick()

    # An invalid params blob: a stray key tripped by ``extra="forbid"``.
    invalid = copy.deepcopy(valid_a)
    invalid["params"] = {
        **invalid["params"],
        "definitely_not_a_real_field": 1,
    }

    # (1) A batch with one invalid row raises and commits nothing: the
    #     override validates every row before the single super().insert,
    #     so neither the valid sibling nor the invalid row should land.
    with pytest.raises(ValidationError):
        table.insert([valid_a, valid_b, invalid])
    assert not (table & restr(valid_a)), (
        f"{cls_name}.insert committed a valid row even though the batch "
        "contained an invalid params blob -- validation must run before "
        "super().insert (all-or-nothing)."
    )
    assert not (table & restr(valid_b)), (
        f"{cls_name}.insert committed a valid row even though the batch "
        "contained an invalid params blob -- validation must run before "
        "super().insert (all-or-nothing)."
    )

    # (2) A fully-valid batch lands every row. ``valid_a`` / ``valid_b`` are
    #     intentionally identical content under two names (this test exercises
    #     bulk *validation*, not the duplicate-content guard), so opt out of
    #     the guard explicitly.
    try:
        table.insert([valid_a, valid_b], allow_duplicate_params=True)
        assert len(table & restr(valid_a)) == 1
        assert len(table & restr(valid_b)) == 1
    finally:
        (table & [restr(valid_a), restr(valid_b)]).delete_quick()


# ---------- quality-metric / auto-curation params ---------------------------


from spyglass.spikesorting.v2._params.metric_curation import (  # noqa: E402
    AUTO_CURATION_RULES_SCHEMA_VERSION,
    QUALITY_METRIC_SCHEMA_VERSION,
    AutoCurationRulesSchema,
    QualityMetricParamsSchema,
)


def test_quality_metric_params_non_default_round_trips():
    """A configured quality-metric blob survives dump -> validate -> dump."""
    schema = QualityMetricParamsSchema(
        metric_names=["snr", "isi_violation", "firing_rate", "nn_advanced"],
        metric_kwargs={
            "nn_advanced": {
                "n_components": 7,
                "n_neighbors": 5,
                "max_spikes": 20000,
                "min_spikes": 10,
                "seed": 0,
            }
        },
        skip_pc_metrics=False,
    )
    blob = schema.model_dump()
    assert blob != QualityMetricParamsSchema(metric_names=["snr"]).model_dump()
    rebuilt = QualityMetricParamsSchema.model_validate(blob).model_dump()
    assert rebuilt == blob
    assert rebuilt["schema_version"] == QUALITY_METRIC_SCHEMA_VERSION


def test_quality_metric_params_default_skip_pc_true():
    """``skip_pc_metrics`` defaults to True (PCA metrics off unless asked)."""
    assert QualityMetricParamsSchema(metric_names=["snr"]).skip_pc_metrics


def test_quality_metric_params_skip_pc_false_requires_pca_metric():
    """``skip_pc_metrics=False`` is meaningful only with a PCA metric.

    Without one no whitened metric analyzer would be built, so the flag is a
    contradiction (and recompute/orphan gating treats skip_pc_metrics=False as
    an exact "metric analyzer exists" signal). The schema rejects it; adding a
    PCA metric (e.g. nn_advanced) makes it valid.
    """
    with pytest.raises(ValidationError, match="no PCA metric"):
        QualityMetricParamsSchema(
            metric_names=["snr", "firing_rate"], skip_pc_metrics=False
        )
    assert not QualityMetricParamsSchema(
        metric_names=["snr", "nn_advanced"], skip_pc_metrics=False
    ).skip_pc_metrics
    # Dual contradiction: skip_pc_metrics=True with PCA metric names would
    # store metrics that are intentionally skipped. Reject any such row so
    # metric_names remains the list of metrics this row actually computes.
    with pytest.raises(ValidationError, match="would be skipped"):
        QualityMetricParamsSchema(
            metric_names=["nn_advanced"], skip_pc_metrics=True
        )
    with pytest.raises(ValidationError, match="would be skipped"):
        QualityMetricParamsSchema(
            metric_names=["snr", "nn_advanced"], skip_pc_metrics=True
        )


def test_quality_metric_params_rejects_unknown_metric_name():
    """An unknown metric name fails validation against SI's exported list."""
    with pytest.raises(ValidationError):
        QualityMetricParamsSchema(metric_names=["snr", "not_a_real_metric"])


@pytest.mark.parametrize("old_name", ["nn_isolation", "nn_noise_overlap"])
def test_quality_metric_params_rejects_renamed_nn_metric(old_name):
    """The 0.99 nn metric *names* are rejected; message points at nn_advanced."""
    with pytest.raises(ValidationError) as excinfo:
        QualityMetricParamsSchema(metric_names=[old_name])
    assert "nn_advanced" in str(excinfo.value)


def test_quality_metric_params_rejects_empty_metric_names():
    """At least one metric name is required."""
    with pytest.raises(ValidationError):
        QualityMetricParamsSchema(metric_names=[])


def test_quality_metric_params_rejects_orphan_kwargs_key():
    """metric_kwargs for a metric not in metric_names is a silent no-op."""
    with pytest.raises(ValidationError) as excinfo:
        QualityMetricParamsSchema(
            metric_names=["snr", "firing_rate"],
            metric_kwargs={"isi_violation": {"isi_threshold_ms": 2.0}},
        )
    assert "isi_violation" in str(excinfo.value)


def test_quality_metric_params_accepts_kwargs_for_requested_metric():
    """metric_kwargs keyed by a requested metric validates cleanly."""
    schema = QualityMetricParamsSchema(
        metric_names=["snr", "isi_violation"],
        metric_kwargs={"isi_violation": {"isi_threshold_ms": 2.0}},
    )
    assert schema.metric_kwargs == {"isi_violation": {"isi_threshold_ms": 2.0}}


def test_auto_curation_rules_non_default_round_trips():
    """A configured rules payload (master + rule rows) round-trips."""
    schema = AutoCurationRulesSchema(
        auto_merge_preset="similarity_correlograms",
        auto_merge_kwargs={"resolve_graph": True},
        rules=[
            {
                "rule_index": 0,
                "rule_name": "nn_noise",
                "metric_name": "nn_noise_overlap",
                "operator": ">",
                "threshold": 0.1,
                "label": "noise",
            },
            {
                "rule_index": 1,
                "rule_name": "nn_reject",
                "metric_name": "nn_noise_overlap",
                "operator": ">",
                "threshold": 0.1,
                "label": "reject",
            },
        ],
    )
    blob = schema.model_dump()
    rebuilt = AutoCurationRulesSchema.model_validate(blob).model_dump()
    assert rebuilt == blob
    assert rebuilt["schema_version"] == AUTO_CURATION_RULES_SCHEMA_VERSION
    assert [r["rule_index"] for r in rebuilt["rules"]] == [0, 1]


def test_auto_curation_rules_allows_none_preset_no_rules():
    """``auto_merge_preset='none'`` with no rules is the inert default."""
    schema = AutoCurationRulesSchema(auto_merge_preset="none", rules=[])
    assert schema.auto_merge_preset == "none"
    assert schema.rules == []


def test_auto_curation_rules_rejects_bad_preset():
    """An unknown auto-merge preset is rejected (Literal guard)."""
    with pytest.raises(ValidationError):
        AutoCurationRulesSchema(auto_merge_preset="not_a_preset", rules=[])


def test_auto_curation_rules_rejects_bad_operator():
    """A rule with an unsupported operator is rejected."""
    with pytest.raises(ValidationError):
        AutoCurationRulesSchema(
            auto_merge_preset="none",
            rules=[
                {
                    "rule_index": 0,
                    "rule_name": "snr_noise",
                    "metric_name": "snr",
                    "operator": "=<",
                    "threshold": 2.0,
                    "label": "noise",
                }
            ],
        )


def test_auto_curation_rules_rejects_duplicate_rule_index():
    """Two rules with the same rule_index are rejected (ordering ambiguity)."""
    rule = {
        "rule_name": "snr_noise",
        "metric_name": "snr",
        "operator": "<",
        "threshold": 2.0,
        "label": "noise",
    }
    with pytest.raises(ValidationError):
        AutoCurationRulesSchema(
            auto_merge_preset="none",
            rules=[
                {**rule, "rule_index": 0},
                {**rule, "rule_index": 0},
            ],
        )
