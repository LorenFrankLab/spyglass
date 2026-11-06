"""Validation tests for the v2 Pydantic parameter models in ``v2._params``.

Mostly DB-free round-trip and rejection cases; tests of table rows need a DB.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spyglass.spikesorting.v2._params.artifact_detection import (
    ARTIFACT_DETECTION_SCHEMA_VERSION,
    ArtifactDetectionParamsSchema,
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
    """A fully non-default blob survives ``dump -> validate -> dump`` unchanged.

    Every field differs from its default so a mangling validator is caught.
    """
    schema = PreprocessingParamsSchema(
        phase_shift={"margin_ms": 50.0},
        bandpass_filter={"freq_min": 250.0, "freq_max": 5000.0},
        common_reference={"operator": "average"},
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
    assert rebuilt["min_segment_length"] == 0.0015
    assert rebuilt["bad_channel_handling"] == "interpolate"


@pytest.mark.parametrize("bad_min_segment", [-0.5, -1e-6])
def test_preprocessing_rejects_negative_min_segment_length(bad_min_segment):
    """``min_segment_length`` is non-negative (``ge=0``); negatives reject.

    It drops disjoint-interval slivers shorter than this many seconds before
    the sorter.
    """
    with pytest.raises(ValidationError):
        PreprocessingParamsSchema(min_segment_length=bad_min_segment)
    # Zero is the documented "drop nothing" boundary and is accepted.
    zero = PreprocessingParamsSchema(min_segment_length=0.0)
    assert zero.min_segment_length == 0.0


def test_preprocessing_rejects_unknown_bad_channel_handling():
    """``bad_channel_handling`` is ``Literal['remove','interpolate']``.

    A typo (e.g. ``'drop'``) rejects instead of falling back to the default.
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
    """``phase_shift`` defaults to ``None`` (the franklab default is a no-op),
    including for a blob that omits the key."""
    schema = PreprocessingParamsSchema()
    assert schema.phase_shift is None
    assert schema.model_dump()["phase_shift"] is None
    without_phase_shift = {
        "schema_version": 4,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "min_segment_length": 1.0,
    }
    assert (
        PreprocessingParamsSchema.model_validate(
            without_phase_shift
        ).phase_shift
        is None
    )


def test_phase_shift_enabled_dumps_margin():
    """Enabling ``phase_shift`` carries ``margin_ms`` into the dumped params."""
    schema = PreprocessingParamsSchema(phase_shift={"margin_ms": 100.0})
    assert schema.phase_shift is not None
    assert schema.phase_shift.margin_ms == 100.0
    assert schema.model_dump()["phase_shift"] == {"margin_ms": 100.0}
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


def test_preprocessing_rejects_recording_stage_whiten():
    """A ``whiten`` key is rejected: no recording cache ever applied it.

    Whitening belongs to ``SorterParameters`` (MS4/MS5) and
    ``AnalyzerWaveformParameters`` (metric analyzer).
    """
    with pytest.raises(ValidationError, match="whiten"):
        PreprocessingParamsSchema(whiten=None)
    with pytest.raises(ValidationError, match="whiten"):
        PreprocessingParamsSchema(whiten={"dtype": "float32"})
    schema = PreprocessingParamsSchema()
    assert "whiten" not in schema.model_dump()


def test_preprocessing_no_filter_is_none():
    """``bandpass_filter=None`` disables filtering; the default keeps it.

    The dumped params carry ``None`` so the runtime skips the step.
    """
    disabled = PreprocessingParamsSchema(bandpass_filter=None)
    assert disabled.bandpass_filter is None
    assert disabled.model_dump()["bandpass_filter"] is None
    # The default still ships a real bandpass.
    assert PreprocessingParamsSchema().bandpass_filter is not None
    assert (
        PreprocessingParamsSchema().model_dump()["bandpass_filter"] is not None
    )


def test_preprocessing_schema_version_is_four():
    """The preprocessing blob schema is at version 4 (no ``whiten`` field)."""
    assert PREPROCESSING_SCHEMA_VERSION == 4
    assert PreprocessingParamsSchema().schema_version == 4


# ---------- artifact detection ---------------------------------------------


def test_artifact_default_field_values():
    """Defaults use the v2 field names and ship the expected values.

    - ``amplitude_threshold_uv == 500.0``: v1's effective Intan value (v1's
      nominal 3000 is in raw counts).
    - ``proportion_above_threshold == 1.0``: v1 parity (every channel).
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

    ``detect=True`` requires at least one threshold; ``detect=False`` does not.
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
    # both thresholds at once (OR)
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

    ``-1`` is SI's "use all channels" sentinel and ``>= 0`` is a radius. These
    values pass the ``ge=-1.0`` bound, so the match checks
    ``_reject_open_interval_radius``'s message.
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

    - ``filter=False``: the recording stage already bandpass-filters.
    - ``whiten=True``: SI 0.104's MS5 default.

    Both toggle, and ``extra="forbid"`` still rejects an unknown kwarg.
    """
    blob = MountainSort5Schema().model_dump()
    assert blob["filter"] is False
    assert blob["whiten"] is True
    # Both toggles round-trip when set explicitly.
    flipped = MountainSort5Schema(filter=True, whiten=False).model_dump()
    assert flipped["filter"] is True
    assert flipped["whiten"] is False
    # The schema stays strict.
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

    - ``outputs`` and ``random_chunk_kwargs`` are not schema fields (stripped
      at runtime in ``Sorting._run_sorter``).
    - ``noise_levels`` is optional; ``threshold_unit='mad'`` gives
      MAD-multiplier semantics.
    - The shipped ``default`` row passes ``noise_levels=[1.0]`` (uV), as v1's
      ``default_clusterless`` does.
    """
    blob = ClusterlessThresholderSchema().model_dump()
    assert blob["detect_threshold"] == 100.0
    assert blob["method"] == "locally_exclusive"
    assert blob["peak_sign"] == "neg"
    # The runtime-stripped fields must not appear in the validated blob.
    assert "outputs" not in blob
    assert "random_chunk_kwargs" not in blob
    # Schema default is None; with threshold_unit='uv' the runtime derives
    # [1.0].
    assert blob["noise_levels"] is None
    explicit = ClusterlessThresholderSchema(noise_levels=[1.0]).model_dump()
    assert explicit["noise_levels"] == [1.0]


def test_clusterless_threshold_unit_explicit():
    """``threshold_unit`` sets the noise_levels unit explicitly.

    - ``"uv"`` (default): runtime derives noise_levels=[1.0].
    - ``"mad"``: MAD multiplier (the simulation fixture's regime).

    Typos are rejected and the schema is at version 4.
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

    In MAD mode without ``noise_levels``, ``detect_threshold`` is a MAD
    multiplier (typically ~3-15); 100 would detect almost nothing. An explicit
    ``noise_levels`` override bypasses the guard.
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

    - Mappings pass through (shallow-copied).
    - Sequences zip against the heading order and must match its length.
    - A bare ``str``/``bytes`` row (as from iterating a DataFrame or CSV path)
      is rejected.
    """
    from spyglass.spikesorting.v2._core.lookup_validation import (
        _insert_row_to_dict,
    )

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
    # A str/bytes row raises.
    for bad in ("params", b"params"):
        with pytest.raises(TypeError, match="mapping or"):
            _insert_row_to_dict(bad, names)
    # Positional rows must match the heading length (zip would truncate).
    for bad in (
        ("mountainsort4", "name"),
        ("mountainsort4", "name", {"x": 1}, "ignored_extra"),
    ):
        with pytest.raises(ValueError, match="wrong length"):
            _insert_row_to_dict(bad, names)


def test_clusterless_noise_levels_length_guard():
    """The clusterless noise_levels length guard rejects bad explicit arrays.

    An explicit array must be length 1 (broadcast) or ``n_channels``;
    ``None`` is valid (SI estimates per-channel MAD).
    """
    from spyglass.spikesorting.v2._core.signal_math import (
        _assert_noise_levels_length,
    )

    n_channels = 4
    # None and the two valid explicit lengths are accepted (no raise).
    _assert_noise_levels_length(None, n_channels)
    _assert_noise_levels_length([1.0], n_channels)  # broadcast
    _assert_noise_levels_length([1.0, 2.0, 3.0, 4.0], n_channels)  # per-chan
    # A wrong explicit length raises and names the expected lengths.
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0], [1.0] * 5):
        with pytest.raises(ValueError, match="length 1.*or.*n_channels=4"):
            _assert_noise_levels_length(bad, n_channels)
    # Rows written by update1 or legacy code can carry non-list shapes.
    for bad in (1.0, "1234", b"1234"):
        with pytest.raises(ValueError, match="numeric sequence"):
            _assert_noise_levels_length(bad, n_channels)


def test_clusterless_default_row_ships_noise_levels_one(dj_conn):
    """The shipped ``clusterless_thresholder`` / ``default`` row has
    ``noise_levels=[1.0]`` in ``params``, so its threshold is in uV as in v1.
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

    Typos like ``detect_signe`` raise on MS4/MS5/Clusterless. SC2, TDC2 and
    KS4 use ``extra='allow'``, so their typos pass.
    """
    with pytest.raises(ValidationError):
        schema_cls.model_validate({typo_field: 1})


def test_kilosort4_schema_accepts_extra_kwargs():
    """KS4 schema accepts undocumented SI kwargs.

    As in v1, ``extra='allow'`` passes fields like ``batch_size`` through to
    SI, which validates them at sort time.
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

    KS4 whitens internally and has no ``whiten`` kwarg, so any value (even
    ``False``) is rejected despite ``extra="allow"``.
    """
    with pytest.raises(ValidationError, match="whiten"):
        Kilosort4Schema.model_validate({"whiten": bad})


@pytest.mark.parametrize("sorter", ["kilosort2_5", "kilosort3"])
@pytest.mark.parametrize("value", [True, False])
def test_no_whiten_kwarg_sorters_reject_any_whiten(sorter, value):
    """KS2.5 / KS3 whiten internally and have no ``whiten`` parameter, so the
    SI-wrapper key check rejects the key at insert whatever its value."""
    from spyglass.spikesorting.v2._params.sorter import (
        validate_sorter_params_against_wrapper,
    )

    with pytest.raises(ValueError, match="'whiten'"):
        validate_sorter_params_against_wrapper(sorter, {"whiten": value})


def test_ironclust_accepts_whiten():
    """IronClust has its own ``whiten`` option (channel whitening during its
    preprocessing); the dispatcher passes it through unchanged."""
    from spyglass.spikesorting.v2._params.sorter import (
        validate_sorter_params_against_wrapper,
    )

    validate_sorter_params_against_wrapper("ironclust", {"whiten": True})


# ---------- quality-metric template (waveform-shape) columns ----------------


def test_quality_metric_schema_defaults_template_columns():
    """A row that omits ``template_metric_columns`` gets ``trough_half_width``.

    ``trough_half_width`` is an SI output column, used for cell typing.
    Post-trough columns (``peak_to_trough_duration``, slopes) clip on the
    narrow hippocampus window, so they are opt-in.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        DEFAULT_TEMPLATE_METRIC_COLUMNS,
        QualityMetricParamsSchema,
    )

    schema = QualityMetricParamsSchema(metric_names=["snr"])
    assert schema.template_metric_columns == ["trough_half_width"]
    assert DEFAULT_TEMPLATE_METRIC_COLUMNS == ["trough_half_width"]
    # Post-trough columns are opt-in, never shipped defaults.
    assert "peak_to_trough_duration" not in schema.template_metric_columns
    assert "recovery_slope" not in schema.template_metric_columns


def test_template_metric_columns_validated():
    """Unknown column raises; a metric *name* raises with the column hint; an
    empty list is accepted (a row that surfaces no shape columns).

    The vocabulary is SI output columns, not metric names; a typo or a metric
    name (``half_width``) fails at insert.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        QualityMetricParamsSchema,
        _available_template_metric_columns,
    )

    with pytest.raises(ValidationError, match="Unknown template metric column"):
        QualityMetricParamsSchema(
            metric_names=["snr"], template_metric_columns=["not_a_real_column"]
        )

    # ``half_width`` is a metric NAME, not a column; its columns are
    # trough_half_width / peak_half_width. Passing it raises with that hint.
    with pytest.raises(ValidationError, match="OUTPUT COLUMN names"):
        QualityMetricParamsSchema(
            metric_names=["snr"], template_metric_columns=["half_width"]
        )

    empty = QualityMetricParamsSchema(
        metric_names=["snr"], template_metric_columns=[]
    )
    assert empty.template_metric_columns == []

    # The validator's vocabulary is the installed SI's single-channel output
    # columns: includes ``trough_half_width`` but never the ``half_width`` name.
    available = _available_template_metric_columns()
    assert "trough_half_width" in available
    assert "peak_to_trough_duration" in available
    assert "half_width" not in available


@pytest.mark.parametrize(
    "schema_cls",
    [SpykingCircus2Schema, Tridesclous2Schema],
)
def test_uncurated_sorter_schemas_accept_arbitrary_kwargs(schema_cls):
    """SC2 and TDC2 keep ``extra='allow'``; SI validates kwargs at sort time."""
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

    The Lookup row stores it alongside the blob; a breaking change bumps it.
    """
    blob = schema_cls().model_dump()
    assert isinstance(blob["schema_version"], int)
    assert blob["schema_version"] >= 1


def _all_params_schemas():
    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        AnalyzerWaveformParamsSchema,
    )
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2._params.metric_curation import (
        AutoCurationRulesSchema,
        QualityMetricParamsSchema,
    )
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2._params.motion_interpolation import (
        MotionInterpolationParamsSchema,
    )
    from spyglass.spikesorting.v2._params.sorter import (
        SorterExecutionParamsSchema,
    )

    return [
        PreprocessingParamsSchema,
        ArtifactDetectionParamsSchema,
        MountainSort4Schema,
        MountainSort5Schema,
        Kilosort4Schema,
        ClusterlessThresholderSchema,
        SpykingCircus2Schema,
        Tridesclous2Schema,
        GenericSorterParamsSchema,
        SorterExecutionParamsSchema,
        AnalyzerWaveformParamsSchema,
        QualityMetricParamsSchema,
        AutoCurationRulesSchema,
        MotionEstimationParamsSchema,
        MotionInterpolationParamsSchema,
        UnitMatchParamsSchema,
    ]


@pytest.mark.parametrize(
    "schema_cls", _all_params_schemas(), ids=lambda cls: cls.__name__
)
@pytest.mark.parametrize("offset", [-1, 1])
def test_schema_accepts_only_its_current_version(schema_cls, offset):
    """A blob stamped with any other ``schema_version`` fails validation.

    Validation is per field, so the ``schema_version`` error is reported even
    for a schema with other required fields.
    """
    current = schema_cls.model_fields["schema_version"].default
    with pytest.raises(ValidationError) as excinfo:
        schema_cls.model_validate({"schema_version": current + offset})
    assert ("schema_version",) in [err["loc"] for err in excinfo.value.errors()]


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


def test_recipe_catalog_rows_copy_inner_schema_version():
    """Recipe rows derive the outer version from the validated params blob."""
    from spyglass.spikesorting.v2._core.recipe_catalog import (
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
    iff the mode is ``"specific"``. ``SortGroupV2.insert1`` / ``insert`` call
    this helper.
    """
    from spyglass.spikesorting.v2._core.reference_resolution import (
        _validate_reference_fields,
    )

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
    """The schema-version markers reflect the current field sets."""
    assert ClusterlessThresholderSchema().schema_version == 4
    assert PreprocessingParamsSchema().schema_version == 4


def test_shipped_rows_carry_current_params_schema_version(dj_conn):
    """No shipped default row is left on a stale ``params_schema_version``.

    Each shipped ``SorterParameters`` / ``PreprocessingParameters`` row's
    ``params_schema_version`` must equal the inner schema's
    ``schema_version``, or ``_assert_schema_version_matches`` rejects it.
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
        assert row["params_schema_version"] == 4, name
        assert row["params"]["schema_version"] == 4, name


# ---------- bulk-insert validation -----------------------------------------

# (module, class, valid_row_a, valid_row_b) for each param Lookup that
# overrides ``insert``; the two rows differ only in primary key. Imported
# lazily because the modules declare ``dj.schema`` at import.
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
        "spyglass.spikesorting.v2.recording",
        "PreprocessingParameters",
        {
            "preprocessing_params_name": "_pytest_bulk_a",
            "params": PreprocessingParamsSchema().model_dump(),
            "params_schema_version": PREPROCESSING_SCHEMA_VERSION,
        },
        {
            "preprocessing_params_name": "_pytest_bulk_b",
            "params": PreprocessingParamsSchema().model_dump(),
            "params_schema_version": PREPROCESSING_SCHEMA_VERSION,
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
    """Bulk ``insert`` validates every row, as ``insert1`` does.

    A batch with one invalid blob raises and lands no row; a valid batch
    lands every row.
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

    # (1) One invalid row: raises and nothing lands.
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

    # (2) A valid batch lands every row. The rows share content, so opt out
    #     of the duplicate-content guard.
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
    AutoCurationRuleSchema,
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

    Recompute and orphan gating read skip_pc_metrics=False as "a metric
    analyzer exists", which needs a PCA metric (e.g. nn_advanced).
    """
    with pytest.raises(ValidationError, match="no PCA metric"):
        QualityMetricParamsSchema(
            metric_names=["snr", "firing_rate"], skip_pc_metrics=False
        )
    assert not QualityMetricParamsSchema(
        metric_names=["snr", "nn_advanced"], skip_pc_metrics=False
    ).skip_pc_metrics
    # skip_pc_metrics=True with PCA metric names is rejected, so metric_names
    # lists only metrics the row computes.
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


def test_auto_curation_rule_missing_policy_validates_and_defaults():
    """Missing-value policy is persisted in normalized rule payloads."""
    base = {
        "rule_index": 0,
        "rule_name": "snr_noise",
        "metric_name": "snr",
        "operator": "<",
        "threshold": 2.0,
        "label": "noise",
    }
    assert AutoCurationRuleSchema(**base).missing_policy == "error"
    for policy in ("error", "fail", "pass"):
        dumped = AutoCurationRuleSchema(
            **base, missing_policy=policy
        ).model_dump()
        assert dumped["missing_policy"] == policy
    for retired in ("skip", "ignore"):
        with pytest.raises(ValidationError):
            AutoCurationRuleSchema(**base, missing_policy=retired)


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


# ---- metric extension-dependency resolution (drift et al.) -----------------

_BASE_EXTENSIONS = frozenset(
    {"random_spikes", "noise_levels", "templates", "waveforms"}
)


@pytest.mark.unit
def test_required_extensions_drift_needs_spike_locations():
    """``drift`` pulls in ``spike_locations`` (read from SI, not hardcoded).

    Otherwise SI's ``compute_quality_metrics`` silently skips ``drift``.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        required_extensions_for_metrics,
    )

    assert required_extensions_for_metrics(["drift"], _BASE_EXTENSIONS) == [
        "spike_locations"
    ]


@pytest.mark.unit
def test_required_extensions_present_dep_is_not_recomputed():
    """A metric whose dependency is already present needs nothing extra."""
    from spyglass.spikesorting.v2._params.metric_curation import (
        required_extensions_for_metrics,
    )

    # snr depends on noise_levels + templates, both base -> nothing extra.
    assert required_extensions_for_metrics(["snr"], _BASE_EXTENSIONS) == []


@pytest.mark.unit
def test_required_extensions_either_or_dep_satisfied_without_spurious_compute():
    """An ``a|b`` dependency is satisfied by either option already present.

    ``amplitude_cutoff`` needs ``spike_amplitudes|amplitude_scalings``. With
    ``spike_amplitudes`` present nothing is added; with neither, the first
    option is requested.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        required_extensions_for_metrics,
    )

    with_amps = _BASE_EXTENSIONS | {"spike_amplitudes"}
    assert (
        required_extensions_for_metrics(["amplitude_cutoff"], with_amps) == []
    )
    assert required_extensions_for_metrics(
        ["amplitude_cutoff"], _BASE_EXTENSIONS
    ) == ["spike_amplitudes"]
