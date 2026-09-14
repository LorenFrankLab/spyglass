"""Pydantic schema-default pins for sorter and preprocessing params.

Pure-schema (no DB) regression guards for intentional v2 design choices: MS4
frequency-band and ``detect_threshold`` defaults, clusterless ``peak_sign`` /
stale-field / production-uV-threshold rules, the common-reference operator
knob, and the ``CurationSource`` enum membership. A future refactor "fixing"
these as drift would silently regress to v1 behavior without these pins.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._params.sorter import MountainSort4Schema


def test_ms4_schema_freq_band_defaults():
    """MS4 schema ships ``freq_min=600`` / ``freq_max=6000``.

    These match v1's tetrode preset; the docstring records the choice but no
    test pinned it. A drift back to SI's bare MS4 defaults (no band) would
    silently change the filtered band the sorter sees.
    """
    schema = MountainSort4Schema()
    assert schema.freq_min == 600.0
    assert schema.freq_max == 6000.0


def test_ms4_schema_detect_threshold_float_and_positive():
    """MS4 ``detect_threshold`` is a positive float (int coerced, 0 rejected)."""
    import pydantic

    coerced = MountainSort4Schema(detect_threshold=3)
    assert coerced.detect_threshold == 3.0
    assert isinstance(coerced.detect_threshold, float)
    with pytest.raises(pydantic.ValidationError):
        MountainSort4Schema(detect_threshold=0)  # gt=0 floor


def test_clusterless_schema_peak_sign_accepts_documented_values():
    """The clusterless schema accepts ``neg`` / ``pos`` / ``both`` and
    rejects an unknown peak_sign."""
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    for sign in ("neg", "pos", "both"):
        assert ClusterlessThresholderSchema(peak_sign=sign).peak_sign == sign
    with pytest.raises(pydantic.ValidationError):
        ClusterlessThresholderSchema(peak_sign="unknown")


def test_clusterless_schema_rejects_stale_fields():
    """The clusterless schema forbids v1's stale ``outputs`` /
    ``random_chunk_kwargs`` (extra='forbid')."""
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    for stale in ({"outputs": "sorting"}, {"random_chunk_kwargs": {}}):
        with pytest.raises(pydantic.ValidationError):
            ClusterlessThresholderSchema(**stale)


def test_common_reference_params_operator_knob():
    """``CommonReferenceParams.operator`` accepts both documented values."""
    import pydantic

    from spyglass.spikesorting.v2._params.preprocessing import (
        CommonReferenceParams,
    )

    for op in ("median", "average"):
        assert CommonReferenceParams(operator=op).operator == op
    with pytest.raises(pydantic.ValidationError):
        CommonReferenceParams(operator="rms")


def test_curation_source_enum_members():
    """``CurationSource`` carries all insert-time provenance values.

    Pinning these members guards against a refactor dropping them, which would
    make a valid curation_source raise at the insert boundary.
    """
    from spyglass.spikesorting.v2.utils import CurationSource

    for value in (
        "manual",
        "analyzer_curation",
        "figpack",
        "curation_evaluation",
    ):
        assert CurationSource(value).value == value
    with pytest.raises(ValueError):
        CurationSource("not_a_member")


def test_clusterless_schema_default_is_production_uv():
    """The clusterless schema's bare default is the production/real-data
    threshold (100 uV under the default 'uv' unit), and a microvolt-scale
    threshold explicitly left in MAD units is rejected.
    The OLD default ``(detect_threshold=100, threshold_unit='mad')`` was a
    100x-MAD threshold that silently detected almost nothing.
    """
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import (
        ClusterlessThresholderSchema,
    )

    # Bare default: the real-data 100 uV threshold ('uv' derives
    # noise_levels=[1.0] at runtime), self-consistent with detect_threshold.
    bare = ClusterlessThresholderSchema()
    assert bare.threshold_unit == "uv"
    assert bare.detect_threshold == 100.0

    # A microvolt-scale threshold explicitly left in MAD units (no
    # noise_levels override) is rejected with a helpful message.
    with pytest.raises(pydantic.ValidationError, match="MAD multiplier"):
        ClusterlessThresholderSchema(
            detect_threshold=100.0, threshold_unit="mad"
        )

    # A sane MAD multiplier (the simulation fixture's regime) is accepted.
    mad = ClusterlessThresholderSchema(
        detect_threshold=5.0, threshold_unit="mad"
    )
    assert mad.threshold_unit == "mad" and mad.detect_threshold == 5.0

    # An explicit noise_levels override bypasses the guard even in MAD mode
    # (the documented advanced-override path is deliberately untouched).
    override = ClusterlessThresholderSchema(
        detect_threshold=100.0, threshold_unit="mad", noise_levels=[2.0]
    )
    assert override.noise_levels == [2.0]


# ---------- MountainSort5 wrapper coverage ---------------------------------


def _ms5_wrapper_defaults() -> dict:
    """SI's MS5 algorithm defaults (no global job kwargs)."""
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    return {
        k: v
        for k, v in sis.get_default_sorter_params("mountainsort5").items()
        if k not in job_keys
    }


def test_ms5_schema_covers_wrapper():
    """Every pinned-wrapper MS5 key is a schema field or a managed key.

    A wrapper key that is neither would be a scientific knob users cannot
    set (the pre-launch state for ``scheme2_training_duration_sec`` /
    ``scheme3_block_duration_sec`` / the PCA and mask-radius fields); a
    schema field the wrapper does not accept would fail at sort time.
    """
    from spyglass.spikesorting.v2._params.sorter import (
        MS5_MANAGED_KEYS,
        MountainSort5Schema,
    )

    wrapper = set(_ms5_wrapper_defaults())
    fields = set(MountainSort5Schema.model_fields) - {"schema_version"}
    assert fields & MS5_MANAGED_KEYS == set()
    assert fields | MS5_MANAGED_KEYS == wrapper, {
        "unexposed": sorted(wrapper - fields - MS5_MANAGED_KEYS),
        "not_in_wrapper": sorted(fields - wrapper),
    }


def test_ms5_schema_defaults_match_wrapper_except_filter():
    """Schema defaults equal the wrapper defaults, except the documented
    ``filter=False`` override (the recording stage already bandpasses)."""
    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema

    dump = MountainSort5Schema().model_dump()
    dump.pop("schema_version")
    wrapper = _ms5_wrapper_defaults()
    for key, value in dump.items():
        if key == "filter":
            assert value is False and wrapper[key] is True
            continue
        assert value == wrapper[key], key


def test_ms5_long_recording_knobs_round_trip_and_typo_rejected():
    """Non-default long-recording values are accepted verbatim; typos fail."""
    import pydantic

    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema

    row = MountainSort5Schema(
        scheme="3",
        scheme3_block_duration_sec=300,
        scheme2_training_duration_sec=120,
        scheme2_training_recording_sampling_mode="initial",
        npca_per_channel=5,
        snippet_mask_radius=100,
    ).model_dump()
    assert row["scheme3_block_duration_sec"] == 300.0
    assert row["scheme2_training_duration_sec"] == 120.0
    assert row["scheme2_training_recording_sampling_mode"] == "initial"
    assert row["npca_per_channel"] == 5
    assert row["snippet_mask_radius"] == 100.0
    with pytest.raises(pydantic.ValidationError):
        MountainSort5Schema(scheme3_block_duration_secs=300)
    with pytest.raises(pydantic.ValidationError):
        MountainSort5Schema(delete_temporary_recording=False)
    with pytest.raises(pydantic.ValidationError, match="freq_min"):
        MountainSort5Schema(filter=True, freq_min=6000, freq_max=300)


def test_wrapper_vocabulary_rejects_unknown_key_with_suggestion():
    """Permissive schemas are checked against the installed SI wrapper."""
    from spyglass.spikesorting.v2._params.sorter import (
        sorter_wrapper_vocabulary,
        validate_sorter_params_against_wrapper,
    )

    # SC2 has a permissive schema; its wrapper vocabulary is knowable.
    assert "apply_preprocessing" in sorter_wrapper_vocabulary("spykingcircus2")
    validate_sorter_params_against_wrapper(
        "spykingcircus2", {"apply_preprocessing": False, "schema_version": 1}
    )
    with pytest.raises(ValueError, match="apply_preprocesing.*did you mean"):
        validate_sorter_params_against_wrapper(
            "spykingcircus2", {"apply_preprocesing": False}
        )
    # The clusterless path is not an SI sorter: vocabulary unknown -> no-op.
    assert sorter_wrapper_vocabulary("clusterless_thresholder") is None
    validate_sorter_params_against_wrapper(
        "clusterless_thresholder", {"anything": 1}
    )
