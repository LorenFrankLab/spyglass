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
