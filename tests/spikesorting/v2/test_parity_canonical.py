"""Unit tests for ``_parity_canonical`` invariant-fingerprint helpers.

The helpers normalize v1 and v2 spike-sorting param payloads into a
common canonical form so the v1↔v2 parity test can assert "same
SI-effective inputs" without tripping on schema-only divergences
(legacy keys v1 carries, ``schema_version`` v2 stamps, nested-vs-flat
shape differences). Tested in isolation -- no DataJoint, no Spyglass
imports -- so a schema regression surfaces here before the slower
integration suite runs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.spikesorting.v2._parity_canonical import (
    _normalize,
    assert_canonical_dict_equal,
    canonical_artifact,
    canonical_preproc,
    canonical_sorter,
)

# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_numpy_float_to_python_float():
    out = _normalize(np.float64(1.5))
    assert out == 1.5
    assert type(out) is float


def test_normalize_numpy_int_to_python_int():
    out = _normalize(np.int64(42))
    assert out == 42
    assert type(out) is int


def test_normalize_tuple_to_list():
    assert _normalize((1, 2, 3)) == [1, 2, 3]


def test_normalize_nested_tuple_to_list():
    assert _normalize((1, (2, 3))) == [1, [2, 3]]


def test_normalize_numpy_array_records_dtype_and_shape():
    arr = np.array([1.0, 2.0, 3.0], dtype="<f8")
    out = _normalize(arr)
    assert out == {
        "_array_data": [1.0, 2.0, 3.0],
        "_array_meta": {"dtype": "float64", "shape": [3]},
    }


def test_normalize_numpy_2d_array_shape_preserved():
    arr = np.zeros((2, 3), dtype="float32")
    out = _normalize(arr)
    assert out["_array_meta"] == {"dtype": "float32", "shape": [2, 3]}


def test_normalize_recurses_into_dict():
    inp = {"a": np.float64(1.5), "b": (1, 2), "c": {"d": np.int64(3)}}
    assert _normalize(inp) == {"a": 1.5, "b": [1, 2], "c": {"d": 3}}


def test_normalize_passes_through_plain_python_types():
    assert _normalize("foo") == "foo"
    assert _normalize(None) is None
    assert _normalize(True) is True
    assert _normalize(1.5) == 1.5


# ---------------------------------------------------------------------------
# assert_canonical_dict_equal
# ---------------------------------------------------------------------------


def test_assert_equal_identical_dicts_passes():
    assert_canonical_dict_equal({"a": 1, "b": "x"}, {"a": 1, "b": "x"})


def test_assert_equal_close_floats_pass():
    # 1e-12 relative drift is below the default rel_tol=1e-9.
    assert_canonical_dict_equal({"x": 1.0}, {"x": 1.0 + 1e-12})


def test_assert_equal_differing_floats_fail_with_path():
    with pytest.raises(AssertionError) as excinfo:
        assert_canonical_dict_equal({"a": {"b": 1.0}}, {"a": {"b": 2.0}})
    msg = str(excinfo.value)
    assert "root.a.b" in msg
    assert "1.0" in msg and "2.0" in msg


def test_assert_equal_missing_key_fails_with_path():
    with pytest.raises(AssertionError) as excinfo:
        assert_canonical_dict_equal({"a": 1}, {"a": 1, "b": 2})
    msg = str(excinfo.value)
    # Either side missing a key surfaces the offending path.
    assert "b" in msg


def test_assert_equal_different_types_fail():
    with pytest.raises(AssertionError) as excinfo:
        assert_canonical_dict_equal({"a": 1}, {"a": "1"})
    msg = str(excinfo.value)
    assert "root.a" in msg


def test_assert_equal_lists_compared_elementwise():
    assert_canonical_dict_equal({"x": [1, 2, 3]}, {"x": [1, 2, 3]})
    with pytest.raises(AssertionError) as excinfo:
        assert_canonical_dict_equal({"x": [1, 2, 3]}, {"x": [1, 2, 4]})
    assert "root.x[2]" in str(excinfo.value)


def test_assert_equal_none_compared_strictly():
    assert_canonical_dict_equal({"x": None}, {"x": None})
    with pytest.raises(AssertionError):
        assert_canonical_dict_equal({"x": None}, {"x": 0})


# ---------------------------------------------------------------------------
# canonical_sorter -- clusterless_thresholder
# ---------------------------------------------------------------------------


SMOKE_CLUSTERLESS_PARAMS = {
    "detect_threshold": 5.0,
    "method": "locally_exclusive",
    "peak_sign": "neg",
    "exclude_sweep_ms": 0.1,
    "local_radius_um": 100.0,
}


def test_canonical_sorter_clusterless_drops_v1_legacy_keys():
    v1_params = {
        **SMOKE_CLUSTERLESS_PARAMS,
        "outputs": "sorting",
        "random_chunk_kwargs": {},
    }
    assert canonical_sorter("clusterless_thresholder", v1_params) == (
        SMOKE_CLUSTERLESS_PARAMS
    )


def test_canonical_sorter_clusterless_v1_and_v2_match():
    v1_params = {
        **SMOKE_CLUSTERLESS_PARAMS,
        "outputs": "sorting",
        "random_chunk_kwargs": {},
    }
    v2_params = dict(SMOKE_CLUSTERLESS_PARAMS)
    assert canonical_sorter(
        "clusterless_thresholder", v1_params
    ) == canonical_sorter("clusterless_thresholder", v2_params)


def test_canonical_sorter_clusterless_noise_levels_none_equals_absent():
    """``noise_levels=None`` means "auto-estimate"; absent means the same.

    The Pydantic-schema default for ``noise_levels`` was ``None``
    after followup #11 closed the 1,400x divergence bug; the v1 row
    omits the field entirely. Canonical form treats them as
    equivalent so the fingerprint check does not regress that fix.
    """
    with_none = {**SMOKE_CLUSTERLESS_PARAMS, "noise_levels": None}
    without = dict(SMOKE_CLUSTERLESS_PARAMS)
    assert canonical_sorter(
        "clusterless_thresholder", with_none
    ) == canonical_sorter("clusterless_thresholder", without)


def test_canonical_sorter_clusterless_explicit_noise_levels_preserved():
    """A non-None ``noise_levels`` is a real config choice and stays.

    Numpy arrays normalize to the ``_array_meta`` sidecar form so
    the fingerprint round-trips through JSON.
    """
    params = {
        **SMOKE_CLUSTERLESS_PARAMS,
        "noise_levels": np.array([1.0, 1.0, 1.0]),
    }
    out = canonical_sorter("clusterless_thresholder", params)
    assert "noise_levels" in out
    assert out["noise_levels"]["_array_meta"] == {
        "dtype": "float64",
        "shape": [3],
    }


# ---------------------------------------------------------------------------
# canonical_sorter -- mountainsort4
# ---------------------------------------------------------------------------


MS4_DEFAULT_PARAMS = {
    "detect_sign": -1,
    "adjacency_radius": 100.0,
    "freq_min": 300.0,
    "freq_max": 6000.0,
    "filter": False,
    "whiten": True,
    "clip_size": 50,
    "detect_threshold": 3.0,
    "detect_interval": 10,
}


def test_canonical_sorter_ms4_strips_outputs_if_present():
    """Mirror clusterless: v1 ``SpikeSorterParameters`` may also carry
    ``outputs`` for MS4 rows. Drop on both sides to keep the canonical
    fingerprint stable."""
    v1_params = {**MS4_DEFAULT_PARAMS, "outputs": "sorting"}
    v2_params = dict(MS4_DEFAULT_PARAMS)
    assert canonical_sorter("mountainsort4", v1_params) == canonical_sorter(
        "mountainsort4", v2_params
    )


def test_canonical_sorter_unknown_sorter_keeps_payload_normalized():
    """Defensive: unknown sorters get pure normalization, no key drops.

    Lets the helper degrade gracefully when a future sorter is added
    to the parity matrix before the canonical rules are extended.
    """
    payload = {"foo": np.float64(1.5), "bar": (1, 2)}
    out = canonical_sorter("kilosort4", payload)
    assert out == {"foo": 1.5, "bar": [1, 2]}


# ---------------------------------------------------------------------------
# canonical_preproc
# ---------------------------------------------------------------------------


def test_canonical_preproc_v1_flat_matches_v2_nested_when_equivalent():
    """v1 ships ``"default"`` flat; v2 ships ``"default"`` nested.

    Both encode the same SI-effective preprocessing (300-6000 Hz bandpass +
    common reference median, 1s min segment length). The canonical
    form must collapse them to identical dicts.
    """
    v1_flat = {
        "frequency_min": 300,
        "frequency_max": 6000,
        "margin_ms": 5,
        "seed": 0,
        "min_segment_length": 1,
    }
    v2_nested = {
        "schema_version": 2,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "whiten": {"dtype": "float32"},
        "min_segment_length": 1.0,
    }
    assert canonical_preproc(v1_flat) == canonical_preproc(v2_nested)


def test_canonical_preproc_freq_min_mismatch_diverges():
    v1_flat = {
        "frequency_min": 300,
        "frequency_max": 6000,
        "margin_ms": 5,
        "seed": 0,
        "min_segment_length": 1,
    }
    v2_nested = {
        "schema_version": 2,
        "bandpass_filter": {"freq_min": 500.0, "freq_max": 6000.0},
        "common_reference": {"operator": "median"},
        "whiten": {"dtype": "float32"},
        "min_segment_length": 1.0,
    }
    with pytest.raises(AssertionError):
        assert_canonical_dict_equal(
            canonical_preproc(v1_flat), canonical_preproc(v2_nested)
        )


def test_canonical_preproc_reference_operator_carries_through():
    """v2-only ``operator="average"`` is NOT a silent equivalence -- it
    must surface as a canonical divergence so the parity test fails."""
    v1_flat = {
        "frequency_min": 300,
        "frequency_max": 6000,
        "min_segment_length": 1,
    }
    v2_avg = {
        "schema_version": 2,
        "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
        "common_reference": {"operator": "average"},
        "min_segment_length": 1.0,
    }
    with pytest.raises(AssertionError):
        assert_canonical_dict_equal(
            canonical_preproc(v1_flat), canonical_preproc(v2_avg)
        )


# ---------------------------------------------------------------------------
# canonical_artifact
# ---------------------------------------------------------------------------


def test_canonical_artifact_strips_v1_job_kwargs():
    """``chunk_duration``, ``n_jobs``, ``progress_bar`` are runtime
    concurrency knobs that v1 stamped into the params blob (now
    factored out to the v2 ``job_kwargs`` row). They are not
    scientific params and must be stripped before comparison."""
    v1_with_job_kwargs = {
        "zscore_thresh": None,
        "amplitude_thresh_uV": 500,
        "proportion_above_thresh": 1.0,
        "removal_window_ms": 1.0,
        "chunk_duration": "10s",
        "n_jobs": 4,
        "progress_bar": "True",
    }
    v2_clean = {
        "schema_version": 2,
        "detect": True,
        "zscore_threshold": None,
        "amplitude_threshold_uv": 500.0,
        "proportion_above_threshold": 1.0,
        "removal_window_ms": 1.0,
        "join_window_ms": 1.0,
        "min_length_s": 1.0,
    }
    # v1 lacks the v2-only ``join_window_ms`` and ``min_length_s``;
    # canonical_artifact treats absent as the v2 default so the two
    # sides agree on the "v1 default + v2 schema defaults" reading.
    assert canonical_artifact(v1_with_job_kwargs) == canonical_artifact(
        v2_clean
    )


def test_canonical_artifact_amplitude_threshold_mismatch_diverges():
    v1 = {
        "zscore_thresh": None,
        "amplitude_thresh_uV": 3000,
        "proportion_above_thresh": 1.0,
        "removal_window_ms": 1.0,
    }
    v2 = {
        "schema_version": 2,
        "detect": True,
        "zscore_threshold": None,
        "amplitude_threshold_uv": 500.0,
        "proportion_above_threshold": 1.0,
        "removal_window_ms": 1.0,
        "join_window_ms": 1.0,
        "min_length_s": 1.0,
    }
    with pytest.raises(AssertionError):
        assert_canonical_dict_equal(
            canonical_artifact(v1), canonical_artifact(v2)
        )


def test_canonical_artifact_none_preset_disables_detect():
    """v1's ``"none"`` preset has ``amplitude_thresh_uV=None``; v2 uses
    ``detect=False``. Both mean "don't detect artifacts" and must
    canonicalize to the same form."""
    v1_none = {
        "zscore_thresh": None,
        "amplitude_thresh_uV": None,
        "chunk_duration": "10s",
        "n_jobs": 4,
        "progress_bar": "True",
    }
    v2_none = {
        "schema_version": 2,
        "detect": False,
        "zscore_threshold": None,
        "amplitude_threshold_uv": None,
        "proportion_above_threshold": 1.0,
        "removal_window_ms": 1.0,
        "join_window_ms": 1.0,
        "min_length_s": 1.0,
    }
    assert canonical_artifact(v1_none) == canonical_artifact(v2_none)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_canonical_sorter_form():
    v1_params = {
        **SMOKE_CLUSTERLESS_PARAMS,
        "outputs": "sorting",
        "random_chunk_kwargs": {},
    }
    canonical = canonical_sorter("clusterless_thresholder", v1_params)
    # JSON serializes + deserializes; both sides of the parity check
    # would write/read the metadata via JSON.
    roundtripped = json.loads(json.dumps(canonical, sort_keys=True))
    assert_canonical_dict_equal(canonical, roundtripped)


def test_json_roundtrip_preserves_numpy_array_canonical_form():
    payload = {"x": np.array([1.0, 2.0, 3.0], dtype="float64")}
    canonical = _normalize(payload)
    roundtripped = json.loads(json.dumps(canonical, sort_keys=True))
    assert_canonical_dict_equal(canonical, roundtripped)


def test_int_keyed_dict_via_string_serialization():
    """Int-keyed dicts (``bad_channel_by_electrode_id``) must
    serialize as JSON strings and re-read to int via the documented
    int-key contract -- canonical comparison must match the int-keyed
    form on both sides."""
    int_keyed = {42: False, 43: True, 100: False}
    json_form = {str(k): v for k, v in int_keyed.items()}
    serialized = json.dumps(json_form, sort_keys=True)
    deserialized = json.loads(serialized)
    reconstructed = {int(k): v for k, v in deserialized.items()}
    assert reconstructed == int_keyed
