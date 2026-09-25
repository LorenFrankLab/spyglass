"""DB-free tests for motion-estimation parameter resolution and estimation.

Resolution is checked against the installed SpikeInterface's own presets and
function signatures, so a SpikeInterface upgrade that moves a default shows up
here rather than silently changing a stored estimate.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def _resolve(params):
    from spyglass.spikesorting.v2._motion import resolve_estimation_params

    return resolve_estimation_params(params)


def _hash(resolved):
    from spyglass.spikesorting.v2._motion import resolved_params_hash

    return resolved_params_hash(resolved)


# ---- resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    "preset, bin_s",
    [("dredge", 1.0), ("dredge_fast", 1.0), ("rigid_fast", 5.0)],
)
def test_resolution_writes_out_the_time_bin_the_estimator_uses(preset, bin_s):
    """``dredge``/``dredge_fast`` name no ``bin_s``; ``dredge_ap``'s own
    signature default (1.0 s) applies. ``rigid_fast`` sets 5.0 s."""
    from spikeinterface.preprocessing.motion import motion_options_preset
    from spikeinterface.sortingcomponents.motion.dredge import dredge_ap

    resolved = _resolve({"preset": preset})

    assert resolved["estimate_motion_kwargs"]["bin_s"] == bin_s
    if preset != "rigid_fast":
        assert (
            "bin_s"
            not in motion_options_preset[preset]["estimate_motion_kwargs"]
        )
        assert inspect.signature(dredge_ap).parameters["bin_s"].default == bin_s


@pytest.mark.parametrize("preset", ["dredge", "dredge_fast", "rigid_fast"])
def test_resolution_pins_cpu_and_excludes_interpolation(preset):
    resolved = _resolve({"preset": preset})

    assert resolved["estimate_motion_kwargs"]["device"] == "cpu"
    assert set(resolved) == {
        "preset",
        "detect_kwargs",
        "select_kwargs",
        "localize_peaks_kwargs",
        "estimate_motion_kwargs",
        "localization_window_ms",
        "noise_levels_kwargs",
    }
    assert resolved["localization_window_ms"] == {
        "ms_before": 0.1,
        "ms_after": 0.3,
    }


def test_estimate_motion_window_defaults_shadow_dredge_ap_defaults():
    """``estimate_motion`` passes ``win_step_um``/``win_scale_um`` to the
    method explicitly, so its 200/300 um defaults -- not ``dredge_ap``'s
    400/450 -- apply when a preset omits them (``rigid_fast``)."""
    resolved = _resolve({"preset": "rigid_fast"})["estimate_motion_kwargs"]

    assert (resolved["win_step_um"], resolved["win_scale_um"]) == (200.0, 300.0)


@pytest.mark.parametrize("preset", ["dredge", "dredge_fast", "rigid_fast"])
def test_resolution_contains_every_preset_value(preset):
    """Every preset value survives resolution unchanged (as SI would pass it)."""
    from spikeinterface.preprocessing.motion import motion_options_preset

    resolved = _resolve({"preset": preset})
    for step in ("detect_kwargs", "localize_peaks_kwargs"):
        for key, value in motion_options_preset[preset][step].items():
            assert resolved[step][key] == value, (step, key)
    for key, value in motion_options_preset[preset][
        "estimate_motion_kwargs"
    ].items():
        assert resolved["estimate_motion_kwargs"][key] == value, key


def test_override_replaces_one_key_and_keeps_the_rest_of_the_step():
    base = _resolve({"preset": "dredge"})
    resolved = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 6.0}}
    )

    assert resolved["detect_kwargs"]["detect_threshold"] == 6.0
    assert resolved["detect_kwargs"]["radius_um"] == 80.0
    unchanged = dict(resolved["detect_kwargs"], detect_threshold=8.0)
    assert unchanged == base["detect_kwargs"]
    assert _hash(resolved) != _hash(base)


def test_nested_override_replaces_the_whole_value():
    """SI merges one level deep: a dict-valued override is not merged."""
    resolved = _resolve(
        {
            "preset": "dredge_fast",
            "localize_peaks_kwargs": {"weight_method": {"mode": "gaussian_2d"}},
        }
    )

    assert resolved["localize_peaks_kwargs"]["weight_method"] == {
        "mode": "gaussian_2d"
    }


def test_integer_and_float_overrides_resolve_to_one_configuration():
    as_float = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 8.0}}
    )
    as_int = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 8}}
    )

    assert _hash(as_int) == _hash(as_float)
    assert _hash(as_int) == _hash(_resolve({"preset": "dredge"}))


def test_fetched_blob_with_numpy_scalars_resolves_identically():
    blob = {
        "preset": "dredge",
        "detect_kwargs": {"detect_threshold": np.float64(6.0)},
        "noise_levels_seed": np.int64(3),
        "schema_version": np.int64(1),
    }
    plain = {
        "preset": "dredge",
        "detect_kwargs": {"detect_threshold": 6.0},
        "noise_levels_seed": 3,
    }

    assert _hash(_resolve(blob)) == _hash(_resolve(plain))


def test_noise_seed_is_part_of_the_resolved_configuration():
    seeded = _resolve({"preset": "dredge", "noise_levels_seed": 7})

    assert seeded["noise_levels_kwargs"] == {
        "method": "mad",
        "num_chunks_per_segment": 20,
        "chunk_duration": "500ms",
        "seed": 7,
    }
    assert _hash(seeded) != _hash(_resolve({"preset": "dredge"}))


def test_resolved_configuration_is_json_stable():
    import json

    resolved = _resolve({"preset": "dredge_fast"})

    assert json.loads(json.dumps(resolved)) == resolved
    assert resolved["estimate_motion_kwargs"]["post_transform"] == (
        "numpy.log1p"
    )


def test_step_kwargs_are_fresh_copies_with_callables_restored():
    from spyglass.spikesorting.v2._motion import spikeinterface_step_kwargs

    resolved = _resolve(
        {"preset": "dredge", "estimate_motion_kwargs": {"xcorr_kw": {}}}
    )
    _, _, estimate = spikeinterface_step_kwargs(resolved)
    estimate["xcorr_kw"]["max_dt_bins"] = 5

    assert estimate["post_transform"] is np.log1p
    assert resolved["estimate_motion_kwargs"]["xcorr_kw"] == {}


@pytest.mark.parametrize("preset", ["medicine", "nonrigid_accurate", "auto"])
def test_unsupported_preset_is_rejected(preset):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _resolve({"preset": preset})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"detect_kwargs": {"noise_levels": [1.0]}}, "noise_levels"),
        ({"detect_kwargs": {"folder": "x"}}, "folder"),
        ({"estimate_motion_kwargs": {"peaks": []}}, "peaks"),
        ({"estimate_motion_kwargs": {"post_transform": "x"}}, "post_transform"),
        ({"estimate_motion_kwargs": {"device": "cuda"}}, "device"),
        ({"localize_peaks_kwargs": {"prototype": [0.0]}}, "prototype"),
        ({"select_kwargs": {"n_peaks": 10}}, "select_kwargs"),
    ],
)
def test_contract_changing_override_is_rejected(overrides, match):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=match):
        _resolve({"preset": "dredge", **overrides})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"detect_kwargs": {"detect_treshold": 6.0}}, "detect_treshold"),
        ({"estimate_motion_kwargs": {"bin_seconds": 2.0}}, "bin_seconds"),
        ({"detect_kwargs": {"method": "by_channel"}}, "method"),
        (
            {"estimate_motion_kwargs": {"method": "decentralized"}},
            "method",
        ),
    ],
)
def test_unknown_key_or_unsupported_method_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _resolve({"preset": "dredge", **overrides})


def test_default_rows_resolve():
    from spyglass.spikesorting.v2._recipe_catalog import (
        motion_estimation_default_contents,
    )

    rows = motion_estimation_default_contents()

    assert [row[0] for row in rows] == ["dredge_v1", "dredge_fast_v1"]
    presets = [_resolve(row[1])["preset"] for row in rows]
    assert presets == ["dredge", "dredge_fast"]
