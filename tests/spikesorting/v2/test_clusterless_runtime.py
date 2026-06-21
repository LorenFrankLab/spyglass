"""Clusterless thresholder runtime behavior and shipped default row.

Covers the singleton ``noise_levels`` broadcast, the shipped
``clusterless_thresholder/default`` row contents, the sort-time MAD-footgun
guard, and the defensive stripping of stale ``outputs`` /
``random_chunk_kwargs`` fields from a params row that bypassed validation.
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_singleton_noise_levels_broadcast(monkeypatch):
    """A singleton ``noise_levels=[1.0]`` is broadcast to length
    ``n_channels`` before ``detect_peaks``.

    SI's ``locally_exclusive`` indexes ``noise_levels[chan] *
    detect_threshold`` per channel, so a length-1 array would read only
    channel 0 (the v1 multi-channel clusterless bug). The runtime broadcasts
    the scalar to one value per channel. We capture the ``method_kwargs``
    that reach ``detect_peaks`` and assert the length + fill.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    n_channels = 6
    rec = si.generate_recording(
        num_channels=n_channels,
        durations=[1.0],
        sampling_frequency=30_000.0,
    )
    # threshold_unit="uv" now scales the recording to uV (scale_to_uV),
    # which requires channel gains/offsets to be set.
    rec.set_channel_gains([1.0] * n_channels)
    rec.set_channel_offsets([0.0] * n_channels)

    captured = {}

    def _capture_detect(recording, *, method, method_kwargs, job_kwargs):
        captured["noise_levels"] = method_kwargs.get("noise_levels")
        # Return an empty detection so the wrapper builds an empty sorting.
        return np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(pd_mod, "detect_peaks", _capture_detect)
    # The runtime imports detect_peaks at call time from this module path.
    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks",
        _capture_detect,
    )

    Sorting._run_clusterless_thresholder(
        sorter_params={"noise_levels": [1.0], "threshold_unit": "uv"},
        recording=rec,
        job_kwargs=None,
    )

    nl = captured["noise_levels"]
    assert nl is not None
    nl = np.asarray(nl)
    assert nl.shape == (n_channels,), (
        f"singleton noise_levels must broadcast to n_channels={n_channels}, "
        f"got shape {nl.shape}"
    )
    assert np.allclose(nl, 1.0)


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_default_row_ships_noise_levels_one():
    """The shipped ``clusterless_thresholder/default`` row carries
    ``noise_levels=[1.0]``.

    Regression guard for the 1,400x divergence bug: v1 read detect_threshold
    in raw uV (noise_levels=[1.0]); a drift to None would reinterpret it as a
    MAD multiplier.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    params = (
        SorterParameters
        & {"sorter": "clusterless_thresholder", "sorter_params_name": "default"}
    ).fetch1("params")
    assert params["noise_levels"] == [1.0]
    # The shipped uv row carries detect_threshold=100 (the production
    # clusterless threshold) via the schema default, paired with its explicit
    # threshold_unit='uv'.
    assert params["detect_threshold"] == 100.0


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_runtime_rejects_bypassed_mad_footgun():
    """Defense-in-depth at SORT time: a clusterless params blob that bypassed
    the SorterParameters insert validator (written via ``update1`` or
    predating the validator) is still caught -- a microvolt-scale
    detect_threshold left in MAD units with no noise_levels raises rather
    than running a silent ~zero-detection sort. The runtime consumes the
    fetched blob without re-validating the schema, so the guard must live in
    ``_run_clusterless_thresholder`` too.
    """
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    # The footgun combo: detect_threshold=100 in MAD units, no noise_levels.
    with pytest.raises(ValueError, match="MAD multiplier"):
        Sorting._run_clusterless_thresholder(
            sorter_params={"detect_threshold": 100.0, "threshold_unit": "mad"},
            recording=rec,
            job_kwargs=None,
        )


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_runtime_strips_stale_fields(monkeypatch):
    """The clusterless runtime defensively strips ``outputs`` /
    ``random_chunk_kwargs`` from a params row that slipped past the Pydantic
    gate (e.g. a raw ``dj.insert1``).

    ``detect_peaks`` rejects those keys; the runtime pops them before the call
    so a stale row does not crash the sort. We capture ``method_kwargs`` and
    assert the stale keys are gone.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    # threshold_unit="uv" requires the recording to carry channel gains
    # (it scales to uV); generate_recording has none. Unity gain/offset make
    # the uv scaling a no-op, so this test exercises field-stripping (its
    # actual purpose) rather than the gains precondition.
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    captured = {}

    def _capture_detect(recording, *, method, method_kwargs, job_kwargs):
        captured["method_kwargs"] = dict(method_kwargs)
        return np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks",
        _capture_detect,
    )
    monkeypatch.setattr(pd_mod, "detect_peaks", _capture_detect)

    # Stale fields that the Pydantic schema would forbid, planted as if a raw
    # insert bypassed validation.
    Sorting._run_clusterless_thresholder(
        sorter_params={
            "detect_threshold": 100.0,
            "threshold_unit": "uv",
            "outputs": "sorting",
            "random_chunk_kwargs": {"num_chunks_per_segment": 5},
        },
        recording=rec,
        job_kwargs=None,
    )
    mk = captured["method_kwargs"]
    assert "outputs" not in mk
    assert "random_chunk_kwargs" not in mk
