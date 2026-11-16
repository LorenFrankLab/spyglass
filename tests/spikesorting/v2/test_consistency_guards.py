"""Consistency guards refuse inputs that differ from what they describe.

Each test starts from a matching input, checks it passes, then changes one
field and expects the guard to raise naming that field.
"""

from __future__ import annotations

import numpy as np
import pytest

_FS = 1000.0


def _recording(n_samples=2000, channel_ids=(3, 7)):
    import spikeinterface as si

    recording = si.NumpyRecording(
        np.zeros((n_samples, len(channel_ids)), dtype=np.float32),
        sampling_frequency=_FS,
        channel_ids=list(channel_ids),
    )
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    return recording


def _estimate():
    return {
        "n_samples": 2000,
        "sampling_frequency": _FS,
        "channel_ids": [3, 7],
        "channel_locations": [[0.0, 0.0], [0.0, 20.0]],
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("n_samples", 1999),
        ("sampling_frequency", 999.0),
        ("channel_ids", [3, 8]),
    ],
)
def test_motion_estimate_refuses_a_source_with_other_frames_rate_or_channels(
    field, value
):
    from spyglass.spikesorting.v2._motion.compute import (
        assert_source_matches_estimate,
    )

    recording = _recording()
    assert_source_matches_estimate({}, recording, _estimate(), 2000, _FS)
    estimate = {**_estimate(), field: value}
    with pytest.raises(ValueError, match=rf"\['{field}'\] differ"):
        assert_source_matches_estimate({}, recording, estimate, 2000, _FS)


def test_motion_estimate_refuses_a_source_with_other_contact_positions():
    from spyglass.spikesorting.v2._motion.compute import (
        assert_geometry_matches_estimate,
    )

    recording = _recording()
    assert_geometry_matches_estimate({}, recording, _estimate())
    estimate = {**_estimate(), "channel_locations": [[0.0, 0.0], [0.0, 25.0]]}
    with pytest.raises(ValueError, match="contact positions"):
        assert_geometry_matches_estimate({}, recording, estimate)


@pytest.fixture(scope="module")
def recipe_analyzer():
    import spikeinterface as si

    sorting = si.NumpySorting.from_unit_dict(
        {1: np.array([200, 600, 1000, 1400])}, sampling_frequency=_FS
    )
    analyzer = si.create_sorting_analyzer(
        sorting, _recording(), sparse=False, format="memory"
    )
    analyzer.compute("random_spikes", max_spikes_per_unit=50, seed=0)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    return analyzer


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("ms_before", 1.5, "waveforms.ms_before"),
        ("ms_after", 3.0, "waveforms.ms_after"),
        ("max_spikes_per_unit", 40, "random_spikes.max_spikes_per_unit"),
        ("whiten", True, "whitening"),
    ],
)
def test_cached_analyzer_refuses_a_recipe_with_other_content(
    recipe_analyzer, field, value, match
):
    from spyglass.spikesorting.v2._sorting.analyzer import (
        assert_analyzer_matches_recipe,
    )

    recipe = {
        "ms_before": 1.0,
        "ms_after": 2.0,
        "max_spikes_per_unit": 50,
        "whiten": False,
    }
    assert_analyzer_matches_recipe(recipe_analyzer, recipe)
    with pytest.raises(ValueError, match=match):
        assert_analyzer_matches_recipe(
            recipe_analyzer, {**recipe, field: value}
        )


@pytest.mark.parametrize(
    "times, frames, match",
    [
        (
            {1: [0.01, 0.02], 2: [0.03]},
            {1: [10, 20], 4: [30]},
            "matching unit IDs",
        ),
        (
            {1: [0.01, 0.02], 2: [0.03]},
            {1: [10], 2: [30]},
            "unit_id=1: spike_times and spike_sample_index must have "
            "matching lengths",
        ),
    ],
)
def test_lazy_merge_refuses_times_and_frames_that_disagree(
    times, frames, match
):
    from spyglass.spikesorting.v2._storage.units_nwb import (
        build_lazy_merged_sorting_from_samples,
    )

    good_frames = {1: [10, 20], 2: [30]}
    merged = build_lazy_merged_sorting_from_samples(
        times, good_frames, [[1, 2]], _FS, delta_s=0.4e-3
    )
    assert merged.get_unit_spike_train(3).tolist() == [10, 20, 30]
    with pytest.raises(ValueError, match=match):
        build_lazy_merged_sorting_from_samples(
            times, frames, [[1, 2]], _FS, delta_s=0.4e-3
        )


_SORTING_ID = "0123abcd-0000-0000-0000-000000000000"


@pytest.mark.parametrize(
    "name",
    [
        f"{_SORTING_ID}__display_waveforms.zarr",  # suffix
        f"{_SORTING_ID}_display_waveforms.analyzer",  # separator
        "__display_waveforms.analyzer",  # sorting id
        f"{_SORTING_ID}__display-waveforms.analyzer",  # payload
    ],
)
def test_analyzer_publication_refuses_a_non_canonical_folder(tmp_path, name):
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        _sorting_id_from_analyzer_folder,
        publish_analyzer_atomically,
    )

    canonical = tmp_path / f"{_SORTING_ID}__display_waveforms.analyzer"
    assert _sorting_id_from_analyzer_folder(canonical) == _SORTING_ID
    built = []
    with pytest.raises(ValueError, match="canonical analyzer-cache"):
        publish_analyzer_atomically(tmp_path / name, built.append)
    assert built == []
    assert list(tmp_path.iterdir()) == []
