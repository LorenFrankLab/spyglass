"""v1 clusterless amplitudes from the real ``UnitWaveformFeatures.populate``.

Runs the v1 chain on the minirec session (recording -> no-op artifact
detection -> ``clusterless_thresholder`` sort -> ``CurationV1`` -> merge ->
``UnitWaveformFeatures``) under SpikeInterface 0.99, where the v0/v1
``WaveformExtractor`` path lives, and checks each stored per-spike amplitude
against the recording traces.

What v1 stores as "amplitude" (feature row ``amplitude`` with
``estimate_peak_time=False``, forced for ``clusterless_thresholder``):

* ``UnitWaveformFeatures._fetch_waveform`` calls ``si.extract_waveforms`` on
  ``SpikeSortingOutput.get_recording`` -- the filtered, referenced v1
  ``SpikeSortingRecording`` NWB series, not whitened -- with SI's default
  ``return_scaled=True``, so waveforms are ``gain * raw + offset``. The
  minirec series is stored with gain 1 and offset 0, so this fixture cannot
  tell scaled from unscaled samples.
* ``_get_peak_amplitude`` returns ``waveforms[:, nbefore]``: every channel at
  the spike frame, sign preserved (``peak_sign`` only matters when the peak
  time is estimated).
* SI 0.99 cuts ``nbefore = int(ms_before * fs / 1000)`` samples before the
  spike and leaves spikes whose window crosses the recording edge as zeros.
* Spike frames come from the stored spike times via ``np.searchsorted``
  against the recording timestamps (``spike_times_to_valid_samples``).

So the oracle is ``recording.get_traces(return_scaled=True)[frame, :]`` for
interior spikes and zeros for edge spikes.

Only the legacy spike-sorting job (SpikeInterface 0.99) collects
``tests/spikesorting/v1``.
"""

import numpy as np
import pytest
import spikeinterface as si
from packaging.version import Version

if Version(si.__version__) >= Version("0.101"):
    pytest.skip(
        "v1 UnitWaveformFeatures extraction needs the SpikeInterface 0.99 "
        f"legacy runtime (have {si.__version__})",
        allow_module_level=True,
    )

_SORTER_PARAMS = {
    "sorter": "clusterless_thresholder",
    "sorter_param_name": "low_thresh",
}
_FEATURES_PARAM = {"features_param_name": "low_thresh_amplitude"}
_MS_BEFORE = 0.2
_MS_AFTER = 0.2


@pytest.fixture(scope="module")
def waveform_features(spike_v1):
    from spyglass.decoding.v1 import waveform_features

    yield waveform_features


@pytest.fixture(scope="module")
def clusterless_merge_id(spike_v1, spike_merge, mini_dict, pop_rec):
    """Merge id of one v1 clusterless curation on the minirec recording."""
    recording_id = pop_rec["recording_id"]

    artifact_key = {"recording_id": recording_id, "artifact_param_name": "none"}
    spike_v1.ArtifactDetectionSelection.insert_selection(artifact_key)
    spike_v1.ArtifactDetection.populate(artifact_key)
    artifact_id = (spike_v1.ArtifactDetectionSelection & artifact_key).fetch1(
        "artifact_id"
    )

    # Low threshold so the short minirec yields spikes (the default finds none)
    spike_v1.SpikeSorterParameters.insert1(
        {
            **_SORTER_PARAMS,
            "sorter_params": {
                "detect_threshold": 10.0,
                # One unit per detected spike
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 1000,
                # Noise level 1.0 keeps the threshold in uV rather than MAD
                "noise_levels": np.asarray([1.0]),
                "random_chunk_kwargs": {},
                "outputs": "sorting",
            },
        },
        skip_duplicates=True,
    )
    sort_key = {
        **_SORTER_PARAMS,
        **mini_dict,
        "recording_id": recording_id,
        "interval_list_name": str(artifact_id),
    }
    spike_v1.SpikeSortingSelection.insert_selection(sort_key)
    sorting_id = (spike_v1.SpikeSortingSelection & sort_key).fetch1(
        "sorting_id"
    )
    spike_v1.SpikeSorting.populate({"sorting_id": sorting_id})

    curation_query = spike_v1.CurationV1 & {"sorting_id": sorting_id}
    if not curation_query:
        spike_v1.CurationV1.insert_curation(sorting_id=sorting_id)
    curation_key = curation_query.fetch("KEY", order_by="curation_id")[0]
    spike_merge.insert(
        [curation_key], part_name="CurationV1", skip_duplicates=True
    )

    yield (spike_merge.CurationV1 & curation_key).fetch1("merge_id")


@pytest.fixture(scope="module")
def features_key(waveform_features, clusterless_merge_id):
    """Populate ``UnitWaveformFeatures`` for the clusterless merge id."""
    waveform_features.WaveformFeaturesParams.insert1(
        {
            **_FEATURES_PARAM,
            "params": {
                "waveform_extraction_params": {
                    "ms_before": _MS_BEFORE,
                    "ms_after": _MS_AFTER,
                    "max_spikes_per_unit": None,
                    "n_jobs": 1,
                    "total_memory": "1G",
                },
                "waveform_features_params": {
                    "amplitude": {
                        "peak_sign": "neg",
                        "estimate_peak_time": False,
                    }
                },
            },
        },
        skip_duplicates=True,
    )
    key = {"spikesorting_merge_id": clusterless_merge_id, **_FEATURES_PARAM}
    waveform_features.UnitWaveformFeaturesSelection.insert1(
        key, skip_duplicates=True
    )
    waveform_features.UnitWaveformFeatures.populate(key)

    yield key


@pytest.mark.slow
def test_unit_waveform_features_amplitude_oracle(
    waveform_features, spike_merge, features_key
):
    """Each stored amplitude is the scaled trace at its spike frame.

    A shifted frame, a reordered or wrong channel, or amplitudes misaligned
    with the stored spike times all fail.
    """
    features = waveform_features.UnitWaveformFeatures & features_key
    assert len(features) == 1, "UnitWaveformFeatures.populate made no row"
    units = features.fetch_nwb()[0]["object_id"]
    assert "amplitude" in units.columns, "amplitude column missing"

    recording = spike_merge.get_recording(
        {"merge_id": features_key["spikesorting_merge_id"]}
    )
    # make() concatenates multi-segment recordings; the frame lookup below
    # indexes one segment directly.
    assert recording.get_num_segments() == 1
    traces = recording.get_traces(return_scaled=True)
    times = recording.get_times()
    n_samples, n_channels = traces.shape
    fs = recording.get_sampling_frequency()
    nbefore = int(_MS_BEFORE * fs / 1000.0)
    nafter = int(_MS_AFTER * fs / 1000.0)

    n_checked = 0
    n_interior = 0
    for unit_id, unit in units.iterrows():
        spike_times = np.asarray(unit["spike_times"])
        amplitudes = np.asarray(unit["amplitude"])
        if spike_times.size == 0:
            continue
        assert amplitudes.shape == (spike_times.size, n_channels), (
            f"unit {unit_id}: amplitude shape {amplitudes.shape} is not "
            f"(n_spikes={spike_times.size}, n_channels={n_channels})"
        )
        frames = np.searchsorted(times, spike_times)
        assert np.all(
            frames < n_samples
        ), f"unit {unit_id}: stored spike times run past the recording"
        interior = (frames >= nbefore) & (frames < n_samples - nafter)
        np.testing.assert_allclose(
            amplitudes[interior],
            traces[frames[interior], :],
            rtol=1e-3,
            atol=1e-2,
            err_msg=(
                f"unit {unit_id}: stored amplitude is not the scaled "
                "recording trace at the spike frame"
            ),
        )
        # SI 0.99 leaves spikes whose window crosses the edge unextracted
        # (zeros in the waveform buffer).
        assert np.all(amplitudes[~interior] == 0.0), (
            f"unit {unit_id}: edge spikes at frames "
            f"{frames[~interior].tolist()} should hold zero amplitude"
        )
        n_checked += spike_times.size
        n_interior += int(interior.sum())

    assert n_checked > 0, "clusterless sort produced no spikes to check"
    assert n_interior >= 0.8 * n_checked, (
        f"too few interior spikes ({n_interior}/{n_checked}) to validate "
        "the amplitude oracle"
    )
