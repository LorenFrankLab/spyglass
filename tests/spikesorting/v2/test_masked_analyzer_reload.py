"""Nonempty masks survive probe projection, analyzer reload and derivation."""

import subprocess
import sys

import numpy as np
import pytest
import spikeinterface as si


@pytest.mark.parametrize("concatenated", [False, True])
@pytest.mark.parametrize("whiten", [False, True])
def test_masked_analyzer_fresh_process_and_derivative(
    tmp_path, concatenated, whiten
):
    from spyglass.spikesorting.v2._analyzer_cache import (
        load_analyzer_extensions,
        load_analyzer_folder,
    )
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
    )

    recording, sorting = si.generate_ground_truth_recording(
        durations=[2.0], num_channels=4, num_units=2, seed=42
    )
    recording = recording.save(folder=tmp_path / "source", n_jobs=1)
    masked = silence_frame_ranges(recording, [(3000, 6000)])
    if concatenated:
        masked = si.concatenate_recordings([masked, masked])
        sorting = si.concatenate_sortings(
            [sorting, sorting],
            total_samples_list=[recording.get_num_samples()] * 2,
        )
    # This clone, used for the pipeline's 3D geometry, drops SI's JSON flag.
    masked = masked.set_probe(masked.get_probe().to_3d(axes="xy"))
    folder = tmp_path / "analyzer"
    build_analyzer(
        sorting,
        masked,
        {"sorting_id": "masked-reload"},
        sorter_row={"sorter": "mountainsort5", "params": {}},
        job_kwargs={"n_jobs": 1, "random_seed": 0},
        analyzer_folder=folder,
        waveform_params={
            "ms_before": 1.0,
            "ms_after": 2.0,
            "max_spikes_per_unit": 100,
            "whiten": whiten,
            "purpose": "metric" if whiten else "display",
            "sparsity": {"method": "dense"},
        },
    )
    analyzer = load_analyzer_folder(folder)
    traces = analyzer.recording.get_traces()
    np.testing.assert_array_equal(traces[3000:6000], 0)
    if concatenated:
        start = recording.get_num_samples() + 3000
        np.testing.assert_array_equal(traces[start : start + 3000], 0)
    expected = tmp_path / "traces.npy"
    np.save(expected, traces)
    # Selecting units exercises SI's independent derivative save path too.
    derivative = tmp_path / "derivative"
    load_analyzer_extensions(analyzer)
    analyzer.select_units(
        analyzer.unit_ids[:1], format="binary_folder", folder=derivative
    )
    script = """
import sys
import numpy as np
import spikeinterface as si
for folder in sys.argv[1:3]:
    analyzer = si.load_sorting_analyzer(folder)
    assert analyzer.has_recording()
    np.testing.assert_array_equal(analyzer.recording.get_traces(), np.load(sys.argv[3]))
    analyzer.compute('spike_amplitudes', n_jobs=1, progress_bar=False)
    assert np.isfinite(analyzer.get_extension('spike_amplitudes').get_data()).all()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(folder),
            str(derivative),
            str(expected),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recordingless_cache_is_invalid(tmp_path):
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder

    recording, sorting = si.generate_ground_truth_recording(
        durations=[1.0], num_channels=4, num_units=2, seed=0
    )
    folder = tmp_path / "analyzer"
    si.create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=folder, sparse=False
    )
    (folder / "recording.json").unlink(missing_ok=True)
    (folder / "recording.pickle").unlink(missing_ok=True)
    with pytest.raises(ValueError, match="recording could not be loaded"):
        load_analyzer_folder(folder)
