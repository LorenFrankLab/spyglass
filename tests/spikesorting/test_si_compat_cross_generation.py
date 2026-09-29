"""Modern SpikeInterface reads extractor folders written by SpikeInterface 0.99.

Existing v0/v1 rows point at recording, sorting and waveform folders that
SpikeInterface 0.99 wrote. The default install pins a newer SpikeInterface,
so ``spyglass.spikesorting._si_compat`` has to read those folders across the
API generation change. These tests load real 0.99-written folders (committed
under ``fixtures/si099``; see its README) and compare what the modern
SpikeInterface reads against what 0.99 itself read back when writing them.

Runs in the main ``run-tests`` job (SpikeInterface 0.104).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si
from packaging.version import Version

if Version(si.__version__) < Version("0.101"):
    pytest.skip(
        "Cross-generation reads need SpikeInterface >= 0.101",
        allow_module_level=True,
    )

SI099_DIR = Path(__file__).parent / "fixtures" / "si099"

# Written with ms_before = ms_after = 1.0 at the minirec sampling rate
# (29959.314 Hz): SpikeInterface computes int(1.0 * fs / 1000) = 29 samples on
# each side of the peak.
N_SPIKES_PER_UNIT = 20
N_SIDE_SAMPLES = 29
N_CHANNELS = 4


@pytest.fixture
def si099(tmp_path):
    """Copy of the committed folders, so a loader cannot write into the repo."""
    copy = tmp_path / "si099"
    shutil.copytree(SI099_DIR, copy)
    return copy


@pytest.fixture
def reference(si099):
    """What SpikeInterface 0.99 read back from the folders it wrote."""
    with np.load(si099 / "reference.npz") as data:
        return {key: data[key] for key in data.files}


def test_load_extractor_reads_si099_recording(si099, reference):
    from spyglass.spikesorting import _si_compat

    recording = _si_compat.load_extractor(si099 / "recording")

    traces = recording.get_traces(return_in_uV=False)
    assert traces.dtype == reference["traces"].dtype == np.int16
    np.testing.assert_array_equal(traces, reference["traces"])
    np.testing.assert_array_equal(
        recording.get_channel_ids(), reference["channel_ids"]
    )


def test_load_extractor_reads_si099_sorting(si099, reference):
    from spyglass.spikesorting import _si_compat

    sorting = _si_compat.load_extractor(si099 / "sorting")

    unit_ids = list(sorting.get_unit_ids())
    assert unit_ids == list(reference["unit_ids"])
    for unit_id in unit_ids:
        expected = reference[f"spike_train_{unit_id}"]
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        assert spike_train.dtype == expected.dtype
        np.testing.assert_array_equal(spike_train, expected)


def test_load_waveforms_reads_si099_waveform_extractor(si099, reference):
    from spyglass.spikesorting import _si_compat

    we = _si_compat.load_waveforms(si099 / "waveforms")

    assert we.nbefore == we.nafter == N_SIDE_SAMPLES
    assert int(reference["nbefore"]) == int(reference["nafter"])
    assert we.nbefore == int(reference["nbefore"])
    assert list(we.unit_ids) == list(reference["unit_ids"])
    for unit_id in we.unit_ids:
        waveforms = we.get_waveforms(unit_id)
        assert waveforms.shape == (
            N_SPIKES_PER_UNIT,
            2 * N_SIDE_SAMPLES,
            N_CHANNELS,
        )
        np.testing.assert_array_equal(
            waveforms, reference[f"waveforms_{unit_id}"]
        )
