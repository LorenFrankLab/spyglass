"""The waveform-folder loader works under either SpikeInterface generation.

``_si_compat.load_waveforms`` is the read path v0 and v1 use for waveform
folders written before the SpikeInterface 0.104 pin. SI 0.99 returns a
``WaveformExtractor``; 0.101+ returns a ``MockWaveformExtractor`` over a
``SortingAnalyzer``. Both expose the ``get_waveforms`` / ``nbefore`` / ``nafter``
surface the callers use, so this module asserts on that surface rather than on
the concrete class.
"""

from __future__ import annotations

import numpy as np
import pytest
from packaging.version import Version

from spyglass.spikesorting import _si_compat


def test_load_waveforms_binary_folder_under_current_si(written_waveforms):
    """A folder written by the installed SI reads back with usable waveforms."""
    we = _si_compat.load_waveforms(written_waveforms.path)

    assert isinstance(we.nbefore, (int, np.integer))
    assert isinstance(we.nafter, (int, np.integer))
    assert we.nbefore + we.nafter == written_waveforms.n_samples

    for unit_id in written_waveforms.unit_ids:
        waveforms = we.get_waveforms(unit_id)
        assert waveforms.shape == (
            written_waveforms.n_spikes_per_unit,
            written_waveforms.n_samples,
            written_waveforms.n_channels,
        )
        assert np.isfinite(waveforms).all()


def test_load_waveforms_zarr_raises_legacy_message(tmp_path):
    """Zarr-format legacy waveforms surface the legacy-environment error."""
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("SpikeInterface 0.99 has no Zarr back-compatibility branch")

    zarr_folder = tmp_path / "waves.zarr"
    zarr_folder.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        _si_compat.load_waveforms(zarr_folder)

    message = str(excinfo.value)
    assert "Zarr" in message
    assert "legacy SpikeInterface 0.99" in message
