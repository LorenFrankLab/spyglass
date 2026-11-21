"""The waveform-folder loader works under either SpikeInterface generation.

Asserts on the ``get_waveforms`` / ``nbefore`` / ``nafter`` surface, not the
class: SI 0.99 returns a ``WaveformExtractor``, 0.101+ a
``MockWaveformExtractor``.
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
    assert "needs SpikeInterface 0.99" in message
    # Binary-folder waveforms stay readable; only extraction is gated.
    assert "binary-folder waveforms, remain readable" in message


def test_untested_spikeinterface_warns_once(monkeypatch):
    """An SI release outside ``TESTED_SI_VERSIONS`` warns once, not raises."""
    import warnings

    import spikeinterface as si

    monkeypatch.setattr(si, "__version__", "0.105.0")
    _si_compat._warn_if_untested.cache_clear()
    try:
        with pytest.warns(RuntimeWarning, match=r"0\.105\.0"):
            _si_compat.quality_metrics_module()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _si_compat.quality_metrics_module()
    finally:
        _si_compat._warn_if_untested.cache_clear()


def test_tested_spikeinterface_does_not_warn():
    """The installed SI is one of ``TESTED_SI_VERSIONS``."""
    import warnings

    _si_compat._warn_if_untested.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _si_compat._warn_if_untested()


def test_capability_without_a_provider_raises(monkeypatch):
    """No provider present names the targets it looked for."""
    monkeypatch.setitem(
        _si_compat._PROVIDERS,
        "load_extractor",
        (
            ("spikeinterface:no_such_loader", {}),
            ("spikeinterface.no_such_module", {}),
        ),
    )
    with pytest.raises(RuntimeError, match="no_such_loader"):
        _si_compat.load_extractor({})


def test_numpy_sorting_constructor_must_keep_sample_indices(monkeypatch):
    """A constructor that reads the samples as anything but frames raises."""
    real = _si_compat._provider("numpy_sorting_from_samples_and_labels")

    def shifted(samples, labels, sampling_frequency, unit_ids=None):
        return real(
            np.asarray(samples) + 1,
            labels,
            sampling_frequency,
            unit_ids=unit_ids,
        )

    empty = _si_compat.numpy_sorting_from_samples_and_labels(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), 30_000.0
    )
    assert len(empty.to_spike_vector()) == 0
    monkeypatch.setattr(_si_compat, "_provider", lambda capability: shifted)
    with pytest.raises(RuntimeError, match=r"5\.\.10"):
        _si_compat.numpy_sorting_from_samples_and_labels(
            np.array([5, 10]), np.zeros(2, dtype=np.int64), 30_000.0
        )
