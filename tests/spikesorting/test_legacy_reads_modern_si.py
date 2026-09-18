"""v0/v1 waveform *reads* are not gated behind the legacy SpikeInterface env.

The default install pins SpikeInterface 0.104. Waveform *extraction* still
needs the 0.99 runtime and stays gated, but a waveform folder written earlier
is readable under the new pin through
``spyglass.spikesorting._si_compat.load_waveforms``. This module pins that
split: the read paths return usable waveforms, and only the extraction branch
raises the legacy-environment error.

Runs in the main ``run-tests`` job (SI 0.104); under SI 0.99 there is no gate
to exercise, so the tests skip.
"""

from __future__ import annotations

import pytest
from packaging.version import Version


def _skip_under_legacy_si() -> None:
    """Skip when the gate under test cannot fire (SpikeInterface 0.99)."""
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("Modern-SI read-path regression test")


def test_v0_load_waveforms_not_gated(dj_conn, monkeypatch, written_waveforms):
    """v0 ``Waveforms.load_waveforms`` reads a saved folder under SI 0.104."""
    _skip_under_legacy_si()

    from spyglass.spikesorting.v0.spikesorting_curation import Waveforms

    monkeypatch.setattr(
        Waveforms,
        "_get_waveform_path",
        lambda self, key: written_waveforms.path,
    )

    we = Waveforms().load_waveforms({})

    unit_id = written_waveforms.unit_ids[0]
    assert we.get_waveforms(unit_id).shape == (
        written_waveforms.n_spikes_per_unit,
        written_waveforms.n_samples,
        written_waveforms.n_channels,
    )
    assert we.nbefore + we.nafter == written_waveforms.n_samples
