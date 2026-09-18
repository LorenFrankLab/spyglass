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

from pathlib import Path

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


class _FakeMetricCurationQuery:
    """Stand-in for ``(MetricCurationSelection & key) * WaveformParameters``.

    ``get_waveforms`` needs exactly the surface implemented here on the
    cached-read path: a length of one and ``fetch1("waveform_params")``.
    """

    def __init__(self, waveform_params: dict):
        self._waveform_params = waveform_params

    def __mul__(self, other):  # ``... * WaveformParameters``
        return self

    def __len__(self):
        return 1

    def fetch1(self, attribute):
        assert attribute == "waveform_params"
        return dict(self._waveform_params)


class _FakeMetricCurationSelection:
    """Stand-in for the ``MetricCurationSelection`` table in ``& key``."""

    def __init__(self, query: _FakeMetricCurationQuery):
        self._query = query

    def __and__(self, key):
        return self._query


def test_v1_get_waveforms_cached_read_not_gated(
    dj_conn, monkeypatch, written_waveforms
):
    """The cached read loads waveforms; only extraction hits the legacy gate."""
    _skip_under_legacy_si()

    from spyglass.spikesorting.v1 import metric_curation
    from spyglass.spikesorting.v1.metric_curation import MetricCuration

    folder = Path(written_waveforms.path)
    monkeypatch.setattr(metric_curation, "temp_dir", str(folder.parent))
    monkeypatch.setattr(
        metric_curation,
        "MetricCurationSelection",
        _FakeMetricCurationSelection(
            _FakeMetricCurationQuery({"sparse": False})
        ),
    )

    def _must_not_be_called(sort_key):
        raise AssertionError("must not be called")

    for name in ("get_recording", "get_sorting"):
        monkeypatch.setattr(
            metric_curation.CurationV1,
            name,
            staticmethod(_must_not_be_called),
        )

    key = {"metric_curation_id": folder.name}

    MetricCuration._waves_cache.clear()
    waveforms = MetricCuration().get_waveforms(key, overwrite=False)

    unit_id = written_waveforms.unit_ids[0]
    assert waveforms.get_waveforms(unit_id).shape == (
        written_waveforms.n_spikes_per_unit,
        written_waveforms.n_samples,
        written_waveforms.n_channels,
    )

    MetricCuration._waves_cache.clear()
    with pytest.raises(RuntimeError, match="extraction"):
        MetricCuration().get_waveforms(key, overwrite=True)
    MetricCuration._waves_cache.clear()
