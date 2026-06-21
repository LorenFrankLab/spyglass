"""``resolve_peak_sign`` maps a sorter params blob to a SI ``peak_sign``.

Per-unit attribution (``Sorting.Unit`` peak channel + amplitude) must honor
the sorter's configured detection polarity instead of hardcoding ``"neg"``
(v1 makes ``peak_sign`` configurable via the ``peak_channel`` metric params;
v2's sort-time attribution previously fell back to SpikeInterface's ``"neg"``
default, mis-attributing units for positive-going detections).

Sorters express polarity differently: ``clusterless_thresholder`` carries
``peak_sign`` directly ("neg"/"pos"/"both"); MountainSort 4/5 carry
``detect_sign`` (-1/0/1); Kilosort4 / SpykingCircus2 / Tridesclous2 / generic
carry neither and default to "neg". Hermetic -- no DB.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2.utils import resolve_peak_sign


def _pos_neg_analyzer():
    """A 2-channel analyzer with one unit whose template has a small POSITIVE
    peak on channel 0 and a larger NEGATIVE deflection on channel 1.

    The abs-max channel is 1; the positive-peak channel is 0, so ``peak_sign``
    decides which electrode a unit is attributed to. Returns
    ``(analyzer, sorting)``.
    """
    import numpy as np
    import spikeinterface as si
    from probeinterface import Probe

    fs = 30000.0
    n = 3000
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((n, 2)) * 2.0).astype("float32")
    spikes = np.array([500, 1000, 1500, 2000, 2500])
    w = 15
    shape = np.exp(-(np.arange(-w, w + 1) ** 2) / (2 * 5.0**2))
    for f in spikes:
        traces[f - w : f + w + 1, 0] += (50.0 * shape).astype("float32")
        traces[f - w : f + w + 1, 1] += (-100.0 * shape).astype("float32")

    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=[[0.0, 0.0], [0.0, 20.0]],
        shapes="circle",
        shape_params={"radius": 5},
    )
    probe.set_device_channel_indices([0, 1])
    rec = rec.set_probe(probe)
    sort = si.NumpySorting.from_samples_and_labels(
        [spikes], [np.zeros(len(spikes), dtype=int)], fs
    )
    analyzer = si.create_sorting_analyzer(sort, rec, sparse=False)
    analyzer.compute(["random_spikes", "templates"])
    return analyzer, sort


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"peak_sign": "neg"}, "neg"),
        ({"peak_sign": "pos"}, "pos"),
        ({"peak_sign": "both"}, "both"),
        ({"detect_sign": -1}, "neg"),
        ({"detect_sign": 1}, "pos"),
        ({"detect_sign": 0}, "both"),
        ({"detect_threshold": 100.0}, "neg"),  # no sign field -> default
        ({}, "neg"),
    ],
)
def test_resolve_peak_sign(params, expected):
    assert resolve_peak_sign(params) == expected


def test_peak_sign_takes_precedence_over_detect_sign():
    """An explicit ``peak_sign`` wins over ``detect_sign`` (defensive; no
    shipped schema carries both)."""
    assert resolve_peak_sign({"peak_sign": "pos", "detect_sign": -1}) == "pos"


def test_non_mapping_defaults_to_neg():
    """A missing/None params blob falls back to ``"neg"`` rather than
    raising."""
    assert resolve_peak_sign(None) == "neg"


def test_peak_sign_pos_attributes_positive_going_channel():
    """A unit whose template has a POSITIVE peak on one channel and a LARGER
    NEGATIVE deflection on another is attributed to the positive channel
    under ``peak_sign='pos'`` and to the negative channel under SI's default
    ``'neg'``.

    v2 threads the sorter's configured ``peak_sign`` into
    ``get_template_extremum_channel`` (via ``resolve_peak_sign``) so a
    positive-going detection lands on its true peak channel; a revert to SI
    defaults would silently mis-attribute it to the most-negative channel
    (audit test-hardening #9). The existing peak-amplitude test runs on a
    trough-aligned fixture where ``pos == neg``, so this is the only case
    that exercises the configured sign actually changing the answer. Hermetic
    synthetic analyzer -- no DB / full pipeline.
    """
    from spikeinterface.core import template_tools

    analyzer, _ = _pos_neg_analyzer()

    pos_chan = template_tools.get_template_extremum_channel(
        analyzer, peak_sign="pos", outputs="id"
    )
    neg_chan = template_tools.get_template_extremum_channel(
        analyzer, peak_sign="neg", outputs="id"
    )
    # The configured sign genuinely changes the attributed channel.
    assert int(pos_chan[0]) == 0, "pos peak_sign must attribute the +channel"
    assert int(neg_chan[0]) == 1, "neg peak_sign attributes the -channel"
    assert int(pos_chan[0]) != int(neg_chan[0])

    # v2 threads the sorter's configured peak_sign through resolve_peak_sign,
    # so _populate_unit_part uses 'pos' when the sorter params ask for it.
    assert resolve_peak_sign({"peak_sign": "pos"}) == "pos"


@pytest.mark.parametrize(
    "sorter_params, expected_electrode_id",
    [
        ({"peak_sign": "pos"}, 0),  # positive-going channel
        ({"detect_sign": 1}, 0),  # MountainSort positive
        ({"peak_sign": "neg"}, 1),  # negative-going channel
        ({"detect_sign": -1}, 1),  # MountainSort negative
    ],
)
def test_peak_sign_persists_attributed_electrode(
    sorter_params, expected_electrode_id
):
    """The sorter's configured polarity flows into the persisted
    ``Sorting.Unit`` row's ``electrode_id``.

    The previous tests cover the two halves in isolation -- ``resolve_peak_sign``
    (the mapping) and ``get_template_extremum_channel`` (SI picks the channel)
    -- but nothing connected them through the row that is actually stored. This
    drives the exact attribution chain ``_populate_unit_part`` runs
    (``resolve_peak_sign`` -> ``get_template_extremum_channel`` ->
    ``build_sorting_unit_rows``) so a regression that hardcoded ``"neg"`` or fed
    the wrong channel mapping into the row builder would change the persisted
    ``electrode_id``. The synthetic unit peaks +on electrode 0 and -on (larger)
    electrode 1, so the two polarities attribute to DIFFERENT electrodes.
    Hermetic -- no DB.
    """
    from spikeinterface.core import template_tools

    from spyglass.spikesorting.v2._sorting_units import (
        build_sorting_unit_rows,
    )

    analyzer, sorting = _pos_neg_analyzer()

    # Mirror _populate_unit_part's attribution sequence (minus the DB fetches).
    peak_sign = resolve_peak_sign(sorter_params)
    peak_channels = template_tools.get_template_extremum_channel(
        analyzer, peak_sign=peak_sign, outputs="id"
    )
    peak_amplitudes = template_tools.get_template_extremum_amplitude(
        analyzer, peak_sign=peak_sign, mode="extremum"
    )
    unit_ids = list(sorting.unit_ids)
    # The sort group owns both channels; ids match the analyzer's channel ids.
    electrode_by_id = {
        ch: {
            "nwb_file_name": "synthetic.nwb",
            "electrode_group_name": "0",
            "electrode_id": int(ch),
        }
        for ch in (0, 1)
    }

    rows = build_sorting_unit_rows(
        unit_ids=unit_ids,
        peak_channels=peak_channels,
        peak_amplitudes=peak_amplitudes,
        n_spikes_by_unit={
            u: len(sorting.get_unit_spike_train(unit_id=u)) for u in unit_ids
        },
        electrode_by_id=electrode_by_id,
        key={"sorting_id": "synthetic", "curation_id": 0},
        sort_group_id=0,
        nwb_file_name="synthetic.nwb",
    )

    assert len(rows) == 1
    assert rows[0]["electrode_id"] == expected_electrode_id, (
        f"sorter params {sorter_params} resolve to peak_sign {peak_sign!r}, "
        f"which must attribute the unit to electrode {expected_electrode_id}"
    )
