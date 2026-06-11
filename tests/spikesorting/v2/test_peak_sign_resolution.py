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
    import numpy as np
    import spikeinterface as si
    from probeinterface import Probe
    from spikeinterface.core import template_tools

    fs = 30000.0
    n = 3000
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((n, 2)) * 2.0).astype("float32")
    spikes = np.array([500, 1000, 1500, 2000, 2500])
    w = 15
    shape = np.exp(-(np.arange(-w, w + 1) ** 2) / (2 * 5.0**2))
    for f in spikes:
        # Channel 0: small POSITIVE peak (+50). Channel 1: larger NEGATIVE
        # deflection (-100). The abs-max channel is 1; the positive-peak
        # channel is 0 -- peak_sign decides which a unit is attributed to.
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
