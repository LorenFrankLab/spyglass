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
