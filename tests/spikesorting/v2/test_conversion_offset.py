"""``resolve_conversion_and_offset`` preserves gain AND offset, rejects bad gain.

v2 writes traces unscaled, so the persisted ElectricalSeries must carry both
``conversion`` (gain) and ``offset`` to recover physical volts on readback
(``volts = raw*conversion + offset``). The prior code wrote only ``conversion``
and dropped the offset -- silently biasing every channel by the DC offset on
readback (real for Intan / Open Ephys) -- and used ``gains[0]`` without a
positivity check (gain==0 -> all-zero recording; negative -> sign flip).

Hermetic -- in-memory NumpyRecording, no DB.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec(gains, offsets):
    import spikeinterface as si

    n_ch = len(gains)
    rec = si.NumpyRecording(
        [np.zeros((100, n_ch), dtype=np.int16)], sampling_frequency=30000.0
    )
    rec.set_channel_gains(list(gains))
    rec.set_channel_offsets(list(offsets))
    return rec


def test_uniform_gain_zero_offset():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    conv, off = resolve_conversion_and_offset(_rec([0.195] * 4, [0.0] * 4))
    assert conv == pytest.approx(0.195e-6)
    assert off == pytest.approx(0.0)


def test_nonzero_offset_is_preserved():
    """The DC offset must survive (volts = raw*gain + offset), not be dropped."""
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    conv, off = resolve_conversion_and_offset(_rec([0.195] * 4, [1000.0] * 4))
    assert conv == pytest.approx(0.195e-6)
    assert off == pytest.approx(1000.0e-6)


def test_heterogeneous_gain_raises():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="heterogeneous channel gains"):
        resolve_conversion_and_offset(_rec([0.195, 0.2, 0.195, 0.2], [0.0] * 4))


@pytest.mark.parametrize("bad_gain", [0.0, -0.195])
def test_nonpositive_gain_raises(bad_gain):
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="non-positive"):
        resolve_conversion_and_offset(_rec([bad_gain] * 4, [0.0] * 4))


def test_heterogeneous_offset_raises():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="offset"):
        resolve_conversion_and_offset(
            _rec([0.195] * 4, [0.0, 1000.0, 0.0, 0.0])
        )
