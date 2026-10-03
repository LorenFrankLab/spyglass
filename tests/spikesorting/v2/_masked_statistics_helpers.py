"""Masked-recording builders for the valid-sample statistics tests.

The sort stage silences artifact frames with zeros, and every
noise/whitening estimator must then draw only from the retained frames.
These helpers build a CLEAN ground-truth recording, a masked twin whose
excluded frame ranges first carry large transients (so any excluded sample
that leaked into an estimate would move it) and are then silenced exactly
as production does, and the statistics spans of the retained frames.

Targets are always computed from the clean twin, never from the recording
that carries transients. DB-free.
"""

from __future__ import annotations

import numpy as np

SAMPLING_FREQUENCY = 30_000.0
DURATION_S = 60.0
N_CHANNELS = 16
TRANSIENT_UV = 800.0
N_EXCLUDED_BLOCKS = 12
# MAD -> Gaussian standard deviation; the constant SpikeInterface's own
# ``_noise_level_chunk`` divides by.
MAD_TO_STD = 0.6744897501960817


def clean_ground_truth():
    """Materialize the clean ground-truth recording once.

    Returns
    -------
    traces : numpy.ndarray
        ``(n_samples, n_channels)`` float32 traces in microvolts (the
        generated recording has gain 1 uV/count, offset 0).
    probe : probeinterface.Probe
        The generated 2D probe.
    sorting : spikeinterface.BaseSorting
        The ground-truth sorting (10 units).
    """
    import spikeinterface as si

    recording, sorting = si.generate_ground_truth_recording(
        num_channels=N_CHANNELS,
        durations=[DURATION_S],
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=0,
    )
    return recording.get_traces(), recording.get_probe(), sorting


def numpy_recording(traces, probe):
    """In-memory recording over ``traces`` with the probe and unit gains."""
    from spikeinterface.core import NumpyRecording

    recording = NumpyRecording([traces], sampling_frequency=SAMPLING_FREQUENCY)
    recording = recording.set_probe(probe)
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    return recording


def excluded_ranges(n_samples: int, fraction: float, *, seed: int = 0):
    """One excluded half-open range per equal block, ``fraction`` of each.

    Twelve blocks keep the excluded ranges spread across the recording, so
    the statistics spans are many and interleaved rather than one long
    retained stretch.
    """
    block = n_samples // N_EXCLUDED_BLOCKS
    length = int(round(fraction * block))
    rng = np.random.default_rng(seed)
    starts = [
        i * block + int(rng.integers(0, block - length + 1))
        for i in range(N_EXCLUDED_BLOCKS)
    ]
    return [(s, s + length) for s in starts]


def with_transients(traces, ranges, *, seed: int = 0):
    """Copy of ``traces`` with +/-800 uV pulses inside ``ranges`` only.

    1.5 ms pulses every 2 ms, an independent random sign per pulse and
    channel. Frames outside ``ranges`` are untouched.
    """
    out = traces.copy()
    rng = np.random.default_rng(seed)
    width = int(1.5e-3 * SAMPLING_FREQUENCY)
    step = int(2e-3 * SAMPLING_FREQUENCY)
    for start, end in ranges:
        offset = np.arange(end - start)
        in_pulse = offset % step < width
        pulse_id = offset // step
        signs = rng.choice([-1.0, 1.0], size=(pulse_id[-1] + 1, N_CHANNELS))
        rows = start + offset[in_pulse]
        out[rows] += (TRANSIENT_UV * signs[pulse_id[in_pulse]]).astype(
            out.dtype
        )
    return out


def masked_twin(traces, probe, fraction: float, *, seed: int = 0):
    """Masked recording, its statistics spans, and its excluded ranges.

    Transients are injected inside the excluded ranges, which are then
    silenced with the production ``silence_frame_ranges``.
    """
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
        statistics_spans,
    )

    n_samples = traces.shape[0]
    ranges = excluded_ranges(n_samples, fraction, seed=seed)
    recording = numpy_recording(
        with_transients(traces, ranges, seed=seed), probe
    )
    masked = silence_frame_ranges(recording, ranges)
    spans = statistics_spans(n_samples, ranges, [(0, n_samples)])
    return masked, spans, ranges


def exact_mad(traces):
    """Per-channel MAD noise over every row of ``traces``."""
    median = np.median(traces, axis=0, keepdims=True)
    return np.median(np.abs(traces - median), axis=0) / MAD_TO_STD
