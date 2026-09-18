"""Multiunit event detection never joins samples across unobserved time."""

import numpy as np
import pytest

from spyglass.utils.spikesorting import contiguous_observed_runs

SAMPLING_FREQUENCY = 1000.0


def test_runs_split_on_unobserved_samples_and_timestamp_gaps():
    time = np.arange(10) / SAMPLING_FREQUENCY
    time[6:] += 1.0  # a one-second jump between sample 5 and sample 6
    mask = np.ones(10, dtype=bool)
    mask[2:4] = False

    runs = contiguous_observed_runs(time, mask, SAMPLING_FREQUENCY)

    assert [run.tolist() for run in runs] == [[0, 1], [4, 5], [6, 7, 8, 9]]
    assert all(run.dtype.kind == "i" for run in runs)


def test_runs_are_empty_when_nothing_is_observed():
    time = np.arange(10) / SAMPLING_FREQUENCY

    assert (
        contiguous_observed_runs(
            time, np.zeros(10, dtype=bool), SAMPLING_FREQUENCY
        )
        == []
    )
    assert (
        contiguous_observed_runs(
            np.array([]), np.array([], dtype=bool), SAMPLING_FREQUENCY
        )
        == []
    )


def test_one_sample_gap_tolerance_keeps_a_jittered_run_intact():
    """A sample-to-sample jitter below 1.5 dt is not a recording gap."""
    time = np.arange(6) / SAMPLING_FREQUENCY
    time[3:] += 0.4 / SAMPLING_FREQUENCY  # 1.4 dt step, still contiguous
    mask = np.ones(6, dtype=bool)

    runs = contiguous_observed_runs(time, mask, SAMPLING_FREQUENCY)

    assert [run.tolist() for run in runs] == [[0, 1, 2, 3, 4, 5]]
