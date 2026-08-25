"""Regression tests for GitHub issue #1273.

``_write_waveform_features_to_nwb`` pairs each unit id taken from the
waveform extractor's sorting with a row of the ``spike_times`` frame. After
curation those two id sets are no longer interchangeable: the curated NWB
units table holds only the accepted (or post-merge) units, so its index is
neither contiguous nor zero-based, while the sorting still reports every
unit it was built from. The pairing must therefore be done by *label*, never
by position.
"""

import numpy as np
import pandas as pd
import pynwb
import pytest


class _StubSorting:
    """Minimal stand-in for ``si.BaseSorting``.

    Parameters
    ----------
    unit_ids : list of int
        Unit ids the stub reports from ``get_unit_ids``.
    """

    def __init__(self, unit_ids):
        self._unit_ids = np.asarray(unit_ids, dtype=np.int64)

    def get_unit_ids(self) -> np.ndarray:
        """Return the sorting's unit ids."""
        return self._unit_ids


class _StubWaveformExtractor:
    """Minimal stand-in for ``si.WaveformExtractor``.

    ``_write_waveform_features_to_nwb`` only reads ``waveforms.sorting``, so
    a stub avoids running a real waveform extraction for this test.

    Parameters
    ----------
    unit_ids : list of int
        Unit ids the wrapped sorting reports.
    """

    def __init__(self, unit_ids):
        self.sorting = _StubSorting(unit_ids)


def _spike_times_series(unit_ids) -> pd.Series:
    """Build a spike-times series indexed by unit id.

    Mirrors ``fetch_nwb(...)[key]["spike_times"]``, which is a column of
    ``pynwb.misc.Units.to_dataframe()`` and so is indexed by the NWB ``id``
    column, i.e. the unit ids.

    Parameters
    ----------
    unit_ids : list of int
        Unit ids to use as the index. Each unit's spike times are made
        distinguishable by offsetting them with the unit id.

    Returns
    -------
    pandas.Series
        Spike times, one array per unit, indexed by unit id.
    """
    return pd.Series(
        [
            np.arange(5, dtype=np.float64) + 100.0 * unit_id
            for unit_id in unit_ids
        ],
        index=pd.Index(list(unit_ids), name="id"),
        name="spike_times",
    )


def _features(unit_ids) -> dict:
    """Build a waveform-features dict keyed by unit id.

    Parameters
    ----------
    unit_ids : list of int
        Unit ids to generate features for.

    Returns
    -------
    dict
        ``{"amplitude": {unit_id: array of shape (5, 1)}}``.
    """
    return {
        "amplitude": {
            int(unit_id): np.full((5, 1), unit_id, dtype=np.float32)
            for unit_id in unit_ids
        }
    }


@pytest.fixture(scope="module")
def write_features_func():
    """The function under test, imported after the test DB is up."""
    from spyglass.decoding.v1.waveform_features import (
        _write_waveform_features_to_nwb,
    )

    yield _write_waveform_features_to_nwb


@pytest.fixture(scope="module")
def read_written_units(mini_copy_name):
    """Return a helper that runs the writer and reads the result back."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    def _run(write_func, waveforms, spike_times, features):
        analysis_file_name, object_id = write_func(
            mini_copy_name, waveforms, spike_times, features
        )
        AnalysisNwbfile().add(mini_copy_name, analysis_file_name)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            units = io.read().objects[object_id].to_dataframe()
        return units

    yield _run


def test_curated_units_written_by_label(
    write_features_func, read_written_units
):
    """Units missing from the curated frame must not shift the pairing.

    The sorting reports every unit it was built from (0-7, ascending from
    zero), but curation dropped all but ids 1, 3, 4 and 7 from the NWB units
    table. Before the fix, the write loop walked the sorting's ids and did
    ``spike_times.loc[0]``, which raises ``KeyError``.
    """
    curated_ids = [1, 3, 4, 7]
    spike_times = _spike_times_series(curated_ids)
    waveforms = _StubWaveformExtractor(list(range(8)))

    units = read_written_units(
        write_features_func, waveforms, spike_times, _features(curated_ids)
    )

    assert list(units.index) == curated_ids, (
        "Written units must be exactly the curated units present in the "
        f"spike-times frame, got {list(units.index)}"
    )
    for unit_id in curated_ids:
        assert np.array_equal(
            np.asarray(units.loc[unit_id, "spike_times"]),
            spike_times.loc[unit_id],
        ), f"Unit {unit_id} was written with another unit's spike times"
        assert np.array_equal(
            np.asarray(units.loc[unit_id, "amplitude"]),
            _features(curated_ids)["amplitude"][unit_id],
        ), f"Unit {unit_id} was written with another unit's features"


def test_noncontiguous_ids_pair_by_label(
    write_features_func, read_written_units
):
    """Non-contiguous ids shared by sorting and frame must stay paired.

    This is the ``apply_merge=True`` shape: merged units are removed and the
    replacement takes ``max(id) + 1``, so both the sorting and the units
    frame carry ids ``[2, 3, 4, 5]``. Positional (``iloc``) indexing would
    read row 2 for unit 2 and silently write unit 4's spike times under id
    2; label indexing keeps them together.
    """
    unit_ids = [2, 3, 4, 5]
    spike_times = _spike_times_series(unit_ids)
    waveforms = _StubWaveformExtractor(unit_ids)

    units = read_written_units(
        write_features_func, waveforms, spike_times, _features(unit_ids)
    )

    assert list(units.index) == unit_ids
    for unit_id in unit_ids:
        assert np.array_equal(
            np.asarray(units.loc[unit_id, "spike_times"]),
            spike_times.loc[unit_id],
        ), f"Unit {unit_id} was written with another unit's spike times"
