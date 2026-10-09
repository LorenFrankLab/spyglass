"""Regression guard for ``SpikeSorting.get_sorting`` with no spikes (#1154).

Issue #1154 asks for explicit handling when a sorting contains no spikes. The
site the issue names -- ``spikesorting.v1.curation`` -- was fixed by PR #1533,
which handles the null entry and writes an empty ``pynwb.misc.Units`` table.

The remaining live defect is in ``spikesorting.v1.sorting``: an empty units
table is a DataFrame with *no columns at all*, so the ``units["spike_times"]``
lookup inside ``SpikeSorting.get_sorting`` raised a bare
``KeyError: 'spike_times'`` -- exactly the failure #1154 describes, one table
upstream of where the issue points.

These tests pin the resolution chosen by #1533 and already implemented by
``CurationV1.get_sorting``: return an empty ``si.NumpySorting`` rather than
raise. They run against a *genuinely* spike-less sorting produced by a real
``SpikeSorting.populate``, not a mock.
"""

import numpy as np
import pynwb
import pytest
import spikeinterface as si


@pytest.fixture(scope="module")
def no_spikes_sort_key(spike_v1, pop_rec, pop_art, mini_dict, pop_sort):
    """A ``SpikeSorting`` primary key whose units table holds no spike times.

    Built by reusing the ``clusterless_thresholder`` sorter with a detection
    threshold no sample can cross, so ``detect_peaks`` returns nothing and
    ``_write_sorting_to_nwb`` takes its ``get_num_units() == 0`` branch,
    writing a real empty ``pynwb.misc.Units`` table to disk.

    Depends on ``pop_sort`` only for ordering: ``pop_sort`` runs a blanket
    ``SpikeSorting.populate()`` and then picks the newest row, so it must
    resolve before this fixture inserts a competing selection.

    Yields
    ------
    dict
        Primary key of the spike-less ``SpikeSorting`` row.
    """
    _ = pop_sort  # ordering only; see docstring

    param_name = "no_spikes_b5"
    base_params = (
        spike_v1.SpikeSorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_param_name": "default_clusterless",
        }
    ).fetch1("sorter_params")

    params = dict(base_params)
    params["detect_threshold"] = 1e9  # uV -- nothing crosses this

    spike_v1.SpikeSorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_param_name": param_name,
            "sorter_params": params,
        },
        skip_duplicates=True,
    )

    key = {
        **mini_dict,
        "sorter": "clusterless_thresholder",
        "sorter_param_name": param_name,
        "recording_id": pop_rec["recording_id"],
        "interval_list_name": str(pop_art["artifact_id"]),
    }
    spike_v1.SpikeSortingSelection.insert_selection(key)
    sel_key = (spike_v1.SpikeSortingSelection & key).fetch1("KEY")
    spike_v1.SpikeSorting.populate(sel_key)

    if not (spike_v1.SpikeSorting & sel_key):
        pytest.fail("Could not build a spike-less SpikeSorting entry.")

    yield sel_key


@pytest.fixture(scope="module")
def no_spikes_units_df(spike_v1, no_spikes_sort_key):
    """Units dataframe read straight from the spike-less sorting's NWB file."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    path = AnalysisNwbfile.get_abs_path(
        (spike_v1.SpikeSorting & no_spikes_sort_key).fetch1(
            "analysis_file_name"
        )
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        yield io.read().units.to_dataframe()


@pytest.fixture(scope="module")
def with_spikes_sort_key(spike_v1, pop_sort):
    """A ``SpikeSorting`` primary key for a sorting that *does* have spikes.

    Derived by restricting on the ``mountainsort4`` selection rather than
    trusting the session-scoped ``pop_sort`` key: ``pop_sort`` populates every
    outstanding selection and returns the newest row, so it can latch onto the
    spike-less sorting built above.

    Yields
    ------
    dict
        Primary key of a spike-bearing ``SpikeSorting`` row.
    """
    _ = pop_sort  # ensure the sorting is populated

    sel = spike_v1.SpikeSortingSelection & {
        "sorter": "mountainsort4",
        "sorter_param_name": "franklab_tetrode_hippocampus_30KHz",
    }
    keys = (spike_v1.SpikeSorting & sel).fetch("KEY", as_dict=True)
    if not keys:
        pytest.fail("No mountainsort4 SpikeSorting row to use as a control.")

    yield keys[0]


def test_fixture_is_genuinely_spike_less(no_spikes_units_df):
    """Guard against a vacuous test: the sorting really has no spike times."""
    assert len(no_spikes_units_df) == 0, "Fixture sorting has units"
    assert (
        "spike_times" not in no_spikes_units_df.columns
    ), "Fixture sorting still has a spike_times column"


def test_get_sorting_no_spikes_does_not_raise(spike_v1, no_spikes_sort_key):
    """Issue #1154: a spike-less sorting must not surface a bare KeyError."""
    try:
        sorting = spike_v1.SpikeSorting.get_sorting(no_spikes_sort_key)
    except KeyError as err:  # pragma: no cover - the regression itself
        pytest.fail(f"get_sorting raised a bare KeyError: {err}")

    assert sorting is not None


def test_get_sorting_no_spikes_returns_empty(spike_v1, no_spikes_sort_key):
    """The returned object is an empty sorting, matching CurationV1."""
    sorting = spike_v1.SpikeSorting.get_sorting(no_spikes_sort_key)
    recording = spike_v1.SpikeSortingRecording.get_recording(
        {
            "recording_id": (
                spike_v1.SpikeSortingSelection & no_spikes_sort_key
            ).fetch1("recording_id")
        }
    )

    assert isinstance(sorting, si.BaseSorting)
    assert sorting.get_num_units() == 0
    assert len(sorting.get_unit_ids()) == 0
    assert sorting.get_num_segments() == 1
    assert (
        sorting.get_sampling_frequency() == recording.get_sampling_frequency()
    )


def test_empty_sorting_is_usable_downstream(
    spike_v1, no_spikes_sort_key, mini_dict
):
    """The empty sorting works where a ``BaseSorting`` is expected.

    A object that merely constructs is not a fix, so this exercises three
    real consumers: spikeinterface's own materialization of the spike vector,
    a ``NumpySorting`` round trip, and ``_write_sorting_to_nwb`` -- the
    function in this very module that is typed to take a ``si.BaseSorting``.
    """
    from spyglass.spikesorting.v1.sorting import _write_sorting_to_nwb

    sorting = spike_v1.SpikeSorting.get_sorting(no_spikes_sort_key)

    assert len(sorting.to_spike_vector()) == 0
    assert si.NumpySorting.from_sorting(sorting).get_num_units() == 0

    recording = spike_v1.SpikeSortingRecording.get_recording(
        {
            "recording_id": (
                spike_v1.SpikeSortingSelection & no_spikes_sort_key
            ).fetch1("recording_id")
        }
    )
    timestamps = recording.get_times()
    sort_interval = np.asarray([[timestamps[0], timestamps[-1]]])

    file_name, _ = _write_sorting_to_nwb(
        sorting, timestamps, sort_interval, mini_dict["nwb_file_name"]
    )

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    with pynwb.NWBHDF5IO(
        AnalysisNwbfile.get_abs_path(file_name), "r", load_namespaces=True
    ) as io:
        assert len(io.read().units.to_dataframe()) == 0


def test_get_sorting_with_spikes_unaffected(spike_v1, with_spikes_sort_key):
    """Back-compat: a sorting that has spikes is read exactly as before."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    path = AnalysisNwbfile.get_abs_path(
        (spike_v1.SpikeSorting & with_spikes_sort_key).fetch1(
            "analysis_file_name"
        )
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        units = io.read().units.to_dataframe()

    assert len(units) > 0, "Control sorting has no units"
    assert "spike_times" in units.columns

    sorting = spike_v1.SpikeSorting.get_sorting(with_spikes_sort_key)

    assert sorting.get_num_units() == len(units)
    assert sorted(sorting.get_unit_ids()) == sorted(units.index)
    assert sum(
        len(sorting.get_unit_spike_train(uid)) for uid in sorting.get_unit_ids()
    ), "Control sorting round-tripped to empty spike trains"
