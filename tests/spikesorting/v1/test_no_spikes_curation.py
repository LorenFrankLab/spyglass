"""Regression guard for no-spikes handling during curation (issue #1154).

Issue #1154 reports that a sorting with no units surfaces as a bare
``KeyError`` from the code in :mod:`spyglass.spikesorting.v1.curation` that
reads spike times out of the units dataframe, and that the extraction
notebooks therefore wrap ``insert_curation`` in ``except KeyError: pass``.

The issue offered two remedies: handle the null entry, or raise a more
specific error. PR #1533 took the first -- ``_write_sorting_to_nwb_with
_curation`` now guards on ``"spike_times" in units.columns`` and writes an
empty ``pynwb.misc.Units`` table -- so the ``KeyError`` no longer occurs and
there is nothing left in ``curation.py`` for a specific exception to replace.

These tests pin that resolution against a *genuinely* spike-less sorting: a
real ``SpikeSorting`` row whose analysis NWB file holds a units table with no
``spike_times`` column. They exist so the ``KeyError`` cannot silently return.
"""

import pynwb
import pytest


@pytest.fixture(scope="module")
def no_spikes_sorting(spike_v1, pop_rec, pop_art, mini_dict):
    """A ``SpikeSorting`` entry whose units table holds no spike times.

    Built by reusing the ``clusterless_thresholder`` sorter with a detection
    threshold no sample can cross, so ``detect_peaks`` returns nothing and
    ``_write_sorting_to_nwb`` writes an empty ``pynwb.misc.Units`` table. This
    is a real populate, not a mock: the analysis NWB file exists on disk.

    Yields
    ------
    str
        ``sorting_id`` of the spike-less sorting.
    """
    param_name = "no_spikes_b2"
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

    yield str(sel_key["sorting_id"])


@pytest.fixture(scope="module")
def no_spikes_units_df(spike_v1, no_spikes_sorting):
    """Units dataframe read straight from the spike-less sorting's NWB file."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    path = AnalysisNwbfile.get_abs_path(
        (spike_v1.SpikeSorting & {"sorting_id": no_spikes_sorting}).fetch1(
            "analysis_file_name"
        )
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        yield io.read().units.to_dataframe()


@pytest.fixture(scope="module")
def no_spikes_curation(spike_v1, no_spikes_sorting):
    """Curation key produced from the spike-less sorting.

    Yields
    ------
    dict
        Primary key of the resulting ``CurationV1`` row.
    """
    key = spike_v1.CurationV1.insert_curation(
        sorting_id=no_spikes_sorting, description="no-spikes guard"
    )
    yield {
        "sorting_id": key["sorting_id"],
        "curation_id": key["curation_id"],
        "analysis_file_name": key["analysis_file_name"],
    }


def test_fixture_is_genuinely_spike_less(no_spikes_units_df):
    """Guard against a vacuous test: the sorting really has no spike times."""
    assert len(no_spikes_units_df) == 0, "Fixture sorting has units"
    assert (
        "spike_times" not in no_spikes_units_df.columns
    ), "Fixture sorting still has a spike_times column"


def test_no_spikes_curation_does_not_raise(spike_v1, no_spikes_sorting):
    """Issue #1154: curating a spike-less sorting must not raise KeyError."""
    try:
        key = spike_v1.CurationV1.insert_curation(
            sorting_id=no_spikes_sorting, description="no-spikes no-raise"
        )
    except KeyError as err:  # pragma: no cover - the regression itself
        pytest.fail(f"insert_curation raised a bare KeyError: {err}")

    assert key["sorting_id"] == no_spikes_sorting


def test_no_spikes_curation_stores_empty_units(spike_v1, no_spikes_curation):
    """The curated analysis file holds an empty units table, not nothing."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    path = AnalysisNwbfile.get_abs_path(
        no_spikes_curation["analysis_file_name"]
    )
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbf = io.read()
        assert nwbf.units is not None, "Curated file has no units table"
        assert len(nwbf.units.to_dataframe()) == 0


def test_no_spikes_curation_reads_back_empty(spike_v1, no_spikes_curation):
    """Readers return a zero-unit sorting rather than raising."""
    pk = {
        "sorting_id": no_spikes_curation["sorting_id"],
        "curation_id": no_spikes_curation["curation_id"],
    }
    assert spike_v1.CurationV1.get_sorting(pk).get_num_units() == 0
    assert spike_v1.CurationV1.get_merged_sorting(pk).get_num_units() == 0
    assert len(spike_v1.CurationV1.get_sorting(pk, as_dataframe=True)) == 0


def test_curation_with_spikes_unaffected(spike_v1, pop_sort):
    """A sorting that does have spikes curates exactly as before.

    Deliberately does not reuse the session-wide ``pop_curation`` fixture:
    that fixture picks an arbitrary ``parent_curation_id=-1`` row, which the
    spike-less curation above would satisfy.
    """
    sorting_id = str(pop_sort["sorting_id"])
    prior = (spike_v1.CurationV1 & {"sorting_id": sorting_id}).fetch(
        "curation_id"
    )
    key = spike_v1.CurationV1.insert_curation(
        sorting_id=sorting_id,
        parent_curation_id=max(prior, default=-1),
        description="b2 with-spikes control",
    )
    pk = {"sorting_id": key["sorting_id"], "curation_id": key["curation_id"]}

    units = spike_v1.CurationV1.get_sorting(pk, as_dataframe=True)
    assert len(units) > 0, "Control sorting has no units"
    assert "spike_times" in units.columns
    assert all(
        len(train) > 0 for train in units["spike_times"]
    ), "Control units have empty spike trains"
    assert spike_v1.CurationV1.get_sorting(pk).get_num_units() == len(units)
