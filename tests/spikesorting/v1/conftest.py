import pytest


@pytest.fixture(scope="session")
def burst_params_key(spike_v1):
    yield dict(burst_params_name="default")


@pytest.fixture(scope="session")
def pop_burst(spike_v1, pop_metric, burst_params_key):
    metric_ids = spike_v1.MetricCuration().fetch("metric_curation_id")
    for metric_id in metric_ids:
        spike_v1.BurstPairSelection().insert_by_curation_id(metric_id)
    burst_key = {**pop_metric, **burst_params_key}
    spike_v1.BurstPair.populate(burst_key)
    yield spike_v1.BurstPair() & burst_key


@pytest.fixture(scope="session")
def pop_figurl(spike_v1, pop_sort, metric_objs):
    # WON'T WORK UNTIL CI/CD KACHERY_CLOUD INIT
    sort_dict = {"sorting_id": pop_sort["sorting_id"], "curation_id": 1}
    curation_uri = spike_v1.FigURLCurationSelection.generate_curation_uri(
        sort_dict
    )
    _, _, metrics = metric_objs
    key = {
        **sort_dict,
        "curation_uri": curation_uri,
        "metrics_figurl": list(metrics.keys()),
    }
    spike_v1.FigURLCurationSelection.insert_selection(key)
    spike_v1.FigURLCuration.populate()

    yield spike_v1.FigURLCuration().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def pop_figurl_json(spike_v1, pop_metric, metric_objs, pop_sort):
    # WON'T WORK UNTIL CI/CD KACHERY_CLOUD INIT
    gh_curation_uri = (
        "gh://LorenFrankLab/sorting-curations/main/khl02007/test/curation.json"
    )
    key = {
        "sorting_id": pop_metric["sorting_id"],
        "curation_id": 1,
        "curation_uri": gh_curation_uri,
        "metrics_figurl": [],
    }
    spike_v1.FigURLCurationSelection.insert_selection(key)
    spike_v1.FigURLCuration.populate()

    labels = spike_v1.FigURLCuration.get_labels(gh_curation_uri)
    merge_groups = spike_v1.FigURLCuration.get_merge_groups(gh_curation_uri)
    _, _, metrics = metric_objs
    spike_v1.CurationV1.insert_curation(
        sorting_id=pop_sort["sorting_id"],
        parent_curation_id=1,
        labels=labels,
        merge_groups=merge_groups,
        metrics=metrics,
        description="after figurl curation",
    )
    yield spike_v1.CurationV1().fetch("KEY", as_dict=True)  # list of dicts


@pytest.fixture(scope="session")
def spike_v1_ua():
    from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation

    yield UnitAnnotation()


@pytest.fixture(scope="session")
def pop_annotations(spike_v1_group, spike_v1_ua, pop_spikes_group):
    spike_times, unit_ids = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_spikes_group, return_unit_ids=True
    )
    for spikes, unit_key in zip(spike_times, unit_ids):
        quant_key = {
            **unit_key,
            "annotation": "spike_count",
            "quantification": len(spikes),
        }
        label_key = {
            **unit_key,
            "annotation": "cell_type",
            "label": "pyridimal" if len(spikes) < 1000 else "interneuron",
        }

        spike_v1_ua.add_annotation(quant_key, skip_duplicates=True)
        spike_v1_ua.add_annotation(label_key, skip_duplicates=True)

    yield (spike_v1_ua.Annotation & {"annotation": "spike_count"})


# ============================================================================
# Mock Helper Functions
# ============================================================================


def create_fake_sorting(
    n_units=10, n_spikes_per_unit=1000, sampling_frequency=30000
):
    """Create fake spike sorting results.

    Parameters
    ----------
    n_units : int
        Number of units
    n_spikes_per_unit : int
        Average spikes per unit
    sampling_frequency : float
        Sampling frequency in Hz

    Returns
    -------
    sorting : object
        Fake sorting object (minimal interface)
    timestamps : np.ndarray
        Timestamps array
    """
    import numpy as np

    class FakeSorting:
        """Minimal sorting interface for mocking."""

        def __init__(self, spike_times, unit_ids):
            self.spike_times = spike_times
            self.unit_ids = unit_ids
            self._sampling_frequency = sampling_frequency

        def get_unit_ids(self):
            return self.unit_ids

        def get_unit_spike_train(self, unit_id):
            return self.spike_times[unit_id]

        def get_sampling_frequency(self):
            return self._sampling_frequency

    # Generate fake spike times
    spike_times = {}
    for unit_id in range(n_units):
        # Random spike times across 10 second recording
        n_spikes = n_spikes_per_unit + np.random.randint(-100, 100)
        spike_times[unit_id] = np.sort(np.random.rand(n_spikes) * 10)

    timestamps = np.arange(0, 10, 1 / sampling_frequency)

    return FakeSorting(spike_times, list(range(n_units))), timestamps


# ============================================================================
# Mock Fixtures for SpikeSorting
# ============================================================================


@pytest.fixture
def mock_spike_sorter():
    """Mock the _run_spike_sorter helper for SpikeSorting.

    This mocks the expensive spikeinterface operations (~90s).
    """

    def _mock_run_sorter(
        self,
        recording_analysis_nwb_file_abs_path,
        artifact_removed_intervals,
        sorter,
        sorter_params,
    ):
        """Mocked version that returns fake sorting instantly."""
        sorting, timestamps = create_fake_sorting(
            n_units=10, n_spikes_per_unit=1000, sampling_frequency=30000
        )
        return sorting, timestamps

    return _mock_run_sorter


@pytest.fixture
def mock_sorting_save():
    """Mock the _save_sorting_results helper for SpikeSorting.

    This mocks file I/O operations (~5s) but still creates the AnalysisNwbfile entry.
    """
    from spyglass.common import AnalysisNwbfile

    def _mock_save_results(
        self, sorting, timestamps, artifact_removed_intervals, nwb_file_name
    ):
        """Mocked version that creates AnalysisNwbfile entry but skips actual file I/O."""
        # Create AnalysisNwbfile entry (required for foreign key)
        nwb_analysis_file = AnalysisNwbfile()
        analysis_file_name = nwb_analysis_file.create(nwb_file_name)

        # Return fake file name and object_id (skip actual NWB write operations)
        object_id = "fake_object_id_123"
        return analysis_file_name, object_id

    return _mock_save_results
