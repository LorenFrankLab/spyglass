import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def mock_netcdf_saves():
    """Globally mock netCDF file writes to avoid HDF5 I/O conflicts in CI.

    This prevents RuntimeError: NetCDF: HDF error during parallel test execution
    by intercepting xarray.Dataset.to_netcdf() calls and using pickle instead.
    """
    import pickle
    from pathlib import Path

    def mock_to_netcdf(
        self,
        path=None,
        mode="w",
        format=None,
        group=None,
        engine=None,
        encoding=None,
        unlimited_dims=None,
        compute=True,
        invalid_netcdf=False,
    ):
        """Mock to_netcdf that writes pickle instead of netCDF."""
        if path is None:
            # Return bytes if no path given (original behavior for some use cases)
            return None

        # Keep the .nc extension to match expectations, but write pickle format
        # This avoids netCDF4/HDF5 errors while maintaining file path compatibility
        with open(path, "wb") as f:
            pickle.dump(self, f)

        return None

    with patch("xarray.Dataset.to_netcdf", mock_to_netcdf):
        yield


@pytest.fixture(scope="session")
def result_coordinates():
    return {
        "encoding_groups",
        "states",
        "state",
        "state_bins",
        "state_ind",
        "time",
        "environments",
    }


@pytest.fixture(scope="session")
def decode_v1(common, trodes_pos_v1):
    from spyglass.decoding import v1

    yield v1


@pytest.fixture(scope="session")
def recording_ids(spike_v1, mini_dict, pop_rec, pop_art):
    _ = pop_rec  # set group by shank

    recording_ids = (spike_v1.SpikeSortingRecordingSelection & mini_dict).fetch(
        "recording_id"
    )
    group_keys = []
    for recording_id in recording_ids:
        key = {
            "recording_id": recording_id,
            "artifact_param_name": "none",
        }
        group_keys.append(key)
        spike_v1.ArtifactDetectionSelection.insert_selection(key)
    spike_v1.ArtifactDetection.populate(group_keys)

    yield recording_ids


@pytest.fixture(scope="session")
def clusterless_params_insert(spike_v1):
    """Low threshold for testing, otherwise no spikes with default."""
    clusterless_params = {
        "sorter": "clusterless_thresholder",
        "sorter_param_name": "low_thresh",
    }
    spike_v1.SpikeSorterParameters.insert1(
        {
            **clusterless_params,
            "sorter_params": {
                "detect_threshold": 10.0,  # was 100
                # Locally exclusive means one unit per spike detected
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 1000,  # was 100
                # noise levels needs to be 1.0 so the units are in uV and not MAD
                "noise_levels": np.asarray([1.0]),
                "random_chunk_kwargs": {},
                # output needs to be set to sorting for the rest of the pipeline
                "outputs": "sorting",
            },
        },
        skip_duplicates=True,
    )
    yield clusterless_params


@pytest.fixture(scope="session")
def clusterless_spikesort(
    spike_v1, recording_ids, mini_dict, clusterless_params_insert
):
    group_keys = []
    for recording_id in recording_ids:
        key = {
            **clusterless_params_insert,
            **mini_dict,
            "recording_id": recording_id,
            "interval_list_name": str(
                (
                    spike_v1.ArtifactDetectionSelection
                    & {
                        "recording_id": recording_id,
                        "artifact_param_name": "none",
                    }
                ).fetch1("artifact_id")
            ),
        }
        group_keys.append(key)
        spike_v1.SpikeSortingSelection.insert_selection(key)
    spike_v1.SpikeSorting.populate()
    yield clusterless_params_insert


@pytest.fixture(scope="session")
def clusterless_params(clusterless_spikesort):
    yield clusterless_spikesort


@pytest.fixture(scope="session")
def clusterless_curate(spike_v1, clusterless_params, spike_merge):

    sorting_ids = (spike_v1.SpikeSortingSelection & clusterless_params).fetch(
        "sorting_id"
    )

    fails = []
    for sorting_id in sorting_ids:
        try:
            spike_v1.CurationV1.insert_curation(sorting_id=sorting_id)
        except KeyError:
            fails.append(sorting_id)

    if len(fails) == len(sorting_ids):
        (spike_v1.SpikeSorterParameters & clusterless_params).delete(
            safemode=False
        )
        raise ValueError("All curation insertions failed.")

    spike_merge.insert(
        spike_v1.CurationV1().fetch("KEY"),
        part_name="CurationV1",
        skip_duplicates=True,
    )
    yield


@pytest.fixture(scope="session")
def waveform_params_tbl(decode_v1):
    params_tbl = decode_v1.waveform_features.WaveformFeaturesParams
    params_tbl.insert_default()
    yield params_tbl


@pytest.fixture(scope="session")
def waveform_params(waveform_params_tbl):
    param_pk = {"features_param_name": "low_thresh_amplitude"}
    waveform_params_tbl.insert1(
        {
            **param_pk,
            "params": {
                "waveform_extraction_params": {
                    "ms_before": 0.2,  # previously 0.5
                    "ms_after": 0.2,  # previously 0.5
                    "max_spikes_per_unit": None,
                    "n_jobs": 1,  # previously 5
                    "total_memory": "1G",  # previously "5G"
                },
                "waveform_features_params": {
                    "amplitude": {
                        "peak_sign": "neg",
                        "estimate_peak_time": False,  # was False
                    }
                },
            },
        },
        skip_duplicates=True,
    )
    yield param_pk


@pytest.fixture(scope="session")
def clusterless_mergeids(
    spike_merge, mini_dict, clusterless_curate, clusterless_params
):
    _ = clusterless_curate  # ensure populated
    yield spike_merge.get_restricted_merge_ids(
        {
            **mini_dict,
            **clusterless_params,
        },
        sources=["v1"],
    )


@pytest.fixture(scope="session")
def pop_unitwave(decode_v1, waveform_params, clusterless_mergeids):
    sel_keys = [
        {
            "spikesorting_merge_id": merge_id,
            **waveform_params,
        }
        for merge_id in clusterless_mergeids
    ]

    wave = decode_v1.waveform_features
    wave.UnitWaveformFeaturesSelection.insert(sel_keys, skip_duplicates=True)
    wave.UnitWaveformFeatures.populate(sel_keys)

    yield wave.UnitWaveformFeatures & sel_keys


@pytest.fixture(scope="session")
def group_unitwave(
    decode_v1,
    mini_dict,
    clusterless_mergeids,
    pop_unitwave,
    waveform_params,
    group_name,
):
    wave = decode_v1.waveform_features
    waveform_selection_keys = (
        wave.UnitWaveformFeaturesSelection() & waveform_params
    ).fetch("KEY", as_dict=True)
    decode_v1.clusterless.UnitWaveformFeaturesGroup().create_group(
        **mini_dict,
        group_name="test_group",
        keys=waveform_selection_keys,
    )
    yield decode_v1.clusterless.UnitWaveformFeaturesGroup & {
        "waveform_features_group_name": group_name,
    }


@pytest.fixture(scope="session")
def pos_merge_keys(pos_merge):
    return (
        (
            pos_merge.TrodesPosV1
            & 'trodes_pos_params_name = "single_led_upsampled"'
        )
        .proj(pos_merge_id="merge_id")
        .fetch(as_dict=True)
    )


@pytest.fixture(scope="session")
def pop_pos_group(decode_v1, pos_merge_keys, group_name, mini_dict):

    decode_v1.core.PositionGroup().create_group(
        **mini_dict,
        group_name=group_name,
        keys=pos_merge_keys,
    )

    yield decode_v1.core.PositionGroup & {
        **mini_dict,
        "position_group_name": group_name,
    }


@pytest.fixture(scope="session")
def pop_pos_group_upsampled(decode_v1, pos_merge_keys, group_name, mini_dict):
    name = group_name + "_upsampled"
    decode_v1.core.PositionGroup().create_group(
        **mini_dict,
        group_name=name,
        keys=pos_merge_keys,
        upsample_rate=250,
    )

    yield decode_v1.core.PositionGroup & {
        **mini_dict,
        "position_group_name": name,
    }


@pytest.fixture(scope="session")
def decode_clusterless_params_insert(decode_v1, track_graph):
    from non_local_detector.environment import Environment
    from non_local_detector.models import ContFragClusterlessClassifier

    graph_entry = track_graph.fetch1()  # Restricted table
    class_kwargs = dict(
        clusterless_algorithm_params={
            "block_size": 10000,
            "position_std": 12.0,
            "waveform_std": 24.0,
        },
        environments=[
            Environment(
                # environment_name=graph_entry["track_graph_name"],
                track_graph=track_graph.get_networkx_track_graph(),
                edge_order=graph_entry["linear_edge_order"],
                edge_spacing=graph_entry["linear_edge_spacing"],
            )
        ],
    )
    params_pk = {"decoding_param_name": "contfrag_clusterless"}
    # decode_v1.core.DecodingParameters.insert_default()
    decode_v1.core.DecodingParameters.insert1(
        {
            **params_pk,
            "decoding_params": ContFragClusterlessClassifier(**class_kwargs),
            "decoding_kwargs": dict(),
        },
        skip_duplicates=True,
    )
    model_params = (decode_v1.core.DecodingParameters & params_pk).fetch1()
    ContFragClusterlessClassifier(**model_params["decoding_params"])

    yield params_pk


@pytest.fixture(scope="session")
def decode_interval(common, mini_dict):
    decode_interval_name = "decode"
    raw_begin = (common.IntervalList & 'interval_list_name LIKE "raw%"').fetch1(
        "valid_times"
    )[0][0]
    # Use a subset of the encoding interval (raw_begin+2 to raw_begin+13)
    # This creates gaps at start and end, ensuring that when
    # estimate_decoding_params=True, there are time points outside the
    # decoding interval that will get interval_labels=-1
    common.IntervalList.insert1(
        {
            **mini_dict,
            "interval_list_name": decode_interval_name,
            "valid_times": [[raw_begin + 2, raw_begin + 13]],
        },
        skip_duplicates=True,
    )
    yield decode_interval_name


@pytest.fixture(scope="session")
def decode_merge(common):
    from spyglass.decoding import DecodingOutput

    yield DecodingOutput()


@pytest.fixture(scope="session")
def decode_sel_key(mini_dict, group_name, pos_interval, decode_interval):
    return {
        **mini_dict,
        "position_group_name": group_name,
        "encoding_interval": pos_interval,
        "decoding_interval": decode_interval,
    }


@pytest.fixture(scope="session")
def clusterless_pop(
    decode_v1,
    decode_sel_key,
    group_name,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
    teardown,
    decode_merge,
    mock_netcdf_saves,
):
    _ = (
        pop_pos_group,
        group_unitwave,
        mock_netcdf_saves,
    )  # ensure populated and mock active

    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": False,
    }

    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    yield decode_v1.clusterless.ClusterlessDecodingV1 & selection_key

    if teardown:
        decode_merge.cleanup()


@pytest.fixture(scope="session")
def clusterless_key(clusterless_pop):
    yield clusterless_pop.fetch("KEY")[0]


@pytest.fixture(scope="session")
def clusterless_pop_estimated(
    decode_v1,
    decode_sel_key,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
    group_name,
    teardown,
    decode_merge,
    mock_netcdf_saves,
):
    _ = (
        pop_pos_group,
        group_unitwave,
        mock_netcdf_saves,
    )  # ensure populated and mock active
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": True,
    }

    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    yield decode_v1.clusterless.ClusterlessDecodingV1 & selection_key

    if teardown:
        decode_merge.cleanup()


@pytest.fixture(scope="session")
def decode_spike_params_insert(decode_v1):
    from non_local_detector.models import ContFragSortedSpikesClassifier

    params_pk = {"decoding_param_name": "contfrag_sorted"}
    decode_v1.core.DecodingParameters.insert1(
        {
            **params_pk,
            "decoding_params": ContFragSortedSpikesClassifier(),
            "decoding_kwargs": dict(),
        },
        skip_duplicates=True,
    )
    yield params_pk


@pytest.fixture(scope="session")
def spikes_decoding(
    decode_spike_params_insert,
    decode_v1,
    decode_sel_key,
    group_name,
    pop_spikes_group,
    pop_pos_group,
    mock_netcdf_saves,
):
    _ = (
        pop_spikes_group,
        pop_pos_group,
        mock_netcdf_saves,
    )  # ensure populated and mock active
    spikes = decode_v1.sorted_spikes
    selection_key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "default_exclusion",
        "estimate_decoding_params": False,
    }
    spikes.SortedSpikesDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    spikes.SortedSpikesDecodingV1.populate(selection_key)

    yield spikes.SortedSpikesDecodingV1 & selection_key


@pytest.fixture(scope="session")
def spikes_decoding_key(spikes_decoding):
    yield spikes_decoding.fetch("KEY")[0]


@pytest.fixture(scope="session")
def spikes_decoding_estimated(
    decode_spike_params_insert,
    decode_v1,
    decode_sel_key,
    group_name,
    pop_spikes_group,
    pop_pos_group,
    mock_netcdf_saves,
):
    _ = (
        pop_spikes_group,
        pop_pos_group,
        mock_netcdf_saves,
    )  # ensure populated and mock active
    spikes = decode_v1.sorted_spikes
    selection_key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "default_exclusion",
        "estimate_decoding_params": True,
    }
    spikes.SortedSpikesDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    spikes.SortedSpikesDecodingV1.populate(selection_key)

    yield spikes.SortedSpikesDecodingV1 & selection_key


# ============================================================================
# Mock Helper Functions for Fast Unit Tests
# ============================================================================


class FakeClassifier:
    """Minimal classifier interface for mocking.

    Module-level class so it's picklable for mock_decoder_save.
    """

    def __init__(self):
        # 2 states (e.g., "continuous" and "fragmented")
        # initial_conditions_: shape (n_states,)
        self.initial_conditions_ = np.array([0.5, 0.5])
        # discrete_state_transitions_: shape (n_states, n_states)
        self.discrete_state_transitions_ = np.array([[0.9, 0.1], [0.1, 0.9]])


def create_fake_classifier():
    """Create a fake classifier object that mimics non_local_detector output."""
    return FakeClassifier()


def create_fake_decoding_results(n_time=100, n_position_bins=50, n_states=2):
    """Create fake decoding results that mimic ClusterlessDetector output.

    Parameters
    ----------
    n_time : int
        Number of time points
    n_position_bins : int
        Number of position bins
    n_states : int
        Number of discrete states (default: 2 for continuous/fragmented)

    Returns
    -------
    results : xr.Dataset
        Fake decoding results matching expected structure with proper dimensions
    """
    import xarray as xr

    time = np.linspace(0, 10, n_time)
    position_bins = np.linspace(0, 100, n_position_bins)
    states = np.arange(n_states)
    state_names = ["continuous", "fragmented"][:n_states]

    # Create realistic-looking posterior probabilities
    position_mean = n_position_bins // 2
    position_std = n_position_bins // 10

    # Posterior shape: (n_time, n_position_bins, n_states)
    posterior = np.zeros((n_time, n_position_bins, n_states))
    for t in range(n_time):
        for s in range(n_states):
            # Gaussian-like posterior for each state
            posterior[t, :, s] = np.exp(
                -((np.arange(n_position_bins) - position_mean) ** 2)
                / (2 * position_std**2)
            )
        # Normalize across position and state
        posterior[t] /= posterior[t].sum()

    # Create all expected coordinates for decoding results
    # Note: Use "states" (plural) as the dimension name to match
    # the real non_local_detector output format
    results = xr.Dataset(
        {
            "posterior": (["time", "position", "states"], posterior),
            "likelihood": (["time", "position", "states"], posterior * 0.8),
        },
        coords={
            "time": time,
            "position": position_bins,
            "states": state_names,  # Dimension coordinate with state names
            "state_ind": ("states", states),  # State indices
            "state_bins": ("position", position_bins),  # Alias for position
            "encoding_groups": ("states", state_names),  # Encoding group names
            "environments": ("states", state_names),  # Environment names
        },
    )

    return results


# ============================================================================
# In-Memory Mock Storage (avoids netCDF4/HDF5 file I/O issues in CI)
# ============================================================================


@pytest.fixture(scope="session")
def mock_results_storage():
    """Session-scoped dictionary to store decoding results in memory.

    This eliminates netCDF4/HDF5 file I/O issues in CI by storing
    xarray datasets and classifiers directly in memory.
    """
    return {"results": {}, "classifiers": {}}


@pytest.fixture(scope="session", autouse=True)
def mock_detector_io_globally(mock_results_storage):
    """Globally mock detector load methods to use netcdf4 engine explicitly.

    This is an autouse fixture that automatically applies to all tests in the
    decoding module, ensuring that all fetch_results() calls use explicit
    netcdf4 engine specification to avoid engine detection errors.
    """
    from unittest.mock import patch
    from non_local_detector.models.base import (
        ClusterlessDetector,
        SortedSpikesDetector,
    )
    import xarray as xr

    def _mock_load_results(filename):
        """Load results with explicit netcdf4 engine."""
        filename_str = str(filename)

        # Try loading from memory first (for tests that use in-memory storage)
        if filename_str in mock_results_storage["results"]:
            return mock_results_storage["results"][filename_str]

        # Load from disk with explicit engine
        try:
            return xr.open_dataset(filename_str, engine="netcdf4")
        except (FileNotFoundError, OSError) as e:
            # OSError with "Unknown file format" means old pickle file exists
            # FileNotFoundError means file doesn't exist at all
            if "Unknown file format" in str(e):
                raise FileNotFoundError(
                    f"Mock result has invalid format (likely old pickle file): {filename_str}. "
                    "Please delete old *_mocked.nc files from tests/_data/analysis/"
                )
            raise FileNotFoundError(f"Mock result not found: {filename_str}")

    def _mock_load_model(filename):
        """Load classifier from in-memory storage."""
        filename_str = str(filename)
        if filename_str in mock_results_storage["classifiers"]:
            return mock_results_storage["classifiers"][filename_str]
        raise FileNotFoundError(
            f"Mock classifier not found in memory: {filename_str}"
        )

    # Patch the detector base classes' load methods globally
    with (
        patch.object(
            ClusterlessDetector,
            "load_results",
            staticmethod(_mock_load_results),
        ),
        patch.object(
            ClusterlessDetector, "load_model", staticmethod(_mock_load_model)
        ),
        patch.object(
            SortedSpikesDetector,
            "load_results",
            staticmethod(_mock_load_results),
        ),
        patch.object(
            SortedSpikesDetector, "load_model", staticmethod(_mock_load_model)
        ),
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def mock_save_decoder_results_globally(mock_results_storage):
    """Globally mock _save_decoder_results to use netcdf4 engine explicitly.

    This prevents NetCDF4/HDF5 engine detection errors in CI by:
    1. Writing netCDF files with explicit netcdf4 engine
    2. Storing actual data in memory for loading via mock_detector_io_globally
    """
    from unittest.mock import patch
    from pathlib import Path
    import uuid
    import pickle
    import os

    def _mock_save_results(self, classifier, results, key):
        """Mocked version that creates files with explicit netcdf4 engine."""
        # Generate unique identifier for this result set
        unique_id = str(uuid.uuid4())[:8]
        nwb_file_name = key["nwb_file_name"].replace("_.nwb", "")

        # Use absolute path to match environment expectations
        base_dir = os.environ.get("SPYGLASS_BASE_DIR", "tests/_data")
        analysis_dir = Path(base_dir) / "analysis"
        subdir = analysis_dir / nwb_file_name
        subdir.mkdir(parents=True, exist_ok=True)

        # Create file paths
        results_path = subdir / f"{nwb_file_name}_{unique_id}_mocked.nc"
        classifier_path = subdir / f"{nwb_file_name}_{unique_id}_mocked.pkl"

        # Write netCDF file with explicit netcdf4 engine
        results.to_netcdf(results_path, engine="netcdf4")

        # Write classifier as pickle (non-xarray object)
        with open(classifier_path, "wb") as f:
            pickle.dump(classifier, f)

        # Store actual data in memory for loading
        results_path_str = str(results_path)
        classifier_path_str = str(classifier_path)
        mock_results_storage["results"][results_path_str] = results
        mock_results_storage["classifiers"][classifier_path_str] = classifier

        return results_path_str, classifier_path_str

    # Import the decoding table classes
    from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
    from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1

    # Patch both tables' _save_decoder_results methods globally
    with (
        patch.object(
            ClusterlessDecodingV1, "_save_decoder_results", _mock_save_results
        ),
        patch.object(
            SortedSpikesDecodingV1, "_save_decoder_results", _mock_save_results
        ),
    ):
        yield


# ============================================================================
# Mock Fixtures for ClusterlessDecodingV1
# ============================================================================


@pytest.fixture
def mock_clusterless_decoder():
    """Mock the _run_decoder helper for ClusterlessDecodingV1.

    Returns a function that can be used with monkeypatch to replace
    the real _run_decoder method.
    """
    import xarray as xr
    from scipy.ndimage import label

    def _mock_run_decoder(
        self,
        key,
        decoding_params,
        decoding_kwargs,
        position_info,
        position_variable_names,
        spike_times,
        spike_waveform_features,
        decoding_interval,
    ):
        """Mocked version of _run_decoder that returns fake results instantly.

        This mocks the expensive non_local_detector operations (~220s)
        while preserving all the Spyglass logic in make().

        Handles both estimate_decoding_params=True and False branches:
        - True: Returns results for ALL time points, with interval_labels
                using scipy.ndimage.label approach (-1 for outside intervals)
        - False: Returns results only for interval time points, with
                interval_labels using enumerate approach (0, 1, 2, ...)
        """
        classifier = create_fake_classifier()

        if key.get("estimate_decoding_params", False):
            # estimate_decoding_params=True branch:
            # Results span ALL time points in position_info
            all_time = position_info.index.to_numpy()

            # Create is_missing mask (same as real code)
            is_missing = np.ones(len(position_info), dtype=bool)
            for interval_start, interval_end in decoding_interval:
                is_missing[
                    np.logical_and(
                        position_info.index >= interval_start,
                        position_info.index <= interval_end,
                    )
                ] = False

            # Create fake results for all time points
            results = create_fake_decoding_results(
                n_time=len(all_time), n_position_bins=50, n_states=2
            )
            results = results.assign_coords(time=all_time)

            # Create interval_labels using scipy.ndimage.label (same as real code)
            labels_arr, _ = label(~is_missing)
            interval_labels = labels_arr - 1

            results = results.assign_coords(
                interval_labels=("time", interval_labels)
            )
        else:
            # estimate_decoding_params=False branch:
            # Results only for time points within intervals
            results_list = []
            interval_labels = []

            for interval_idx, (interval_start, interval_end) in enumerate(
                decoding_interval
            ):
                # Get time points for this interval
                interval_time = position_info.loc[
                    interval_start:interval_end
                ].index.to_numpy()

                if interval_time.size == 0:
                    continue

                # Create fake results for this interval
                interval_results = create_fake_decoding_results(
                    n_time=len(interval_time), n_position_bins=50, n_states=2
                )
                # Update time coordinates to match actual interval times
                interval_results = interval_results.assign_coords(
                    time=interval_time
                )
                results_list.append(interval_results)
                interval_labels.extend([interval_idx] * len(interval_time))

            # Concatenate along time dimension (as the real code now does)
            if len(results_list) == 1:
                results = results_list[0]
            else:
                results = xr.concat(results_list, dim="time")

            # Add interval_labels coordinate (as the real code now does)
            results = results.assign_coords(
                interval_labels=("time", interval_labels)
            )

        # Add metadata (same as real implementation)
        # initial_conditions: shape (n_states,) with explicit dims
        results["initial_conditions"] = xr.DataArray(
            classifier.initial_conditions_,
            dims=("state",),
            name="initial_conditions",
        )
        # discrete_state_transitions: shape (n_states, n_states) with explicit dims
        # Use different dim names to avoid "duplicate dimension" warning
        # Don't set coords - xarray will create default integer indices
        results["discrete_state_transitions"] = xr.DataArray(
            classifier.discrete_state_transitions_,
            dims=("state_from", "state_to"),
            name="discrete_state_transitions",
        )

        return classifier, results

    return _mock_run_decoder


@pytest.fixture
def mock_decoder_save(mock_results_storage):
    """Mock the _save_decoder_results helper for ClusterlessDecodingV1.

    Returns a function that can be used with monkeypatch to replace
    the real _save_decoder_results method. Creates netCDF files using
    the netcdf4 engine explicitly to avoid engine detection issues.
    """
    from pathlib import Path
    import uuid
    import pickle
    import os

    def _mock_save_results(self, classifier, results, key):
        """Mocked version that creates files with explicit netcdf4 engine."""
        # Generate unique identifier for this result set
        unique_id = str(uuid.uuid4())[:8]
        nwb_file_name = key["nwb_file_name"].replace("_.nwb", "")

        # Use absolute path to match environment expectations
        base_dir = os.environ.get("SPYGLASS_BASE_DIR", "tests/_data")
        analysis_dir = Path(base_dir) / "analysis"
        subdir = analysis_dir / nwb_file_name
        subdir.mkdir(parents=True, exist_ok=True)

        # Create file paths
        results_path = subdir / f"{nwb_file_name}_{unique_id}_mocked.nc"
        classifier_path = subdir / f"{nwb_file_name}_{unique_id}_mocked.pkl"

        # Write netCDF file with explicit netcdf4 engine
        results.to_netcdf(results_path, engine="netcdf4")

        # Write classifier as pickle (non-xarray object)
        with open(classifier_path, "wb") as f:
            pickle.dump(classifier, f)

        # Store actual data in memory for loading
        results_path_str = str(results_path)
        classifier_path_str = str(classifier_path)
        mock_results_storage["results"][results_path_str] = results
        mock_results_storage["classifiers"][classifier_path_str] = classifier

        return results_path_str, classifier_path_str

    return _mock_save_results


@pytest.fixture
def mock_detector_load_results(mock_results_storage):
    """Mock the detector load_results methods to use netcdf4 engine.

    This mocks both ClusterlessDetector.load_results and
    SortedSpikesDetector.load_results to read netCDF files with
    explicit netcdf4 engine specification.
    """
    import xarray as xr

    def _mock_load_results(filename):
        """Load results from disk with explicit netcdf4 engine."""
        # Convert Path to string if needed
        filename_str = str(filename)

        # Try loading from memory first (for tests that use in-memory storage)
        if filename_str in mock_results_storage["results"]:
            return mock_results_storage["results"][filename_str]

        # Load from disk with explicit engine
        try:
            return xr.open_dataset(filename_str, engine="netcdf4")
        except (FileNotFoundError, OSError) as e:
            # OSError with "Unknown file format" means old pickle file exists
            if "Unknown file format" in str(e):
                raise FileNotFoundError(
                    f"Mock result has invalid format (likely old pickle file): {filename_str}. "
                    "Please delete old *_mocked.nc files from tests/_data/analysis/"
                )
            raise FileNotFoundError(
                f"Mock result not found: {filename_str}. "
                "This usually means the test setup didn't properly mock "
                "the _save_decoder_results method."
            )

    return _mock_load_results


@pytest.fixture
def mock_detector_load_model(mock_results_storage):
    """Mock the detector load_model methods to use in-memory storage."""

    def _mock_load_model(filename):
        """Load classifier from in-memory storage instead of disk."""
        filename_str = str(filename)

        if filename_str in mock_results_storage["classifiers"]:
            return mock_results_storage["classifiers"][filename_str]

        raise FileNotFoundError(
            f"Mock classifier not found in memory: {filename_str}"
        )

    return _mock_load_model


# ============================================================================
# Mock Fixtures for SortedSpikesDecodingV1
# ============================================================================


@pytest.fixture
def mock_sorted_spikes_decoder():
    """Mock the _run_decoder helper for SortedSpikesDecodingV1."""
    import xarray as xr
    from scipy.ndimage import label

    def _mock_run_decoder(
        self,
        key,
        decoding_params,
        decoding_kwargs,
        position_info,
        position_variable_names,
        spike_times,
        decoding_interval,
    ):
        """Mocked version that returns fake results instantly.

        Handles both estimate_decoding_params=True and False branches:
        - True: Returns results for ALL time points, with interval_labels
                using scipy.ndimage.label approach (-1 for outside intervals)
        - False: Returns results only for interval time points, with
                interval_labels using enumerate approach (0, 1, 2, ...)
        """
        classifier = create_fake_classifier()

        if key.get("estimate_decoding_params", False):
            # estimate_decoding_params=True branch:
            # Results span ALL time points in position_info
            all_time = position_info.index.to_numpy()

            # Create is_missing mask (same as real code)
            is_missing = np.ones(len(position_info), dtype=bool)
            for interval_start, interval_end in decoding_interval:
                is_missing[
                    np.logical_and(
                        position_info.index >= interval_start,
                        position_info.index <= interval_end,
                    )
                ] = False

            # Create fake results for all time points
            results = create_fake_decoding_results(
                n_time=len(all_time), n_position_bins=50, n_states=2
            )
            results = results.assign_coords(time=all_time)

            # Create interval_labels using scipy.ndimage.label (same as real code)
            labels_arr, _ = label(~is_missing)
            interval_labels = labels_arr - 1

            results = results.assign_coords(
                interval_labels=("time", interval_labels)
            )
        else:
            # estimate_decoding_params=False branch:
            # Results only for time points within intervals
            results_list = []
            interval_labels = []

            for interval_idx, (interval_start, interval_end) in enumerate(
                decoding_interval
            ):
                # Get time points for this interval
                interval_time = position_info.loc[
                    interval_start:interval_end
                ].index.to_numpy()

                if interval_time.size == 0:
                    continue

                # Create fake results for this interval
                interval_results = create_fake_decoding_results(
                    n_time=len(interval_time), n_position_bins=50, n_states=2
                )
                # Update time coordinates to match actual interval times
                interval_results = interval_results.assign_coords(
                    time=interval_time
                )
                results_list.append(interval_results)
                interval_labels.extend([interval_idx] * len(interval_time))

            # Concatenate along time dimension (as the real code now does)
            if len(results_list) == 1:
                results = results_list[0]
            else:
                results = xr.concat(results_list, dim="time")

            # Add interval_labels coordinate (as the real code now does)
            results = results.assign_coords(
                interval_labels=("time", interval_labels)
            )

        # Add metadata (same as real implementation)
        # initial_conditions: shape (n_states,) with explicit dims
        results["initial_conditions"] = xr.DataArray(
            classifier.initial_conditions_,
            dims=("state",),
            name="initial_conditions",
        )
        # discrete_state_transitions: shape (n_states, n_states) with explicit dims
        # Use different dim names to avoid "duplicate dimension" warning
        # Don't set coords - xarray will create default integer indices
        results["discrete_state_transitions"] = xr.DataArray(
            classifier.discrete_state_transitions_,
            dims=("state_from", "state_to"),
            name="discrete_state_transitions",
        )

        return classifier, results

    return _mock_run_decoder
