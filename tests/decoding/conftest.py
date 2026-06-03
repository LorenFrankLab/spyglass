from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_netcdf_saves():
    """Globally mock netCDF file writes to avoid HDF5 I/O conflicts in CI.

    This prevents RuntimeError: NetCDF: HDF error during parallel test execution
    by intercepting xarray.Dataset.to_netcdf() calls and using pickle instead.
    """
    import pickle

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

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Keep the .nc extension to match expectations, but write pickle format
        # This avoids netCDF4/HDF5 errors while maintaining file path compatibility
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except (FileNotFoundError, PermissionError, OSError):
            # Copilot suggested that this is where a file might throw error
            # during teardown, attempted automatic cleanup.
            pass

        return None

    with patch("xarray.Dataset.to_netcdf", mock_to_netcdf):
        yield


@pytest.fixture(scope="session")
def result_coordinates():
    """Expected coordinates in decoding results.

    Note: interval_labels is added by the new concatenation approach
    that removed the intervals dimension.
    """
    return {
        "interval_labels",
        "states",
        "state_bins",
        "time",
    }


@pytest.fixture(scope="session")
def decode_v1(common, trodes_pos_v1):
    from spyglass.decoding import v1

    yield v1


# ============================================================================
# Synthetic, SpikeInterface-version-agnostic spike fixtures
# ============================================================================
# The legacy v1 spikesorting pipeline (SpikeSorting -> CurationV1 -> waveform
# extraction) uses SpikeInterface APIs removed in SI >= 0.101, which produces
# setup ERRORs under the SI 0.104 CI job. These fixtures synthesize the
# decoding inputs directly on the minirec session (which already carries Trodes
# position) so the decoding tests run under ANY SI version. minirec supplies
# position; we synthesize the spikes. Everything keys off the single minirec
# ``nwb_file_name`` so the decoding selections' same-session FKs hold.


@pytest.fixture(scope="session")
def synthetic_spike_window(common, mini_dict, pos_interval, decode_interval):
    """Time window that lies inside BOTH the encoding and decoding intervals.

    Synthetic spike times must fall inside the encoding interval
    (``pos_interval``) and the decoding interval (``decode_interval``) so they
    survive ``fetch_spike_data``'s interval filtering. Return the overlap
    ``[max(starts), min(ends)]`` of the two intervals' valid_times.
    """
    enc = (
        common.IntervalList & {**mini_dict, "interval_list_name": pos_interval}
    ).fetch1("valid_times")
    dec = (
        common.IntervalList
        & {**mini_dict, "interval_list_name": decode_interval}
    ).fetch1("valid_times")
    start = max(float(enc[0][0]), float(dec[0][0]))
    end = min(float(enc[-1][-1]), float(dec[-1][-1]))
    assert end > start, "Encoding/decoding intervals do not overlap."
    yield (start, end)


@pytest.fixture(scope="session")
def imported_merge_id(common, mini_dict, mini_insert, synthetic_spike_window):
    """Synthetic ImportedSpikeSorting -> SpikeSortingOutput merge_id on minirec.

    Appends a single synthetic ``units`` table (one unit, id 0) to the
    already-registered minirec ``_`` copy on disk, reconciles the DataJoint
    ``filepath@raw`` external checksum to the modified file, then ingests it via
    ``ImportedSpikeSorting.insert_from_nwbfile``. That method already runs
    ``SpikeSortingOutput._merge_insert``, so the merge_id is available from the
    merge part table.

    The checksum landmine (resolved via the in-place + refresh fallback,
    option 2): ``ImportedSpikeSorting.fetch_nwb`` resolves the raw file through
    the external store (``download_filepath``), which checks the stored
    ``size``/``contents_hash`` against the on-disk file. Appending units changes
    both, so we update the external row to match. ``Nwbfile.get_abs_path``
    (used by the position reads) builds the path from ``raw_dir + name`` and
    never touches the external store, which is why position was unaffected and
    writing units BEFORE registration (option 1) was unnecessary.
    """
    import pynwb
    from datajoint.hash import uuid_from_file

    from spyglass.common import Nwbfile
    from spyglass.common.common_nwbfile import schema as nwbfile_schema
    from spyglass.spikesorting.imported import ImportedSpikeSorting
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.utils.nwb_helper_fn import close_nwb_files

    nwb_file_name = mini_dict["nwb_file_name"]
    start, end = synthetic_spike_window

    # 30 spikes evenly spaced inside the overlap window (margin off the edges)
    margin = 0.05 * (end - start)
    spike_times = np.linspace(start + margin, end - margin, 30)

    abs_path = Nwbfile.get_abs_path(nwb_file_name)

    # Spyglass caches open NWB handles (read-only) in a module-level dict; the
    # minirec ``_`` copy is already cached by the time the position/ingest
    # fixtures have run, so opening it ``mode="a"`` would raise "file is already
    # open for read-only". Close cached handles first; the next reader re-opens
    # a fresh handle that sees the appended units. (Position data is already
    # materialized into PositionGroup/pos_merge, so subsequent re-reads are
    # safe.) This mirrors the ``mini_insert`` teardown's ``close_nwb_files``.
    close_nwb_files()

    # Append a units table to the registered ``_`` copy in place, only if it
    # does not already carry one (idempotent across --no-teardown reruns where
    # the persistent raw file already carries the synthetic units).
    with pynwb.NWBHDF5IO(path=abs_path, mode="a", load_namespaces=True) as io:
        nwbf = io.read()
        if getattr(nwbf, "units", None) is None:
            nwbf.add_unit(spike_times=spike_times, id=0)
            io.write(nwbf)
    close_nwb_files()

    # Reconcile the external ``filepath@raw`` row to the current on-disk file.
    # Done unconditionally so a prior partial run (units appended but external
    # row not yet refreshed) is also healed before ingestion reads the file.
    ext = nwbfile_schema.external["raw"]
    ext_key = (ext & 'filepath = "%s"' % Path(abs_path).name).fetch1("KEY")
    ext.update1(
        {
            **ext_key,
            "size": Path(abs_path).stat().st_size,
            "contents_hash": uuid_from_file(abs_path),
        }
    )

    # Idempotent across --no-teardown reruns: ``insert_from_nwbfile`` inserts
    # with ``skip_duplicates=False`` and would raise on a pre-existing row.
    if not (ImportedSpikeSorting & mini_dict):
        ImportedSpikeSorting().insert_from_nwbfile(nwb_file_name)

    merge_id = (SpikeSortingOutput.ImportedSpikeSorting & mini_dict).fetch1(
        "merge_id"
    )
    yield merge_id


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
def clusterless_params():
    """No-op restriction.

    With the synthetic pipeline there is no sorter parameter set to restrict
    on; the merge_id comes from ``imported_merge_id`` instead. Kept (yielding
    ``{}``) so dependent fixtures/keys that spread it stay valid.
    """
    yield {}


@pytest.fixture(scope="session")
def clusterless_mergeids(imported_merge_id):
    """Single synthetic SpikeSortingOutput merge_id (one unit total)."""
    yield [imported_merge_id]


@pytest.fixture(scope="session")
def pop_unitwave(decode_v1, waveform_params, clusterless_mergeids, mini_dict):
    """Direct-insert synthetic ``UnitWaveformFeatures`` rows (no populate).

    Reuses the production writer ``_write_waveform_features_to_nwb`` to build a
    units table carrying ``spike_times`` plus an ``amplitude`` feature of shape
    ``(n_spikes, 3)``, then inserts the selection + computed rows directly (a
    fixture shortcut for ``UnitWaveformFeatures.make``). Avoids all SI waveform
    extraction so this works under any SI version. Exactly one merge_id, one
    feature row, one unit (id 0) -> clusterless group resolves to a single unit.
    """
    import pandas as pd

    from spyglass.common import AnalysisNwbfile
    from spyglass.decoding.v1.waveform_features import (
        _write_waveform_features_to_nwb,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    wave = decode_v1.waveform_features
    nwb_file_name = mini_dict["nwb_file_name"]

    class _StubWaveforms:
        """Minimal ``waveforms`` surface used by the writer when features are
        supplied: only ``.sorting.get_unit_ids()`` is read (the SI waveform
        extraction path is not entered)."""

        class _Sorting:
            @staticmethod
            def get_unit_ids():
                return np.array([0])

        sorting = _Sorting()

    sel_keys = []
    for merge_id in clusterless_mergeids:
        # Read this unit's spike_times back from the synthetic units table so
        # the feature rows align 1:1 with the persisted spikes.
        nwb_file = (SpikeSortingOutput & {"merge_id": merge_id}).fetch_nwb()[0]
        unit_field = "object_id" if "object_id" in nwb_file else "units"
        unit_spike_times = np.asarray(
            nwb_file[unit_field]["spike_times"].iloc[0]
        )
        n_spikes = len(unit_spike_times)

        # ``_write_waveform_features_to_nwb`` calls ``spike_times.loc[unit_id]``
        # and expects the per-unit spike array back. Production passes
        # ``units["spike_times"]`` -- a pandas *Series* indexed by unit_id whose
        # cells are 1D arrays -- so mirror that (a DataFrame would yield a
        # Series row and break ``add_unit``).
        spike_times_df = pd.Series({0: unit_spike_times}, name="spike_times")
        waveform_features = {
            "amplitude": {
                0: np.zeros((n_spikes, 3), dtype=np.float32),
            }
        }

        analysis_file_name, object_id = _write_waveform_features_to_nwb(
            nwb_file_name,
            _StubWaveforms(),
            spike_times_df,
            waveform_features,
        )
        AnalysisNwbfile().add(nwb_file_name, analysis_file_name)

        sel_key = {"spikesorting_merge_id": merge_id, **waveform_params}
        wave.UnitWaveformFeaturesSelection.insert1(
            sel_key, skip_duplicates=True
        )
        wave.UnitWaveformFeatures.insert1(
            {
                **sel_key,
                "analysis_file_name": analysis_file_name,
                "object_id": object_id,
            },
            skip_duplicates=True,
            allow_direct_insert=True,  # dj.Computed; fixture bypasses make()
        )
        sel_keys.append(sel_key)

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
def pop_spikes_group(group_name, mini_dict, imported_merge_id):
    """Build the sorted ``SortedSpikesGroup`` from the synthetic merge_id.

    Shadows the root SI-pipeline ``pop_spikes_group`` (tests/conftest.py),
    which is built on ``pop_spike_merge`` (CurationV1 -> SI waveform path) and
    errors under SI >= 0.101. The root chain is left untouched for the
    spikesorting suite; this override only changes how the decoding suite's
    sorted group is sourced.
    """
    from spyglass.spikesorting.analysis.v1 import group as spike_v1_group

    spike_v1_group.UnitSelectionParams().insert_default()
    spike_v1_group.SortedSpikesGroup().create_group(
        **mini_dict,
        group_name=group_name,
        keys=[{"spikesorting_merge_id": imported_merge_id}],
        unit_filter_params_name="default_exclusion",
    )
    yield spike_v1_group.SortedSpikesGroup().fetch("KEY", as_dict=True)[0]


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
        try:
            decode_merge.cleanup()
        except Exception:
            pass


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
        try:
            decode_merge.cleanup()
        except Exception:
            pass


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

    In non_local_detector, state_bins is a MultiIndex combining position and state,
    so n_state_bins = n_position_bins * n_states. The initial_conditions_ has shape
    (n_state_bins,) and discrete_state_transitions_ has shape (n_states, n_states).
    """

    def __init__(self, n_state_bins=100, n_states=2):
        # initial_conditions_: shape (n_state_bins,) - one value per (position, state) combination
        # Initialize with uniform distribution
        self.initial_conditions_ = np.ones(n_state_bins) / n_state_bins

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

    Notes
    -----
    In non_local_detector, state_bins is a MultiIndex combining (position, state),
    so n_state_bins = n_position_bins * n_states. The posterior has shape
    (n_time, n_state_bins) in the flattened form.
    """
    import xarray as xr

    time = np.linspace(0, 10, n_time)
    state_names = ["Continuous", "Fragmented"][:n_states]

    # n_state_bins is the product of position bins and states
    n_state_bins = n_position_bins * n_states

    # Create state_bins coordinate values (flattened position x state)
    # This mimics the MultiIndex structure in non_local_detector
    state_bins_values = np.arange(n_state_bins)

    # Create realistic-looking posterior probabilities
    # Shape: (n_time, n_state_bins) - flattened across position and state
    # Center posterior at middle of position bins
    POSITION_CENTER_FRACTION = 0.5
    position_mean = int(n_position_bins * POSITION_CENTER_FRACTION)
    # Spread posterior over ~10% of position range for realistic Gaussian shape
    POSITION_SPREAD_FRACTION = 0.1
    position_std = max(1, int(n_position_bins * POSITION_SPREAD_FRACTION))

    posterior = np.zeros((n_time, n_state_bins))
    for t in range(n_time):
        for s in range(n_states):
            # Each state gets a slice of state_bins
            start_idx = s * n_position_bins
            end_idx = (s + 1) * n_position_bins
            # Gaussian-like posterior for each state
            posterior[t, start_idx:end_idx] = np.exp(
                -((np.arange(n_position_bins) - position_mean) ** 2)
                / (2 * position_std**2)
            )
        # Normalize across all state_bins
        posterior[t] /= posterior[t].sum()

    # Create results matching non_local_detector output structure
    # Primary dimensions: time, state_bins
    results = xr.Dataset(
        {
            "acausal_posterior": (["time", "state_bins"], posterior),
            "causal_posterior": (["time", "state_bins"], posterior * 0.9),
        },
        coords={
            "time": time,
            "state_bins": state_bins_values,
            # states coordinate with state names
            "states": ("states", state_names),
        },
    )

    return results


# ============================================================================
# In-Memory Mock Storage (avoids netCDF4/HDF5 file I/O issues in CI)
# ============================================================================


def _create_mock_save_function(mock_results_storage):
    """Create a mock save function for decoder results.

    This shared helper contains the common logic for both the global
    mock_save_decoder_results_globally fixture and the mock_decoder_save
    fixture. It saves results to netCDF with explicit engine and stores
    data in memory for loading.

    Parameters
    ----------
    mock_results_storage : dict
        Dictionary with 'results' and 'classifiers' keys for in-memory storage

    Returns
    -------
    callable
        Mock save function compatible with _save_decoder_results signature
    """
    import os
    import pickle
    import uuid
    from pathlib import Path

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

    import xarray as xr
    from non_local_detector.models.base import (
        ClusterlessDetector,
        SortedSpikesDetector,
    )

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
            # (from a prior --no-teardown run where mock_netcdf_saves wrote
            # pickle format). Fall back to pickle to keep reruns working.
            if "Unknown file format" in str(e):
                import pickle

                try:
                    with open(filename_str, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    raise FileNotFoundError(
                        f"Mock result has invalid format: {filename_str}. "
                        "Please delete old *_mocked.nc files from tests/_data/analysis/"
                    )
            raise FileNotFoundError(f"Mock result not found: {filename_str}")

    def _mock_load_model(filename):
        """Load classifier from in-memory storage or disk."""
        filename_str = str(filename)
        if filename_str in mock_results_storage["classifiers"]:
            return mock_results_storage["classifiers"][filename_str]
        # Fall back to disk for --no-teardown reruns where in-memory is empty
        import pickle

        try:
            with open(filename_str, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, OSError):
            raise FileNotFoundError(
                f"Mock classifier not found in memory or on disk: {filename_str}"
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

    # Use shared helper to create mock save function
    _mock_save_results = _create_mock_save_function(mock_results_storage)

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
# Shared Mock Decoder Logic
# ============================================================================


def _create_mock_decoder_results(key, position_info, decoding_interval):
    """Create mock decoder results for testing.

    This shared helper contains the common logic for both clusterless and
    sorted spikes mock decoders. It handles both estimate_decoding_params
    branches and returns a fake classifier and results dataset.

    Parameters
    ----------
    key : dict
        Decoding selection key containing estimate_decoding_params flag
    position_info : pd.DataFrame
        Position data with time index
    decoding_interval : np.ndarray
        Array of (start, end) interval tuples

    Returns
    -------
    classifier : object
        Fake classifier with required attributes
    results : xr.Dataset
        Mock decoding results with interval_labels coordinate
    """
    import xarray as xr
    from scipy.ndimage import label

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
    results["initial_conditions"] = xr.DataArray(
        classifier.initial_conditions_,
        dims=("state_bins",),
        coords={"state_bins": results.coords["state_bins"]},
        name="initial_conditions",
    )
    results["discrete_state_transitions"] = xr.DataArray(
        classifier.discrete_state_transitions_,
        dims=("states_from", "states_to"),
        name="discrete_state_transitions",
    )

    return classifier, results


# ============================================================================
# Mock Fixtures for ClusterlessDecodingV1
# ============================================================================


@pytest.fixture
def mock_clusterless_decoder():
    """Mock the _run_decoder helper for ClusterlessDecodingV1.

    Returns a function that can be used with monkeypatch to replace
    the real _run_decoder method.
    """

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
        """Mocked version of _run_decoder that returns fake results instantly."""
        return _create_mock_decoder_results(
            key, position_info, decoding_interval
        )

    return _mock_run_decoder


@pytest.fixture
def mock_decoder_save(mock_results_storage):
    """Mock the _save_decoder_results helper for ClusterlessDecodingV1.

    Returns a function that can be used with monkeypatch to replace
    the real _save_decoder_results method. Creates netCDF files using
    the netcdf4 engine explicitly to avoid engine detection issues.
    """
    # Use shared helper to create mock save function
    return _create_mock_save_function(mock_results_storage)


# ============================================================================
# Mock Fixtures for SortedSpikesDecodingV1
# ============================================================================


@pytest.fixture
def mock_sorted_spikes_decoder():
    """Mock the _run_decoder helper for SortedSpikesDecodingV1.

    Returns a function that can be used with monkeypatch to replace
    the real _run_decoder method.
    """

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
        """Mocked version of _run_decoder that returns fake results instantly."""
        return _create_mock_decoder_results(
            key, position_info, decoding_interval
        )

    return _mock_run_decoder
