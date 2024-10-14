import numpy as np
import pytest


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
    common.IntervalList.insert1(
        {
            **mini_dict,
            "interval_list_name": decode_interval_name,
            "valid_times": [[raw_begin, raw_begin + 15]],
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
):
    _ = pop_pos_group, group_unitwave  # ensure populated
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
):
    _ = pop_pos_group, group_unitwave
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
):
    _ = pop_spikes_group, pop_pos_group  # ensure populated
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
):
    _ = pop_spikes_group, pop_pos_group  # ensure populated
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
