import matplotlib.pyplot as plt
import numpy as np
import pytest

# from non_local_detector.models import (
#     ContFragClusterlessClassifier,
#     ContFragSortedSpikesClassifier,
# )
#
# import spyglass.position as sgp
# import spyglass.spikesorting.v1 as spike_v1
# from spyglass.common import IntervalList
# from spyglass.decoding import SortedSpikesDecodingSelection
# from spyglass.decoding.decoding_merge import DecodingOutput
# from spyglass.decoding.v1.clusterless import (
#     ClusterlessDecodingSelection,
#     ClusterlessDecodingV1,
#     UnitWaveformFeaturesGroup,
# )
# from spyglass.decoding.v1.core import DecodingParameters, PositionGroup
# from spyglass.decoding.v1.sorted_spikes import (
#     SortedSpikesDecodingV1,
#     SortedSpikesGroup,
# )
# from spyglass.decoding.v1.waveform_features import (
#     UnitWaveformFeatures,
#     UnitWaveformFeaturesSelection,
#     WaveformFeaturesParams,
# )
# from spyglass.position import PositionOutput
# from spyglass.spikesorting.analysis.v1.group import (
#     SortedSpikesGroup,
#     UnitSelectionParams,
# )
# from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
# from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
#
# nwb_file_name = "mediumnwb20230802.nwb"
# nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
# mini_dict = {"nwb_file_name": nwb_copy_file_name}
# interval_list_name = "pos 0 valid times"
# trodes_s_key = {
#     "nwb_file_name": nwb_copy_file_name,
#     "interval_list_name": interval_list_name,
#     "trodes_pos_params_name": "default",
# }


@pytest.fixture(scope="session")
def decode_v1(common, trodes_pos_v1):
    from spyglass.decoding import v1

    yield v1


@pytest.fixture(scope="session")
def recording_ids(spike_v1, mini_dict, pop_rec, pop_art):
    _ = pop_rec  # set group by shank

    # FROM NOTEBOOK, trying existing items from pop_rec
    # sort_group_ids = (spike_v1.SortGroup & mini_dict).fetch("sort_group_id")
    # group_keys = []
    # for sort_group_id in sort_group_ids:
    #     key = {
    #         "nwb_file_name": nwb_copy_file_name,
    #         "sort_group_id": sort_group_id,
    #         "interval_list_name": interval_list_name,
    #         "preproc_param_name": "default",
    #         "team_name": "Alison Comrie",
    #     }
    #     group_keys.append(key)
    #     sgs.SpikeSortingRecordingSelection.insert_selection(key)
    # sgs.SpikeSortingRecording.populate(group_keys)

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
def waveform_params(decode_v1):
    param_pk = {"features_param_name": "low_thresh_amplitude"}
    decode_v1.waveform_features.WaveformFeaturesParams.insert1(
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
                        "estimate_peak_time": False,
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
    decode_v1, mini_dict, clusterless_mergeids, pop_unitwave, waveform_params
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
        "waveform_features_group_name": "test_group"
    }


@pytest.fixture(scope="session")
def pop_pos_group(decode_v1, mini_dict, pos_merge):
    merge_key = (
        (
            pos_merge.TrodesPosV1
            & 'trodes_pos_params_name = "single_led_upsampled"'
        )
        .proj(pos_merge_id="merge_id")
        .fetch(as_dict=True)
    )

    decode_v1.core.PositionGroup().create_group(
        **mini_dict,
        group_name="test_group",
        keys=merge_key,
    )

    yield decode_v1.core.PositionGroup & {
        **mini_dict,
        "position_group_name": "test_group",
    }


# position_merge_ids = (
#     PositionOutput.TrodesPosV1
#     & {
#         "nwb_file_name": nwb_copy_file_name,
#         "interval_list_name": "pos 0 valid times",
#         "trodes_pos_params_name": "default_decoding",
#     }
# ).fetch("merge_id")
#
# PositionGroup().create_group(
#     nwb_file_name=nwb_copy_file_name,
#     group_name="test_group",
#     keys=[{"pos_merge_id": merge_id} for merge_id in position_merge_ids],
#     upsample_rate=500,
# )
#
# (
#     PositionGroup
#     & {"nwb_file_name": nwb_copy_file_name, "position_group_name": "test_group"}
# ).fetch1("position_variables")
#
#
# ContFragClusterlessClassifier(
#     clusterless_algorithm_params={
#         "block_size": 10000,
#         "position_std": 12.0,
#         "waveform_std": 24.0,
#     },
# )
#
#
# DecodingParameters.insert1(
#     {
#         "decoding_param_name": "contfrag_clusterless",
#         "decoding_params": ContFragClusterlessClassifier(),
#         "decoding_kwargs": dict(),
#     },
#     skip_duplicates=True,
# )
#
#
# model_params = (
#     DecodingParameters & {"decoding_param_name": "contfrag_clusterless"}
# ).fetch1()
#
# ContFragClusterlessClassifier(**model_params["decoding_params"])
#
#
# decoding_interval_valid_times = [
#     [1625935714.6359036, 1625935714.6359036 + 15.0]
# ]
#
# IntervalList.insert1(
#     {
#         "nwb_file_name": "mediumnwb20230802_.nwb",
#         "interval_list_name": "test decoding interval",
#         "valid_times": decoding_interval_valid_times,
#     },
#     skip_duplicates=True,
# )
#
#
# selection_key = {
#     "waveform_features_group_name": "test_group",
#     "position_group_name": "test_group",
#     "decoding_param_name": "contfrag_clusterless",
#     "nwb_file_name": nwb_copy_file_name,
#     "encoding_interval": "pos 0 valid times",
#     "decoding_interval": "test decoding interval",
#     "estimate_decoding_params": False,
# }
#
# ClusterlessDecodingSelection.insert1(
#     selection_key,
#     skip_duplicates=True,
# )
# ClusterlessDecodingV1.populate(selection_key)
#
#
# decoding_results = (ClusterlessDecodingV1 & selection_key).fetch_results()
#
#
# DecodingOutput().cleanup()
#
#
# sgp.v1.TrodesPosParams.insert_default()
#
# interval_list_name = "pos 0 valid times"
#
# trodes_s_key = {
#     "nwb_file_name": nwb_copy_file_name,
#     "interval_list_name": interval_list_name,
#     "trodes_pos_params_name": "default",
# }
# sgp.v1.TrodesPosSelection.insert1(
#     trodes_s_key,
#     skip_duplicates=True,
# )
# sgp.v1.TrodesPosV1.populate(trodes_s_key)
#
#
# spike_v1.SortGroup.set_group_by_shank(nwb_file_name=nwb_copy_file_name)
#
# sort_group_ids = (
#     spike_v1.SortGroup & {"nwb_file_name": nwb_copy_file_name}
# ).fetch("sort_group_id")
#
# group_keys = []
# for sort_group_id in sort_group_ids:
#     key = {
#         "nwb_file_name": nwb_copy_file_name,
#         "sort_group_id": sort_group_id,
#         "interval_list_name": interval_list_name,
#         "preproc_param_name": "default",
#         "team_name": "Alison Comrie",
#     }
#     group_keys.append(key)
#     spike_v1.SpikeSortingRecordingSelection.insert_selection(key)
#
# spike_v1.SpikeSortingRecording.populate(group_keys)
#
#
# recording_ids = (
#     spike_v1.SpikeSortingRecordingSelection
#     & {"nwb_file_name": nwb_copy_file_name}
# ).fetch("recording_id")
#
# group_keys = []
# for recording_id in recording_ids:
#     key = {
#         "recording_id": recording_id,
#         "artifact_param_name": "none",
#     }
#     group_keys.append(key)
#     spike_v1.ArtifactDetectionSelection.insert_selection(key)
#
# spike_v1.ArtifactDetection.populate(group_keys)
#
#
# group_keys = []
# for recording_id in recording_ids:
#     key = {
#         "recording_id": recording_id,
#         "sorter": "clusterless_thresholder",
#         "sorter_param_name": "default_clusterless",
#         "nwb_file_name": nwb_copy_file_name,
#         "interval_list_name": str(
#             (
#                 spike_v1.ArtifactDetectionSelection
#                 & {"recording_id": recording_id}
#             ).fetch1("artifact_id")
#         ),
#     }
#     group_keys.append(key)
#     spike_v1.SpikeSortingSelection.insert_selection(key)
#
# spike_v1.SpikeSorting.populate(group_keys)
#
#
# sorting_ids = (
#     spike_v1.SpikeSortingSelection & {"nwb_file_name": nwb_copy_file_name}
# ).fetch("sorting_id")
#
# for sorting_id in sorting_ids:
#     try:
#         spike_v1.CurationV1.insert_curation(sorting_id=sorting_id)
#     except KeyError:
#         pass
#
# SpikeSortingOutput.insert(
#     spike_v1.CurationV1().fetch("KEY"),
#     part_name="CurationV1",
#     skip_duplicates=True,
# )
#
#
# waveform_extraction_params = {
#     "ms_before": 0.5,
#     "ms_after": 0.5,
#     "max_spikes_per_unit": None,
#     "n_jobs": 5,
#     "total_memory": "5G",
# }
# waveform_feature_params = {
#     "amplitude": {
#         "peak_sign": "neg",
#         "estimate_peak_time": False,
#     }
# }
#
# WaveformFeaturesParams.insert1(
#     {
#         "features_param_name": "amplitude",
#         "params": {
#             "waveform_extraction_params": waveform_extraction_params,
#             "waveform_feature_params": waveform_feature_params,
#         },
#     },
#     skip_duplicates=True,
# )
#
# merge_ids = (
#     (SpikeSortingOutput.CurationV1 * spike_v1.SpikeSortingSelection)
#     & {
#         "nwb_file_name": nwb_copy_file_name,
#         "sorter": "clusterless_thresholder",
#         "sorter_param_name": "default_clusterless",
#     }
# ).fetch("merge_id")
#
#
# selection_keys = [
#     {
#         "spikesorting_merge_id": merge_id,
#         "features_param_name": "amplitude",
#     }
#     for merge_id in merge_ids
# ]
# UnitWaveformFeaturesSelection.insert(selection_keys, skip_duplicates=True)
#
#
# UnitWaveformFeatures.populate(selection_keys)
#
#
# spike_times, spike_waveform_features = (
#     UnitWaveformFeatures & selection_keys
# ).fetch_data()
#
#
# tetrode_ind = 1
# plt.scatter(
#     spike_waveform_features[tetrode_ind][:, 0],
#     spike_waveform_features[tetrode_ind][:, 1],
#     s=1,
# )
#
#
# sorter_keys = {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorter": "clusterless_thresholder",
#     "sorter_param_name": "default_clusterless",
# }
#
# feature_key = {"features_param_name": "amplitude"}
#
# (
#     spike_v1.SpikeSortingSelection & sorter_keys
# ) * SpikeSortingOutput.CurationV1 * (
#     UnitWaveformFeaturesSelection.proj(merge_id="spikesorting_merge_id")
#     & feature_key
# )
#
#
# spikesorting_merge_id = (
#     (spike_v1.SpikeSortingSelection & sorter_keys)
#     * SpikeSortingOutput.CurationV1
#     * (
#         UnitWaveformFeaturesSelection.proj(merge_id="spikesorting_merge_id")
#         & feature_key
#     )
# ).fetch("merge_id")
#
# waveform_selection_keys = [
#     {"spikesorting_merge_id": merge_id, "features_param_name": "amplitude"}
#     for merge_id in spikesorting_merge_id
# ]
#
#
# UnitWaveformFeaturesGroup().create_group(
#     nwb_file_name=nwb_copy_file_name,
#     group_name="test_group",
#     keys=waveform_selection_keys,
# )
# UnitWaveformFeaturesGroup & {"waveform_features_group_name": "test_group"}
#
#
# UnitWaveformFeaturesGroup.UnitFeatures & {
#     "nwb_file_name": nwb_copy_file_name,
#     "waveform_features_group_name": "test_group",
# }
#
#
# sgp.v1.TrodesPosParams.insert1(
#     {
#         "trodes_pos_params_name": "default_decoding",
#         "params": {
#             "max_LED_separation": 9.0,
#             "max_plausible_speed": 300.0,
#             "position_smoothing_duration": 0.125,
#             "speed_smoothing_std_dev": 0.100,
#             "orient_smoothing_std_dev": 0.001,
#             "led1_is_front": 1,
#             "is_upsampled": 1,
#             "upsampling_sampling_rate": 250,
#             "upsampling_interpolation_method": "linear",
#         },
#     },
#     skip_duplicates=True,
# )
#
# trodes_s_key = {
#     "nwb_file_name": nwb_copy_file_name,
#     "interval_list_name": "pos 0 valid times",
#     "trodes_pos_params_name": "default_decoding",
# }
# sgp.v1.TrodesPosSelection.insert1(
#     trodes_s_key,
#     skip_duplicates=True,
# )
# sgp.v1.TrodesPosV1.populate(trodes_s_key)
#
#
# position_merge_ids = (
#     PositionOutput.TrodesPosV1
#     & {
#         "nwb_file_name": nwb_copy_file_name,
#         "interval_list_name": "pos 0 valid times",
#         "trodes_pos_params_name": "default_decoding",
#     }
# ).fetch("merge_id")
#
# PositionGroup().create_group(
#     nwb_file_name=nwb_copy_file_name,
#     group_name="test_group",
#     keys=[{"pos_merge_id": merge_id} for merge_id in position_merge_ids],
# )
#
# PositionGroup & {
#     "nwb_file_name": nwb_copy_file_name,
#     "position_group_name": "test_group",
# }
#
# (
#     PositionGroup
#     & {"nwb_file_name": nwb_copy_file_name, "position_group_name": "test_group"}
# ).fetch1("position_variables")
#
# PositionGroup.Position & {
#     "nwb_file_name": nwb_copy_file_name,
#     "position_group_name": "test_group",
# }
#
#
# ContFragClusterlessClassifier(
#     clusterless_algorithm_params={
#         "block_size": 10000,
#         "position_std": 12.0,
#         "waveform_std": 24.0,
#     },
# )
#
#
# DecodingParameters.insert1(
#     {
#         "decoding_param_name": "contfrag_clusterless",
#         "decoding_params": ContFragClusterlessClassifier(),
#         "decoding_kwargs": dict(),
#     },
#     skip_duplicates=True,
# )
#
# DecodingParameters & {"decoding_param_name": "contfrag_clusterless"}
#
#
# model_params = (
#     DecodingParameters & {"decoding_param_name": "contfrag_clusterless"}
# ).fetch1()
#
# ContFragClusterlessClassifier(**model_params["decoding_params"])
#
#
# decoding_interval_valid_times = [
#     [1625935714.6359036, 1625935714.6359036 + 15.0]
# ]
#
# IntervalList.insert1(
#     {
#         "nwb_file_name": "mediumnwb20230802_.nwb",
#         "interval_list_name": "test decoding interval",
#         "valid_times": decoding_interval_valid_times,
#     },
#     skip_duplicates=True,
# )
#
#
# selection_key = {
#     "waveform_features_group_name": "test_group",
#     "position_group_name": "test_group",
#     "decoding_param_name": "contfrag_clusterless",
#     "nwb_file_name": nwb_copy_file_name,
#     "encoding_interval": "pos 0 valid times",
#     "decoding_interval": "test decoding interval",
#     "estimate_decoding_params": False,
# }
#
# ClusterlessDecodingSelection.insert1(
#     selection_key,
#     skip_duplicates=True,
# )
#
#
# ClusterlessDecodingV1.populate(selection_key)
#
#
# decoding_results = (ClusterlessDecodingV1 & selection_key).fetch_results()
# decoding_results
#
#
# DecodingOutput().cleanup()
#
#
# UnitSelectionParams().insert_default()
#
# unit_filter_params_name = "default_exclusion"
#
# sorter_keys = {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorter": "mountainsort4",
#     "curation_id": 1,
# }
# (
#     spike_v1.SpikeSortingSelection & sorter_keys
# ) * SpikeSortingOutput.CurationV1 & sorter_keys
#
#
# spikesorting_merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
#     sorter_keys, restrict_by_artifact=False
# )
#
# unit_filter_params_name = "default_exclusion"
# SortedSpikesGroup().create_group(
#     group_name="test_group",
#     nwb_file_name=nwb_copy_file_name,
#     keys=[
#         {"spikesorting_merge_id": merge_id}
#         for merge_id in spikesorting_merge_ids
#     ],
#     unit_filter_params_name=unit_filter_params_name,
# )
# SortedSpikesGroup & {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorted_spikes_group_name": "test_group",
# }
#
# SortedSpikesGroup.Units & {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorted_spikes_group_name": "test_group",
#     "unit_filter_params_name": unit_filter_params_name,
# }
#
# # BOOKMARK
#
# DecodingParameters.insert1(
#     {
#         "decoding_param_name": "contfrag_sorted",
#         "decoding_params": ContFragSortedSpikesClassifier(),
#         "decoding_kwargs": dict(),
#     },
#     skip_duplicates=True,
# )
#
# DecodingParameters()
#
#
# selection_key = {
#     "sorted_spikes_group_name": "test_group",
#     "unit_filter_params_name": "default_exclusion",
#     "position_group_name": "test_group",
#     "decoding_param_name": "contfrag_sorted",
#     "nwb_file_name": "mediumnwb20230802_.nwb",
#     "encoding_interval": "pos 0 valid times",
#     "decoding_interval": "test decoding interval",
#     "estimate_decoding_params": False,
# }
#
# SortedSpikesDecodingSelection.insert1(
#     selection_key,
#     skip_duplicates=True,
# )
#
# SortedSpikesDecodingV1.populate(selection_key)
#
#
# DecodingOutput.SortedSpikesDecodingV1 & selection_key
#
#
# results = (SortedSpikesDecodingV1 & selection_key).fetch_results()
#
#
# UnitSelectionParams().insert_default()
#
# unit_filter_params_name = "default_exclusion"
# print(
#     (
#         UnitSelectionParams()
#         & {"unit_filter_params_name": unit_filter_params_name}
#     ).fetch1()
# )
# UnitSelectionParams()
#
#
# nwb_copy_file_name = "mediumnwb20230802_.nwb"
#
# sorter_keys = {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorter": "mountainsort4",
#     "curation_id": 1,
# }
# (spike_v1.SpikeSortingSelection & sorter_keys) * SpikeSortingOutput.CurationV1
#
# spikesorting_merge_ids = (
#     (spike_v1.SpikeSortingSelection & sorter_keys)
#     * SpikeSortingOutput.CurationV1
# ).fetch("merge_id")
#
# unit_filter_params_name = "default_exclusion"
# SortedSpikesGroup().create_group(
#     group_name="test_group",
#     nwb_file_name=nwb_copy_file_name,
#     keys=[
#         {"spikesorting_merge_id": merge_id}
#         for merge_id in spikesorting_merge_ids
#     ],
#     unit_filter_params_name=unit_filter_params_name,
# )
# SortedSpikesGroup & {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorted_spikes_group_name": "test_group",
# }
#
# SortedSpikesGroup.Units & {
#     "nwb_file_name": nwb_copy_file_name,
#     "sorted_spikes_group_name": "test_group",
#     "unit_filter_params_name": unit_filter_params_name,
# }
#
#
# DecodingParameters.insert1(
#     {
#         "decoding_param_name": "contfrag_sorted",
#         "decoding_params": ContFragSortedSpikesClassifier(),
#         "decoding_kwargs": dict(),
#     },
#     skip_duplicates=True,
# )
#
# selection_key = {
#     "sorted_spikes_group_name": "test_group",
#     "unit_filter_params_name": "default_exclusion",
#     "position_group_name": "test_group",
#     "decoding_param_name": "contfrag_sorted",
#     "nwb_file_name": "mediumnwb20230802_.nwb",
#     "encoding_interval": "pos 0 valid times",
#     "decoding_interval": "test decoding interval",
#     "estimate_decoding_params": False,
# }
#
#
# SortedSpikesDecodingSelection.insert1(
#     selection_key,
#     skip_duplicates=True,
# )
#
#
# SortedSpikesDecodingV1.populate(selection_key)
#
#
# results = (SortedSpikesDecodingV1 & selection_key).fetch_results()
