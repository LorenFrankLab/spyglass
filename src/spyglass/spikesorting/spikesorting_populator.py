from ..common import IntervalList
from .spikesorting_recording import (
    SortGroup,
    SortInterval,
    SpikeSortingRecordingSelection,
    SpikeSortingRecording,
)
from .spikesorting_artifact import (
    ArtifactDetectionSelection,
    ArtifactDetection,
    ArtifactRemovedIntervalList,
)
from .spikesorting_sorting import SpikeSortingSelection, SpikeSorting
from .spikesorting_curation import (
    Curation,
    WaveformSelection,
    Waveforms,
    MetricSelection,
    QualityMetrics,
    AutomaticCurationSelection,
    AutomaticCuration,
    CuratedSpikeSortingSelection,
    CuratedSpikeSorting,
)
from .curation_figurl import CurationFigurlSelection, CurationFigurl


def spikesorting_pipeline_populator(
    nwb_file_name: str,
    team_name: str,
    fig_url_repo: str,
    interval_list_name: str,
    sort_interval_name: str = None,
    artifact_parameters: str = "ampl_2000_prop_75",
    preproc_params_name: str = "franklab_tetrode_hippocampus",
    sorter: str = "mountainsort4",
    sorter_params_name: str = "franklab_tetrode_hippocampus_30KHz_tmp",
    waveform_params_name: str = "default_whitened",
    metric_params_name: str = "peak_offest_num_spikes_2",
    auto_curation_params_name: str = "mike_noise_03_offset_2_isi_0025_mua",
):
    """Function top auomatically populate the spike sorting pipeline for a given epoch

    Parameters
    ----------
    nwb_file_name : str
        Session ID
    team_name : str
        Which team to assign the spike sorting to
    fig_url_repo : str
        Whewre to store the curation figurl json files (e.x. 'gh://LorenFrankLab/sorting-curations/main/sambray/')
    interval_list_name : str,
        if sort_interval_name not provided, will create a sort interval for the given interval with the same name
    sort_interval_name : str, default None
        if provided, will use the given sort interval, requires making this interval yourself

    artifact_parameters : str, optional
        parameter set for artifact detection, by default "ampl_2000_prop_75"
    preproc_params_name : str, optional
        parameter set for spikesorting recording, by default "franklab_tetrode_hippocampus"
    sorter : str, optional
        which spikesorting algorithm to use, by default "mountainsort4"
    sorter_params_name : str, optional
        parameters for the spike sorting algorithm, by default "franklab_tetrode_hippocampus_30KHz_tmp"
    waveform_params_name : str, optional
        Parameters for spike waveform extraction, by default "default_whitened"
    metric_params_name : str, optional
        Parameters defining which QualityMetrics to calculate and how, by default "peak_offest_num_spikes_2"
    auto_curation_params_name : str, optional
        Thresholds applied to Quality metrics for automatic unit curation, by default "mike_noise_03_offset_2_isi_0025_mua"
    """

    ## Sorting
    ## Sort groups
    ## Sort intervals
    ## Spike sorting recording
    ## Artifact detection
    ## Spike sorting

    # make sort groups only if not currently available (don't overwrite existing ones!)
    if len(SortGroup() & {"nwb_file_name": nwb_file_name}) == 0:
        print("Generating sort groups")
        SortGroup().set_group_by_shank(nwb_file_name)
    sort_group_id_list = (SortGroup & {"nwb_file_name": nwb_file_name}).fetch(
        "sort_group_id"
    )

    # Define sort interval
    if sort_interval_name is not None:
        print(f"Using sort interval {sort_interval_name}")
        if (
            len(
                SortInterval()
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_interval_name": sort_interval_name,
                }
            )
            == 0
        ):
            raise KeyError(f"Sort interval {sort_interval_name} not found")
    else:
        print(f"Generating sort interval from {interval_list_name}")
        interval_list = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")[0]
        sort_interval_name = interval_list_name
        sort_interval = interval_list
        SortInterval.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "sort_interval_name": sort_interval_name,
                "sort_interval": sort_interval,
            },
            skip_duplicates=True,
        )

    # make spike sorting recording
    print("Generating spike sorting recording")
    for sort_group_id in sort_group_id_list:
        ssr_key = dict(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,  # See SortGroup
            sort_interval_name=sort_interval_name,  # First N seconds above
            preproc_params_name=preproc_params_name,  # See preproc_params
            interval_list_name=interval_list_name,
            team_name=team_name,
        )
        SpikeSortingRecordingSelection.insert1(ssr_key, skip_duplicates=True)
    ssr_pj = (
        SpikeSortingRecordingSelection()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).proj()
    SpikeSortingRecording.populate([ssr_pj])

    # Artifact detection
    print("Running artifact detection")
    artifact_key_list = (ssr_pj).fetch("KEY")
    for artifact_key in artifact_key_list:
        artifact_key["artifact_params_name"] = artifact_parameters
        ArtifactDetectionSelection().insert1(artifact_key, skip_duplicates=True)

    art_pj = (
        ArtifactDetectionSelection()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).proj()
    ArtifactDetection.populate([art_pj])

    # Spike sorting
    print("Running spike sorting")
    for artifact_key in artifact_key_list:
        ss_key = dict(
            **(ArtifactDetection & artifact_key).fetch1("KEY"),
            **(ArtifactRemovedIntervalList() & artifact_key).fetch1("KEY"),
            sorter=sorter,
            sorter_params_name=sorter_params_name,
        )
        ss_key.pop("artifact_params_name")
        SpikeSortingSelection.insert1(ss_key, skip_duplicates=True)
    ss_proj = (
        SpikeSortingSelection
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).proj()
    SpikeSorting.populate([ss_proj])

    ## Curation
    ## Initial curation
    ## Extract waveforms
    ## Quality Metrics
    ## Automatic Curation
    ## Curated Spike Sorting
    ## Curation Figurl

    # initial curation
    print("Begginning curation")
    sorting_key_list = (
        SpikeSorting()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("KEY")
    for sorting_key in sorting_key_list:
        Curation.insert_curation(sorting_key)

    # Extract waveforms
    print("Extracting waveforms")
    curation_key_list = (
        Curation()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("KEY")
    for curation_key in curation_key_list:
        curation_key["waveform_params_name"] = waveform_params_name
        WaveformSelection.insert1(curation_key, skip_duplicates=True)
    wave_pj = (
        WaveformSelection()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).proj()
    Waveforms.populate([wave_pj])

    # Quality Metrics
    print("Calculating quality metrics")
    waveform_key_list = (
        Waveforms()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("KEY")
    for waveform_key in waveform_key_list:
        waveform_key["metric_params_name"] = metric_params_name
        MetricSelection.insert1(waveform_key, skip_duplicates=True)
    metrics_pj = (
        MetricSelection()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).proj()
    QualityMetrics().populate([metrics_pj])

    # Automatic Curation
    print("Creating automatic curation")
    metric_key_list = (
        QualityMetrics()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("KEY")
    for metric_key in metric_key_list:
        metric_key["auto_curation_params_name"] = auto_curation_params_name
        AutomaticCurationSelection.insert1(metric_key, skip_duplicates=True)
    auto_pj = (
        AutomaticCurationSelection
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).proj()
    AutomaticCuration().populate([auto_pj])

    # Curated Spike Sorting
    print("Creating curated spike sorting")
    auto_key_list = (
        AutomaticCuration()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("auto_curation_key")
    for auto_key in auto_key_list:
        curation_auto_key = (Curation() & auto_key).fetch1("KEY")
        CuratedSpikeSortingSelection.insert1(
            curation_auto_key, skip_duplicates=True
        )
    cur_proj = CuratedSpikeSortingSelection() & {
        "nwb_file_name": nwb_file_name,
        "sort_interval_name": sort_interval_name,
    }
    CuratedSpikeSorting.populate([cur_proj])

    # Curation Figurl
    print("Creating curation figurl")
    sort_interval_name = interval_list_name + f"_entire"
    for auto_id in (
        AutomaticCuration()
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch("auto_curation_key"):
        tetrode = auto_id["sort_group_id"]
        session_id = nwb_file_name + "_" + sort_interval_name
        github_url = (
            fig_url_repo
            + str(session_id)
            + "/"
            + str(tetrode)
            + "/curation.json"
        )
        auto_curation_out_key = (Curation() & auto_id).fetch1("KEY")
        auto_curation_out_key["new_curation_uri"] = github_url
        CurationFigurlSelection.insert1(
            auto_curation_out_key, skip_duplicates=True
        )
        CurationFigurl.populate(auto_curation_out_key)
