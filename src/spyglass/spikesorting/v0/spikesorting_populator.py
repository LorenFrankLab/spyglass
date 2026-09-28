import datajoint as dj

from spyglass.common import ElectrodeGroup, IntervalList
from spyglass.spikesorting.v0.curation_figurl import (
    CurationFigurl,
    CurationFigurlSelection,
)
from spyglass.spikesorting.v0.spikesorting_artifact import (
    ArtifactDetection,
    ArtifactDetectionSelection,
    ArtifactRemovedIntervalList,
)
from spyglass.spikesorting.v0.spikesorting_curation import (
    AutomaticCuration,
    AutomaticCurationSelection,
    CuratedSpikeSorting,
    CuratedSpikeSortingSelection,
    Curation,
    MetricSelection,
    QualityMetrics,
    Waveforms,
    WaveformSelection,
)
from spyglass.spikesorting.v0.spikesorting_recording import (
    SortGroup,
    SortInterval,
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from spyglass.spikesorting.v0.spikesorting_sorting import (
    SpikeSorting,
    SpikeSortingSelection,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_sorting")


@schema
class SpikeSortingPipelineParameters(SpyglassMixin, dj.Manual):
    """Parameters for the spike sorting pipeline

    Attributes
    ----------
    pipeline_parameters_name : str
        A name for this set of parameters
    preproc_params_name : str
        Name of the preprocessing parameters to use
    artifact_parameters : str
        Name of the artifact detection parameters to use
    sorter : str
        Name of the sorting algorithm to use
    sorter_params_name : str
        Name of the sorting parameters to use
    waveform_params_name : str
        Name of the waveform parameters to use
    metric_params_name : str
        Name of the metric parameters to use
    auto_curation_params_name : str
        Name of the automatic curation parameters to use
    """

    definition = """
    pipeline_parameters_name: varchar(200)
    ---
    preproc_params_name: varchar(200)
    artifact_parameters: varchar(200)
    sorter: varchar(200)
    sorter_params_name: varchar(200)
    waveform_params_name: varchar(200)
    metric_params_name: varchar(200)
    auto_curation_params_name: varchar(200)
    """


def spikesorting_pipeline_populator(
    nwb_file_name: str,
    team_name: str,
    fig_url_repo: str = None,
    interval_list_name: str = None,
    sort_interval_name: str = None,
    pipeline_parameters_name: str = None,
    probe_restriction: dict = {},
    artifact_parameters: str = "ampl_2000_prop_75",
    preproc_params_name: str = "franklab_tetrode_hippocampus",
    sorter: str = "mountainsort4",
    sorter_params_name: str = "franklab_tetrode_hippocampus_30KHz_tmp",
    waveform_params_name: str = "default_whitened",
    metric_params_name: str = "peak_offest_num_spikes_2",
    auto_curation_params_name: str = "mike_noise_03_offset_2_isi_0025_mua",
):
    """Automatically populate the spike sorting pipeline for a given epoch

    Parameters
    ----------
    nwb_file_name : str
        Session ID
    team_name : str
        Which team to assign the spike sorting to
    fig_url_repo : str, optional
        Where to store the curation figurl json files (e.g.,
        'gh://LorenFrankLab/sorting-curations/main/user/'). Default None to
        skip figurl
    interval_list_name : str,
        if sort_interval_name not provided, will create a sort interval for the
        given interval with the same name
    sort_interval_name : str, default None
        if provided, will use the given sort interval, requires making this
        interval yourself
    pipeline_parameters_name : str, optional
        If provided, will lookup pipeline parameters from the
        SpikeSortingPipelineParameters table, supersedes other values provided,
        by default None
    probe_restriction : dict, optional
        Restricts analysis to sort groups with matching keys. Can use keys from
        the SortGroup and ElectrodeGroup Tables (e.g. electrode_group_name,
        probe_id, target_hemisphere), by default {}
    artifact_parameters : str, optional
        parameter set for artifact detection, by default "ampl_2000_prop_75"
    preproc_params_name : str, optional
        parameter set for spikesorting recording, by default
        "franklab_tetrode_hippocampus"
    sorter : str, optional
        which spikesorting algorithm to use, by default "mountainsort4"
    sorter_params_name : str, optional
        parameters for the spike sorting algorithm, by default
        "franklab_tetrode_hippocampus_30KHz_tmp"
    waveform_params_name : str, optional
        Parameters for spike waveform extraction. If empty string, will skip
        automatic curation steps, by default "default_whitened"
    metric_params_name : str, optional
        Parameters defining which QualityMetrics to calculate and how. If empty
        string, will skip automatic curation steps, by default
        "peak_offest_num_spikes_2"
    auto_curation_params_name : str, optional
        Thresholds applied to Quality metrics for automatic unit curation. If
        empty string, will skip automatic curation steps, by default
        "mike_noise_03_offset_2_isi_0025_mua"
    """
    nwbf_dict = dict(nwb_file_name=nwb_file_name)
    # Define pipeline parameters
    if pipeline_parameters_name is not None:
        logger.info(f"Using pipeline parameters {pipeline_parameters_name}")
        (
            artifact_parameters,
            preproc_params_name,
            sorter,
            sorter_params_name,
            waveform_params_name,
            metric_params_name,
            auto_curation_params_name,
        ) = (
            SpikeSortingPipelineParameters
            & {"pipeline_parameters_name": pipeline_parameters_name}
        ).fetch1(
            "artifact_parameters",
            "preproc_params_name",
            "sorter",
            "sorter_params_name",
            "waveform_params_name",
            "metric_params_name",
            "auto_curation_params_name",
        )

    # make sort groups only if not currently available
    # don't overwrite existing ones!
    if not SortGroup() & nwbf_dict:
        logger.info("Generating sort groups")
        SortGroup().set_group_by_shank(nwb_file_name)

    # Define sort interval
    interval_dict = dict(**nwbf_dict, interval_list_name=interval_list_name)

    if sort_interval_name is not None:
        logger.info(f"Using sort interval {sort_interval_name}")
        if not (
            SortInterval
            & nwbf_dict
            & {"sort_interval_name": sort_interval_name}
        ):
            raise KeyError(f"Sort interval {sort_interval_name} not found")
    else:
        logger.info(f"Generating sort interval from {interval_list_name}")
        interval_list = (IntervalList & interval_dict).fetch1("valid_times")[0]

        sort_interval_name = interval_list_name
        sort_interval = interval_list

        SortInterval.insert1(
            {
                **nwbf_dict,
                "sort_interval_name": sort_interval_name,
                "sort_interval": sort_interval,
            },
            skip_duplicates=True,
        )

    sort_dict = dict(**nwbf_dict, sort_interval_name=sort_interval_name)

    # find desired sort group(s) for these settings
    sort_group_id_list = (
        (SortGroup.SortGroupElectrode * ElectrodeGroup)
        & nwbf_dict
        & probe_restriction
    ).fetch("sort_group_id")

    # make spike sorting recording
    logger.info("Generating spike sorting recording")
    for sort_group_id in sort_group_id_list:
        ssr_key = dict(
            **sort_dict,
            sort_group_id=sort_group_id,  # See SortGroup
            preproc_params_name=preproc_params_name,  # See preproc_params
            interval_list_name=interval_list_name,
            team_name=team_name,
        )
        SpikeSortingRecordingSelection.insert1(ssr_key, skip_duplicates=True)

    SpikeSortingRecording.populate(sort_dict)

    # Artifact detection
    logger.info("Running artifact detection")
    artifact_keys = [
        {**k, "artifact_params_name": artifact_parameters}
        for k in (SpikeSortingRecordingSelection() & sort_dict).fetch("KEY")
    ]
    ArtifactDetectionSelection().insert(artifact_keys, skip_duplicates=True)
    ArtifactDetection.populate(sort_dict)

    # Spike sorting
    logger.info("Running spike sorting")
    for artifact_key in artifact_keys:
        ss_key = dict(
            **(ArtifactDetection & artifact_key).fetch1("KEY"),
            **(ArtifactRemovedIntervalList() & artifact_key).fetch1("KEY"),
            sorter=sorter,
            sorter_params_name=sorter_params_name,
        )
        ss_key.pop("artifact_params_name")
        SpikeSortingSelection.insert1(ss_key, skip_duplicates=True)
    SpikeSorting.populate(sort_dict)

    # initial curation
    logger.info("Beginning curation")
    for sorting_key in (SpikeSorting() & sort_dict).fetch("KEY"):
        if not (Curation() & sorting_key):
            Curation.insert_curation(sorting_key)

    # Calculate quality metrics and perform automatic curation if specified
    if (
        len(waveform_params_name) > 0
        and len(metric_params_name) > 0
        and len(auto_curation_params_name) > 0
    ):
        # Extract waveforms
        logger.info("Extracting waveforms")
        curation_keys = [
            {**k, "waveform_params_name": waveform_params_name}
            for k in (Curation() & sort_dict & {"curation_id": 0}).fetch("KEY")
        ]
        WaveformSelection.insert(curation_keys, skip_duplicates=True)
        Waveforms.populate(sort_dict)

        # Quality Metrics
        logger.info("Calculating quality metrics")
        waveform_keys = [
            {**k, "metric_params_name": metric_params_name}
            for k in (Waveforms() & sort_dict).fetch("KEY")
        ]
        MetricSelection.insert(waveform_keys, skip_duplicates=True)
        QualityMetrics().populate(sort_dict)

        # Automatic Curation
        logger.info("Creating automatic curation")
        metric_keys = [
            {**k, "auto_curation_params_name": auto_curation_params_name}
            for k in (QualityMetrics() & sort_dict).fetch("KEY")
        ]
        AutomaticCurationSelection.insert(metric_keys, skip_duplicates=True)
        AutomaticCuration().populate(sort_dict)

        # Curated Spike Sorting
        # get curation keys of the automatic curation to populate into curated
        # spike sorting selection
        logger.info("Creating curated spike sorting")
        auto_key_list = (AutomaticCuration() & sort_dict).fetch(
            "auto_curation_key"
        )
        for auto_key in auto_key_list:
            curation_auto_key = (Curation() & auto_key).fetch1("KEY")
            CuratedSpikeSortingSelection.insert1(
                curation_auto_key, skip_duplicates=True
            )

    else:
        # Perform no automatic curation, just populate curated spike sorting
        # selection with the initial curation. Used in case of clusterless
        # decoding
        logger.info("Creating curated spike sorting")
        curation_keys = (Curation() & sort_dict).fetch("KEY")
        for curation_key in curation_keys:
            CuratedSpikeSortingSelection.insert1(
                curation_key, skip_duplicates=True
            )

    # Populate curated spike sorting
    CuratedSpikeSorting.populate(sort_dict)

    if fig_url_repo:
        # Curation Figurl
        logger.info("Creating curation figurl")
        sort_interval_name = interval_list_name + "_entire"
        gh_url = (
            fig_url_repo
            + str(nwb_file_name + "_" + sort_interval_name)  # session id
            + "/{}"  # tetrode using auto_id['sort_group_id']
            + "/curation.json"
        )

        for auto_id in (AutomaticCuration() & sort_dict).fetch(
            "auto_curation_key"
        ):
            auto_curation_out_key = dict(
                **(Curation() & auto_id).fetch1("KEY"),
                new_curation_uri=gh_url.format(str(auto_id["sort_group_id"])),
            )
            CurationFigurlSelection.insert1(
                auto_curation_out_key, skip_duplicates=True
            )
            CurationFigurl.populate(auto_curation_out_key)
