from typing import List, Union

import click
import yaml


@click.group(help="Spyglass command-line client")
def cli():
    pass


@click.command(
    help="Insert a session from a .nwb file that lives in the $SPYGLASS_BASE_DIR directory"
)
@click.argument("nwb_file_name")
def insert_session(nwb_file_name: str):
    import spyglass as nd

    nd.insert_sessions(nwb_file_name)


@click.command(help="List all sessions")
def list_sessions():
    import spyglass.common as sgc

    results = sgc.Session & {}
    print(results)


sample_lab_team_key = {
    "team_name": "your_team_name",
    "team_description": "optional_team_description",
}


@click.command(help="Insert a new lab team")
@click.argument("yaml_file_name", required=False)
def insert_lab_team(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(yaml.safe_dump(sample_lab_team_key, sort_keys=False))
        return

    import spyglass.common as sgc

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_lab_team_key.keys()}
    sgc.LabTeam.insert1(x)


@click.command(help="List all lab teams")
def list_lab_teams():
    import spyglass.common as sgc

    results = sgc.LabTeam & {}
    print(results)


sample_lab_member_key = {
    "lab_member_name": "first last",
    "first_name": "firstname",
    "last_name": "lastname",
}


@click.command(help="Insert a new lab member")
@click.argument("yaml_file_name", required=False)
def insert_lab_member(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("Note that lab_member_name is the primary key")
        print("==========================================")
        print(yaml.safe_dump(sample_lab_member_key, sort_keys=False))
        return

    import spyglass.common as sgc

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_lab_member_key.keys()}
    sgc.LabMember.insert1(x)


@click.command(help="List all lab members")
def list_lab_members():
    import spyglass.common as sgc

    results = sgc.LabMember & {}
    print(results)


sample_lab_team_member_key = {
    "team_name": "TeamName",
    "lab_member_name": "Labmember Name",
}


@click.command(help="Insert a new lab team member")
@click.argument("yaml_file_name", required=False)
def insert_lab_team_member(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(yaml.safe_dump(sample_lab_team_member_key, sort_keys=False))
        return

    import spyglass.common as sgc

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_lab_team_member_key.keys()}
    sgc.LabTeam.LabTeamMember.insert1(x)


@click.command(help="List all lab team members for a team")
@click.argument("team_name")
def list_lab_team_members(team_name: str):
    import spyglass.common as sgc

    results = sgc.LabTeam.LabTeamMember & {"team_name": team_name}
    print(results)


@click.command(
    help="List sort groups for a session. Note that nwb_file_name should include the trailing underscore."
)
@click.argument("nwb_file_name")
def list_sort_groups(nwb_file_name: str):
    import spyglass.spikesorting as nds

    results = nds.SortGroup & {"nwb_file_name": nwb_file_name}
    print(results)


@click.command(
    help="List sort group electrodes for a session. Note that nwb_file_name should include the trailing underscore."
)
@click.argument("nwb_file_name")
def list_sort_group_electrodes(nwb_file_name: str):
    import spyglass.spikesorting as nds

    results = nds.SortGroup.SortGroupElectrode & {"nwb_file_name": nwb_file_name}
    print(results)


@click.command(help="List interval lists for a session.")
@click.argument("nwb_file_name")
def list_interval_lists(nwb_file_name: str):
    import spyglass.common as sgc

    results = sgc.IntervalList & {"nwb_file_name": nwb_file_name}
    print(results)


@click.command(help="List sort intervals for a session.")
@click.argument("nwb_file_name")
def list_sort_intervals(nwb_file_name: str):
    import spyglass.spikesorting as nds

    results = nds.SortInterval & {"nwb_file_name": nwb_file_name}
    print(results)


sample_spike_sorting_preprocessing_parameters = {
    "preproc_params_name": "default",
    "preproc_params": {
        "frequency_min": 300,
        "frequency_max": 6000,
        "margin_ms": 5,
        "seed": 0,
        "whiten": True,
    },
}


@click.command(help="Insert spike sorting preprocessing parameters")
@click.argument("yaml_file_name", required=False)
def insert_spike_sorting_preprocessing_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(
            yaml.safe_dump(
                sample_spike_sorting_preprocessing_parameters, sort_keys=False
            )
        )
        return

    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_spike_sorting_preprocessing_parameters.keys()}
    nds.SpikeSortingPreprocessingParameters.insert1(x)


@click.command(help="List spike sorting preprocessing parameters.")
def list_spike_sorting_preprocessing_parameters():
    import spyglass.spikesorting as nds

    results = nds.SpikeSortingPreprocessingParameters & {}
    print(results)


sample_artifact_detection_parameters = {
    "artifact_params_name": "example",
    "artifact_params": {
        "zscore_thresh": None,
        "amplitude_thresh": 3000,
        "proportion_above_thresh": 1.0,
        "removal_window_ms": 1.0,
    },
}


@click.command(help="Insert artifact detection parameters")
@click.argument("yaml_file_name", required=False)
def insert_artifact_detection_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(yaml.safe_dump(sample_artifact_detection_parameters, sort_keys=False))
        return

    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_artifact_detection_parameters.keys()}
    nds.ArtifactDetectionParameters.insert1(x)


@click.command(help="List artifact detection parameters.")
def list_artifact_detection_parameters():
    import spyglass.spikesorting as nds

    results = nds.ArtifactDetectionParameters & {}
    print(results)


sample_spike_sorting_recording_selection_key = {
    "nwb_file_name": "FileName_.nwb",
    "sort_group_id": 0,
    "sort_interval_name": "sort_interval_name",
    "preproc_params_name": "default",
    "interval_list_name": "interval_list_name",
    "team_name": "TeamName",
}


@click.command(help="Create a spike sorting recording")
@click.argument("yaml_file_name", required=False)
def create_spike_sorting_recording(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(
            yaml.safe_dump(
                sample_spike_sorting_recording_selection_key, sort_keys=False
            )
        )
        return

    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_spike_sorting_recording_selection_key.keys()}
    nds.SpikeSortingRecordingSelection.insert1(x, skip_duplicates=True)
    nds.SpikeSortingRecording.populate(
        [(nds.SpikeSortingRecordingSelection & x).proj()]
    )


@click.command(help="List spike sorting recordings for a session.")
@click.argument("nwb_file_name")
def list_spike_sorting_recordings(nwb_file_name: str):
    import spyglass.spikesorting as nds

    results = nds.SpikeSortingRecording & {"nwb_file_name": nwb_file_name}
    print(results)


@click.command(help="Create a spike sorting recording view")
@click.argument("yaml_file_name", required=False)
@click.option("--replace", is_flag=True)
def create_spike_sorting_recording_view(
    yaml_file_name: Union[str, None], replace: bool
):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(
            yaml.safe_dump(
                sample_spike_sorting_recording_selection_key, sort_keys=False
            )
        )
        return

    import spyglass.figurl_views as ndf
    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    x = {k: x[k] for k in sample_spike_sorting_recording_selection_key.keys()}
    if replace:
        (ndf.SpikeSortingRecordingView & x).delete()
    ndf.SpikeSortingRecordingView.populate([(nds.SpikeSortingRecording & x).proj()])
    figurl = (ndf.SpikeSortingRecordingView & x).fetch1("figurl")
    print(figurl)


sample_spike_sorter_params_key = {
    "sorter_params_name": "example",
    "sorter": "mountainsort4",
    "sorter_params": {
        "detect_sign": -1,
        "adjacency_radius": 100,
        "filter": False,
        "whiten": False,
        "clip_size": 50,
        "detect_threshold": 3,
        "detect_interval": 10,
    },
}


@click.command(help="Insert spike sorter parameters")
@click.argument("yaml_file_name", required=False)
def insert_spike_sorter_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(yaml.safe_dump(sample_spike_sorter_params_key, sort_keys=False))
        return

    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    x = {k: x[k] for k in sample_spike_sorter_params_key.keys()}
    nds.SpikeSorterParameters.insert1(x)


@click.command(help="List spike sorter parameters.")
def list_spike_sorter_parameters():
    import spyglass.spikesorting as nds

    results = nds.SpikeSorterParameters & {}
    print(results)


sample_spike_sorting_key = dict(
    sample_spike_sorting_recording_selection_key,
    **{"artifact_params_name": "default", "sorter_params_name": "default"},
)


@click.command(help="Run spike sorting")
@click.argument("yaml_file_name", required=False)
def run_spike_sorting(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print("You must specify a yaml file. Sample content:")
        print("==========================================")
        print(yaml.safe_dump(sample_spike_sorting_key, sort_keys=False))
        return

    import spyglass.spikesorting as nds

    with open(yaml_file_name, "r") as f:
        x = yaml.safe_load(f)
    # restrict to relevant keys
    x = {k: x[k] for k in sample_spike_sorting_key.keys()}
    spike_sorting_recording_query = {
        k: x[k] for k in sample_spike_sorting_recording_selection_key.keys()
    }
    spike_sorting_recording_key = (
        (nds.SpikeSortingRecording & spike_sorting_recording_query).proj().fetch1()
    )

    artifact_key = dict(
        spike_sorting_recording_key,
        **{"artifact_params_name": x["artifact_params_name"]},
    )
    nds.ArtifactDetectionSelection.insert1(artifact_key, skip_duplicates=True)
    nds.ArtifactDetection.populate(
        [(nds.ArtifactDetectionSelection & artifact_key).proj()]
    )
    artifact_removed_interval_list_name = (nds.ArtifactDetection & artifact_key).fetch1(
        "artifact_removed_interval_list_name"
    )

    sorter_params_name = x["sorter_params_name"]
    sorter = (
        nds.SpikeSorterParameters & {"sorter_params_name": sorter_params_name}
    ).fetch1("sorter")

    sorting_key = dict(
        spike_sorting_recording_key,
        **{
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_removed_interval_list_name": artifact_removed_interval_list_name,
        },
    )

    nds.SpikeSortingSelection.insert1(sorting_key, skip_duplicates=True)
    nds.SpikeSorting.populate([(nds.SpikeSortingSelection & sorting_key).proj()])


@click.command(help="List spike sortings for a session.")
@click.argument("nwb_file_name")
def list_spike_sortings(nwb_file_name: str):
    import spyglass.spikesorting as nds

    results = nds.SpikeSorting & {"nwb_file_name": nwb_file_name}
    print(results)


@click.command(help="Create spyglass view of a session group.")
@click.argument("session_group_name")
def create_spyglass_view(session_group_name: str):
    import spyglass.common as sgc

    F = sgc.SessionGroup.create_spyglass_view(session_group_name)
    url = F.url(label=session_group_name)
    print(url)


cli.add_command(insert_session)
cli.add_command(list_sessions)
cli.add_command(insert_lab_team)
cli.add_command(list_lab_teams)
cli.add_command(insert_lab_member)
cli.add_command(list_lab_members)
cli.add_command(insert_lab_team_member)
cli.add_command(list_lab_team_members)
cli.add_command(list_sort_groups)
cli.add_command(list_sort_group_electrodes)
cli.add_command(list_interval_lists)
cli.add_command(list_sort_intervals)
cli.add_command(insert_spike_sorting_preprocessing_parameters)
cli.add_command(list_spike_sorting_preprocessing_parameters)
cli.add_command(insert_artifact_detection_parameters)
cli.add_command(list_artifact_detection_parameters)
cli.add_command(create_spike_sorting_recording)
cli.add_command(list_spike_sorting_recordings)
cli.add_command(create_spike_sorting_recording_view)
cli.add_command(insert_spike_sorter_parameters)
cli.add_command(list_spike_sorter_parameters)
cli.add_command(run_spike_sorting)
cli.add_command(list_spike_sortings)
cli.add_command(create_spyglass_view)
