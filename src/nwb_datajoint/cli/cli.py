from typing import Union
import click
from pyrsistent import optional
import yaml

@click.group(help="Spyglass command-line client")
def cli():
    pass

@click.command(help="Insert a session from a .nwb file that lives in the $NWB_DATAJOINT_BASE_DIR directory")
@click.argument('nwb_file_name')
def insert_session(nwb_file_name: str):
    import nwb_datajoint as nd
    nd.insert_sessions(nwb_file_name)

@click.command(help="List all sessions")
def list_sessions():
    import nwb_datajoint.common as ndc
    a = ndc.Session & {}
    print(a)

@click.command(help="Insert a new lab team")
@click.argument('yaml_file_name', required=False)
def insert_lab_team(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'team_name': 'your_team_name',
            'team_description': 'optional_team_description'
        })
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        labteam_dict = yaml.safe_load(f)
    ndc.LabTeam.insert1(labteam_dict)

@click.command(help="List all lab teams")
def list_lab_teams():
    import nwb_datajoint.common as ndc
    a = ndc.LabTeam & {}
    print(a)

@click.command(help="Insert a new lab member")
@click.argument('yaml_file_name', required=False)
def insert_lab_member(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('Note that lab_member_name is the primary key')
        print('==========================================')
        a = yaml.safe_dump({
            'lab_member_name': 'first last',
            'first_name': 'firstname',
            'last_name': 'lastname'
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.LabMember.insert1(x)

@click.command(help="List all lab members")
def list_lab_members():
    import nwb_datajoint.common as ndc
    a = ndc.LabMember & {}
    print(a)

@click.command(help="Insert a new lab team member")
@click.argument('yaml_file_name', required=False)
def insert_lab_team_member(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'team_name': 'TeamName',
            'lab_member_name': 'Labmember Name'
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.LabTeam.LabTeamMember.insert1(x)

@click.command(help="List all lab team members for a team")
@click.argument('team_name')
def list_lab_team_members(team_name: str):
    import nwb_datajoint.common as ndc
    a = ndc.LabTeam.LabTeamMember & {'team_name': team_name}
    print(a)

@click.command(help="List sort groups for a session. Note that nwb_file_name should include the trailing underscore.")
@click.argument('nwb_file_name')
def list_sort_groups(nwb_file_name: str):
    import nwb_datajoint.common as ndc
    a = ndc.SortGroup & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="List sort group electrodes for a session. Note that nwb_file_name should include the trailing underscore.")
@click.argument('nwb_file_name')
def list_sort_group_electrodes(nwb_file_name: str):
    import nwb_datajoint.common as ndc
    a = ndc.SortGroup.SortGroupElectrode & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="List interval lists for a session.")
@click.argument('nwb_file_name')
def list_interval_lists(nwb_file_name: str):
    import nwb_datajoint.common as ndc
    a = ndc.IntervalList & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="List sort intervals for a session.")
@click.argument('nwb_file_name')
def list_sort_intervals(nwb_file_name: str):
    import nwb_datajoint.common as ndc
    a = ndc.SortInterval & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="Insert spike sorting preprocessing parameters")
@click.argument('yaml_file_name', required=False)
def insert_spike_sorting_preprocessing_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'preproc_params_name': 'default',
            'preproc_params': {
                'frequency_min': 300,
                'frequency_max': 6000,
                'margin_ms': 5,
                'seed': 0
            }
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.SpikeSortingPreprocessingParameters.insert1(x)

@click.command(help="List spike sorting preprocessing parameters.")
def list_spike_sorting_preprocessing_parameters():
    import nwb_datajoint.common as ndc
    a = ndc.SpikeSortingPreprocessingParameters & {}
    print(a)

@click.command(help="Insert artifact detection parameters")
@click.argument('yaml_file_name', required=False)
def insert_artifact_detection_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'artifact_params_name': 'example',
            'artifact_params': {
                'zscore_thresh': None,
                'amplitude_thresh': 3000,
                'proportion_above_thresh': 1.0,
                'removal_window_ms': 1.0,
            }
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.ArtifactDetectionParameters.insert1(x)

@click.command(help="List artifact detection parameters.")
def list_artifact_detection_parameters():
    import nwb_datajoint.common as ndc
    a = ndc.ArtifactDetectionParameters & {}
    print(a)

@click.command(help="Insert spike sorting recording selection")
@click.argument('yaml_file_name', required=False)
def insert_spike_sorting_recording_selection(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'nwb_file_name': 'FileName_.nwb',
            'sort_group_id': 0,
            'sort_interval_name': 'sort_interval_name',
            'preproc_params_name': 'default',
            'interval_list_name': 'interval_list_name',
            'team_name': 'TeamName'
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.SpikeSortingRecordingSelection.insert1(x)

@click.command(help="List spike sorting recording selections for a session.")
@click.argument('nwb_file_name')
def list_spike_sorting_recording_selections(nwb_file_name):
    import nwb_datajoint.common as ndc
    a = ndc.SpikeSortingRecordingSelection & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="Create a spike sorting recording")
@click.argument('yaml_file_name', required=True)
def create_spike_sorting_recording(yaml_file_name: str):
    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.SpikeSortingRecording.populate([(ndc.SpikeSortingRecordingSelection & x).proj()])

@click.command(help="List spike sorting recordings for a session.")
@click.argument('nwb_file_name')
def list_spike_sorting_recordings(nwb_file_name):
    import nwb_datajoint.common as ndc
    a = ndc.SpikeSortingRecording & {'nwb_file_name': nwb_file_name}
    print(a)

@click.command(help="Create a spike sorting recording view")
@click.argument('yaml_file_name', required=True)
def create_spike_sorting_recording_view(yaml_file_name: str):
    import nwb_datajoint.common as ndc
    import nwb_datajoint.figurl_views as ndf
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    a = ndf.SpikeSortingRecordingView.populate([(ndc.SpikeSortingRecording & x).proj()])
    print(a)

@click.command(help="Insert spike sorter parameters")
@click.argument('yaml_file_name', required=False)
def insert_spike_sorter_parameters(yaml_file_name: Union[str, None]):
    if yaml_file_name is None:
        print('You must specify a yaml file. Sample content:')
        print('==========================================')
        a = yaml.safe_dump({
            'sorter_params_name': 'example',
            'sorter': 'mountainsort4',
            'sorter_params': {
                'detect_sign': -1,
                'adjacency_radius': 100,
                'filter': False,
                'whiten': False,
                'clip_size': 50,
                'detect_threshold': 3,
                'detect_interval': 10
            }
        }, sort_keys=False)
        print(a)
        return

    import nwb_datajoint.common as ndc
    with open(yaml_file_name, 'r') as f:
        x = yaml.safe_load(f)
    ndc.SpikeSorterParameters.insert1(x)

@click.command(help="List spike sorter parameters.")
def list_spike_sorter_parameters():
    import nwb_datajoint.common as ndc
    a = ndc.SpikeSorterParameters & {}
    print(a)

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
cli.add_command(insert_spike_sorting_recording_selection)
cli.add_command(list_spike_sorting_recording_selections)
cli.add_command(create_spike_sorting_recording)
cli.add_command(list_spike_sorting_recordings)
cli.add_command(create_spike_sorting_recording_view)
cli.add_command(insert_spike_sorter_parameters)
cli.add_command(list_spike_sorter_parameters)