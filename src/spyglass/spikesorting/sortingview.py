import datajoint as dj

import sortingview as sv

from ..common.common_lab import LabMember, LabTeam
from .sortingview_helper_fn import (
    _create_spikesortingview_workspace,
    _generate_url,
)
from .spikesorting_curation import Curation
from .spikesorting_recording import SpikeSortingRecording
from .spikesorting_sorting import SpikeSorting

import spikeinterface as si

schema = dj.schema("spikesorting_sortingview")


@schema
class SortingviewWorkspaceSelection(dj.Manual):
    definition = """
    -> Curation
    """


@schema
class SortingviewWorkspace(dj.Computed):
    definition = """
    -> SortingviewWorkspaceSelection
    ---
    workspace_uri: varchar(1000)
    sortingview_recording_id: varchar(30)
    sortingview_sorting_id: varchar(30)
    channel = NULL : varchar(80)        # the name of kachery channel for data sharing (for kachery daemon, deprecated)
    """

    # make class for parts table to hold URLs
    class URL(dj.Part):
        # Table for holding URLs
        definition = """
        -> SortingviewWorkspace
        ---
        curation_url: varchar(1000)   # URL with sortingview data
        curation_jot: varchar(200)   # URI for saving manual curation tags
        """

    def make(self, key: dict):
        """Create a Sortingview workspace

        Parameters
        ----------
        key : dict
            primary key of an entry from SortingviewWorkspaceSelection table
        """

        # fetch
        recording_path = (SpikeSortingRecording & key).fetch1("recording_path")
        sorting_path = (SpikeSorting & key).fetch1("sorting_path")
        merge_groups = (Curation & key).fetch1("merge_groups")
        workspace_label = SpikeSortingRecording._get_recording_name(key)
        recording_label = SpikeSortingRecording._get_recording_name(key)
        sorting_label = SpikeSorting._get_sorting_name(key)
        metrics = (Curation & key).fetch1("quality_metrics")
        curation_labels = (Curation & key).fetch1("curation_labels")
        team_name = (SpikeSortingRecording & key).fetch1()["team_name"]
        team_members = (LabTeam.LabTeamMember & {"team_name": team_name}).fetch(
            "lab_member_name"
        )
        google_user_ids = []
        for team_member in team_members:
            google_user_id = (
                LabMember.LabMemberInfo & {"lab_member_name": team_member}
            ).fetch("google_user_name")
            if len(google_user_id) != 1:
                print(
                    f"Google user ID for {team_member} does not exist or more than one ID detected;\
                        permission not given to {team_member}, skipping..."
                )
                continue
            google_user_ids.append(google_user_id[0])

        # do
        workspace_uri, recording_id, sorting_id = _create_spikesortingview_workspace(
            recording_path=recording_path,
            sorting_path=sorting_path,
            merge_groups=merge_groups,
            workspace_label=workspace_label,
            recording_label=recording_label,
            sorting_label=sorting_label,
            metrics=metrics,
            curation_labels=curation_labels,
            google_user_ids=google_user_ids,
        )

        # insert
        key["workspace_uri"] = workspace_uri
        key["sortingview_recording_id"] = recording_id
        key["sortingview_sorting_id"] = sorting_id
        self.insert1(key)

        # insert URLs
        # remove non-primary keys
        del key["workspace_uri"]
        del key["sortingview_recording_id"]
        del key["sortingview_sorting_id"]

        # generate URLs and add to key
        url = self.url_trythis(key)
        # url = _generate_url(key)
        # print("URL:", url)
        key["curation_url"] = url
        key["curation_jot"] = "not ready yet"

        SortingviewWorkspace.URL.insert1(key)

    def remove_sorting_from_workspace(self, key):
        return NotImplementedError

    def url_trythis(self, key: dict, sortingview_sorting_id: str = None):
        """Generate a URL for visualizing and curating a sorting on the web.
        Will print instructions on how to do the curation.
        Parameters
        ----------
        key : dict
            An entry from SortingviewWorkspace table
        sortingview_sorting_id : str, optional
            sortingview sorting ID to visualize; if None then chooses the first one
        Returns
        -------
        url : str
        """
        workspace_uri = (self & key).fetch1("workspace_uri")
        workspace = sv.load_workspace(workspace_uri)
        recording_id = workspace.recording_ids[0]
        if sortingview_sorting_id is None:
            sortingview_sorting_id = workspace.sorting_ids[0]

        R = workspace.get_recording_extractor(recording_id)
        S = workspace.get_sorting_extractor(sortingview_sorting_id)

        initial_labels = (Curation & key).fetch("curation_labels")[0]
        for k, v in initial_labels.items():
            new_list = []
            for item in v:
                if item not in new_list:
                    new_list.append(item)
            initial_labels[k] = new_list
        initial_curation = {"labelsByUnit": initial_labels}

        # custom metrics
        unit_metrics = workspace.get_unit_metrics_for_sorting(sortingview_sorting_id)

        # This will print some instructions on how to do the curation
        # old: sv.trythis_start_sorting_curation
        url = _generate_url(
            recording=R,
            sorting=S,
            label=workspace.label,
            initial_curation=initial_curation,
            raster_plot_subsample_max_firing_rate=50,
            spike_amplitudes_subsample_max_firing_rate=50,
            unit_metrics=unit_metrics,
        )
        return url

    def insert_manual_curation(
        self, key: dict, url: str, description="manually curated"
    ):
        """Based on information in key for an SortingviewWorkspace, loads the
        curated sorting from sortingview, saves it (with labels and the
        optional description) and inserts it to CuratedSorting

        Assumes that the workspace corresponding to the recording and (original) sorting exists

        Parameters
        ----------
        key : dict
            primary key of AutomaticCuration
        description: str, optional
            description of curated sorting
        """

        # get the labels and remove the non-primary merged units
        # labels = workspace.get_sorting_curation(sorting_id=sortingview_sorting_id)
        # labels = sv.trythis_load_sorting_curation('jot://xTzzyDieQPkW')
        labels = sv.trythis_load_sorting_curation(url)

        # turn labels to list of str, only including accepted units.
        # if bool(labels["mergeGroups"]):
        if bool(labels.get("mergeGroups", [])):
            # clusters were merged, so we empty out metrics
            metrics = {}
        else:
            # get the metrics from the parent curation
            metrics = (Curation & key).fetch1("quality_metrics")

        # insert this curation into the  Table
        return Curation.insert_curation(
            key,
            parent_curation_id=key["curation_id"],
            labels=labels["labelsByUnit"],
            merge_groups=labels.get("mergeGroups", []),
            metrics=metrics,
            description=description,
        )
