import datajoint as dj

import sortingview as sv

from ..common.common_lab import LabMember, LabTeam
from .sortingview_helper_fn import (
    _add_metrics_to_sorting_in_workspace,
    _create_spikesortingview_workspace,
    _set_workspace_permission,
)
from .spikesorting_curation import Curation
from .spikesorting_recording import SpikeSortingRecording
from .spikesorting_sorting import SpikeSorting

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

    def remove_sorting_from_workspace(self, key):
        return NotImplementedError

    def add_metrics_to_sorting(
        self, key: dict, metrics: dict, sortingview_sorting_id: str = None
    ):
        """Adds a metrics to the specified sorting.

        Parameters
        ----------
        key : dict
            primary key of an entry from SortingviewWorkspace table
        metrics : dict
            Quality metrics.
            Key: name of quality metric
            Value: another dict in which key: unit ID (must be str),
                                         value: metric value (float)
        sortingview_sorting_id : str, optional
            if not specified, just uses the first sorting ID of the workspace
        """

        # fetch
        workspace_uri = (self & key).fetch1("workspace_uri")
        workspace = sv.load_workspace(workspace_uri)

        # do
        _add_metrics_to_sorting_in_workspace(workspace, metrics, sortingview_sorting_id)

    def set_workspace_permission(self, key, curators):
        """Gives curation permission to lab members not in the team associated
        with the recording (team members are given permission by default).

        Parameters
        ----------
        key : dict
            primary key of an entry from SortingviewWorkspace table
        curators : List[str]
            names of lab members to be given permission to curate
        sortingview_sorting_id : str
        """
        workspace_uri = (self & key).fetch1("workspace_uri")
        workspace = sv.load_workspace(workspace_uri)
        if sortingview_sorting_id is None:
            sortingview_sorting_id = workspace.sorting_ids[0]
        google_user_ids = []
        for curator in curators:
            google_user_id = (
                LabMember.LabMemberInfo & {"lab_member_name": curator}
            ).fetch("google_user_name")
            if len(google_user_id) != 1:
                print(
                    f"Google user ID for {curator} does not exist or more than one ID detected;\
                        permission not given to {curator}, skipping..."
                )
                continue
            google_user_ids.append(google_user_id[0])
        workspace = _set_workspace_permission(
            workspace, google_user_ids, sortingview_sorting_id
        )
        return workspace

    def set_snippet_len(self, key: dict, snippet_len: int):
        """Sets the snippet length of a workspace specified by the key

        Parameters
        ----------
        key : dict
            primary key of an entry from SortingviewWorkspace table
        """
        workspace_uri = (self & key).fetch1("workspace_uri")
        workspace = sv.load_workspace(workspace_uri)
        workspace.set_snippet_len(snippet_len)

    def url(self, key: dict, sortingview_sorting_id: str = None):
        """Generate a URL for visualizing the sorting on the web.

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
        url = workspace.spikesortingview(
            recording_id=recording_id,
            sorting_id=sortingview_sorting_id,
            label=workspace.label,
        )

        return url

    def insert_manual_curation(self, key: dict, description="manually curated"):
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
        workspace_uri = (self & key).fetch("workspace_uri")
        if workspace_uri.size == 0:
            raise ValueError("First create a sortingview workspace for this entry.")
        workspace = sv.load_workspace(workspace_uri=workspace_uri[0])

        sortingview_sorting_id = (SortingviewWorkspace & key).fetch1(
            "sortingview_sorting_id"
        )

        # get the labels and remove the non-primary merged units
        labels = workspace.get_sorting_curation(sorting_id=sortingview_sorting_id)

        # turn labels to list of str, only including accepted units.
        if bool(labels["mergeGroups"]):
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
            merge_groups=labels["mergeGroups"],
            metrics=metrics,
            description=description,
        )
