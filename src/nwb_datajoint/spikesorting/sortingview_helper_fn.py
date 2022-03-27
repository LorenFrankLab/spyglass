"Sortingview helper functions"

from typing import List

import sortingview as sv

from ..common.common_lab import LabMember


def set_workspace_permission(workspace_uri: str,
                             team_members: List[str],
                             sortingview_sorting_id: str = None):
    """Sets permission curate specified sorting on sortingview workspace based on google ID

    Parameters
    ----------
    workspace_uri : str
    team_members : List[str]
        list of team members to be given permission to edit the workspace
    sortingview_sorting_id : str
    """
    workspace = sv.load_workspace(workspace_uri)
    if sortingview_sorting_id is None:
        sortingview_sorting_id = workspace.sorting_ids[0]
    # check if team_members is empty
    if len(team_members) == 0:
        raise ValueError('The specified team does not exist or there are no members in the team;\
                          create or change the entry in LabTeam table first.')
    google_user_ids = []
    for team_member in team_members:
        google_user_id = (LabMember.LabMemberInfo & {
                          'lab_member_name': team_member}).fetch('google_user_name')
        if len(google_user_id) != 1:
            print(f'Google user ID for {team_member} does not exist or more than one ID detected;\
                    permission not given to {team_member}, skipping...')
            continue
        google_user_ids.append(google_user_id[0])
    workspace.set_sorting_curation_authorized_users(
        sorting_id=sortingview_sorting_id, user_ids=google_user_ids)
    print(
        f'Permissions to curate sorting {sortingview_sorting_id} given to {google_user_ids}.')
    return workspace_uri
