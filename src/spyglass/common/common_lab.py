"""Schema for institution, lab team/name/members. Session-independent."""
import datajoint as dj

from ..utils.nwb_helper_fn import get_nwb_file
from .common_nwbfile import Nwbfile

schema = dj.schema("common_lab")


@schema
class LabMember(dj.Manual):
    definition = """
    lab_member_name: varchar(80)
    ---
    first_name: varchar(200)
    last_name: varchar(200)
    """

    # NOTE that names must be unique here. If there are two neuroscientists
    # named Jack Black that have data in this database, this will create an
    # incorrect linkage. NWB does not yet provide unique IDs for names.

    class LabMemberInfo(dj.Part):
        definition = """
        # Information about lab member in the context of Frank lab network
        -> LabMember
        ---
        google_user_name: varchar(200)              # used for permission to curate
        datajoint_user_name = "": varchar(200)      # used for permission to delete entries
        """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Insert lab member information from an NWB file.

        Parameters
        ----------
        nwbf: pynwb.NWBFile
            The NWB file with experimenter information.
        """
        if isinstance(nwbf, str):
            nwb_file_abspath = Nwbfile.get_abs_path(nwbf, new_file=True)
            nwbf = get_nwb_file(nwb_file_abspath)

        if nwbf.experimenter is None:
            print("No experimenter metadata found.\n")
            return

        for experimenter in nwbf.experimenter:
            cls.insert_from_name(experimenter)

            # each person is by default the member of their own LabTeam
            # (same as their name)

            full_name, _, _ = decompose_name(experimenter)
            LabTeam.create_new_team(
                team_name=full_name, team_members=[full_name]
            )

    @classmethod
    def insert_from_name(cls, full_name):
        """Insert a lab member by name.

        Parameters
        ----------
        full_name : str
            The name to be added.
        """
        _, first, last = decompose_name(full_name)
        cls.insert1(
            dict(
                lab_member_name=f"{first} {last}",
                first_name=first,
                last_name=last,
            ),
            skip_duplicates=True,
        )


@schema
class LabTeam(dj.Manual):
    definition = """
    team_name: varchar(80)
    ---
    team_description = "": varchar(2000)
    """

    class LabTeamMember(dj.Part):
        definition = """
        -> LabTeam
        -> LabMember
        """

    @classmethod
    def create_new_team(
        cls, team_name: str, team_members: list, team_description: str = ""
    ):
        """Create a new team with a list of team members.

        If the lab member does not exist in the database, they will be added.

        Parameters
        ----------
        team_name : str
            The name of the team.
        team_members : str
            The full names of the lab members that are part of the team.
        team_description: str
            The description of the team.
        """
        labteam_dict = dict()
        labteam_dict["team_name"] = team_name
        labteam_dict["team_description"] = team_description
        cls.insert1(labteam_dict, skip_duplicates=True)

        for team_member in team_members:
            LabMember.insert_from_name(team_member)
            query = (
                LabMember.LabMemberInfo() & {"lab_member_name": team_member}
            ).fetch("google_user_name")
            if not query:
                print(
                    f"Please add the Google user ID for {team_member} in the "
                    + "LabMember.LabMemberInfo table to help manage permissions."
                )
            labteammember_dict = dict()
            labteammember_dict["team_name"] = team_name
            labteammember_dict["lab_member_name"] = team_member
            cls.LabTeamMember.insert1(labteammember_dict, skip_duplicates=True)


@schema
class Institution(dj.Manual):
    definition = """
    institution_name: varchar(80)
    ---
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Insert institution information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with institution information.
        """
        if nwbf.institution is None:
            print("No institution metadata found.\n")
            return
        cls.insert1(
            dict(institution_name=nwbf.institution), skip_duplicates=True
        )


@schema
class Lab(dj.Manual):
    definition = """
    lab_name: varchar(80)
    ---
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Insert lab name information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with lab name information.
        """
        if nwbf.lab is None:
            print("No lab metadata found.\n")
            return
        cls.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)


def decompose_name(full_name: str) -> tuple:
    """Return the full, first and last name parts of a full name.

    Names capitalized. Decomposes either comma- or space-separated.

    Parameters
    ----------
    full_name: str
        The full name to decompose.

    Returns
    -------
    tuple
        A tuple of the full, first, last name parts.
    """
    if full_name.count(", ") != 1 and full_name.count(" ") != 1:
        raise ValueError(
            f"Names should be stored as 'last, first'. Skipping {full_name}"
        )
    elif ", " in full_name:
        last, first = full_name.title().split(", ")
    else:
        first, last = full_name.title().split(" ")
    full = f"{first} {last}"
    return full, first, last
