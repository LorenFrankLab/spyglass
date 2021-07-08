"""Schema for institution name, lab name, and lab members (experimenters)."""
import datajoint as dj

schema = dj.schema("common_lab")


@schema
class LabMember(dj.Manual):
    definition = """
    lab_member_name: varchar(80)
    ---
    first_name: varchar(80)
    last_name: varchar(80)
    """
    # TODO automatically splitting first and last name using the method below is not always appropriate / correct
    # consider not splitting full name into first and last name
    # NOTE that names must be unique here. If there are two neuroscientists named Jack Black that have data in this
    # database, this will create an incorrect linkage. NWB does not yet provide unique IDs for names.

    class LabMemberInfo(dj.Part):
        definition = """
        # Information about lab member in the context of Frank lab network
        -> master
        ---
        google_user_name: varchar(80)
        """

    def insert_from_nwbfile(self, nwbf):
        """Insert lab member information from an NWB file.

        Parameters
        ----------
        nwbf: pynwb.NWBFile
            The NWB file with experimenter information.
        """
        if nwbf.experimenter is None:
            print('No experimenter metadata found.\n')
            return
        for experimenter in nwbf.experimenter:
            self.add_from_name(experimenter)
            # each person is by default the member of their own LabTeam (same as their name)
            LabTeam.create_new_team(team_name=experimenter, team_members=[experimenter])

    def add_from_name(self, full_name):
        """Insert a lab member by name.

        The first name is the part of the name that precedes a space, and the last name is the part of the name that
        follows the last space.

        Parameters
        ----------
        full_name : str
            The name to be added.
        """
        labmember_dict = dict()
        labmember_dict['lab_member_name'] = full_name
        labmember_dict['first_name'] = str.split(full_name)[0]
        labmember_dict['last_name'] = str.split(full_name)[-1]
        self.insert1(labmember_dict, skip_duplicates=True)


@schema
class LabTeam(dj.Manual):
    definition = """
    team_name: varchar(80)
    ---
    team_description='': varchar(200)
    """

    class LabTeamMember(dj.Part):
        definition = """
        -> master
        -> LabMember
        """

    def create_new_team(self, team_name: str, team_members: list, team_description: str = ''):
        """
        Shortcut for adding a new team with a list of team members
        """
        labteam_dict = dict()
        labteam_dict['team_name'] = team_name
        labteam_dict['team_description'] = team_description
        LabTeam.insert1(labteam_dict, skip_duplicates=True)
        for team_member in team_members:
            LabMember.add_from_name(team_member)
            if len((LabMember.LabMemberInfo & {'lab_member_name': team_member}).fetch('google_user_name')) == 0:
                print(f'Please add the Google user ID for {team_member} in LabMember.LabMemberInfo table '
                      'if you want to give them permission to manually curate sortings by this team.')
            labteammember_dict = dict()
            labteammember_dict['team_name'] = team_name
            labteammember_dict['lab_member_name'] = team_member
            LabTeam.LabTeamMember.insert1(labteammember_dict, skip_duplicates=True)


@schema
class Institution(dj.Manual):
    definition = """
    institution_name: varchar(80)
    ---
    """

    def insert_from_nwbfile(self, nwbf):
        """Insert institution information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with institution information.
        """
        self.initialize()
        if nwbf.institution is None:
            print('No institution metadata found.\n')
            return
        self.insert1(dict(institution_name=nwbf.institution), skip_duplicates=True)


@schema
class Lab(dj.Manual):
    definition = """
    lab_name: varchar(80)
    ---
    """

    def insert_from_nwbfile(self, nwbf):
        """Insert lab name information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with lab name information.
        """
        self.initialize()
        if nwbf.lab is None:
            print('No lab metadata found.\n')
            return
        self.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)
