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
        for labmember in nwbf.experimenter:
            labmember_dict = dict()
            labmember_dict['lab_member_name'] = str(labmember)
            labmember_dict['first_name'] = str.split(labmember)[0]
            labmember_dict['last_name'] = str.split(labmember)[-1]
            self.insert1(labmember_dict, skip_duplicates=True)
            # each person is by default the member of their own LabTeam (same as their name)
            LabTeam.insert1({'team_name' : labmember_dict['lab_member_name']}, skip_duplicates=True)
            LabTeam.LabTeamMember.insert1({'team_name' : labmember_dict['lab_member_name'], 
                                           'lab_member_name' : labmember_dict['lab_member_name']}, skip_duplicates=True)

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
        LabTeam.insert1([team_name, team_description],skip_duplicates=True)
        for team_member in team_members:
            LabMember.insert1([team_member, team_member.split()[0], team_member.split()[1]], skip_duplicates=True)
            if len((LabMember.LabMemberInfo & {'lab_member_name': team_member}).fetch('google_user_name'))==0:
                print(f'Please add the Google user ID for {team_member} in LabMember.LabMemberInfo table \
                        if you want to give them permission to manually curate sortings by this team')
            LabTeam.LabTeamMember.insert1([team_name, team_member], skip_duplicates=True)
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
        self.insert1(dict(institution_name=nwbf.institution),
                     skip_duplicates=True)


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
        self.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)
