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

    def insert_from_nwbfile(self, nwbf):
        """Insert lab member information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with experimenter information.
        """
        if nwbf.experimenter is None:
            print('No experimenter metadata found.\n')
            return
        for experimenter in nwbf.experimenter:
            self.add_from_name(experimenter)

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
        labmember_dict['lab_member_name'] = str(full_name)
        labmember_dict['first_name'] = str.split(full_name)[0]
        labmember_dict['last_name'] = str.split(full_name)[-1]
        self.insert1(labmember_dict, skip_duplicates=True)



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
