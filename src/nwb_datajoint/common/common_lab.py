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

    def insert_from_nwbfile(self, nwbf):
        """Insert lab member information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with experimenter information.
        """
        for labmember in nwbf.experimenter:
            labmember_dict = dict()
            labmember_dict['lab_member_name'] = str(labmember)
            labmember_dict['first_name'] = str.split(labmember)[0]
            labmember_dict['last_name'] = str.split(labmember)[-1]
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
        self.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)
