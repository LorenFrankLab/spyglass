import datajoint as dj
from spyglass.common.common_ephys import Electrode
from spyglass.common.common_session import Session

schema = dj.schema("lfp_electrodes")


@schema
class LFPElectrodeGroup(dj.Manual):
    definition = """
     -> Session                             # the session to which this LFP belongs
     lfp_electrode_group_name: varchar(200) # the name of this group of electrodes
     """

    class LFPElectrode(dj.Part):
        definition = """
        -> LFPElectrodeGroup # the group of electrodes to be filtered
        -> Electrode        # the electrode to be filtered
        """

    @staticmethod
    def create_lfp_electrode_group(
        nwb_file_name: str, group_name: str, electrode_list: list[int]
    ):
        """Adds an LFPElectrodeGroup and the individual electrodes

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file (e.g. the session)
        group_name : str
            The name of this group (< 200 char)
        electrode_list : list
            A list of the electrode ids to include in this group.
        """
        # remove the session and then recreate the session and Electrode list
        # check to see if the user allowed the deletion
        key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": group_name,
        }
        LFPElectrodeGroup().insert1(key, skip_duplicates=True)

        # TODO: do this in a better way
        all_electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch(
            as_dict=True
        )
        primary_key = Electrode.primary_key
        for e in all_electrodes:
            # create a dictionary so we can insert the electrodes
            if e["electrode_id"] in electrode_list:
                lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                lfpelectdict["lfp_electrode_group_name"] = group_name
                LFPElectrodeGroup().LFPElectrode.insert1(
                    lfpelectdict, skip_duplicates=True
                )


# @schema
# class LFPElectrodeGroups(dj.Manual):
#     definition = """
#     -> common_session.Session
#     group_id: uint
#     target: varchar(32)
#     ---
#     lfp_electrode_group_name: varchar(32) # the name of this group of electrodes
#     parent_set: group_id of upstream set
#     """

#     def insert_entry(key, **kwargs):
#         if paramset_idx is None:
#             paramset_idx = (
#                 dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
#             ) + 1

#     class Electrodes(dj.Part):
#         definition="""

#         electrode_id: int
#         """


#     class LFP(dj.Part):
#         definition = """
#         id: int
#         """

#     class LFPBand(dj.Part):
#         definition = """
#         id: int
#         """

#     class Ripple(dj.Part):
#         definition = """
#         id: int
#         """

#     def validate_insert(key, **kwargs):
#         pass

#     def validate_delete(restriction=None, **kwargs):
#         pass
