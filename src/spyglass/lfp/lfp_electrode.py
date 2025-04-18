from typing import List

import datajoint as dj
from numpy import ndarray

from spyglass.common.common_ephys import Electrode
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("lfp_electrode")


@schema
class LFPElectrodeGroup(SpyglassMixin, dj.Manual):
    definition = """
     -> Session                             # the session for this LFP
     lfp_electrode_group_name: varchar(200) # name for this group of electrodes
     """

    class LFPElectrode(SpyglassMixin, dj.Part):
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
        if isinstance(electrode_list, ndarray):
            # convert to list if it is an numpy array
            electrode_list = list(electrode_list.astype(int).reshape(-1))
        for e in all_electrodes:
            # create a dictionary so we can insert the electrodes
            if e["electrode_id"] in electrode_list:
                lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                lfpelectdict["lfp_electrode_group_name"] = group_name
                LFPElectrodeGroup().LFPElectrode.insert1(
                    lfpelectdict, skip_duplicates=True
                )

    def cautious_insert(
        self, session_key: dict, electrode_ids: List[int], group_name: str
    ) -> str:
        """Insert the electrode group, or return name if it already exists.

        Parameters
        ----------
        session_key : dict
            The session key associated with the electrode group.
        electrode_ids : list
            The set of electrode ids to insert into the group.
        group_name : str
            The name of the electrode group to insert.

        Returns
        ----------
        dictionary
            The key of the inserted group, or the existing group if it already exists.
        """
        e_ids = set(electrode_ids)  # remove duplicates

        # Collect existing ids into comma separated string to avoid multi-fetch
        aggregated = (self & session_key).aggr(
            self.LFPElectrode,
            ids="GROUP_CONCAT(electrode_id ORDER BY electrode_id ASC)",
        )

        # group for this set of electrodes already exists
        sorted_str = ",".join(map(str, sorted(e_ids)))
        if len(query := aggregated & f"ids='{sorted_str}'"):
            return query.fetch("KEY")[0]  # could be mult

        # group with this name already exists for a different set of electrodes
        if len(aggregated & {"lfp_electrode_group_name": group_name}):
            raise ValueError(
                f"LFP Group name {group_name} already exists"
                + "for a different set of electrode ids."
            )

        # Unique group and set of electrodes, insert
        master_insert = dict(**session_key, lfp_electrode_group_name=group_name)
        electrode_inserts = [dict(master_insert, electrode_id=e) for e in e_ids]

        self.insert1(master_insert)
        self.LFPElectrode.insert(electrode_inserts)
        return master_insert
