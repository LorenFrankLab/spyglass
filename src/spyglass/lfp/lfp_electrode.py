from typing import List, Union

import datajoint as dj
import numpy as np

from spyglass.common.common_ephys import Electrode
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import logger
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
        nwb_file_name: str,
        group_name: str,
        electrode_list: Union[list[int], np.ndarray],
        **kwargs,
    ) -> None:
        """Adds an LFPElectrodeGroup and the individual electrodes

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file (e.g. the session)
        group_name : str
            The name of this group (< 200 char)
        electrode_list : list[int] or np.ndarray
            A list of the electrode ids to include in this group.
        **kwargs : dict
            Additional keyword arguments to pass to the insert method.


        Raises
        ------
        ValueError
            If the session is not found in the Session table or
            if the electrode list is empty or
            if the electrodes are not valid for this session.
        """
        # Validate inputs
        session_key = {"nwb_file_name": nwb_file_name}
        if not (Session() & session_key):
            raise ValueError(
                f"Session '{nwb_file_name}' not found in Session table."
            )

        if not isinstance(electrode_list, (list, np.ndarray)):
            raise ValueError(
                "electrode_list must be a list or numpy array of integers."
            )

        if len(electrode_list) == 0:
            raise ValueError("electrode_list cannot be empty.")

        if isinstance(electrode_list, np.ndarray):
            # convert to list[int] if numpy array
            electrode_list = electrode_list.astype(int).ravel().tolist()

        # Sort and remove duplicates
        electrode_list = sorted(set(electrode_list))

        # Check against valid electrodes for this session in the database
        electrode_table = Electrode() & session_key
        if not electrode_table:
            raise ValueError(
                f"No electrodes found for session '{nwb_file_name}'."
            )
        if np.any(
            np.isin(
                electrode_list,
                electrode_table.fetch("electrode_id"),
                invert=True,
            )
        ):
            raise ValueError(
                f"Invalid electrode_id(s) provided for "
                f"nwb_file_name '{nwb_file_name}'. They do not exist in the "
                f"Electrode table for this session."
            )

        master_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": group_name,
        }

        restriction_str = (
            f"electrode_id = {electrode_list[0]}"
            if len(electrode_list) == 1
            else f"electrode_id in {tuple(electrode_list)}"
        )

        electrode_keys_to_insert = (electrode_table & restriction_str).fetch(
            "KEY"
        )
        part_keys = [
            {**master_key, **electrode_key}
            for electrode_key in electrode_keys_to_insert
        ]

        # Insert within a transaction for atomicity
        # (Ensures master and parts are inserted together or not at all)
        with LFPElectrodeGroup.connection.transaction:
            # Insert master table entry
            LFPElectrodeGroup().insert1(master_key, **kwargs)
            # Insert part table entries
            LFPElectrodeGroup.LFPElectrode.insert(part_keys, **kwargs)

        logger.info(
            f"Successfully created/updated LFPElectrodeGroup {nwb_file_name}, {group_name} "
            f"with {len(electrode_list)} electrodes."
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
        electrode_keys = (
            Electrode()
            & session_key
            & [{"electrode_id": e_id} for e_id in e_ids]
        ).fetch("KEY")
        e_group_dict = dict(lfp_electrode_group_name=group_name)
        electrode_inserts = [
            dict(e_key, **e_group_dict) for e_key in electrode_keys
        ]

        self.insert1(master_insert)
        self.LFPElectrode.insert(electrode_inserts)
        return master_insert
