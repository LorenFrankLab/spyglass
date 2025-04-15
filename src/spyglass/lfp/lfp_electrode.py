from typing import Union

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
    ):
        """Adds an LFPElectrodeGroup and the individual electrodes

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file (e.g. the session)
        group_name : str
            The name of this group (< 200 char)
        electrode_list : list or np.ndarray
            A list of the electrode ids to include in this group.

        Raises
        ------
        ValueError
            If the electrode list is empty or if the electrodes are not valid
            for this session.
        """

        # Validate inputs
        session_key = {"nwb_file_name": nwb_file_name}
        if not (Session() & session_key):
            raise ValueError(
                f"Session '{nwb_file_name}' not found in Session table."
            )

        if isinstance(electrode_list, np.ndarray):
            # convert to list if numpy array
            electrode_list = electrode_list.astype(int).ravel().tolist()

        if not electrode_list:
            raise ValueError(
                "The provided electrode list for"
                f" '{nwb_file_name}', '{group_name}' is empty."
            )

        electrode_list = sorted(list(set(electrode_list)))

        # Check against valid electrodes for this session in the database
        valid_electrodes = (Electrode & session_key).fetch("electrode_id")

        if np.any(np.isin(electrode_list, valid_electrodes, invert=True)):
            raise ValueError(
                f"Invalid electrode_id(s) provided for "
                f"nwb_file_name '{nwb_file_name}'. They do not exist in the "
                f"Electrode table for this session."
            )

        master_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": group_name,
        }

        part_list = [
            {**master_key, "electrode_id": eid} for eid in electrode_list
        ]

        # Insert within a transaction for atomicity
        # (Ensures master and parts are inserted together or not at all)
        connection = LFPElectrodeGroup.connection
        with connection.transaction:
            # Insert master table entry (skips if already exists)
            LFPElectrodeGroup().insert1(master_key, skip_duplicates=True)

            # Insert part table entries (skips duplicates)
            # Check if part_list is not empty before inserting
            if part_list:
                LFPElectrodeGroup.LFPElectrode().insert(
                    part_list, skip_duplicates=True
                )
        logger.info(
            f"Successfully created/updated LFPElectrodeGroup {nwb_file_name}, {group_name} "
            f"with {len(electrode_list)} electrodes."
        )
