from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd

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
        electrode_list = sorted(list(set(electrode_list)))

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
        electrode_table = pd.DataFrame(electrode_table)
        part_list = electrode_table.loc[
            electrode_table["electrode_id"].isin(electrode_list),
            Electrode.primary_key,
        ]
        for key_name in master_key:
            part_list[key_name] = master_key[key_name]

        # Insert within a transaction for atomicity
        # (Ensures master and parts are inserted together or not at all)
        connection = LFPElectrodeGroup.connection
        with connection.transaction:
            # Insert master table entry (skips if already exists)
            LFPElectrodeGroup().insert1(master_key, skip_duplicates=True)
            # Insert part table entries (skips duplicates)
            LFPElectrodeGroup.LFPElectrode.insert(
                part_list.to_dict(orient="records"), skip_duplicates=True
            )

        logger.info(
            f"Successfully created/updated LFPElectrodeGroup {nwb_file_name}, {group_name} "
            f"with {len(electrode_list)} electrodes."
        )
