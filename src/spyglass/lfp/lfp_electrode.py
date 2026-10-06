from typing import List, Union

import datajoint as dj
import numpy as np

from spyglass.common.common_ephys import Electrode
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.settings import test_mode
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

        if not test_mode:
            logger.info(
                "Successfully created/updated LFPElectrodeGroup "
                + f"{nwb_file_name}, {group_name} with {len(electrode_list)} "
                + "electrodes."
            )

    def plan_cautious_insert(
        self, session_key: dict, electrode_ids: List[int], group_name: str
    ) -> tuple:
        """Resolve an electrode group to a key plus the entries it still needs.

        The read half of `cautious_insert`, split out so a caller planning an
        ingestion can hold the entries and write them under its own gate --
        `insert_from_nwbfile(dry_run=True)` must not write.

        Every check here reads the database, so a plan is only correct against
        what is stored. A caller planning several groups before writing any of
        them has to dedupe electrode sets and pick names itself: this would
        otherwise hand back the same name twice, having no way to see the
        group the previous call planned. See
        `ImportedLFP._plan_electrode_group`.

        Parameters
        ----------
        session_key : dict
            The session key associated with the electrode group.
        electrode_ids : list
            The set of electrode ids to insert into the group.
        group_name : str
            The name of the electrode group to insert.

        Returns
        -------
        tuple of (dict, dict)
            The group's key, and the entries needed to create it keyed by
            table, master before part. The entries are empty when a stored
            group already holds these electrodes, under whatever name.

        Raises
        ------
        ValueError
            If the name is taken by a stored group holding different
            electrodes.
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
            return query.fetch("KEY")[0], dict()  # could be mult

        # group with this name already exists for a different set of electrodes
        if len(aggregated & {"lfp_electrode_group_name": group_name}):
            raise ValueError(
                f"LFP Group name {group_name} already exists"
                + "for a different set of electrode ids."
            )

        # Unique group and set of electrodes, plan the insert
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

        return master_insert, {
            LFPElectrodeGroup: [master_insert],
            LFPElectrodeGroup.LFPElectrode: electrode_inserts,
        }

    def cautious_insert(
        self, session_key: dict, electrode_ids: List[int], group_name: str
    ) -> dict:
        """Insert the electrode group if not already exist. Return group key.

        Parameters
        ----------
        session_key : dict
            The session key associated with the electrode group.
        electrode_ids : list
            The set of electrode ids to insert into the group.
        group_name : str
            The name of the electrode group to insert.

        Returns
        -------
        dict
            The key of the inserted group, or of the existing group if it
            already exists.
        """
        group_key, entries = self.plan_cautious_insert(
            session_key, electrode_ids, group_name
        )

        for table, table_entries in entries.items():
            table().insert(table_entries)

        return group_key
