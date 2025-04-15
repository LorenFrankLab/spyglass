from typing import List, Union

import datajoint as dj
from numpy import ndarray

from spyglass.common.common_ephys import Electrode
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_session import Session
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("lfp_band_v1")


@schema
class LFPBandSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> LFPOutput.proj(lfp_merge_id='merge_id')                            # the LFP data to be filtered
    -> FirFilterParameters                                                # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int                                           # the sampling rate for this band
    ---
    min_interval_len = 1.0: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> LFPBandSelection # the LFP band selection
        -> LFPElectrodeGroup.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
        """

    # --- REFACTORED METHOD ---
    def set_lfp_band_electrodes(
        self,
        nwb_file_name: str,
        lfp_merge_id: str,
        electrode_list: Union[List[int], ndarray],
        filter_name: str,
        interval_list_name: str,
        reference_electrode_list: Union[List[int], ndarray],
        lfp_band_sampling_rate: int,
    ):
        """Populates LFPBandSelection and its part table LFPBandElectrode.

        Performs validation checks before inserting. Uses batch insert for parts.

        Parameters
        ----------
        nwb_file_name: str
            The name of the NWB file containing the LFP data.
        lfp_merge_id: str
            The merge_id of the LFP data entry in LFPOutput.
        electrode_list: list[int] | np.ndarray
            A list/array of electrode IDs to be filtered.
        filter_name: str
            The name of the filter parameters in FirFilterParameters.
        interval_list_name: str
            The name of the target interval list in IntervalList.
        reference_electrode_list: list[int] | np.ndarray
            Reference electrode IDs. Must have 1 element (common ref) or
            same number of elements as electrode_list after filtering for uniqueness.
            Use -1 for no reference.
        lfp_band_sampling_rate: int
            Desired output sampling rate. Must be a divisor of the source LFP sampling rate.

        Raises
        ------
        ValueError
            If inputs are invalid (e.g., non-existent session, LFP entry, electrodes,
            filter, interval, mismatched reference list length, invalid sampling rate,
            empty electrode list).
        """
        # === Validation ===

        # 1. Check Session & LFP Output entry existence
        session_key = {"nwb_file_name": nwb_file_name}
        if not (Session() & session_key):
            raise ValueError(f"Session '{nwb_file_name}' not found.")

        lfp_output_key = {"merge_id": lfp_merge_id}
        lfp_entry = LFPOutput & lfp_output_key
        if not lfp_entry:
            raise ValueError(
                f"LFPOutput entry with merge_id '{lfp_merge_id}' not found."
            )
        try:
            lfp_parent_info = LFPOutput.merge_get_parent(
                lfp_output_key
            ).fetch1()
            lfp_sampling_rate = lfp_parent_info["lfp_sampling_rate"]
            lfp_electrode_group_name = lfp_parent_info[
                "lfp_electrode_group_name"
            ]
        except Exception as e:
            raise dj.DataJointError(
                f"Could not fetch parent info for LFP merge_id {lfp_merge_id}: {e}"
            )

        # 2. Process and Validate Electrodes & References
        if isinstance(electrode_list, ndarray):
            electrode_list = electrode_list.astype(int).ravel().tolist()
        if isinstance(reference_electrode_list, ndarray):
            reference_electrode_list = (
                reference_electrode_list.astype(int).ravel().tolist()
            )

        if not electrode_list:
            raise ValueError("Input 'electrode_list' cannot be empty.")

        # Ensure uniqueness and sort electrodes, preparing aligned references
        if len(reference_electrode_list) == 1:
            common_ref = reference_electrode_list[0]
            # Create pairs, ensure unique electrodes, sort by electrode
            paired_list = sorted(
                list(set([(e, common_ref) for e in electrode_list]))
            )
        elif len(reference_electrode_list) == len(electrode_list):
            # Zip, ensure unique pairs (electrode, ref), sort by electrode
            paired_list = sorted(
                list(set(zip(electrode_list, reference_electrode_list)))
            )
        else:
            raise ValueError(
                "reference_electrode_list must contain either 1 element (common reference)"
                f" or {len(electrode_list)} elements (one per unique electrode specified)."
            )

        if not paired_list:
            raise ValueError(
                "Processed electrode/reference list is empty (perhaps duplicates removed?)."
            )

        electrode_list_final, ref_list_final = zip(*paired_list)

        # Check if all final electrodes exist within the specific LFPElectrodeGroup used by the source LFP
        electrode_group_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
        }
        available_electrodes_in_group = (
            LFPElectrodeGroup.LFPElectrode & electrode_group_key
        ).fetch("electrode_id")
        invalid_electrodes = set(electrode_list_final) - set(
            available_electrodes_in_group
        )
        if invalid_electrodes:
            raise ValueError(
                f"Electrode IDs {sorted(list(invalid_electrodes))} are not part of the required "
                f"LFPElectrodeGroup '{lfp_electrode_group_name}' for LFP merge_id '{lfp_merge_id}'."
            )

        # Check if all final references are valid (-1 or exist in the group)
        available_refs = set(available_electrodes_in_group).union(
            {-1}
        )  # -1 is always valid
        invalid_refs = set(ref_list_final) - available_refs
        if invalid_refs:
            raise ValueError(
                f"Reference Electrode IDs {sorted(list(invalid_refs))} are not valid "
                f"(must be -1 or an electrode in group '{lfp_electrode_group_name}')."
            )

        # 3. Validate Sampling Rate & Filter
        if lfp_sampling_rate % lfp_band_sampling_rate != 0:
            logger.warning(
                f"LFP sampling rate {lfp_sampling_rate} is not perfectly "
                f"divisible by band sampling rate {lfp_band_sampling_rate}. "
                f"Using integer division for decimation."
            )
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if decimation <= 0:
            raise ValueError(
                f"lfp_band_sampling_rate ({lfp_band_sampling_rate} Hz) must be less than "
                f"or equal to LFP sampling rate ({lfp_sampling_rate} Hz)."
            )
        # Use the sampling rate resulting from integer decimation
        actual_lfp_band_sampling_rate = int(lfp_sampling_rate / decimation)

        filter_key = {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not (FirFilterParameters() & filter_key):
            raise ValueError(
                f"Filter '{filter_name}' with sampling rate {lfp_sampling_rate} Hz "
                "is not in the FirFilterParameters table."
            )

        # 4. Validate Interval List
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
        if not (IntervalList() & interval_key):
            raise ValueError(
                f"Interval list '{interval_list_name}' is not in the IntervalList table."
            )

        # === Insertion ===
        # 1. Prepare Master Key for LFPBandSelection table
        master_key = dict(
            lfp_merge_id=lfp_merge_id,
            filter_name=filter_name,
            filter_sampling_rate=int(lfp_sampling_rate),
            target_interval_list_name=interval_list_name,
            lfp_band_sampling_rate=actual_lfp_band_sampling_rate,
        )

        # 2. Prepare Part Keys for LFPBandElectrode part table
        part_keys = [
            {
                **master_key,
                "nwb_file_name": nwb_file_name,
                "lfp_electrode_group_name": lfp_electrode_group_name,
                "electrode_id": electrode_id,
                "reference_elect_id": reference_id,
            }
            for electrode_id, reference_id in zip(
                electrode_list_final, ref_list_final
            )
        ]

        # 3. Insert within transaction
        connection = self.connection
        with connection.transaction:
            # Insert master selection entry
            self.insert1(master_key, skip_duplicates=True)
            # Insert part table entries (electrodes and their references)
            if part_keys:
                self.LFPBandElectrode().insert(part_keys, skip_duplicates=True)

        logger.info(
            f"Successfully set LFP Band Electrodes for selection:\n{master_key}"
        )
