"""High-level function for running the Spyglass MUA V1 pipeline."""

from typing import List, Union

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import IntervalList, Nwbfile
from spyglass.mua.v1 import MuaEventsParameters, MuaEventsV1
from spyglass.position import PositionOutput
from spyglass.spikesorting.analysis.v1 import SortedSpikesGroup
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_mua_v1(
    nwb_file_name: str,
    sorted_spikes_group_name: str,
    unit_filter_params_name: str,
    pos_merge_id: str,
    detection_intervals: Union[str, List[str]],
    mua_param_name: str = "default",
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Multi-Unit Activity (MUA) detection pipeline
    for one or more specified detection intervals.

    This function acts like a populator for the `MuaEventsV1` table, checking
    for necessary upstream data (spike sorting group, position data, MUA parameters)
    and then triggering the computation for the specified interval(s).

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file (must exist in `Nwbfile` table).
    sorted_spikes_group_name : str
        The name of the spike group defined in `SortedSpikesGroup`.
    unit_filter_params_name : str
        The name of the unit filter parameters used in the `SortedSpikesGroup`.
    pos_merge_id : str
        The merge ID (UUID) from the `PositionOutput` table containing the
        animal's position and speed data.
    detection_intervals : Union[str, List[str]]
        The name of the interval list (or a list of names) defined in
        `IntervalList` during which MUA events should be detected.
    mua_param_name : str, optional
        The name of the MUA detection parameters in `MuaEventsParameters`.
        Defaults to "default".
    skip_duplicates : bool, optional
        If True (default), checks if the entry already exists before attempting
        population. Note that DataJoint's `populate` typically handles this.
    **kwargs : dict
        Additional keyword arguments passed to the `populate` call
        (e.g., `display_progress=True`, `reserve_jobs=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist, or if any
        specified detection interval is not found.
    DataJointError
        If there are issues during DataJoint table operations.

    Examples
    --------
    ```python
    # Example prerequisites (ensure these are populated)
    nwb_file = 'my_session_.nwb'
    spikes_group = 'my_spike_group'
    unit_filter = 'default_exclusion'
    # pos_key = {'nwb_file_name': nwb_file, 'position_info_param_name': 'my_pos_params', ...}
    # position_id = (PositionOutput & pos_key).fetch1('merge_id')
    position_id = 'replace_with_actual_position_merge_id' # Placeholder
    detection_interval_name = 'pos 0 valid times' # Or another valid interval

    # Run MUA detection for a single interval
    populate_spyglass_mua_v1(
        nwb_file_name=nwb_file,
        sorted_spikes_group_name=spikes_group,
        unit_filter_params_name=unit_filter,
        pos_merge_id=position_id,
        detection_intervals=detection_interval_name,
        mua_param_name='default',
        display_progress=True
    )

    # Run MUA detection for multiple intervals
    detection_interval_list = ['pos 0 valid times', 'pos 1 valid times']
    populate_spyglass_mua_v1(
        nwb_file_name=nwb_file,
        sorted_spikes_group_name=spikes_group,
        unit_filter_params_name=unit_filter,
        pos_merge_id=position_id,
        detection_intervals=detection_interval_list,
        mua_param_name='default',
        display_progress=True
    )
    ```
    """

    # --- Input Validation ---
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")

    mua_params_key = {"mua_param_name": mua_param_name}
    if not (MuaEventsParameters & mua_params_key):
        raise ValueError(f"MuaEventsParameters not found: {mua_param_name}")

    spike_group_key = {
        "nwb_file_name": nwb_file_name,
        "sorted_spikes_group_name": sorted_spikes_group_name,
        "unit_filter_params_name": unit_filter_params_name,
    }
    if not (SortedSpikesGroup & spike_group_key):
        raise ValueError(
            "SortedSpikesGroup entry not found for:"
            f" {nwb_file_name}, {sorted_spikes_group_name},"
            f" {unit_filter_params_name}"
        )

    pos_key = {"merge_id": pos_merge_id}
    if not (PositionOutput & pos_key):
        raise ValueError(f"PositionOutput entry not found for: {pos_merge_id}")

    # Ensure detection_intervals is a list
    if isinstance(detection_intervals, str):
        detection_intervals = [detection_intervals]
    if not isinstance(detection_intervals, list):
        raise TypeError(
            "detection_intervals must be a string or a list of strings."
        )

    # Validate each interval
    valid_intervals = []
    for interval_name in detection_intervals:
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
        if not (IntervalList & interval_key):
            raise ValueError(
                f"IntervalList entry not found for: {nwb_file_name},"
                f" {interval_name}"
            )
        valid_intervals.append(interval_name)

    if not valid_intervals:
        logger.error("No valid detection intervals found. Aborting.")
        return

    # --- Loop through Intervals and Populate ---
    successful_intervals = 0
    failed_intervals = 0
    for interval_name in valid_intervals:
        pipeline_description = (
            f"{nwb_file_name} | Spikes {sorted_spikes_group_name} |"
            f" Pos {pos_merge_id} | Interval {interval_name} |"
            f" Params {mua_param_name}"
        )

        # Construct the primary key for MuaEventsV1
        mua_population_key = {
            **mua_params_key,
            **spike_group_key,
            "pos_merge_id": pos_merge_id,
            "detection_interval": interval_name,
        }

        # Check if already computed
        if skip_duplicates and (MuaEventsV1 & mua_population_key):
            logger.warning(
                f"MUA events already computed for {pipeline_description}."
                " Skipping population."
            )
            successful_intervals += (
                1  # Count skipped as success in terms of availability
            )
            continue

        # Populate MuaEventsV1
        logger.info(f"---- Populating MUA Events | {pipeline_description} ----")
        try:
            MuaEventsV1.populate(mua_population_key, **kwargs)
            # Verify insertion (optional, populate should raise error if it fails)
            if MuaEventsV1 & mua_population_key:
                logger.info(
                    "---- MUA event detection population complete for"
                    f" {pipeline_description} ----"
                )
                successful_intervals += 1
            else:
                logger.error(
                    "---- MUA event detection population failed for"
                    f" {pipeline_description} (entry not found after populate) ----"
                )
                failed_intervals += 1

        except dj.errors.DataJointError as e:
            logger.error(
                f"DataJoint Error populating MUA events for {pipeline_description}: {e}"
            )
            failed_intervals += 1
        except Exception as e:
            logger.error(
                f"General Error populating MUA events for {pipeline_description}: {e}",
                exc_info=True,  # Include traceback for debugging
            )
            failed_intervals += 1

    # --- Final Log ---
    logger.info(
        f"---- MUA pipeline population finished for {nwb_file_name} ----"
    )
    logger.info(
        f"    Successfully processed/found: {successful_intervals} intervals."
    )
    logger.info(f"    Failed to process: {failed_intervals} intervals.")
