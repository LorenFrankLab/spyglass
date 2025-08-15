"""High-level function for running the Spyglass Position Linearization V1 pipeline."""

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import IntervalList, TrackGraph
from spyglass.linearization.merge import LinearizedPositionOutput
from spyglass.linearization.v1 import (
    LinearizationParameters,
    LinearizationSelection,
    LinearizedPositionV1,
)
from spyglass.position import PositionOutput
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_linearization_v1(
    pos_merge_id: str,
    track_graph_name: str,
    linearization_param_name: str,
    target_interval_list_name: str,
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Position Linearization pipeline.

    Automates selecting position data, a track graph, and parameters,
    computing the linearized position, and inserting into the merge table.

    Parameters
    ----------
    pos_merge_id : str
        The merge ID from the `PositionOutput` table containing the position
        data to be linearized.
    track_graph_name : str
        The name of the track graph defined in `TrackGraph`.
    linearization_param_name : str
        The name of the parameters in `LinearizationParameters`.
    target_interval_list_name : str
        The name of the interval defined in `IntervalList` over which to
        linearize the position data.
    skip_duplicates : bool, optional
        If True, skips insertion if a matching selection entry exists.
        Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to the `populate` call
        (e.g., `display_progress=True`, `reserve_jobs=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist.
    DataJointError
        If there are issues during DataJoint table operations.

    Examples
    --------
    ```python
    # --- Example Prerequisites (Ensure these are populated) ---
    # Assume 'my_pos_output_id' exists in PositionOutput
    # Assume 'my_track_graph' exists in TrackGraph
    # Assume 'default' params exist in LinearizationParameters
    # Assume 'run_interval' exists in IntervalList for the session

    pos_id = 'replace_with_actual_position_merge_id' # Placeholder
    track_name = 'my_track_graph'
    lin_params = 'default'
    interval = 'run_interval'
    nwb_file = 'my_session_.nwb' # Needed to check interval exists

    # Check interval exists (optional, function does basic check)
    # assert len(IntervalList & {'nwb_file_name': nwb_file, 'interval_list_name': interval}) == 1

    # --- Run Linearization ---
    populate_spyglass_linearization_v1(
        pos_merge_id=pos_id,
        track_graph_name=track_name,
        linearization_param_name=lin_params,
        target_interval_list_name=interval,
        display_progress=True
    )
    ```
    """

    # --- Input Validation ---
    pos_key = {"merge_id": pos_merge_id}
    if not (PositionOutput & pos_key):
        raise ValueError(f"PositionOutput entry not found: {pos_merge_id}")
    # Need nwb_file_name from position source to check track graph and interval
    pos_entry = (PositionOutput & pos_key).fetch("nwb_file_name")
    if not pos_entry:
        raise ValueError(
            f"Could not retrieve source NWB file for PositionOutput {pos_merge_id}"
        )
    nwb_file_name = pos_entry[0][
        "nwb_file_name"
    ]  # Assuming fetch returns list of dicts

    track_key = {"track_graph_name": track_graph_name}
    if not (TrackGraph & track_key):
        raise ValueError(f"TrackGraph not found: {track_graph_name}")
    # Check if track graph is associated with this NWB file (optional but good practice)
    if not (TrackGraph & track_key & {"nwb_file_name": nwb_file_name}):
        logger.warning(
            f"TrackGraph '{track_graph_name}' is not directly associated with NWB file '{nwb_file_name}'. Ensure it is applicable."
        )

    params_key = {"linearization_param_name": linearization_param_name}
    if not (LinearizationParameters & params_key):
        raise ValueError(
            f"LinearizationParameters not found: {linearization_param_name}"
        )

    interval_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": target_interval_list_name,
    }
    if not (IntervalList & interval_key):
        raise ValueError(
            f"IntervalList not found: {nwb_file_name}, {target_interval_list_name}"
        )

    # --- Construct Selection Key ---
    selection_key = {
        "pos_merge_id": pos_merge_id,
        "track_graph_name": track_graph_name,
        "linearization_param_name": linearization_param_name,
        "target_interval_list_name": target_interval_list_name,
    }

    pipeline_description = (
        f"Pos {pos_merge_id} | Track {track_graph_name} | "
        f"Params {linearization_param_name} | Interval {target_interval_list_name}"
    )

    final_key = None

    try:
        # --- 1. Insert Selection ---
        logger.info(
            f"---- Step 1: Selection Insert | {pipeline_description} ----"
        )
        if not (LinearizationSelection & selection_key):
            LinearizationSelection.insert1(
                selection_key, skip_duplicates=skip_duplicates
            )
        else:
            logger.warning(
                f"Linearization Selection already exists for {pipeline_description}"
            )
            if not skip_duplicates:
                raise dj.errors.DataJointError(
                    "Duplicate selection entry exists."
                )

        # Ensure selection exists before populating
        if not (LinearizationSelection & selection_key):
            raise dj.errors.DataJointError(
                f"Selection key missing after insert attempt for {pipeline_description}"
            )

        # --- 2. Populate Linearization ---
        logger.info(
            f"---- Step 2: Populate Linearization | {pipeline_description} ----"
        )
        if not (LinearizedPositionV1 & selection_key):
            LinearizedPositionV1.populate(
                selection_key, reserve_jobs=True, **kwargs
            )
        else:
            logger.info(
                f"LinearizedPositionV1 already populated for {pipeline_description}"
            )

        # Ensure population succeeded
        if not (LinearizedPositionV1 & selection_key):
            raise dj.errors.DataJointError(
                f"LinearizedPositionV1 population failed for {pipeline_description}"
            )
        final_key = (LinearizedPositionV1 & selection_key).fetch1("KEY")

        # --- 3. Insert into Merge Table ---
        if final_key:
            logger.info(
                f"---- Step 3: Merge Table Insert | {pipeline_description} ----"
            )
            if not (
                LinearizedPositionOutput.LinearizedPositionV1() & final_key
            ):
                LinearizedPositionOutput._merge_insert(
                    [final_key],
                    part_name="LinearizedPositionV1",
                    skip_duplicates=skip_duplicates,
                )
            else:
                logger.warning(
                    f"Final linearized position {final_key} already in merge table for {pipeline_description}."
                )
        else:
            logger.error(
                f"Final key not generated, cannot insert into merge table for {pipeline_description}"
            )

        logger.info(
            f"==== Completed Linearization Pipeline for {pipeline_description} ===="
        )

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Linearization for {pipeline_description}: {e}"
        )
    except Exception as e:
        logger.error(
            f"General Error processing Linearization for {pipeline_description}: {e}",
            exc_info=True,
        )
