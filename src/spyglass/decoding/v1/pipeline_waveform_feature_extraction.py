"""High-level function for running the Spyglass Waveform Features Extraction V1 pipeline."""

from typing import List, Union

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import IntervalList
from spyglass.decoding.v1.waveform_features import (
    UnitWaveformFeatures,
    UnitWaveformFeaturesSelection,
    WaveformFeaturesParams,
)
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_waveform_features_v1(
    spikesorting_merge_ids: Union[str, List[str]],
    features_param_name: str = "amplitude",
    interval_list_name: str = None,  # Optional interval to restrict spikes
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Waveform Features extraction pipeline.

    Automates selecting spike sorting results and parameters, and computing
    waveform features (often used for clusterless decoding).

    Parameters
    ----------
    spikesorting_merge_ids : Union[str, List[str]]
        A single merge ID or list of merge IDs from the `SpikeSortingOutput`
        table containing the spike sorting results to process.
    features_param_name : str, optional
        The name of the parameters in `WaveformFeaturesParams`.
        Defaults to "amplitude".
    interval_list_name : str, optional
        The name of an interval list used to select a temporal subset of spikes
        before extracting features. If None, all spikes are used. Defaults to None.
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
    # Assume SpikeSortingOutput merge entry 'my_sorting_merge_id' exists
    # Assume 'amplitude' params exist in WaveformFeaturesParams
    # Assume 'run_interval' exists in IntervalList for the session

    merge_id = 'replace_with_actual_spikesorting_merge_id' # Placeholder
    feat_params = 'amplitude'
    interval = 'run_interval'

    # --- Run Waveform Feature Extraction for all spikes ---
    populate_spyglass_waveform_features_v1(
        spikesorting_merge_ids=merge_id,
        features_param_name=feat_params,
        display_progress=True
    )

    # --- Run Waveform Feature Extraction restricted to an interval ---
    # populate_spyglass_waveform_features_v1(
    #     spikesorting_merge_ids=merge_id,
    #     features_param_name=feat_params,
    #     interval_list_name=interval,
    #     display_progress=True
    # )

    # --- Run for multiple merge IDs ---
    # merge_ids = ['id1', 'id2']
    # populate_spyglass_waveform_features_v1(
    #     spikesorting_merge_ids=merge_ids,
    #     features_param_name=feat_params,
    #     display_progress=True
    # )
    ```
    """

    # --- Input Validation ---
    params_key = {"features_param_name": features_param_name}
    if not (WaveformFeaturesParams & params_key):
        raise ValueError(
            f"WaveformFeaturesParams not found: {features_param_name}"
        )

    if isinstance(spikesorting_merge_ids, str):
        spikesorting_merge_ids = [spikesorting_merge_ids]
    if not isinstance(spikesorting_merge_ids, list):
        raise TypeError("spikesorting_merge_ids must be a string or list.")

    valid_merge_ids = []
    nwb_file_name = None  # To check interval list
    for merge_id in spikesorting_merge_ids:
        ss_key = {"merge_id": merge_id}
        if not (SpikeSortingOutput & ss_key):
            raise ValueError(f"SpikeSortingOutput entry not found: {merge_id}")
        valid_merge_ids.append(merge_id)
        # Get nwb_file_name from the first valid merge_id to check interval
        if nwb_file_name is None:
            try:
                nwb_file_name = (
                    SpikeSortingOutput.merge_get_parent(ss_key) & ss_key
                ).fetch1("nwb_file_name")
            except Exception as e:
                logger.warning(
                    f"Could not fetch nwb_file_name for merge_id {merge_id}: {e}"
                )
                # Continue, interval check might fail later if interval_list_name is provided

    if interval_list_name and nwb_file_name:
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
        if not (IntervalList & interval_key):
            raise ValueError(
                f"IntervalList not found: {nwb_file_name}, {interval_list_name}"
            )
    elif interval_list_name and not nwb_file_name:
        raise ValueError(
            "Cannot check IntervalList without a valid nwb_file_name from SpikeSortingOutput."
        )

    # --- Loop through Merge IDs and Populate ---
    successful_items = 0
    failed_items = 0
    for merge_id in valid_merge_ids:
        # --- Construct Selection Key ---
        selection_key = {
            "merge_id": merge_id,
            "features_param_name": features_param_name,
            "interval_list_name": (
                interval_list_name if interval_list_name else ""
            ),
        }
        pipeline_description = (
            f"SpikeSortingMergeID {merge_id} | Features {features_param_name} | "
            f"Interval {interval_list_name if interval_list_name else 'None'}"
        )

        try:
            # --- 1. Insert Selection ---
            logger.info(
                f"---- Step 1: Selection Insert | {pipeline_description} ----"
            )
            if not (UnitWaveformFeaturesSelection & selection_key):
                UnitWaveformFeaturesSelection.insert1(
                    selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"Waveform Features Selection already exists for {pipeline_description}"
                )
                if not skip_duplicates:
                    raise dj.errors.DataJointError(
                        "Duplicate selection entry exists."
                    )

            # Ensure selection exists before populating
            if not (UnitWaveformFeaturesSelection & selection_key):
                raise dj.errors.DataJointError(
                    f"Selection key missing after insert attempt for {pipeline_description}"
                )

            # --- 2. Populate Waveform Features ---
            logger.info(
                f"---- Step 2: Populate Waveform Features | {pipeline_description} ----"
            )
            if not (UnitWaveformFeatures & selection_key):
                UnitWaveformFeatures.populate(
                    selection_key, reserve_jobs=True, **kwargs
                )
            else:
                logger.info(
                    f"UnitWaveformFeatures already populated for {pipeline_description}"
                )

            # Verify population
            if UnitWaveformFeatures & selection_key:
                logger.info(
                    f"==== Completed Waveform Features Extraction for {pipeline_description} ===="
                )
                successful_items += 1
            else:
                raise dj.errors.DataJointError(
                    f"UnitWaveformFeatures population failed for {pipeline_description}"
                )

        except dj.errors.DataJointError as e:
            logger.error(
                f"DataJoint Error processing Waveform Features for {pipeline_description}: {e}"
            )
            failed_items += 1
        except Exception as e:
            logger.error(
                f"General Error processing Waveform Features for {pipeline_description}: {e}",
                exc_info=True,
            )
            failed_items += 1

    # --- Final Log ---
    logger.info("---- Waveform Features Extraction finished ----")
    logger.info(f"    Successfully processed/found: {successful_items} items.")
    logger.info(f"    Failed to process: {failed_items} items.")
