"""High-level functions for running the Spyglass Spike Sorting V1 pipeline."""

from typing import Dict, Optional

import datajoint as dj
import numpy as np

# --- Spyglass Imports ---
# Import tables and classes directly used by these functions
from spyglass.common import ElectrodeGroup, IntervalList, LabTeam, Nwbfile
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1 import (
    ArtifactDetection,
    ArtifactDetectionParameters,
    ArtifactDetectionSelection,
    CurationV1,
    MetricCuration,
    MetricCurationParameters,
    MetricCurationSelection,
    MetricParameters,
    SortGroup,
    SpikeSorterParameters,
    SpikeSorting,
    SpikeSortingPreprocessingParameters,
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
    SpikeSortingSelection,
    WaveformParameters,
)
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import NonDaemonPool  # For parallel processing

# --- Helper Function for Parallel Processing ---


def _process_single_sort_group(args_tuple: tuple) -> bool:
    """Processes a single sort group for the v1 pipeline.

    Intended for use with multiprocessing pool within
    `populate_spyglass_spike_sorting_v1`.

    Handles recording preprocessing, artifact detection, spike sorting,
    initial curation, optional metric curation, and insertion into the
    SpikeSortingOutput merge table.

    Parameters
    ----------
    args_tuple : tuple
        A tuple containing all necessary arguments corresponding to the
        parameters of `populate_spyglass_spike_sorting_v1`.

    Returns
    -------
    bool
        True if processing for the sort group completed successfully (including
        merge table insertion), False otherwise.
    """
    (
        nwb_file_name,
        sort_interval_name,
        sort_group_id,
        team_name,
        preproc_param_name,
        artifact_param_name,
        sorter_name,
        sorting_param_name,
        run_metric_curation,
        waveform_param_name,
        metric_param_name,
        metric_curation_param_name,
        apply_curation_merges,
        description,
        skip_duplicates,
        kwargs,
    ) = args_tuple

    # Base key for this specific sort group run
    base_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": int(sort_group_id),  # Ensure correct type
        "interval_list_name": sort_interval_name,
    }
    sg_description = (
        f"{nwb_file_name} | Sort Group {sort_group_id} | "
        f"Interval {sort_interval_name}"
    )
    final_curation_key = None  # Initialize

    try:
        # --- 1. Recording Selection and Population ---
        logger.info(f"---- Step 1: Recording | {sg_description} ----")
        recording_selection_key = {
            **base_key,
            "preproc_param_name": preproc_param_name,
            "team_name": team_name,
        }
        # insert_selection generates the UUID and handles skip_duplicates
        recording_id_dict = SpikeSortingRecordingSelection.insert_selection(
            recording_selection_key
        )
        if not recording_id_dict:
            logger.warning(
                "Skipping recording step due to potential duplicate or"
                f" insertion error for {sg_description}"
            )
            # Attempt to fetch the existing key if skipping duplicates
            existing_recording = (
                SpikeSortingRecordingSelection & recording_selection_key
            ).fetch("KEY", limit=1)
            if not existing_recording:
                logger.error(
                    f"Failed to find or insert recording selection for {sg_description}"
                )
                return False
            recording_id_dict = existing_recording[0]
            if not (SpikeSortingRecording & recording_id_dict):
                logger.info(
                    f"Populating existing recording selection for {sg_description}"
                )
                SpikeSortingRecording.populate(
                    recording_id_dict, reserve_jobs=True, **kwargs
                )
        else:
            SpikeSortingRecording.populate(
                recording_id_dict, reserve_jobs=True, **kwargs
            )

        # --- 2. Artifact Detection Selection and Population ---
        logger.info(f"---- Step 2: Artifact Detection | {sg_description} ----")
        # Use the fetched/validated recording_id_dict which contains recording_id
        artifact_selection_key = {
            "recording_id": recording_id_dict["recording_id"],
            "artifact_param_name": artifact_param_name,
        }
        artifact_id_dict = ArtifactDetectionSelection.insert_selection(
            artifact_selection_key
        )
        if not artifact_id_dict:
            logger.warning(
                "Skipping artifact detection step due to potential duplicate"
                f" or insertion error for {sg_description}"
            )
            existing_artifact = (
                ArtifactDetectionSelection & artifact_selection_key
            ).fetch("KEY", limit=1)
            if not existing_artifact:
                logger.error(
                    f"Failed to find or insert artifact selection for {sg_description}"
                )
                return False
            artifact_id_dict = existing_artifact[0]
            if not (ArtifactDetection & artifact_id_dict):
                logger.info(
                    f"Populating existing artifact selection for {sg_description}"
                )
                ArtifactDetection.populate(
                    artifact_id_dict, reserve_jobs=True, **kwargs
                )
        else:
            ArtifactDetection.populate(
                artifact_id_dict, reserve_jobs=True, **kwargs
            )

        # --- 3. Spike Sorting Selection and Population ---
        logger.info(f"---- Step 3: Spike Sorting | {sg_description} ----")
        artifact_interval_name = str(artifact_id_dict["artifact_id"])
        sorting_selection_key = {
            "recording_id": recording_id_dict["recording_id"],
            "sorter": sorter_name,
            "sorter_param_name": sorting_param_name,
            "nwb_file_name": nwb_file_name,  # Required for IntervalList FK
            "interval_list_name": artifact_interval_name,
        }
        sorting_id_dict = SpikeSortingSelection.insert_selection(
            sorting_selection_key
        )
        if not sorting_id_dict:
            logger.warning(
                "Skipping spike sorting step due to potential duplicate or"
                f" insertion error for {sg_description}"
            )
            existing_sorting = (
                SpikeSortingSelection & sorting_selection_key
            ).fetch("KEY", limit=1)
            if not existing_sorting:
                logger.error(
                    f"Failed to find or insert sorting selection for {sg_description}"
                )
                return False
            sorting_id_dict = existing_sorting[0]
            if not (SpikeSorting & sorting_id_dict):
                logger.info(
                    f"Populating existing sorting selection for {sg_description}"
                )
                SpikeSorting.populate(
                    sorting_id_dict, reserve_jobs=True, **kwargs
                )
        else:
            SpikeSorting.populate(sorting_id_dict, reserve_jobs=True, **kwargs)

        # --- 4. Initial Curation ---
        logger.info(f"---- Step 4: Initial Curation | {sg_description} ----")
        # Check if initial curation (curation_id=0, parent=-1) already exists
        initial_curation_check_key = {
            "sorting_id": sorting_id_dict["sorting_id"],
            "curation_id": 0,
        }
        if CurationV1 & initial_curation_check_key:
            logger.warning(
                f"Initial curation already exists for {sg_description}, fetching key."
            )
            initial_curation_key = (
                CurationV1 & initial_curation_check_key
            ).fetch1("KEY")
        else:
            initial_curation_key = CurationV1.insert_curation(
                sorting_id=sorting_id_dict["sorting_id"],
                description=f"Initial: {description} (Group {sort_group_id})",
            )
            if not initial_curation_key:
                logger.error(
                    f"Failed to insert initial curation for {sg_description}"
                )
                return False
        final_curation_key = initial_curation_key  # Default final key

        # --- 5. Metric-Based Curation (Optional) ---
        if run_metric_curation:
            logger.info(f"---- Step 5: Metric Curation | {sg_description} ----")
            metric_selection_key = {
                **initial_curation_key,
                "waveform_param_name": waveform_param_name,
                "metric_param_name": metric_param_name,
                "metric_curation_param_name": metric_curation_param_name,
            }
            metric_curation_id_dict = MetricCurationSelection.insert_selection(
                metric_selection_key
            )
            if not metric_curation_id_dict:
                logger.warning(
                    "Skipping metric curation selection: duplicate or error"
                    f" for {sg_description}"
                )
                existing_metric_curation = (
                    MetricCurationSelection & metric_selection_key
                ).fetch("KEY", limit=1)
                if not existing_metric_curation:
                    logger.error(
                        f"Failed to find or insert metric curation selection for {sg_description}"
                    )
                    return False
                metric_curation_id_dict = existing_metric_curation[0]
                if not (MetricCuration & metric_curation_id_dict):
                    logger.info(
                        f"Populating existing metric curation selection for {sg_description}"
                    )
                    MetricCuration.populate(
                        metric_curation_id_dict, reserve_jobs=True, **kwargs
                    )
            else:
                MetricCuration.populate(
                    metric_curation_id_dict, reserve_jobs=True, **kwargs
                )

            # Check if the MetricCuration output exists before inserting final curation
            if not (MetricCuration & metric_curation_id_dict):
                logger.error(
                    f"Metric Curation failed or did not populate for {sg_description}"
                )
                return False

            logger.info(
                "---- Inserting Metric Curation Result |"
                f" {sg_description} ----"
            )
            # Check if the result of this metric curation already exists
            metric_curation_result_check_key = {
                "sorting_id": sorting_id_dict["sorting_id"],
                "parent_curation_id": initial_curation_key["curation_id"],
                "description": f"metric_curation_id: {metric_curation_id_dict['metric_curation_id']}",
            }
            # Note: This check might be too simple if descriptions vary slightly.
            # Relying on insert_curation's internal checks might be better.
            if CurationV1 & metric_curation_result_check_key:
                logger.warning(
                    f"Metric curation result already exists for {sg_description}, fetching key."
                )
                final_key = (
                    CurationV1 & metric_curation_result_check_key
                ).fetch1("KEY")
            else:
                final_key = CurationV1.insert_metric_curation(
                    metric_curation_id_dict, apply_merge=apply_curation_merges
                )
                if not final_key:
                    logger.error(
                        f"Failed to insert metric curation result for {sg_description}"
                    )
                    return False
            final_curation_key = final_key  # Update final key

        # --- 6. Insert into Merge Table ---
        # Ensure we have a valid final curation key before proceeding
        if final_curation_key is None:
            logger.error(
                f"Could not determine final curation key for merge insert for {sg_description}"
            )
            return False

        logger.info(f"---- Step 6: Merge Table Insert | {sg_description} ----")
        # Check if this specific curation is already in the merge table part
        if not (SpikeSortingOutput.CurationV1() & final_curation_key):
            SpikeSortingOutput._merge_insert(
                [final_curation_key],  # Must be a list of dicts
                part_name="CurationV1",  # Specify the correct part table name
                skip_duplicates=skip_duplicates,
            )
        else:
            logger.warning(
                f"Final curation {final_curation_key} already in merge table for {sg_description}. Skipping merge insert."
            )

        logger.info(f"==== Completed Sort Group ID: {sort_group_id} ====")
        return True  # Indicate success for this group

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Sort Group ID {sort_group_id}: {e}"
        )
        return False  # Indicate failure for this group
    except Exception as e:
        logger.error(
            f"General Error processing Sort Group ID {sort_group_id}: {e}",
            exc_info=True,  # Include traceback for debugging
        )
        return False  # Indicate failure for this group


# --- Main Populator Function ---


def populate_spyglass_spike_sorting_v1(
    nwb_file_name: str,
    sort_interval_name: str,
    team_name: str,
    probe_restriction: Optional[Dict] = None,
    preproc_param_name: str = "default",
    artifact_param_name: str = "default",
    sorter_name: str = "mountainsort4",
    sorting_param_name: str = "franklab_tetrode_hippocampus_30KHz",
    run_metric_curation: bool = True,
    waveform_param_name: str = "default_whitened",
    metric_param_name: str = "franklab_default",
    metric_curation_param_name: str = "default",
    apply_curation_merges: bool = False,
    description: str = "Standard pipeline run",
    skip_duplicates: bool = True,
    max_processes: Optional[int] = None,
    **kwargs,
) -> None:
    """Runs the standard Spyglass v1 spike sorting pipeline for specified sort groups,
    potentially in parallel across groups, and inserts results into the merge table.

    This function acts like a populator, simplifying the process by encapsulating
    the common sequence of DataJoint table selections, insertions, and
    population calls required for a typical spike sorting workflow across one or
    more sort groups within a session determined by the probe_restriction. It also
    inserts the final curated result into the SpikeSortingOutput merge table.

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file (must exist in `Nwbfile` table).
    sort_interval_name : str
        The name of the interval defined in `IntervalList` to use for sorting.
    team_name : str
        The name of the lab team defined in `LabTeam`.
    probe_restriction : dict, optional
        Restricts analysis to sort groups with matching keys from `SortGroup`
        and `ElectrodeGroup`. Defaults to {}, processing all sort groups.
    preproc_param_name : str, optional
        Parameters for preprocessing. Defaults to "default".
    artifact_param_name : str, optional
        Parameters for artifact detection. Defaults to "default".
    sorter_name : str, optional
        The spike sorting algorithm name. Defaults to "mountainsort4".
    sorting_param_name : str, optional
        Parameters for the chosen sorter. Defaults to "franklab_tetrode_hippocampus_30KHz".
    run_metric_curation : bool, optional
        If True, run waveform extraction, metrics, and metric-based curation. Defaults to True.
    waveform_param_name : str, optional
        Parameters for waveform extraction. Defaults to "default_whitened".
    metric_param_name : str, optional
        Parameters for quality metric calculation. Defaults to "franklab_default".
    metric_curation_param_name : str, optional
        Parameters for applying curation based on metrics. Defaults to "default".
    apply_curation_merges : bool, optional
        If True and metric curation runs, applies merges defined by metric curation params. Defaults to False.
    description : str, optional
        Optional description for the final curation entry. Defaults to "Standard pipeline run".
    skip_duplicates : bool, optional
        Allows skipping insertion of duplicate selection entries. Defaults to True.
    max_processes : int, optional
        Maximum number of parallel processes to run for sorting groups.
        If None or 1, runs sequentially. Defaults to None.
    **kwargs : dict
        Additional keyword arguments passed to `populate` calls (e.g., `display_progress=True`).

    Raises
    ------
    ValueError
        If required upstream entries do not exist or `probe_restriction` finds no groups.
    """

    # --- Input Validation ---
    required_tables = [Nwbfile, IntervalList, LabTeam, SortGroup]
    required_keys = [
        {"nwb_file_name": nwb_file_name},
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": sort_interval_name,
        },
        {"team_name": team_name},
        {"nwb_file_name": nwb_file_name},  # Check if any sort group exists
    ]
    for TableClass, check_key in zip(required_tables, required_keys):
        if not (TableClass & check_key):
            raise ValueError(
                f"Required entry not found in {TableClass.__name__} for key: {check_key}"
            )

    # Check parameter tables exist (if defaults aren't guaranteed by DB setup)
    # Minimal check - assumes defaults exist or user provided valid names
    if not (
        SpikeSortingPreprocessingParameters
        & {"preproc_param_name": preproc_param_name}
    ):
        raise ValueError(
            f"Preprocessing parameters not found: {preproc_param_name}"
        )
    if not (
        ArtifactDetectionParameters
        & {"artifact_param_name": artifact_param_name}
    ):
        raise ValueError(
            f"Artifact parameters not found: {artifact_param_name}"
        )
    if not (
        SpikeSorterParameters
        & {"sorter": sorter_name, "sorter_param_name": sorting_param_name}
    ):
        raise ValueError(
            f"Sorting parameters not found: {sorter_name}, {sorting_param_name}"
        )
    if run_metric_curation:
        if not (
            WaveformParameters & {"waveform_param_name": waveform_param_name}
        ):
            raise ValueError(
                f"Waveform parameters not found: {waveform_param_name}"
            )
        if not (MetricParameters & {"metric_param_name": metric_param_name}):
            raise ValueError(
                f"Metric parameters not found: {metric_param_name}"
            )
        if not (
            MetricCurationParameters
            & {"metric_curation_param_name": metric_curation_param_name}
        ):
            raise ValueError(
                f"Metric curation parameters not found: {metric_curation_param_name}"
            )

    # --- Identify Sort Groups ---
    sort_group_query = (SortGroup.SortGroupElectrode * ElectrodeGroup) & {
        "nwb_file_name": nwb_file_name
    }
    if probe_restriction:
        sort_group_query &= probe_restriction

    sort_group_ids = np.unique(sort_group_query.fetch("sort_group_id"))

    if len(sort_group_ids) == 0:
        raise ValueError(
            f"No sort groups found for nwb_file_name '{nwb_file_name}' "
            f"and probe_restriction: {probe_restriction}"
        )

    logger.info(
        f"Found {len(sort_group_ids)} sort group(s) to process:"
        f" {sort_group_ids}"
    )

    # --- Prepare arguments for each sort group ---
    process_args_list = []
    for sort_group_id in sort_group_ids:
        process_args_list.append(
            (
                nwb_file_name,
                sort_interval_name,
                sort_group_id,
                team_name,
                preproc_param_name,
                artifact_param_name,
                sorter_name,
                sorting_param_name,
                run_metric_curation,
                waveform_param_name,
                metric_param_name,
                metric_curation_param_name,
                apply_curation_merges,
                description,
                skip_duplicates,
                kwargs,
            )
        )

    # --- Run Pipeline ---
    if max_processes is None or max_processes <= 1 or len(sort_group_ids) <= 1:
        logger.info("Running spike sorting pipeline sequentially...")
        results = [
            _process_single_sort_group(args) for args in process_args_list
        ]
    else:
        logger.info(
            "Running spike sorting pipeline in parallel with"
            f" {max_processes} processes..."
        )
        try:
            with NonDaemonPool(processes=max_processes) as pool:
                results = list(
                    pool.map(_process_single_sort_group, process_args_list)
                )
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Attempting sequential processing...")
            results = [
                _process_single_sort_group(args) for args in process_args_list
            ]

    # --- Final Log ---
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    logger.info(f"---- Pipeline population finished for {nwb_file_name} ----")
    logger.info(f"    Successfully processed: {success_count} sort groups.")
    logger.info(f"    Failed to process: {fail_count} sort groups.")
