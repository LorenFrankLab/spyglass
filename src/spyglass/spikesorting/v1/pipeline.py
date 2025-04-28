"""High-level functions for running the Spyglass Spike Sorting V1 pipeline."""

import time
from itertools import starmap
from typing import Any, Dict, List, Optional, Union

import datajoint as dj
import numpy as np

from spyglass.common import (
    ElectrodeGroup,
    IntervalList,
    LabMember,
    LabTeam,
    Nwbfile,
    Probe,
)
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

# --- Constants ---
INITIAL_CURATION_ID = 0
PARENT_CURATION_ID = -1


# --- Helper Function for DataJoint Population Pattern ---
def _ensure_selection_and_populate(
    selection_table: dj.Table,
    computed_table: dj.Table,
    selection_key: Dict[str, Any],
    description: str,
    reserve_jobs: bool = True,
    populate_kwargs: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ensures a selection entry exists (by inserting or fetching) and populates
    the corresponding computed table if the entry was newly inserted.

    Handles the return signature of insert_selection (list for existing,
    dict for new) based on the user-provided implementation.

    Parameters
    ----------
    selection_table : dj.Table
        The DataJoint Selection table class (e.g., SpikeSortingRecordingSelection).
    computed_table : dj.Table
        The DataJoint Computed table class linked to the selection table.
    selection_key : Dict[str, Any]
        The key defining the selection (used as input for insert_selection).
    description : str
        A description of the step for logging.
    reserve_jobs : bool, optional
        Passed to `populate`. Defaults to True.
    populate_kwargs : Optional[Dict], optional
        Additional keyword arguments for the `populate` call. Defaults to None.

    Returns
    -------
    Optional[Dict[str, Any]]
        The primary key dictionary of the selection entry if successful,
        otherwise None.
    """
    if populate_kwargs is None:
        populate_kwargs = {}

    logger.debug(
        f"Ensuring selection for {description} with key: {selection_key}"
    )
    final_key: Optional[Dict[str, Any]] = None

    try:
        inserted_or_fetched_key: Union[Dict, List[Dict]] = (
            selection_table.insert_selection(selection_key)
        )

        if isinstance(inserted_or_fetched_key, list):
            if not inserted_or_fetched_key:
                logger.error(
                    f"insert_selection found existing entries but returned an empty list for {description}."
                )
                return None
            if len(inserted_or_fetched_key) > 1:
                raise ValueError(
                    f"Multiple entries found for {description}: {inserted_or_fetched_key}"
                )
            final_key = inserted_or_fetched_key[0]
            logger.info(
                f"Using existing selection entry for {description}: {final_key}"
            )

        elif isinstance(inserted_or_fetched_key, dict):
            final_key = inserted_or_fetched_key
            logger.info(
                f"New selection entry inserted for {description}: {final_key}"
            )
        else:
            logger.error(
                f"Unexpected return type from insert_selection for {description}: {type(inserted_or_fetched_key)}"
            )
            return None

        if final_key:
            try:
                computed_table.populate(
                    final_key, reserve_jobs=reserve_jobs, **populate_kwargs
                )
            except dj.errors.DataJointError as e:
                logger.warning(
                    f"DataJointError checking computed table {computed_table.__name__} for {description}: {e}. Assuming population needed."
                )

            return final_key
        else:
            return None

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJointError during selection/population for {description}: {e}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during selection/population for {description}: {e}",
            exc_info=True,
        )
        return None


# --- Worker Function for Parallel Processing ---
def _process_single_sort_group(
    nwb_file_name: str,
    sort_interval_name: str,
    sort_group_id: int,
    team_name: str,
    preproc_param_name: str,
    artifact_param_name: str,
    sorter_name: str,
    sorting_param_name: str,
    waveform_param_name: str,
    metric_param_name: str,
    metric_curation_param_name: str,
    run_metric_curation: bool,
    apply_curation_merges: bool,
    base_curation_description: str,
    skip_duplicates: bool,
    reserve_jobs: bool,
    populate_kwargs: Dict,
) -> bool:
    """Processes a single sort group for the v1 pipeline (worker function)."""
    sg_description = (
        f"{nwb_file_name} | SG {sort_group_id} | Intvl {sort_interval_name}"
    )
    final_curation_key: Optional[Dict[str, Any]] = None

    try:
        # --- 1. Recording Selection and Population ---
        logger.info(f"---- Step 1: Recording | {sg_description} ----")
        recording_selection_key = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": sort_interval_name,
            "preproc_param_name": preproc_param_name,
            "team_name": team_name,
        }
        recording_id_dict = _ensure_selection_and_populate(
            SpikeSortingRecordingSelection,
            SpikeSortingRecording,
            recording_selection_key,
            f"Recording | {sg_description}",
            reserve_jobs,
            populate_kwargs,
        )
        if not recording_id_dict:
            logger.error(f"Recording step failed for {sg_description}.")
            return False

        # --- 2. Artifact Detection Selection and Population (optional) ---
        logger.info(f"---- Step 2: Artifact Detection | {sg_description} ----")
        artifact_selection_key = {
            "recording_id": recording_id_dict["recording_id"],
            "artifact_param_name": artifact_param_name,
        }
        artifact_id_dict = _ensure_selection_and_populate(
            ArtifactDetectionSelection,
            ArtifactDetection,
            artifact_selection_key,
            f"Artifact Detection | {sg_description}",
            reserve_jobs,
            populate_kwargs,
        )
        if not artifact_id_dict:
            logger.error(
                f"Artifact Detection step failed for {sg_description}."
            )
            return False

        # --- 3. Spike Sorting Selection and Population ---
        logger.info(f"---- Step 3: Spike Sorting | {sg_description} ----")
        sorting_selection_key = {
            "recording_id": recording_id_dict["recording_id"],
            "sorter": sorter_name,
            "sorter_param_name": sorting_param_name,
            "nwb_file_name": nwb_file_name,
            "interval_list_name": str(artifact_id_dict["artifact_id"]),
        }
        sorting_id_dict = _ensure_selection_and_populate(
            SpikeSortingSelection,
            SpikeSorting,
            sorting_selection_key,
            f"Spike Sorting | {sg_description}",
            reserve_jobs,
            populate_kwargs,
        )
        if not sorting_id_dict:
            logger.error(f"Spike Sorting step failed for {sg_description}.")
            return False

        # --- 4. Initial Automatic Curation ---
        logger.info(
            f"---- Step 4: Initial Automatic Metric Curation | {sg_description} ----"
        )
        initial_curation_key_base = {
            "sorting_id": sorting_id_dict["sorting_id"],
            "curation_id": INITIAL_CURATION_ID,
        }
        initial_curation_key = None
        if CurationV1 & initial_curation_key_base:
            logger.warning(
                f"Initial curation already exists for {sg_description}, fetching key."
            )
            initial_curation_key = (
                CurationV1 & initial_curation_key_base
            ).fetch1("KEY")
        else:
            try:
                inserted: Union[Dict, List[Dict]] = CurationV1.insert_curation(
                    sorting_id=sorting_id_dict["sorting_id"],
                    parent_curation_id=PARENT_CURATION_ID,
                    description=f"Initial: {base_curation_description} (SG {sort_group_id})",
                )
                if not inserted:
                    logger.error(
                        f"CurationV1.insert_curation returned None/empty for initial curation for {sg_description}"
                    )
                    if CurationV1 & initial_curation_key_base:
                        initial_curation_key = (
                            CurationV1 & initial_curation_key_base
                        ).fetch1("KEY")
                    else:
                        return False
                elif isinstance(inserted, list):
                    initial_curation_key = inserted[0]
                elif isinstance(inserted, dict):
                    initial_curation_key = inserted

                if not (
                    initial_curation_key
                    and initial_curation_key["curation_id"]
                    == INITIAL_CURATION_ID
                ):
                    logger.error(
                        f"Initial curation key mismatch or not found after insertion attempt for {sg_description}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Failed to insert initial curation for {sg_description}: {e}",
                    exc_info=True,
                )
                return False
        final_curation_key = initial_curation_key

        # --- 5. Metric-Based Curation (Optional) ---
        if run_metric_curation:
            logger.info(f"---- Step 5: Metric Curation | {sg_description} ----")
            metric_selection_key = {
                **initial_curation_key,
                "waveform_param_name": waveform_param_name,
                "metric_param_name": metric_param_name,
                "metric_curation_param_name": metric_curation_param_name,
            }
            metric_curation_id_dict = _ensure_selection_and_populate(
                MetricCurationSelection,
                MetricCuration,
                metric_selection_key,
                f"Metric Curation Selection | {sg_description}",
                reserve_jobs,
                populate_kwargs,
            )
            if not metric_curation_id_dict:
                logger.error(
                    f"Metric Curation Selection/Population step failed for {sg_description}."
                )
                return False

            if not (MetricCuration & metric_curation_id_dict):
                logger.error(
                    f"Metric Curation table check failed after populate call for {sg_description} | Key: {metric_curation_id_dict}"
                )
                return False

            logger.info(
                f"---- Inserting Metric Curation Result into CurationV1 | {sg_description} ----"
            )
            metric_result_description = f"metric_curation_id: {metric_curation_id_dict['metric_curation_id']}"
            metric_curation_result_check_key = {
                "sorting_id": sorting_id_dict["sorting_id"],
                "parent_curation_id": initial_curation_key["curation_id"],
                "description": metric_result_description,
            }
            final_metric_curation_key = None
            if CurationV1 & metric_curation_result_check_key:
                logger.warning(
                    f"Metric curation result already in CurationV1 for {sg_description}, fetching key."
                )
                final_metric_curation_key = (
                    CurationV1 & metric_curation_result_check_key
                ).fetch1("KEY")
            else:
                try:
                    inserted = CurationV1.insert_metric_curation(
                        metric_curation_id_dict,
                        apply_merge=apply_curation_merges,
                    )
                    if not inserted:
                        logger.error(
                            f"CurationV1.insert_metric_curation returned None/empty for {sg_description}"
                        )
                        if CurationV1 & metric_curation_result_check_key:
                            final_metric_curation_key = (
                                CurationV1 & metric_curation_result_check_key
                            ).fetch1("KEY")
                        else:
                            return False
                    elif isinstance(inserted, list):
                        final_metric_curation_key = inserted[0]
                    elif isinstance(inserted, dict):
                        final_metric_curation_key = inserted

                    if not final_metric_curation_key:
                        logger.error(
                            f"Metric curation result key not obtained after insertion attempt for {sg_description}"
                        )
                        return False
                except Exception as e:
                    logger.error(
                        f"Failed to insert metric curation result for {sg_description}: {e}",
                        exc_info=True,
                    )
                    return False
            final_curation_key = final_metric_curation_key

        # --- 6. Insert into Merge Table ---
        if final_curation_key is None:
            logger.error(
                f"Final curation key is None before Merge Table Insert for {sg_description}. Aborting."
            )
            return False

        logger.info(f"---- Step 6: Merge Table Insert | {sg_description} ----")
        logger.debug(f"Merge table insert key: {final_curation_key}")

        merge_part_table = SpikeSortingOutput.CurationV1()
        if not (merge_part_table & final_curation_key):
            try:
                SpikeSortingOutput.insert(
                    [final_curation_key],
                    part_name="CurationV1",
                    skip_duplicates=skip_duplicates,
                )
                logger.info(
                    f"Successfully inserted final curation into merge table for {sg_description}."
                )
            except Exception as e:
                logger.error(
                    f"Failed to insert into merge table for {sg_description}: {e}",
                    exc_info=True,
                )
                return False
        else:
            logger.warning(
                f"Final curation {final_curation_key} already in merge table part "
                f"{merge_part_table.table_name} for {sg_description}. Skipping merge insert."
            )

        logger.info(
            f"==== Successfully Completed Sort Group ID: {sort_group_id} ===="
        )
        return True

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Sort Group ID {sort_group_id}: {e}",
            exc_info=True,
        )
        return False
    except Exception as e:
        logger.error(
            f"General Error processing Sort Group ID {sort_group_id}: {e}",
            exc_info=True,
        )
        return False


# --- Main Populator Function ---
def _check_param_exists(table: dj.Table, pkey: Dict[str, Any], desc: str):
    """Checks if a parameter entry exists in the given table."""
    if not (table & pkey):
        raise ValueError(f"{desc} parameters not found: {pkey}")


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
    reserve_jobs: bool = True,
    max_processes: Optional[int] = None,
    **kwargs,
) -> None:
    """Runs the standard Spyglass v1 spike sorting pipeline for specified sort groups.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to process.
    sort_interval_name : str
        The name of the interval list to use for sorting.
    team_name : str
        The name of the lab team responsible for the data.
    probe_restriction : Optional[Dict], optional
        A dictionary to restrict the probe selection (e.g., {'probe_id': 1}).
    preproc_param_name : str, optional
        The name of the preprocessing parameters to use. Defaults to "default".
    artifact_param_name : str, optional
        The name of the artifact detection parameters to use. Defaults to "default".
    sorter_name : str, optional
        The name of the spike sorter to use. Defaults to "mountainsort4".
    sorting_param_name : str, optional
        The name of the sorting parameters to use. Defaults to "franklab_tetrode_hippocampus_30KHz".
    run_metric_curation : bool, optional
        If True, runs metric curation. Defaults to True.
    waveform_param_name : str, optional
        The name of the waveform parameters to use. Defaults to "default_whitened".
    metric_param_name : str, optional
        The name of the metric parameters to use. Defaults to "franklab_default".
    metric_curation_param_name : str, optional
        The name of the metric curation parameters to use. Defaults to "default".
    apply_curation_merges : bool, optional
        If True, applies merges during curation. Defaults to False.
    description : str, optional
        A description of the pipeline run. Defaults to "Standard pipeline run".
    skip_duplicates : bool, optional
        If True, skips entries that already exist in the database. Defaults to True.
    reserve_jobs : bool, optional
        If True, coordinates parallel population of tables in datajoint. Defaults to True.
        See: https://docs.datajoint.com/core/datajoint-python/latest/compute/distributed/
    max_processes : Optional[int], optional
        The maximum number of parallel processes to use. Defaults to None (no limit).
    **kwargs : Any
        Additional keyword arguments for the population process.

    Raises
    ------
    ValueError
        If any required entries or parameters do not exist in the database.

    """
    # --- Input Validation ---
    logger.info(
        f"Initiating V1 pipeline for: {nwb_file_name}, Interval: {sort_interval_name}"
    )
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")
    if not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": sort_interval_name,
        }
    ):
        raise ValueError(
            f"IntervalList not found: {nwb_file_name}, {sort_interval_name}"
        )
    if not (LabTeam & {"team_name": team_name}):
        msg = "No LabTeam found. Do you want to create a new one with your username?"
        if dj.utils.user_choice(msg).lower() not in ["yes", "y"]:
            raise ValueError(
                f"LabTeam not found: {team_name}. Use `sgc.LabTeam().create_new_team` "
                "to add your spikesorting team."
            )
        else:
            logger.info(
                f"Creating new LabTeam entry for {team_name} with username."
            )
            # Create a new LabTeam entry with the current username
            team_members = [
                (LabMember & {"username": dj.config["user"]}).fetch1(
                    "lab_member_name"
                )
            ]
            LabTeam.create_new_team(
                team_name=team_name,
                team_members=team_members,
                team_description="",
            )

    if not (SortGroup & {"nwb_file_name": nwb_file_name}):
        logger.info(
            f"Sort groups not found for {nwb_file_name}. Attempting creation by shank."
        )
        try:
            SortGroup.set_group_by_shank(nwb_file_name=nwb_file_name)
            if not (SortGroup & {"nwb_file_name": nwb_file_name}):
                raise ValueError(
                    f"Failed to create/find SortGroups for {nwb_file_name} after set_group_by_shank."
                )
            logger.info(f"Successfully created SortGroups for {nwb_file_name}.")
        except Exception as e:
            raise ValueError(
                f"Failed to create SortGroups for {nwb_file_name}: {e}"
            ) from e
    else:
        logger.info(f"Sort groups already exist for {nwb_file_name}.")

    _check_param_exists(
        SpikeSortingPreprocessingParameters,
        {"preproc_param_name": preproc_param_name},
        "Preprocessing",
    )
    _check_param_exists(
        ArtifactDetectionParameters,
        {"artifact_param_name": artifact_param_name},
        "Artifact",
    )
    _check_param_exists(
        SpikeSorterParameters,
        {"sorter": sorter_name, "sorter_param_name": sorting_param_name},
        "Sorting",
    )

    if run_metric_curation:
        _check_param_exists(
            WaveformParameters,
            {"waveform_param_name": waveform_param_name},
            "Waveform",
        )
        _check_param_exists(
            MetricParameters, {"metric_param_name": metric_param_name}, "Metric"
        )
        _check_param_exists(
            MetricCurationParameters,
            {"metric_curation_param_name": metric_curation_param_name},
            "Metric Curation",
        )

    # --- Identify Sort Groups ---
    logger.info(
        "Identifying valid sort groups joined with ElectrodeGroup and Probe..."
    )
    try:
        # This ensures we only consider sort groups linked to valid probe electrodes
        base_query_with_probe = (SortGroup * ElectrodeGroup * Probe) & {
            "nwb_file_name": nwb_file_name
        }
    except dj.errors.QueryError as e:
        raise ValueError(
            f"Error joining SortGroup, ElectrodeGroup, Probe for {nwb_file_name}. Ensure valid entries exist. Details: {e}"
        )

    # Check if any valid groups exist *after* the mandatory join
    if not base_query_with_probe:
        raise ValueError(
            f"No SortGroups found associated with valid Electrodes and Probes for {nwb_file_name}. Cannot proceed."
        )
    else:
        logger.info(
            f"Found {len(base_query_with_probe)} potential sort group entries linked to probes."
        )

    # Now apply the optional probe_restriction to the validated base query
    if probe_restriction:
        logger.info(f"Applying probe restriction: {probe_restriction}")
        sort_group_query = base_query_with_probe & probe_restriction
        # Check if the restriction resulted in an empty set
        if not sort_group_query:
            raise ValueError(
                f"No sort groups found for nwb_file_name '{nwb_file_name}' "
                f"after applying probe_restriction: {probe_restriction}"
            )
    else:
        # No restriction provided, use all valid groups found by the join
        sort_group_query = base_query_with_probe

    # Fetch the unique sort group IDs from the final query
    sort_group_ids = np.unique(sort_group_query.fetch("sort_group_id"))

    # Final check
    if len(sort_group_ids) == 0:
        raise ValueError(
            f"No processable sort groups identified for nwb_file_name '{nwb_file_name}' "
            f"(restriction applied: {bool(probe_restriction)})."
        )

    logger.info(
        f"Identified {len(sort_group_ids)} sort group(s) to process: {sort_group_ids.tolist()}"
    )

    # --- Prepare arguments for each sort group ---
    process_args_list: List[tuple] = []
    for sort_group_id in sort_group_ids:
        process_args_list.append(
            (
                nwb_file_name,
                sort_interval_name,
                int(sort_group_id),
                team_name,
                preproc_param_name,
                artifact_param_name,
                sorter_name,
                sorting_param_name,
                waveform_param_name,
                metric_param_name,
                metric_curation_param_name,
                run_metric_curation,
                apply_curation_merges,
                description,
                skip_duplicates,
                reserve_jobs,
                kwargs,
            )
        )

    # --- Run Pipeline ---
    start_time = time.time()
    results: List[bool] = []
    use_parallel = (
        max_processes is not None
        and max_processes > 0
        and len(sort_group_ids) > 1
    )

    if not use_parallel:
        effective_processes = 1
        logger.info("Running spike sorting pipeline sequentially...")
        results = list(starmap(_process_single_sort_group, process_args_list))
    else:
        effective_processes = min(max_processes, len(sort_group_ids))
        logger.info(
            f"Running spike sorting pipeline in parallel with up to {effective_processes} processes..."
        )
        try:
            with NonDaemonPool(processes=effective_processes) as pool:
                results = list(
                    pool.starmap(_process_single_sort_group, process_args_list)
                )
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}", exc_info=True)
            logger.warning("Attempting sequential processing as fallback...")
            try:
                results = list(
                    starmap(_process_single_sort_group, process_args_list)
                )
                effective_processes = 1
            except Exception as seq_e:
                logger.critical(
                    f"Sequential fallback failed after parallel error: {seq_e}",
                    exc_info=True,
                )
                results = [False] * len(sort_group_ids)
                effective_processes = 0

    # --- Final Log ---
    end_time = time.time()
    duration = end_time - start_time
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count

    final_process_count_str = (
        f"{effective_processes} process(es)"
        if effective_processes > 0
        else "failed execution"
    )

    logger.info(
        f"---- Pipeline processing finished for {nwb_file_name} ({duration:.2f} seconds, {final_process_count_str}) ----"
    )
    logger.info(
        f"    Successfully processed: {success_count} / {len(sort_group_ids)} sort groups."
    )
    if fail_count > 0:
        failed_ids: List[int] = [
            int(gid)
            for i, gid in enumerate(sort_group_ids)
            if results[i] is False
        ]
        logger.warning(
            f"    Failed to process: {fail_count} / {len(sort_group_ids)} sort groups."
        )
        logger.warning(f"    Failed Sort Group IDs: {failed_ids}")
