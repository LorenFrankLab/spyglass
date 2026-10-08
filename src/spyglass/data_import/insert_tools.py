import os
import stat
import warnings
from pathlib import Path
from typing import List, Union

import pynwb

from spyglass.common import Nwbfile, get_raw_eseries, populate_all_common
from spyglass.settings import debug_mode, raw_dir, test_mode
from spyglass.utils import logger
from spyglass.utils.ingestion_plan import IngestionPlan, Problem
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def _plan_raw_file(copy_name: str, raw_path: Path):
    """Plan an ingestion from the raw file, before anything is registered.

    Reads the raw file and plans under the *copy's* name, which is what a real
    ingestion keys its entries by. The two describe one session -- the copy
    holds links where the raw file holds ephys data, and an NWB `object_id` is
    the same either way -- so what is planned is what would be inserted.

    Parameters
    ----------
    copy_name : str
        The `_.nwb` name ingestion would register.
    raw_path : pathlib.Path
        The raw file to read.

    Returns
    -------
    IngestionPlan
        Staged and reported.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.common.populate_all_common import lab_config
    from spyglass.data_import.planner import plan_nwbfile

    logger.info(f"Planning {raw_path.name} without registering it")

    with pynwb.NWBHDF5IO(
        path=str(raw_path), mode="r", load_namespaces=True
    ) as io:
        plan = plan_nwbfile(
            copy_name,
            config=lab_config(),  # the sidecar needs a registered file
            nwb_file=io.read(),
            nwb_path=str(raw_path),
        )

    IngestionPlanLog().stage(plan)
    plan.report()

    return plan


def insert_sessions(
    nwb_file_names: Union[str, List[str]],
    rollback_on_fail: bool = False,
    raise_err: bool = False,
    reinsert: bool = False,
    dry_run: bool = False,
    on_divergence: str = "report",
):
    """Populate the database with new sessions.

    Parameters
    ----------
    nwb_file_names : str or List of str
        File names in raw directory ($SPYGLASS_RAW_DIR) pointing to
        existing .nwb files. Each file represents a session. Also accepts
        strings with glob wildcards (e.g., *) so long as the wildcard specifies
        exactly one file.
    rollback_on_fail : bool, optional
        If True, undo all inserts if an error occurs. Default is False.
    raise_err : bool, optional
        If True, raise an error if an error occurs. Default is False.
    reinsert : bool, optional
        If True and the nwb file already exists in the Nwbfile table,
        reinsert the data. Default is False.
    dry_run : bool, optional
        If True, report what each file would insert and insert none of it.
        Every problem in a file is reported at once, rather than stopping at
        the first table that raises. Default False.

        A dry run writes nothing at all: no `_.nwb` copy, no `Nwbfile` row,
        and no `reinsert` delete. A file already in `Nwbfile` is planned from
        its copy, where a real run would warn and skip it; a file Spyglass has
        never seen is planned from the raw file, keyed by the `_.nwb` name
        ingestion would give it. Either way every table is checked: the plan
        runs tables in the order an insert would, so a table resolving a
        reference to one this ingestion also fills sees the planned rows.

    on_divergence : str, optional
        What to do when the file disagrees with a stored row: `report`
        keeps the stored rows, warns, and inserts the rest; `raise` declines.
        Default `report`. Nothing prompts.

    Returns
    -------
    list
        One `IngestionPlan` per file processed. Files skipped because they are
        already in the Nwbfile table contribute no entry, except on a dry run.
    """

    if not isinstance(nwb_file_names, list):
        nwb_file_names = [nwb_file_names]

    results = []

    for nwb_file_name in nwb_file_names:
        # Accepts a Path or a name with directories in front of it; the file
        # always lives in the raw directory, which get_abs_path supplies.
        nwb_file_name = Path(str(nwb_file_name)).name
        nwb_file_abs_path = Path(
            Nwbfile.get_abs_path(nwb_file_name, new_file=True)
        )

        if not nwb_file_abs_path.exists():
            possible_matches = sorted(Path(raw_dir).glob(f"*{nwb_file_name}*"))

            if len(possible_matches) == 1:
                nwb_file_abs_path = possible_matches[0]
                nwb_file_name = nwb_file_abs_path.name

            else:
                raise FileNotFoundError(
                    f"File not found: {nwb_file_abs_path}\n\t"
                    + f"{len(possible_matches)} possible matches:"
                    + f"{possible_matches}"
                )

        # file name for the copied raw data
        out_nwb_file_name = get_nwb_copy_filename(nwb_file_abs_path.name)

        # Check whether the file already exists in the Nwbfile table
        query = Nwbfile() & {"nwb_file_name": out_nwb_file_name}
        file_exists = bool(query)

        if dry_run:
            # None of the branches below: copying the file, registering it in
            # Nwbfile and deleting an existing session are all writes, and a
            # dry run writes to log tables only.
            if file_exists:
                # Registered: plan the copy, which is what a real run reads,
                # and whose config sidecar is reachable.
                results.append(
                    populate_all_common(out_nwb_file_name, dry_run=True)
                )
            else:
                # Never seen: plan the raw file, so a report is available
                # before the copy and the Nwbfile row exist.
                results.append(
                    _plan_raw_file(out_nwb_file_name, nwb_file_abs_path)
                )
            continue

        if file_exists and not reinsert:
            warnings.warn(
                f"Cannot insert data from {nwb_file_name}: {out_nwb_file_name}"
                + " is already in Nwbfile table."
            )
            # A result per file, including this one. Returning nothing for a
            # skipped file left the list shorter than the input, so a caller
            # with several files could not tell which had been skipped, or
            # line results up with what it asked for.
            results.append(
                IngestionPlan(
                    nwb_file_name=out_nwb_file_name,
                    fatal=(
                        Problem(
                            severity="info",
                            code="file_already_registered",
                            message=(
                                f"{out_nwb_file_name} is already in Nwbfile; "
                                + "skipped. Pass reinsert=True to replace it."
                            ),
                        ),
                    ),
                )
            )
            continue
        elif file_exists and reinsert:
            logger.info(
                f"Reinserting data from {nwb_file_name}: {out_nwb_file_name}"
            )
            query.delete(safemode=False)

        # Make a copy of the NWB file that ends with '_'.
        # This has everything except the raw data but has a link to
        # the raw data in the original file
        copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name)
        Nwbfile().insert_from_relative_file_name(out_nwb_file_name)
        results.append(
            populate_all_common(
                out_nwb_file_name,
                rollback_on_fail=rollback_on_fail,
                raise_err=raise_err,
                on_divergence=on_divergence,
            )
        )

    # One result per file. Previously returned from inside the loop, so only
    # the first file of a list was ever processed.
    return results


def copy_nwb_link_raw_ephys(
    nwb_file_name, out_nwb_file_name, keep_existing=False
):
    """Copies an NWB file with a link to raw ephys data.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to be copied.
    out_nwb_file_name : str
        The name of the new NWB file with the link to raw ephys data.
    keep_existing : bool, optional
        If True, will not overwrite an existing file. Default is False.

    Returns
    -------
    str
        The absolute path of the new NWB file.
    """
    if not test_mode:
        logger.info(
            f"Creating a copy of NWB file {nwb_file_name} "
            + f"with link to raw ephys data: {out_nwb_file_name}"
        )

    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name, new_file=True)

    if not os.path.exists(nwb_file_abs_path):
        raise FileNotFoundError(f"Could not find raw file: {nwb_file_abs_path}")

    out_nwb_file_abs_path = Nwbfile.get_abs_path(
        out_nwb_file_name, new_file=True
    )

    if os.path.exists(out_nwb_file_abs_path):
        if debug_mode or keep_existing:
            return out_nwb_file_abs_path
        if not test_mode:
            logger.warning(
                f"Output file exists, will be overwritten: {out_nwb_file_abs_path}"
            )

    with pynwb.NWBHDF5IO(
        path=nwb_file_abs_path, mode="r", load_namespaces=True
    ) as input_io:
        nwbf = input_io.read()

        # pop off acquisition electricalseries
        eseries_list = get_raw_eseries(nwbf)
        for eseries in eseries_list:
            nwbf.acquisition.pop(eseries.name)

        # pop off analog processing module
        analog_processing = nwbf.processing.get("analog")
        if analog_processing:
            nwbf.processing.pop("analog")

        # export the new NWB file
        with pynwb.NWBHDF5IO(
            path=out_nwb_file_abs_path, mode="w", manager=input_io.manager
        ) as export_io:
            export_io.export(input_io, nwbf)

    # add link from new file back to raw ephys data in raw data file using
    # fresh build manager and container cache where the acquisition
    # electricalseries objects have not been removed
    with pynwb.NWBHDF5IO(
        path=nwb_file_abs_path, mode="r", load_namespaces=True
    ) as input_io:
        nwbf_raw = input_io.read()
        eseries_list = get_raw_eseries(nwbf_raw)
        analog_processing = nwbf_raw.processing.get("analog")

        with pynwb.NWBHDF5IO(
            path=out_nwb_file_abs_path, mode="a", manager=input_io.manager
        ) as export_io:
            nwbf_export = export_io.read()

            # add link to raw ephys ElectricalSeries in raw data file
            for eseries in eseries_list:
                nwbf_export.add_acquisition(eseries)

            # add link to processing module in raw data file
            if analog_processing:
                nwbf_export.add_processing_module(analog_processing)

            nwbf_export.set_modified()
            export_io.write(nwbf_export)

    # change the permissions to only allow owner to write
    permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    os.chmod(out_nwb_file_abs_path, permissions)

    return out_nwb_file_abs_path
