import os
import stat
import warnings
from pathlib import Path
from typing import List, Union

import pynwb

from spyglass.common import Nwbfile, get_raw_eseries, populate_all_common
from spyglass.common.common_nwbfile import schema as nwbfile_schema
from spyglass.settings import debug_mode, raw_dir
from spyglass.utils import logger
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def insert_sessions(
    nwb_file_names: Union[str, List[str]],
    rollback_on_fail: bool = False,
    raise_err: bool = False,
    reinsert: bool = False,
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
    """

    if not isinstance(nwb_file_names, list):
        nwb_file_names = [nwb_file_names]

    for nwb_file_name in nwb_file_names:
        nwb_file_name = str(nwb_file_name)  # in case it's a Path object

        if "/" in nwb_file_name:
            nwb_file_name = nwb_file_name.split("/")[-1]

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
        if file_exists and not reinsert:
            warnings.warn(
                f"Cannot insert data from {nwb_file_name}: {out_nwb_file_name}"
                + " is already in Nwbfile table."
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
        return populate_all_common(
            out_nwb_file_name,
            rollback_on_fail=rollback_on_fail,
            raise_err=raise_err,
        )


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
