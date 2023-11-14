import os
import stat
import warnings
from pathlib import Path
from typing import List, Union

import pynwb

from ..common import Nwbfile, get_raw_eseries, populate_all_common
from ..settings import debug_mode, raw_dir
from ..utils.nwb_helper_fn import get_nwb_copy_filename


def insert_sessions(nwb_file_names: Union[str, List[str]]):
    """
    Populate the dj database with new sessions.

    Parameters
    ----------
    nwb_file_names : str or List of str
        File names in raw directory ($SPYGLASS_RAW_DIR) pointing to
        existing .nwb files. Each file represents a session. Also accepts
        strings with glob wildcards (e.g., *) so long as the wildcard specifies
        exactly one file.
    """

    if not isinstance(nwb_file_names, list):
        nwb_file_names = [nwb_file_names]

    for nwb_file_name in nwb_file_names:
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
        if len(Nwbfile() & {"nwb_file_name": out_nwb_file_name}):
            warnings.warn(
                f"Cannot insert data from {nwb_file_name}: {out_nwb_file_name}"
                + " is already in Nwbfile table."
            )
            continue

        # Make a copy of the NWB file that ends with '_'.
        # This has everything except the raw data but has a link to
        # the raw data in the original file
        copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name)
        Nwbfile().insert_from_relative_file_name(out_nwb_file_name)
        populate_all_common(out_nwb_file_name)


def copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name):
    """Copies an NWB file with a link to raw ephys data.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to be copied.
    out_nwb_file_name : str
        The name of the new NWB file with the link to raw ephys data.

    Returns
    -------
    str
        The absolute path of the new NWB file.
    """
    print(
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
        if debug_mode:
            return out_nwb_file_abs_path
        warnings.warn(
            f"Output file {out_nwb_file_abs_path} exists and will be "
            + "overwritten."
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

    # add link from new file back to raw ephys data in raw data file using fresh build manager and container cache
    # where the acquisition electricalseries objects have not been removed
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
