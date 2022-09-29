import os
import stat
import warnings
from typing import List, Union

import pynwb

from ..common import Nwbfile, get_raw_eseries, populate_all_common
from .storage_dirs import check_env


def insert_sessions(nwb_file_names: Union[str, List[str]]):
    """
    Populate the dj database with new sessions.

    Parameters
    ----------
    nwb_file_names : str or List of str
        File paths (relative to $SPYGLASS_BASE_DIR) pointing to
        existing .nwb files. Each file represents a session.
    """
    check_env()

    if isinstance(nwb_file_names, str):
        nwb_file_names = [nwb_file_names]

    for nwb_file_name in nwb_file_names:
        assert not nwb_file_name.startswith(
            "/"
        ), f"You must use relative paths. nwb_file_name: {nwb_file_name}"

        # file name for the copied raw data
        out_nwb_file_name = os.path.splitext(nwb_file_name)[0] + "_.nwb"

        # Check whether the file already exists in the Nwbfile table
        if len(Nwbfile() & {"nwb_file_name": out_nwb_file_name}):
            warnings.warn(
                f"Cannot insert data from {nwb_file_name}: {out_nwb_file_name} is already in Nwbfile table."
            )
            continue

        # Make a copy of the NWB file that ends with '_'.
        # This has everything except the raw data but has a link to the raw data in the original file
        copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name)
        Nwbfile().insert_from_relative_file_name(out_nwb_file_name)
        populate_all_common(out_nwb_file_name)


def copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name):
    print(
        f"Creating a copy of NWB file {nwb_file_name} with link to raw ephys data: {out_nwb_file_name}"
    )

    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
    assert os.path.exists(
        nwb_file_abs_path
    ), f"File does not exist: {nwb_file_abs_path}"

    out_nwb_file_abs_path = Nwbfile.get_abs_path(out_nwb_file_name)
    if os.path.exists(out_nwb_file_name):
        warnings.warn(
            f"Output file {out_nwb_file_abs_path} exists and will be overwritten."
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
