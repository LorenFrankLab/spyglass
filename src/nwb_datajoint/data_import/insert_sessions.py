import os
import warnings

import pynwb

from ..common import Nwbfile, get_raw_eseries, populate_all_common
from .storage_dirs import check_env


def insert_sessions(nwb_file_names):
    """
    Populate the dj database with new sessions.

    Parameters
    ----------
    nwb_file_names : string or List of strings
        nwb_file_names is a list of relative file paths, relative to $NWB_DATAJOINT_BASE_DIR, pointing to
        existing .nwb files. Each file represents a session.
    """
    check_env()

    if type(nwb_file_names) is str:
        nwb_file_names = [nwb_file_names]

    for nwb_file_name in nwb_file_names:
        assert not nwb_file_name.startswith(
            '/'), f'You must use relative paths. nwb_file_name: {nwb_file_name}'

        # file name for the copied raw data
        out_nwb_file_name = os.path.splitext(nwb_file_name)[0] + '_.nwb'

        # Check whether the file already exists in the Nwbfile table
        if len(Nwbfile() & {'nwb_file_name': out_nwb_file_name}):
            warnings.warn(
                f'Cannot insert data from {nwb_file_name}: {out_nwb_file_name} is already in Nwbfile table.')
            continue

        # Make a copy of the NWB file that ends with '_'.
        # This has everything except the raw data but has a link to the raw data in the original file
        copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name)
        Nwbfile().insert_from_relative_file_name(out_nwb_file_name)
        populate_all_common(out_nwb_file_name)


def copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name):
    print(
        f'Creating a copy of NWB file {nwb_file_name} with link to raw ephys data: {out_nwb_file_name}')

    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
    assert os.path.exists(
        nwb_file_abs_path), f'File does not exist: {nwb_file_abs_path}'

    out_nwb_file_abs_path = Nwbfile.get_abs_path(out_nwb_file_name)
    if os.path.exists(out_nwb_file_name):
        warnings.warn(
            f'Output file {out_nwb_file_abs_path} exists and will be overwritten.')

    with pynwb.NWBHDF5IO(path=nwb_file_abs_path, mode='r', load_namespaces=True) as input_io:
        nwbf = input_io.read()

        # pop off acquisition electricalseries
        eseries_list = get_raw_eseries(nwbf)
        for eseries in eseries_list:
            nwbf.acquisition.pop(eseries.name)

        # export the new NWB file
        with pynwb.NWBHDF5IO(path=out_nwb_file_abs_path, mode='w', manager=input_io.manager) as export_io:
            export_io.export(input_io, nwbf)

    # add link from new file back to raw ephys data in raw data file using fresh build manager and container cache
    # where the acquisition electricalseries objects have not been removed
    with pynwb.NWBHDF5IO(path=nwb_file_abs_path, mode='r', load_namespaces=True) as input_io:
        nwbf_raw = input_io.read()
        eseries_list = get_raw_eseries(nwbf_raw)

        with pynwb.NWBHDF5IO(path=out_nwb_file_abs_path, mode='a', manager=input_io.manager) as export_io:
            nwbf_export = export_io.read()

            # add link to raw ephys ElectricalSeries in raw data file
            for eseries in eseries_list:
                nwbf_export.add_acquisition(eseries)
            nwbf_export.set_modified()  # workaround until the above sets modified=True on the file

            export_io.write(nwbf_export)

    return out_nwb_file_abs_path
