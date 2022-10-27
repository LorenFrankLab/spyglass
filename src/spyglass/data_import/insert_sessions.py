import os
import stat
import warnings
from typing import List, Union

import pynwb
import datajoint as dj

from ..common import Nwbfile, get_raw_eseries
from .storage_dirs import check_env

from ..common.common_behav import (
    PositionSource,
    RawPosition,
    StateScriptFile,
    VideoFile,
)
from ..common.common_dio import DIOEvents
from ..common.common_ephys import Electrode, Raw, SampleCount
from ..common.common_nwbfile import Nwbfile
from ..common.common_session import ExperimenterList, Session
from ..common.common_task import TaskEpoch


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
        assert not os.path.isabs(
            nwb_file_name
        ), f"You must provide the relative path for {nwb_file_name}."

        # File name for the copied raw data
        out_nwb_file_name = os.path.splitext(nwb_file_name)[0] + "_.nwb"

        # Check whether the file already exists in the Nwbfile table
        if len(Nwbfile & {"nwb_file_name": out_nwb_file_name}):
            warnings.warn(
                f"{out_nwb_file_name} is already in Nwbfile table, "
                 "so skipping all downstream populate steps. If you want to insert this file anew, "
                 "then first delete it from Nwbfile table."
            )
            continue

        # Make a copy of the NWB file that ends with '_'.
        # This has everything except the raw data but has a link to the raw data in the original file
        copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name)
        # conn = dj.conn()
        # with conn.transaction:
        populate_all_common(out_nwb_file_name)


def populate_all_common(nwb_file_name: str):
    """Populate tables in common from the NWB file.

    Parameters
    ----------
    nwb_file_name : str
        name of the nwb file (with _)
    """
    from ..spikesorting.spikesorting_sorting import ImportedSpikeSorting

    # conn = dj.conn()

    # with conn.transaction:
    # Insert to Nwbfile table
    print("Populate Nwbfile...")
    nwbfile_key = Nwbfile.insert_from_relative_file_name(nwb_file_name)
    print(nwbfile_key)
    print()

    fp = [(Nwbfile & {"nwb_file_name": nwb_file_name}).proj()]

    Session.populate(fp)

    # If we use Kachery for data sharing we can uncomment the following two lines. TBD
    # print('Populate NwbfileKachery...')
    # NwbfileKachery.populate()

    print("Populate ExperimenterList...")
    ExperimenterList.populate(fp)
    print()

    # print('Populate ElectrodeGroup...')
    # ElectrodeGroup.populate(fp)
    # print()

    print("Populate Raw...")
    Raw.populate(fp)
    print()

    print("Populate Electrode...")
    Electrode.populate(fp)
    print()

    print("Populate SampleCount...")
    SampleCount.populate(fp)
    print()

    print("Populate DIOEvents...")
    DIOEvents.populate(fp)
    print()

    # sensor data (from analog ProcessingModule) is temporarily removed from NWBFile
    # to reduce file size while it is not being used. add it back in by commenting out
    # the removal code in spyglass/data_import/insert_sessions.py when ready
    # print('Populate SensorData')
    # SensorData.populate(fp)

    print("Populate TaskEpoch...")
    TaskEpoch.populate(fp)
    print()

    print("Populate StateScriptFile...")
    StateScriptFile.populate(fp)
    print()

    print("Populate VideoFile...")
    VideoFile.populate(fp)
    print()

    print("Populate RawPosition...")
    PositionSource.insert_from_nwbfile(nwb_file_name)
    RawPosition.populate(fp)
    print()

    print("Populate ImportedSpikeSorting...")
    ImportedSpikeSorting.populate(fp)

    # print('HeadDir...')
    # HeadDir().populate()
    # print('Speed...')
    # Speed().populate()
    # print('LinPos...')
    # LinPos().populate()


def copy_nwb_link_raw_ephys(nwb_file_name, out_nwb_file_name):
    print(
        f"Creating a copy of NWB file {nwb_file_name} with link to raw ephys data: {out_nwb_file_name}"
    )
    print()

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
