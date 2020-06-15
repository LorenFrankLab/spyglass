import os
import uuid
from datetime import datetime
import pynwb
from pynwb.file import Subject
import pytz
import numpy as np
from nwb_datajoint import insert_sessions

from ._temporarydirectory import TemporaryDirectory

def test_1(tmp_path):
    tmpdir = str(tmp_path)
    os.environ['NWB_DATAJOINT_BASE_DIR'] = tmpdir + '/nwb-data'
    os.environ['KACHERY_STORAGE_DIR'] = tmpdir + '/nwb-data/kachery-storage'
    os.mkdir(os.environ['NWB_DATAJOINT_BASE_DIR'])
    os.mkdir(os.environ['KACHERY_STORAGE_DIR'])

    nwb_content = pynwb.NWBFile(
        session_description='test-session-description',
        experimenter='test-experimenter-name',
        lab='test-lab',
        institution='test-institution',
        session_start_time=datetime.now(),
        timestamps_reference_time=datetime.fromtimestamp(0, pytz.utc),
        identifier=str(uuid.uuid1()),
        session_id='test-session-id',
        notes='test-notes',
        experiment_description='test-experiment-description',
        subject=Subject(
            description='test-subject-description',
            genotype='test-subject-genotype',
            sex='female',
            species='test-subject-species',
            subject_id='test-subject-id',
            weight='2 lbs'
        ),
    )
    nwb_content.add_epoch(
        start_time=float(0), # start time
        stop_time=float(100), # end time (what are the units?)
        tags='epoch1'
    )

    nwb_content.create_device(
        name='test-device'
    )
    nwb_content.create_electrode_group(
        name='test-electrode-group',
        location='test-electrode-group-location',
        device=nwb_content.devices['test-device'],
        description='test-electrode-group-description'
    )
    num_channels = 8
    for m in range(num_channels):
        location = [0, m]
        grp = nwb_content.electrode_groups['test-electrode-group']
        impedance = -1.0
        nwb_content.add_electrode(
            id=m,
            x=float(location[0]), y=float(location[1]), z=float(0),
            imp=impedance,
            location='test-electrode-group-location',
            filtering='none',
            group=grp
        )

    nwb_fname = os.environ['NWB_DATAJOINT_BASE_DIR'] + '/test.nwb'  
    with pynwb.NWBHDF5IO(path=nwb_fname, mode='w') as io:
        io.write(nwb_content)
        io.close()

    with pynwb.NWBHDF5IO(path=nwb_fname, mode='r') as io:
        nwbf = io.read()

    insert_sessions(['test.nwb'])