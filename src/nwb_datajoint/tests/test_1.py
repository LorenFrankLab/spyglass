import os
import shutil
import uuid
from datetime import datetime
import pynwb
from pynwb.file import Subject
import pytz
import numpy as np
import kachery as ka

from ._temporarydirectory import TemporaryDirectory

def test_1(tmp_path, datajoint_server):
    from nwb_datajoint.common import Session, DataAcquisitionDevice, CameraDevice, Probe
    from nwb_datajoint.data_import import insert_sessions
    tmpdir = str(tmp_path)
    os.environ['NWB_DATAJOINT_BASE_DIR'] = tmpdir + '/nwb-data'
    os.environ['KACHERY_STORAGE_DIR'] = tmpdir + '/nwb-data/kachery-storage'
    os.mkdir(os.environ['NWB_DATAJOINT_BASE_DIR'])
    os.mkdir(os.environ['KACHERY_STORAGE_DIR'])

    os.mkdir(os.environ['NWB_DATAJOINT_BASE_DIR'] + '/raw')  # TODO cleanup
    nwb_fname = os.environ['NWB_DATAJOINT_BASE_DIR'] + '/raw/test.nwb'  

    with ka.config(fr='default_readonly'):
        ka.load_file('sha1://8ed68285c327b3766402ee75730d87994ac87e87/beans20190718_no_eseries_no_behavior.nwb', dest=nwb_fname)

    with pynwb.NWBHDF5IO(path=nwb_fname, mode='r', load_namespaces=True) as io:
        nwbf = io.read()

    insert_sessions(['test.nwb'])

    x = (Session() & {'nwb_file_name': 'test.nwb'}).fetch1()
    assert x['nwb_file_name'] == 'test.nwb'
    assert x['subject_id'] == 'Beans'
    assert x['institution_name'] == 'University of California, San Francisco'
    assert x['lab_name'] == 'Loren Frank'
    assert x['session_id'] == 'beans_01'
    assert x['session_description'] == 'Reinforcement leaarning'
    assert x['session_start_time'] == datetime(2019, 7, 18, 15, 29, 47)
    assert x['timestamps_reference_time'] == datetime(1970, 1, 1, 0, 0)
    assert x['experiment_description'] == 'Reinforcement learning'

    x = DataAcquisitionDevice().fetch()
    # TODO No data acquisition devices?
    assert len(x) == 0
    
    x = CameraDevice().fetch()
    # TODO No camera devices?
    assert len(x) == 0

    x = Probe().fetch()
    assert len(x) == 1
    assert x[0]['probe_type'] == '128c-4s8mm6cm-20um-40um-sl' 
    assert x[0]['probe_description'] == '128 channel polyimide probe'
    assert x[0]['num_shanks'] == 4
    assert x[0]['contact_side_numbering'] == 'True'

# This is how I created the file: beans20190718_no_eseries_no_behavior.nwb
def _create_beans20190718_no_eseries_no_behavior():
    # Use: pip install git+https://github.com/flatironinstitute/h5_to_json
    import os
    import shutil
    import h5_to_json as h5j

    basepath = '/workspaces/nwb_datajoint/devel/data/nwb_builder_test_data/'
    nwb_path = basepath + '/beans20190718.nwb'
    x = h5j.h5_to_dict(
        nwb_path,
        exclude_groups=['/acquisition/e-series', '/processing/behavior'],
        include_datasets=True
    )
    h5j.dict_to_h5(x, basepath + '/beans20190718_no_eseries_no_behavior.nwb')

def _old_method_for_creating_test_file():
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