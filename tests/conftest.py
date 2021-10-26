# directory-specific hook implementations
import datajoint as dj
import pathlib
import os
import shutil
import sys
import tempfile

from .datajoint._config import DATAJOINT_SERVER_PORT
from .datajoint._datajoint_server import run_datajoint_server, kill_datajoint_server


thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thisdir)


global __PROCESS
__PROCESS = None


def pytest_addoption(parser):
    parser.addoption('--current', action='store_true', dest="current",
                     default=False, help="run only tests marked as current")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "current: for convenience -- mark one test as current"
    )

    markexpr_list = []

    if config.option.current:
        markexpr_list.append('current')

    if len(markexpr_list) > 0:
        markexpr = ' and '.join(markexpr_list)
        setattr(config.option, 'markexpr', markexpr)

    _set_env()

    # note that in this configuration, every test will use the same datajoint server
    # this may create conflicts and dependencies between tests
    # it may be better but significantly slower to start a new server for every test
    # but the server needs to be started before tests are collected because datajoint runs when the source
    # files are loaded, not when the tests are run. one solution might be to restart the server after every test
    global __PROCESS
    __PROCESS = run_datajoint_server()


def pytest_unconfigure(config):
    if __PROCESS:
        print('Terminating datajoint compute resource process')
        __PROCESS.terminate()
        # TODO handle ResourceWarning: subprocess X is still running
        # __PROCESS.join()

    kill_datajoint_server()
    shutil.rmtree(os.environ['NWB_DATAJOINT_BASE_DIR'])


def _set_env():
    """Set environment variables."""
    print('Setting datajoint and kachery environment variables.')

    nwb_datajoint_base_dir = pathlib.Path(tempfile.mkdtemp())

    spike_sorting_storage_dir = nwb_datajoint_base_dir / 'spikesorting'
    kachery_storage_dir = nwb_datajoint_base_dir / 'kachery-storage'
    tmp_dir = nwb_datajoint_base_dir / 'tmp'

    os.environ['NWB_DATAJOINT_BASE_DIR'] = str(nwb_datajoint_base_dir)
    print('NWB_DATAJOINT_BASE_DIR set to', nwb_datajoint_base_dir)
    os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'
    os.environ['SPIKE_SORTING_STORAGE_DIR'] = str(spike_sorting_storage_dir)
    # export KACHERY_DAEMON_HOST=...
    # export KACHERY_DAEMON_PORT=...
    os.environ['KACHERY_TEMP_DIR'] = str(tmp_dir)
    os.environ['NWB_DATAJOINT_TEMP_DIR'] = str(tmp_dir)
    os.environ['KACHERY_STORAGE_DIR'] = str(kachery_storage_dir)
    # os.environ['FIGURL_CHANNEL'] = 'franklab2'

    os.mkdir(spike_sorting_storage_dir)
    os.mkdir(tmp_dir)
    os.mkdir(kachery_storage_dir)

    raw_dir = nwb_datajoint_base_dir / 'raw'
    analysis_dir = nwb_datajoint_base_dir / 'analysis'

    os.mkdir(raw_dir)
    os.mkdir(analysis_dir)

    dj.config['database.host'] = 'localhost'
    dj.config['database.port'] = DATAJOINT_SERVER_PORT
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'tutorial'

    dj.config['stores'] = {
        'raw': {
            'protocol': 'file',
            'location': str(raw_dir),
            'stage': str(raw_dir)
        },
        'analysis': {
            'protocol': 'file',
            'location': str(analysis_dir),
            'stage': str(analysis_dir)
        }
    }
