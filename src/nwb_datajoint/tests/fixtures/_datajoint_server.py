import os
import multiprocessing
import pytest
import time
import hither as hi
from urllib import request
from ._config import DATAJOINT_SERVER_PORT

def run_service_datajoint_server():
    # The following cleanup is needed because we terminate this compute resource process
    # See: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    from pytest_cov.embed import cleanup_on_sigterm
    cleanup_on_sigterm()

    os.environ['RUNNING_PYTEST'] = 'TRUE'

    with hi.ConsoleCapture(label='[datajoint-server]'):
        ss = hi.ShellScript(f"""
        #!/bin/bash
        set -ex

        docker kill datajoint-server-fixture > /dev/null 2>&1 || true
        docker rm datajoint-server-fixture > /dev/null 2>&1 || true
        exec docker run --name datajoint-server-fixture -e MYSQL_ROOT_PASSWORD=tutorial -p {DATAJOINT_SERVER_PORT}:3306 datajoint/mysql
        """, redirect_output_to_stdout=True)
        ss.start()
        ss.wait()

@pytest.fixture()
def datajoint_server():
    print('Starting datajoint server')

    ss_pull = hi.ShellScript("""
    #!/bin/bash
    set -ex

    exec docker pull datajoint/mysql
    """)
    ss_pull.start()
    ss_pull.wait()

    process = multiprocessing.Process(target=run_service_datajoint_server, kwargs=dict())
    process.start()
    
    try:
        _wait_for_datajoint_server_to_start()
    except:
        _kill_datajoint_server()
        raise

    yield process

    process.terminate()    
    _kill_datajoint_server()

def _kill_datajoint_server():
    print('Terminating datajoint server')

    ss2 = hi.ShellScript(f"""
    #!/bin/bash

    set -ex

    docker kill datajoint-server-fixture || true
    docker rm datajoint-server-fixture
    """)
    ss2.start()
    ss2.wait()

def _wait_for_datajoint_server_to_start():
    timer = time.time()
    while True:
        try:
            from nwb_datajoint.common import Session
            return
        except:
            pass
        elapsed = time.time() - timer
        if elapsed > 300:
            raise Exception('Timeout while waiting for datajoint server to start')
        time.sleep(5)