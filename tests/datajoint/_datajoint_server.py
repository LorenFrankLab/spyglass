import hither2 as hi
import kachery_client as kc
import multiprocessing
import os

import time
from ._config import DATAJOINT_SERVER_PORT


DOCKER_IMAGE_NAME = 'datajoint-server-pytest'


def run_service_datajoint_server():
    # The following cleanup is needed because we terminate this compute resource process
    # See: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    from pytest_cov.embed import cleanup_on_sigterm
    cleanup_on_sigterm()

    os.environ['RUNNING_PYTEST'] = 'TRUE'

    with hi.ConsoleCapture(label='[datajoint-server]'):
        ss = kc.ShellScript(f"""
        #!/bin/bash
        set -ex

        docker kill {DOCKER_IMAGE_NAME} > /dev/null 2>&1 || true
        docker rm {DOCKER_IMAGE_NAME} > /dev/null 2>&1 || true
        exec docker run --name {DOCKER_IMAGE_NAME} -e MYSQL_ROOT_PASSWORD=tutorial -p {DATAJOINT_SERVER_PORT}:3306 datajoint/mysql
        """, redirect_output_to_stdout=True)  # noqa: E501
        ss.start()
        ss.wait()


def run_datajoint_server():
    print('Starting datajoint server')

    ss_pull = kc.ShellScript("""
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
    except Exception:
        kill_datajoint_server()
        raise

    return process
    # yield process

    # process.terminate()
    # kill_datajoint_server()


def kill_datajoint_server():
    print('Terminating datajoint server')

    ss2 = kc.ShellScript(f"""
    #!/bin/bash

    set -ex

    docker kill {DOCKER_IMAGE_NAME} || true
    docker rm {DOCKER_IMAGE_NAME}
    """)
    ss2.start()
    ss2.wait()


def _wait_for_datajoint_server_to_start():
    timer = time.time()
    while True:
        try:
            from nwb_datajoint.common import Session  # noqa: F401
            return
        except Exception as e:
            print('DataJoint server not started yet. Connection error:', e)
            pass
        elapsed = time.time() - timer
        if elapsed > 300:
            raise Exception('Timeout while waiting for datajoint server to start')
        time.sleep(5)
