import multiprocessing
import os
import time
import traceback

import kachery_client as kc
from pymysql.err import OperationalError

from ._config import DATAJOINT_SERVER_PORT

DOCKER_IMAGE_NAME = "datajoint-server-pytest"


def run_service_datajoint_server():
    # The following cleanup is needed because we terminate this compute resource process
    # See: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    from pytest_cov.embed import cleanup_on_sigterm

    cleanup_on_sigterm()

    os.environ["RUNNING_PYTEST"] = "TRUE"

    ss = kc.ShellScript(
        f"""
    #!/bin/bash
    set -ex

    docker kill {DOCKER_IMAGE_NAME} > /dev/null 2>&1 || true
    docker rm {DOCKER_IMAGE_NAME} > /dev/null 2>&1 || true
    exec docker run --name {DOCKER_IMAGE_NAME} -e MYSQL_ROOT_PASSWORD=tutorial -p {DATAJOINT_SERVER_PORT}:3306 datajoint/mysql
    """,
        redirect_output_to_stdout=True,
    )  # noqa: E501
    ss.start()
    ss.wait()


def run_datajoint_server():
    print("Starting datajoint server")

    ss_pull = kc.ShellScript(
        """
    #!/bin/bash
    set -ex

    exec docker pull datajoint/mysql
    """
    )
    ss_pull.start()
    ss_pull.wait()

    process = multiprocessing.Process(
        target=run_service_datajoint_server, kwargs=dict()
    )
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
    print("Terminating datajoint server")

    ss2 = kc.ShellScript(
        f"""
    #!/bin/bash

    set -ex

    docker kill {DOCKER_IMAGE_NAME} || true
    docker rm {DOCKER_IMAGE_NAME}
    """
    )
    ss2.start()
    ss2.wait()


def _wait_for_datajoint_server_to_start():
    time.sleep(15)  # it takes a while to start the server
    timer = time.time()
    print("Waiting for DataJoint server to start. Time", timer)
    while True:
        try:
            from spyglass.common import Session  # noqa: F401

            return
        except OperationalError as e:  # e.g. Connection Error
            print("DataJoint server not yet started. Time", time.time())
            print(e)
        except Exception:
            print("Failed to import Session. Time", time.time())
            print(traceback.format_exc())
        current_time = time.time()
        elapsed = current_time - timer
        if elapsed > 300:
            raise Exception(
                "Timeout while waiting for datajoint server to start and "
                "import Session to succeed. Time",
                current_time,
            )
        time.sleep(5)
