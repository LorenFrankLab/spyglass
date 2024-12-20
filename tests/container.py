import atexit
import time

import datajoint as dj
import docker
from datajoint import logger


class DockerMySQLManager:
    """Manage Docker container for MySQL server

    Parameters
    ----------
    image_name : str
        Docker image name. Default 'datajoint/mysql'.
    mysql_version : str
        MySQL version. Default '8.0'.
    container_name : str
        Docker container name. Default 'spyglass-pytest'.
    port : str
        Port to map to DJ's default 3306. Default '330[mysql_version]'
        (i.e., 3308 if testing 8.0).
    null_server : bool
        If True, do not start container. Return on all methods. Default False.
        Useful for iterating on tests in existing container.
    restart : bool
        If True, stop and remove existing container on startup. Default True.
    shutdown : bool
        If True, stop and remove container on exit from python. Default True.
    verbose : bool
        If True, print container status on startup. Default False.
    """

    def __init__(
        self,
        image_name="datajoint/mysql",
        mysql_version="8.0",
        container_name="spyglass-pytest",
        port=None,
        null_server=False,
        restart=True,
        shutdown=True,
        verbose=False,
    ) -> None:
        self.image_name = image_name
        self.mysql_version = mysql_version
        self.container_name = container_name
        self.port = port or "330" + self.mysql_version[0]
        self.client = None if null_server else docker.from_env()
        self.null_server = null_server
        self.password = "tutorial"
        self.user = "root"
        self.host = "localhost"
        self._ran_container = None
        self.logger = logger
        self.logger.setLevel("INFO" if verbose else "ERROR")

        if not self.null_server:
            if shutdown:
                atexit.register(self.stop)  # stop container on python exit
            if restart:
                self.stop()  # stop container if it exists
            self.start()

    @property
    def container(self) -> docker.models.containers.Container:
        if self.null_server:
            return self.container_name
        return self.client.containers.get(self.container_name)

    @property
    def container_status(self) -> str:
        if self.null_server:
            return None
        try:
            self.container.reload()
            return self.container.status
        except docker.errors.NotFound:
            return None

    @property
    def container_health(self) -> str:
        if self.null_server:
            return None
        try:
            self.container.reload()
            return self.container.health
        except docker.errors.NotFound:
            return None

    @property
    def msg(self) -> str:
        return f"Container {self.container_name} "

    def start(self) -> str:
        if self.null_server:
            return None

        elif self.container_status in ["created", "running", "restarting"]:
            self.logger.info(
                self.msg + "starting: " + self.container_status + "."
            )

        elif self.container_status == "exited":
            self.logger.info(self.msg + "restarting.")
            self.container.restart()

        else:
            self._ran_container = self.client.containers.run(
                image=f"{self.image_name}:{self.mysql_version}",
                name=self.container_name,
                ports={3306: self.port},
                environment=[
                    f"MYSQL_ROOT_PASSWORD={self.password}",
                    "MYSQL_DEFAULT_STORAGE_ENGINE=InnoDB",
                ],
                detach=True,
                tty=True,
            )
            self.logger.info(self.msg + "starting new.")

        return self.container.name

    def wait(self, timeout=120, wait=3) -> None:
        """Wait for healthy container.

        Parameters
        ----------
        timeout : int
            Timeout in seconds. Default 120.
        wait : int
            Time to wait between checks in seconds. Default 5.
        """
        if self.null_server:
            return None
        if not self.container_status or self.container_status == "exited":
            self.start()

        print("")
        for i in range(timeout // wait):
            if self.container.health == "healthy":
                break
            self.logger.info(f"Container {self.container_name} starting... {i}")
            time.sleep(wait)
        self.logger.info(
            f"Container {self.container_name}, {self.container.health}."
        )

    @property
    def _add_sql(self) -> str:
        ESC = r"\_%"
        return (
            "CREATE USER IF NOT EXISTS 'basic'@'%' IDENTIFIED BY "
            + f"'{self.password}'; GRANT USAGE ON `%`.* TO 'basic'@'%';"
            + "GRANT SELECT ON `%`.* TO 'basic'@'%';"
            + f"GRANT ALL PRIVILEGES ON `common{ESC}`.* TO `basic`@`%`;"
            + f"GRANT ALL PRIVILEGES ON `spikesorting{ESC}`.* TO `basic`@`%`;"
            + f"GRANT ALL PRIVILEGES ON `lfp{ESC}`.* TO `basic`@`%`;"
            + f"GRANT ALL PRIVILEGES ON `position{ESC}`.* TO `basic`@`%`;"
            + f"GRANT ALL PRIVILEGES ON `ripple{ESC}`.* TO `basic`@`%`;"
            + f"GRANT ALL PRIVILEGES ON `linearization{ESC}`.* TO `basic`@`%`;"
        ).strip()

    def add_user(self) -> int:
        """Add 'basic' user to container."""
        if self.null_server:
            return None

        if self._container_running():
            result = self.container.exec_run(
                cmd=[
                    "mysql",
                    "-u",
                    self.user,
                    f"--password={self.password}",
                    "-e",
                    self._add_sql,
                ],
                stdout=False,
                stderr=False,
                tty=True,
            )
            if result.exit_code == 0:
                self.logger.info("Container added user.")
            else:
                logger.error("Failed to add user.")
            return result.exit_code
        else:
            logger.error(f"Container {self.container_name} does not exist.")
            return None

    @property
    def credentials(self):
        """Datajoint credentials for this container."""
        return {
            "database.host": "localhost",
            "database.password": self.password,
            "database.user": self.user,
            "database.port": int(self.port),
            "safemode": "false",
            "custom": {"test_mode": True, "debug_mode": False},
        }

    @property
    def connected(self) -> bool:
        self.wait()
        dj.config.update(self.credentials)
        return dj.conn().is_connected

    def stop(self, remove=True) -> None:
        """Stop and remove container."""
        if self.null_server:
            return None
        if not self.container_status or self.container_status == "exited":
            return

        container_name = self.container_name
        self.container.stop()  # Logger I/O operations close during teardown
        print(f"Container {container_name} stopped.")

        if remove:
            self.container.remove()
            print(f"Container {container_name} removed.")
