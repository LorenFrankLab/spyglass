import atexit
import hashlib
import os
import re
import socket
import subprocess
import time
from pathlib import Path

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
    vol_dir : str
        Parent directory for the container's MySQL data dir, bind-mounted as
        `<vol_dir>/<container_name>` -> /var/lib/mysql. Default None, letting
        Docker manage storage on its own root disk.
    """

    def __init__(
        self,
        image_name: str = "datajoint/mysql",
        mysql_version: str = "8.0",
        null_server: bool = False,
        restart: bool = True,
        shutdown: bool = True,
        verbose: bool = False,
        container_name: str = None,
        port: int = None,
        vol_dir: str = None,
    ) -> None:
        self.image_name = image_name
        self.mysql_version = mysql_version
        self.client = None if null_server else docker.from_env()
        self.null_server = null_server
        self.password = "tutorial"
        self.user = "root"
        self.host = "localhost"
        self.branch_name = None
        self._ran_container = None
        self.logger = logger
        self.logger.setLevel("INFO" if verbose else "ERROR")

        if container_name is None:
            container_name = self._resolve_container_name()
        if port is None:
            port = self._resolve_port(container_name)

        self.container_name = container_name
        self.port = port
        self.vol_dir = self._resolve_vol_dir(
            vol_dir or os.environ.get("SPYGLASS_TEST_DOCKER_VOL_DIR")
        )

        if not self.null_server:
            if shutdown:
                atexit.register(self.stop)  # stop container on python exit
            if restart:
                self.stop()  # stop container if it exists
            self.start()

    def _get_existing_container_port(self, container_name: str) -> int:
        """Get port of existing container with given name, if it exists.

        Parameters
        ----------
        container_name : str
            Name of container to check

        Returns
        -------
        int or None
            Port number if container exists, None otherwise
        """
        try:
            container = self.client.containers.get(container_name)
            # Container exists, get its port mapping
            # Format: {'3306/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '47811'}]}
            port_bindings = container.attrs["NetworkSettings"]["Ports"]
            if "3306/tcp" in port_bindings and port_bindings["3306/tcp"]:
                host_port = port_bindings["3306/tcp"][0]["HostPort"]
                self.logger.info(
                    f"Found container {container_name} on port {host_port}"
                )
                return int(host_port)
        except docker.errors.NotFound:  # Container doesn't exist, that's fine
            pass
        except Exception as e:
            self.logger.warning(
                f"Error checking existing container {container_name}: {e}"
            )
        return None

    def _resolve_container_name(self) -> str:
        """Generate a container name from the current git branch.

        Resolved independently of `port` so that an explicit `--container-
        name` is never silently discarded just because `--container-port`
        was omitted (and vice versa).
        """
        default_name = "spyglass-pytest"

        if self.null_server:
            return default_name

        try:
            branch_name = (
                subprocess.check_output(["git", "branch", "--show-current"])
                .decode("utf-8")
                .strip()
            )
        except Exception as e:
            logger.error(f"Failed to get git branch name: {e}")
            return default_name

        self.branch_name = branch_name
        return f"spyglass-pytest-{branch_name}"

    def _resolve_port(self, container_name: str) -> int:
        """Pick a port for `container_name`.

        Reuses the port of an existing container with that name if one is
        running; otherwise deterministically derives one from the name.
        """
        # default is 3308, as a holdover from mysql version testing
        default_port = 3300 + int(self.mysql_version[0])

        if self.null_server:
            return default_port

        # Check if container with this name already exists and get its port
        existing_port = self._get_existing_container_port(container_name)
        if existing_port is not None:
            return existing_port

        # Otherwise, find an available port
        port = self.string_to_port(container_name)
        while self.port_in_use(port):
            port += 1

        return port

    def _resolve_vol_dir(self, vol_dir: str = None) -> Path:
        """Resolve the host directory to bind-mount as MySQL's data dir.

        Keeps container data off the root disk, which can be small relative to
        the space a populated test database needs.

        Parameters
        ----------
        vol_dir : str, optional
            Parent directory for container volumes. If not given, returns None
            and Docker manages storage itself.

        Returns
        -------
        Path or None
            `<vol_dir>/<container_name>`, else None.
        """
        if not vol_dir or self.null_server:
            return None

        # container_name may derive from a branch name (e.g. "feature/foo");
        # strip path separators and other non-filename characters so it can't
        # create nested directories or escape vol_dir via ".." segments.
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", self.container_name)
        safe_name = safe_name.strip(".") or "_"  # reject bare "." / ".."
        path = Path(vol_dir).expanduser().absolute() / safe_name
        self.logger.info(f"{self.msg}data dir: {path}")

        return path

    @staticmethod
    def string_to_port(
        name: str, min_port: int = 10240, max_port: int = 60000
    ) -> int:
        """Deterministically convert a string to a valid TCP port number."""
        h = hashlib.sha256(name.encode()).hexdigest()
        val = int(h, 16)
        port_range = max_port - min_port + 1
        return min_port + (val % port_range)

    @staticmethod
    def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is currently in use on the given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False  # port is available
            except OSError:
                return True  # port is in use

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
            self._warn_if_vol_dir_ignored()
            self.logger.info(
                self.msg + "starting: " + self.container_status + "."
            )

        elif self.container_status == "exited":
            self._warn_if_vol_dir_ignored()
            self.logger.info(self.msg + "restarting.")
            self.container.restart()

        else:
            # Optionally place the MySQL data dir on a larger disk. Pass
            # ``--container-vol-dir`` (or set SPYGLASS_TEST_DOCKER_VOL_DIR) to a
            # host directory to bind-mount per-container data there instead of
            # Docker's default storage; unset keeps the default anonymous
            # volume. Avoids filling the root disk with test-container data.
            # self.vol_dir (resolved in __init__ via _resolve_vol_dir) already
            # includes the per-container subdirectory.
            volumes = None
            if self.vol_dir:
                self.vol_dir.mkdir(parents=True, exist_ok=True)
                volumes = {
                    str(self.vol_dir): {
                        "bind": "/var/lib/mysql",
                        "mode": "rw",
                    }
                }
            self._ran_container = self.client.containers.run(
                image=f"{self.image_name}:{self.mysql_version}",
                name=self.container_name,
                ports={3306: self.port},
                environment=[
                    f"MYSQL_ROOT_PASSWORD={self.password}",
                    "MYSQL_DEFAULT_STORAGE_ENGINE=InnoDB",
                ],
                volumes=volumes,
                detach=True,
                tty=True,
            )
            self.logger.info(self.msg + "starting new.")

        return self.container.name

    def _warn_if_vol_dir_ignored(self) -> None:
        """Warn that `vol_dir` cannot apply to a container we didn't create."""
        if self.vol_dir is not None:
            self.logger.warning(
                self.msg
                + "reusing an existing container; --container-vol-dir has "
                + "no effect unless the container is removed and recreated "
                + "(omit --no-teardown, or pick a fresh --container-name)."
            )

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
        if self.container.health == "healthy":
            return

        print("")
        self.logger.info(f"Container {self.container_name} starting...")
        for _ in range(timeout // wait):
            if self.container.health == "healthy":
                break
            print(".", end="")
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
        """Stop and remove container, clearing its data dir if removed.

        `vol_dir` bind-mounts outlive the container itself: removing the
        container without also clearing `vol_dir` leaves a later run that
        reuses this container name pointed at a stale MySQL data dir, even
        though its on-disk artifacts (NWB files, DLC projects, etc.) were
        already deleted by that earlier run's teardown. Clearing `vol_dir`
        here whenever the container is removed keeps the two in sync.

        An "exited" container is already stopped, but still gets removed
        when `remove` is True -- otherwise `start()`'s restart branch would
        bring it back on the old mount/port after its data dir was cleared.
        """
        if self.null_server:
            return None

        if self.container_status:  # present, whether running or exited
            if self.container_status != "exited":
                self.container.stop()  # Logger I/O closes during teardown
                logline = f"Container {self.container_name} stopped"
            else:
                logline = f"Container {self.container_name} (exited)"

            if remove:
                self.container.remove()
                logline += " and removed"

            print(f"{logline}.")

        if remove and self.vol_dir is not None:
            self._clear_vol_dir()

    def _clear_vol_dir(self) -> None:
        """Remove `vol_dir` so a later run starts from scratch."""
        if not self.vol_dir.exists():
            return
        try:
            self.client.containers.run(
                image="alpine",
                command=["sh", "-c", "rm -rf /data/..?* /data/.[!.]* /data/*"],
                volumes={str(self.vol_dir): {"bind": "/data", "mode": "rw"}},
                remove=True,
                detach=False,
            )
            self.vol_dir.rmdir()
        except Exception as e:
            self.logger.warning(f"{self.msg}failed to clear vol_dir: {e}")
