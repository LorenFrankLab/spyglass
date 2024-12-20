from functools import cached_property
from pathlib import Path
from shutil import copy as shutil_copy
from subprocess import DEVNULL, Popen
from sys import stderr, stdout
from time import sleep as time_sleep
from typing import Dict, Union

from datajoint import logger as dj_logger

BASE_URL = "https://ucsf.box.com/shared/static/"

NON_DLC = 3  # First N items below are not for DeepLabCut
FILE_PATHS = [
    {
        "relative_dir": "raw",
        "target_name": "minirec20230622.nwb",
        "url": BASE_URL + "k3sgql6z475oia848q1rgms4zdh4rkjn.nwb",
    },
    {
        "relative_dir": "video",
        "target_name": "20230622_minirec_01_s1.1.h264",
        "url": BASE_URL + "ykep8ek4ogad20wz4p0vuyuqfo60cv3w.h264",
    },
    {
        "relative_dir": "video",
        "target_name": "20230622_minirec_02_s2.1.h264",
        "url": BASE_URL + "d2jjk0y565ru75xqojio3hymmehzr5he.h264",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "CollectedData_sc_eb.csv",
        "url": BASE_URL + "3nzqdfty51vrga7470rn2vayrtoor3ot.csv",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "CollectedData_sc_eb.h5",
        "url": BASE_URL + "sx30rqljppeisi4jdyu53y51na0q9rff.h5",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "img000.png",
        "url": BASE_URL + "wrvgncfbpjuzfhopkfaizzs069tb1ruu.png",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "img001.png",
        "url": BASE_URL + "czbkxeinemat7jj7j0877pcosfqo9psh.png",
    },
]


class DataDownloader:
    def __init__(
        self,
        file_paths=FILE_PATHS,
        base_dir=".",
        download_dlc=True,
        verbose=True,
    ):
        if verbose:
            self.cmd_kwargs = dict(stdout=stdout, stderr=stderr)
        else:
            self.cmd_kwargs = dict(stdout=DEVNULL, stderr=DEVNULL)

        self.verbose = verbose
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.download_dlc = download_dlc
        self.file_paths = file_paths if download_dlc else file_paths[:NON_DLC]
        self.base_dir.mkdir(exist_ok=True)

        # Start downloads
        _ = self.file_downloads

    @cached_property  # Only make list of processes once
    def file_downloads(self) -> Dict[str, Union[Popen, None]]:
        """{File: POpen/None} for each file. If exists/finished, None."""
        ret = dict()
        for path in self.file_paths:
            target_dir = self.base_dir / path["relative_dir"]
            target_dir.mkdir(exist_ok=True, parents=True)

            target = path["target_name"]
            dest = target_dir / target

            if dest.exists():
                cmd = ["echo", f"Already have {target}"]
                ret[target] = "Done"
            else:
                cmd = ["curl", "-L", "--output", str(dest), f"{path['url']}"]
                ret[target] = Popen(cmd, **self.cmd_kwargs)

        return ret

    def wait_for(self, target: str, timeout: int = 50, interval=5):
        """Wait for target to finish downloading, and clean up if needed.

        Parameters
        ----------
        target : str
            Name of file to wait for.
        timeout : int, optional
            Maximum time to wait for download to finish.
        interval : int, optional
            Time between checks for download completion.

        Raises
        ------
        ValueError
            If download failed or target not being downloaded.
        TimeoutError
            If download took too long.
        """
        process = self.file_downloads.get(target)
        if not process:
            raise ValueError(f"No active download process for target: {target}")
        if process == "Done":
            return

        elapsed_time = 0
        try:  # Refactored to clean up process streams
            while (status := process.poll()) is None:
                if elapsed_time >= timeout:
                    process.terminate()  # Terminate on timeout
                    process.wait()
                    raise TimeoutError(f"Timeout waiting for {target}.")
                time_sleep(interval)
                elapsed_time += interval
            if status != 0:
                raise ValueError(f"Error occurred during download of {target}.")
        finally:  # Ensure process streams are closed and cleaned up
            process.stdout and process.stdout.close()
            process.stderr and process.stderr.close()
            self.file_downloads[target] = "Done"  # Remove target from dict

    def move_dlc_items(self, dest_dir: Path):
        """Move completed DLC files to dest_dir."""
        if not self.download_dlc:
            return
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        for path in self.file_paths[NON_DLC:]:
            target = path["target_name"]
            self.wait_for(target)  # Could be faster if moved finished first

            src_path = self.base_dir / path["relative_dir"] / target
            dest_path = dest_dir / src_path.name
            if not dest_path.exists():
                shutil_copy(str(src_path), str(dest_path))
                dj_logger.info(f"Moved: {src_path} -> {dest_path}")
