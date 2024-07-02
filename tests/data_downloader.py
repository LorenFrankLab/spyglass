from functools import cached_property
from os import environ as os_environ
from pathlib import Path
from shutil import copy as shutil_copy
from subprocess import DEVNULL, Popen
from sys import stderr, stdout
from time import sleep as time_sleep
from typing import Dict, Union

from datajoint import logger as dj_logger

UCSF_BOX_USER = os_environ.get("UCSF_BOX_USER")
UCSF_BOX_TOKEN = os_environ.get("UCSF_BOX_TOKEN")
BASE_URL = "ftps://ftp.box.com/trodes_to_nwb_test_data/"

NON_DLC = 3  # First N items below are not for DeepLabCut
FILE_PATHS = [
    {
        "relative_dir": "raw",
        "target_name": "minirec20230622.nwb",
        "url": BASE_URL + "minirec20230622.nwb",
    },
    {
        "relative_dir": "video",
        "target_name": "20230622_minirec_01_s1.1.h264",
        "url": BASE_URL + "20230622_sample_01_a1/20230622_sample_01_a1.1.h264",
    },
    {
        "relative_dir": "video",
        "target_name": "20230622_minirec_02_s2.1.h264",
        "url": BASE_URL + "20230622_sample_02_a1/20230622_sample_02_a1.1.h264",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "CollectedData_sc_eb.csv",
        "url": BASE_URL + "minirec_dlc_items/CollectedData_sc_eb.csv",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "CollectedData_sc_eb.h5",
        "url": BASE_URL + "minirec_dlc_items/CollectedData_sc_eb.h5",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "img000.png",
        "url": BASE_URL + "minirec_dlc_items/img000.png",
    },
    {
        "relative_dir": "deeplabcut",
        "target_name": "img001.png",
        "url": BASE_URL + "minirec_dlc_items/img001.png",
    },
]


class DataDownloader:
    def __init__(
        self,
        nwb_file_name,
        file_paths=FILE_PATHS,
        base_dir=".",
        download_dlc=True,
        verbose=True,
    ):
        if not all([UCSF_BOX_USER, UCSF_BOX_TOKEN]):
            raise ValueError(
                "Missing os.environ credentials: UCSF_BOX_USER, UCSF_BOX_TOKEN."
            )
        if nwb_file_name != file_paths[0]["target_name"]:
            raise ValueError(
                f"Please adjust data_downloader.py to match: {nwb_file_name}"
            )

        self.cmd = [
            "wget",
            "--recursive",
            "--no-host-directories",
            "--no-directories",
            "--user",
            UCSF_BOX_USER,
            "--password",
            UCSF_BOX_TOKEN,
            "-P",  # Then need relative path, then url
        ]

        self.verbose = verbose
        if not verbose:
            self.cmd.insert(self.cmd.index("--recursive") + 1, "--no-verbose")
            self.cmd_kwargs = dict(stdout=DEVNULL, stderr=DEVNULL)
        else:
            self.cmd_kwargs = dict(stdout=stdout, stderr=stderr)

        self.base_dir = Path(base_dir).resolve()
        self.download_dlc = download_dlc
        self.file_paths = file_paths if download_dlc else file_paths[:NON_DLC]
        self.base_dir.mkdir(exist_ok=True)

        # Start downloads
        _ = self.file_downloads

    def rename_files(self):
        """Redundant, but allows rerun later in startup process of conftest."""
        for path in self.file_paths:
            target, url = path["target_name"], path["url"]
            target_dir = self.base_dir / path["relative_dir"]
            orig = target_dir / url.split("/")[-1]
            dest = target_dir / target

            if orig.exists():
                orig.rename(dest)

    @cached_property  # Only make list of processes once
    def file_downloads(self) -> Dict[str, Union[Popen, None]]:
        """{File: POpen/None} for each file. If exists/finished, None."""
        ret = dict()
        self.rename_files()
        for path in self.file_paths:
            target, url = path["target_name"], path["url"]
            target_dir = self.base_dir / path["relative_dir"]
            target_dir.mkdir(exist_ok=True, parents=True)
            dest = target_dir / target
            cmd = (
                ["echo", f"Already have {target}"]
                if dest.exists()
                else self.cmd + [target_dir, url]
            )
            ret[target] = Popen(cmd, **self.cmd_kwargs)
        return ret

    def wait_for(self, target: str):
        """Wait for target to finish downloading."""
        status = self.file_downloads.get(target).poll()
        limit = 10
        while status is None and limit > 0:
            time_sleep(5)  # Some
            limit -= 1
            status = self.file_downloads.get(target).poll()
        if status != 0:
            raise ValueError(f"Error downloading: {target}")
        if limit < 1:
            raise TimeoutError(f"Timeout downloading: {target}")

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
