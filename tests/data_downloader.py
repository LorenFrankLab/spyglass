from os import environ as os_environ
from pathlib import Path
from subprocess import DEVNULL, Popen
from sys import stderr, stdout

UCSF_BOX_USER = os_environ.get("UCSF_BOX_USER")
UCSF_BOX_TOKEN = os_environ.get("UCSF_BOX_TOKEN")
BASE_URL = "ftps://ftp.box.com/trodes_to_nwb_test_data/"
EXTRA_PATHS = [
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
        extra_paths=EXTRA_PATHS,
        target_dir=".",
        download_extras=True,
        verbose=True,
    ):
        if not all([UCSF_BOX_USER, UCSF_BOX_TOKEN]):
            raise ValueError(
                "Missing os.environ credentials: UCSF_BOX_USER, UCSF_BOX_TOKEN."
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
            "-P",
            target_dir,
        ]
        if not verbose:
            self.cmd.insert(self.cmd.index("--recursive") + 1, "--no-verbose")
            self.cmd_kwargs = dict(stdout=DEVNULL, stderr=DEVNULL)
        else:
            self.cmd_kwargs = dict(stdout=stdout, stderr=stderr)

        self.file_name = nwb_file_name
        self.base_dir = Path(target_dir).resolve()
        self.extra_paths = extra_paths
        self.verbose = verbose
        self.base_dir.mkdir(exist_ok=True)

        # Start downloads
        _ = self.download_nwb
        if download_extras:
            _ = self.download_extras

    def _single_wget(self, url):
        return Popen(self.cmd + [url], **self.cmd_kwargs)

    def _dirs_from_info(self, info):
        """Return source and destination paths for a given info dict.

        For convenience, all wget downloads are placed in the target_dir and
        then moved to the correct location.
        """
        src = Path(self.base_dir) / info["url"].split("/")[-1]
        dest = Path(self.base_dir) / info["relative_dir"] / info["target_name"]
        return src.absolute(), dest.absolute()

    @property
    def download_nwb(self):
        if (self.base_dir / "raw" / self.file_name).exists():
            return None
        return self._single_wget(BASE_URL + self.file_name)

    @property
    def download_extras(self):
        ret = []
        for path in self.extra_paths:
            dl_dir, dest = self._dirs_from_info(path)
            if dest.exists():
                continue  # Have the file
            if dl_dir.exists():
                dl_dir.rename(dest)  # Move the file
                continue
            ret.append(self._single_wget(path["url"]))
        return ret

    def check_download(self, download, info):
        if download is not None:
            download.wait()
            if download.returncode:
                return download
        return None

    def check_move(self, item):
        src, dest = self._dirs_from_info(item)
        if src.exists() and not dest.exists():
            src.rename(dest)
        elif src.exists():
            src.unlink()
        if not dest.exists():
            return "Move error: " + dest
        return None

    @property
    def extras_errors(self):
        ret = []
        for download, item in zip(self.download_extras, self.extra_paths):
            if d_status := self.check_download(download, item):
                ret.append(d_status)
                continue
            if m_status := self.check_move(item):
                ret.append(m_status)
        return ret
