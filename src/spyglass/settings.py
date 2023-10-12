import os
from pathlib import Path

import datajoint as dj


class SpyglassConfig:
    """Gets Spyglass dirs from dj.config or environment variables.

    Uses SpyglassConfig.relative_dirs to (a) gather user
    settings from dj.config or os environment variables or defaults relative to
    base, in that order (b) set environment variables, and (c) make dirs that
    don't exist. NOTE: when passed a base_dir, it will ignore env vars to
    facilitate testing.
    """

    def __init__(self, base_dir=None):
        """
        Initializes a new instance of the class.

        Parameters
        ----------
            base_dir (str): The base directory.

        Returns
        -------
            None
        """
        self.supplied_base_dir = base_dir
        self._config = dict()
        self.config_defaults = dict(prepopulate=True)

        self.relative_dirs = {
            # {PREFIX}_{KEY}_DIR, default dir relative to base_dir
            "spyglass": {
                "raw": "raw",
                "analysis": "analysis",
                "recording": "recording",
                "sorting": "spikesorting",
                "waveforms": "waveforms",
                "temp": "tmp",
                "video": "video",
            },
            "kachery": {
                "cloud": "kachery_storage",
                "storage": "kachery_storage",
                "temp": "tmp",
            },
        }

        self.env_defaults = {
            "FIGURL_CHANNEL": "franklab2",
            "DJ_SUPPORT_FILEPATH_MANAGEMENT": "TRUE",
            "KACHERY_CLOUD_EPHEMERAL": "TRUE",
        }

    def load_config(self, force_reload=False):
        """
        Loads the configuration settings for the object.

        Order of precedence, where X is base, raw, analysis, etc.:
            1. SpyglassConfig(base_dir="string") for base dir only
            2. dj.config['custom']['{spyglass/kachery}_dirs']['X']
            3. os.environ['{SPYGLASS/KACHERY}_{X}_DIR']
            4. resolved_base_dir/X for non-base dirs

        Parameters
        ----------
        force_reload: bool
            Optional. Default False. Default skip load if already completed.

        Raises
        ------
        ValueError
            If base_dir is not set in either dj.config or os.environ.

        Returns
        -------
        dict
            list of relative_dirs and other settings (e.g., prepopulate).
        """
        if self._config and not force_reload:
            return self._config

        dj_custom = dj.config.get("custom", {})
        dj_spyglass = dj_custom.get("spyglass_dirs", {})
        dj_kachery = dj_custom.get("kachery_dirs", {})

        resolved_base = (
            self.supplied_base_dir
            or dj_spyglass.get("base")
            or os.environ.get("SPYGLASS_BASE_DIR")
        )

        if not resolved_base:
            raise ValueError(
                "SPYGLASS_BASE_DIR not defined in dj.config or os env vars"
            )

        config_dirs = {"SPYGLASS_BASE_DIR": resolved_base}
        for prefix, dirs in self.relative_dirs.items():
            for dir, dir_str in dirs.items():
                dir_env_fmt = self.dir_to_var(dir, prefix)

                env_loc = (  # Ignore env vars if base was passed to func
                    os.environ.get(dir_env_fmt)
                    if not self.supplied_base_dir
                    else None
                )

                dir_location = (
                    dj_spyglass.get(dir)
                    or dj_kachery.get(dir)
                    or env_loc
                    or resolved_base + "/" + dir_str
                ).replace('"', "")

                config_dirs.update({dir_env_fmt: dir_location})

        kachery_zone_dict = {
            "KACHERY_ZONE": (
                os.environ.get("KACHERY_ZONE")
                or dj.config.get("custom", {}).get("kachery_zone")
                or "franklab.default"
            )
        }

        loaded_env = self._load_env_vars()
        self._set_env_with_dict(
            {**config_dirs, **kachery_zone_dict, **loaded_env}
        )
        self._mkdirs_from_dict_vals(config_dirs)
        self._set_dj_config_stores(config_dirs)

        self._config = dict(
            debug_mode=dj_custom.get("debug_mode", False),
            **self.config_defaults,
            **config_dirs,
            **kachery_zone_dict,
            **loaded_env,
        )
        return self._config

    def _load_env_vars(self):
        loaded_dict = {}
        for var, val in self.env_defaults.items():
            loaded_dict[var] = os.getenv(var, val)
        return loaded_dict

    @staticmethod
    def _set_env_with_dict(env_dict):
        for var, val in env_dict.items():
            os.environ[var] = str(val)

    @staticmethod
    def _mkdirs_from_dict_vals(dir_dict):
        for dir_str in dir_dict.values():
            Path(dir_str).mkdir(exist_ok=True)

    @staticmethod
    def _set_dj_config_stores(dir_dict, check_match=True, set_stores=True):
        """
        Checks dj.config['stores'] match resolved dirs. Ensures stores set.

        Parameters
        ----------
        dir_dict: dict
            Dictionary of resolved dirs.
        check_match: bool
            Optional. Default True. Check that dj.config['stores'] match resolved dirs.
        set_stores: bool
            Optional. Default True. Set dj.config['stores'] to resolved dirs.
        """
        raw_dir = Path(dir_dict["SPYGLASS_RAW_DIR"])
        analysis_dir = Path(dir_dict["SPYGLASS_ANALYSIS_DIR"])

        if check_match:
            dj_stores = dj.config.get("stores", {})
            store_raw = dj_stores.get("raw", {}).get("location")
            store_analysis = dj_stores.get("analysis", {}).get("location")

            err_template = (
                "dj.config['stores'] does not match resolved dir."
                + "\n\tdj.config['stores']['{0}']['location']:\n\t\t{1}"
                + "\n\tSPYGLASS_{2}_DIR:\n\t\t{3}."
            )
            if store_raw and Path(store_raw) != raw_dir:
                raise ValueError(
                    err_template.format("raw", store_raw, "RAW", raw_dir)
                )
            if store_analysis and Path(store_analysis) != analysis_dir:
                raise ValueError(
                    err_template.format(
                        "analysis", store_analysis, "ANALYSIS", analysis_dir
                    )
                )

        if set_stores:
            dj.config["stores"] = {
                "raw": {
                    "protocol": "file",
                    "location": str(raw_dir),
                    "stage": str(raw_dir),
                },
                "analysis": {
                    "protocol": "file",
                    "location": str(analysis_dir),
                    "stage": str(analysis_dir),
                },
            }

    def dir_to_var(self, dir: str, dir_type: str = "spyglass"):
        """Converts a dir string to an env variable name."""
        dir_string = self.relative_dirs.get(dir_type, {}).get(dir, "base")
        return f"{dir_type.upper()}_{dir_string.upper()}_DIR"

    @property
    def config(self) -> dict:
        if not self._config:
            self.load_config()
        return self._config

    @property
    def base_dir(self) -> str:
        return self.config.get(self.dir_to_var("base"))

    @property
    def raw_dir(self) -> str:
        return self.config.get(self.dir_to_var("raw"))

    @property
    def analysis_dir(self) -> str:
        return self.config.get(self.dir_to_var("analysis"))

    @property
    def recording_dir(self) -> str:
        return self.config.get(self.dir_to_var("recording"))

    @property
    def sorting_dir(self) -> str:
        return self.config.get(self.dir_to_var("sorting"))

    @property
    def temp_dir(self) -> str:
        return self.config.get(self.dir_to_var("temp"))

    @property
    def waveform_dir(self) -> str:
        return self.config.get(self.dir_to_var("waveform"))

    @property
    def debug_mode(self) -> bool:
        return self.config.get("debug_mode", False)


sg_config = SpyglassConfig()
config = sg_config.config
base_dir = sg_config.base_dir
raw_dir = sg_config.raw_dir
recording_dir = sg_config.recording_dir
temp_dir = sg_config.temp_dir
analysis_dir = sg_config.analysis_dir
sorting_dir = sg_config.sorting_dir
waveform_dir = sg_config.waveform_dir
debug_mode = sg_config.debug_mode
