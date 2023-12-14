import json
import os
import warnings
from pathlib import Path

import datajoint as dj
import yaml
from pymysql.err import OperationalError


class SpyglassConfig:
    """Gets Spyglass dirs from dj.config or environment variables.

    Uses SpyglassConfig.relative_dirs to (a) gather user
    settings from dj.config or os environment variables or defaults relative to
    base, in that order (b) set environment variables, and (c) make dirs that
    don't exist. NOTE: when passed a base_dir, it will ignore env vars to
    facilitate testing.
    """

    def __init__(self, base_dir: str = None, **kwargs):
        """
        Initializes a new instance of the class.

        Parameters
        ----------
        base_dir (str)
            The base directory.
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

        self.dj_defaults = {
            "database.host": kwargs.get("database_host", "lmf-db.cin.ucsf.edu"),
            "database.user": kwargs.get("database_user"),
            "database.port": kwargs.get("database_port", 3306),
            "database.use_tls": kwargs.get("database_use_tls", True),
            "filepath_checksum_size_limit": 1 * 1024**3,
            "enable_python_native_blobs": True,
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

        if not resolved_base or not Path(resolved_base).exists():
            raise ValueError(
                f"Could not find SPYGLASS_BASE_DIR: {resolved_base}"
                + "\n\tCheck dj.config['custom']['spyglass_dirs']['base']"
                + "\n\tand os.environ['SPYGLASS_BASE_DIR']"
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
                    or str(Path(resolved_base) / dir_str)
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

        self._config = dict(
            debug_mode=dj_custom.get("debug_mode", False),
            **self.config_defaults,
            **config_dirs,
            **kachery_zone_dict,
            **loaded_env,
        )

        self._set_dj_config_stores(config_dirs)

        return self._config

    def _load_env_vars(self):
        loaded_dict = {}
        for var, val in self.env_defaults.items():
            loaded_dict[var] = os.getenv(var, val)
        return loaded_dict

    @staticmethod
    def _set_env_with_dict(env_dict):
        # NOTE: Kept for backwards compatibility. Should be removed in future.
        for var, val in env_dict.items():
            os.environ[var] = str(val)

    @staticmethod
    def _mkdirs_from_dict_vals(dir_dict):
        for dir_str in dir_dict.values():
            Path(dir_str).mkdir(exist_ok=True)

    def _set_dj_config_stores(self, check_match=True, set_stores=True):
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
        if check_match:
            dj_stores = dj.config.get("stores", {})
            store_raw = dj_stores.get("raw", {}).get("location")
            store_analysis = dj_stores.get("analysis", {}).get("location")

            err_template = (
                "dj.config['stores'] does not match resolved dir."
                + "\n\tdj.config['stores']['{0}']['location']:\n\t\t{1}"
                + "\n\tSPYGLASS_{2}_DIR:\n\t\t{3}."
            )
            if store_raw and Path(store_raw) != Path(self.raw_dir):
                raise ValueError(
                    err_template.format("raw", store_raw, "RAW", self.raw_dir)
                )
            if store_analysis and Path(store_analysis) != Path(
                self.analysis_dir
            ):
                raise ValueError(
                    err_template.format(
                        "analysis",
                        store_analysis,
                        "ANALYSIS",
                        self.analysis_dir,
                    )
                )

        if set_stores:
            dj.config.update(self._dj_stores)

    def dir_to_var(self, dir: str, dir_type: str = "spyglass"):
        """Converts a dir string to an env variable name."""
        dir_string = self.relative_dirs.get(dir_type, {}).get(dir, "base")
        return f"{dir_type.upper()}_{dir_string.upper()}_DIR"

    def _generate_dj_config(
        self,
        base_dir: str = None,
        database_user: str = None,
        database_host: str = "lmf-db.cin.ucsf.edu",
        database_port: int = 3306,
        database_use_tls: bool = True,
        **kwargs,
    ):
        """Generate a datajoint configuration file.

        Parameters
        ----------
        base_dir : str, optional
            The base directory. If not provided, will use the env variable or
            existing config.
        database_user : str, optional
            The database user. If not provided, resulting config will not
            specify.
        database_host : str, optional
            Default lmf-db.cin.ucsf.edu. MySQL host name.
        dapabase_port : int, optional
            Default 3306. Port number for MySQL server.
        database_use_tls : bool, optional
            Default True. Use TLS encryption.
        **kwargs: dict, optional
            Any other valid datajoint configuration parameters.
            Note: python will raise error for params with `.` in name.
        """

        if base_dir:
            self.supplied_base_dir = base_dir
            self.load_config(force_reload=True)

        if database_user:
            kwargs.update({"database.user": database_user})

        kwargs.update(
            {
                "database.host": database_host,
                "database.port": database_port,
                "database.use_tls": database_use_tls,
            }
        )

        # `|` merges dictionaries
        return self.dj_defaults | self._dj_stores | self._dj_custom | kwargs

    def save_dj_config(
        self,
        save_method: str = "global",
        filename: str = None,
        base_dir=None,
        database_user=None,
        set_password=True,
        **kwargs,
    ):
        """Set the dj.config parameters, set password, and save config to file.

        Parameters
        ----------
        save_method : {'local', 'global', 'custom'}, optional
            The method to use to save the config. If either 'local' or 'global',
            datajoint builtins will be used to save.
        filename : str or Path, optional
            Default to datajoint global config. If save_method = 'custom', name
            of file to generate. Must end in either be either yaml or json.
        base_dir : str, optional
            The base directory. If not provided, will default to the env var
        database_user : str, optional
            The database user. If not provided, resulting config will not
            specify.
        set_password : bool, optional
            Default True. Set the database password.
        """
        if save_method == "local":
            filepath = Path(".") / dj.settings.LOCALCONFIG
        elif not filename or save_method == "global":
            save_method = "global"
            filepath = Path("~").expanduser() / dj.settings.GLOBALCONFIG

        dj.config.update(
            self._generate_dj_config(
                base_dir=base_dir, database_user=database_user, **kwargs
            )
        )

        if set_password:
            try:
                dj.set_password()
            except OperationalError as e:
                warnings.warn(f"Database connection issues. Wrong pass? {e}")
                # NOTE: Save anyway? Or raise error?

        user_warn = (
            f"Replace existing file? {filepath.resolve()}\n\t"
            + "\n\t".join([f"{k}: {v}" for k, v in config.items()])
            + "\n"
        )

        if filepath.exists() and dj.utils.user_choice(user_warn)[0] != "y":
            return dj.config

        if save_method == "global":
            dj.config.save_global(verbose=True)
            return

        if save_method == "local":
            dj.config.save_local(verbose=True)
            return

        with open(filename, "w") as outfile:
            if filename.endswith("json"):
                json.dump(dj.config, outfile, indent=2)
            else:
                yaml.dump(dj.config, outfile, default_flow_style=False)

    @property
    def _dj_stores(self) -> dict:
        self.load_config()
        return {
            "stores": {
                "raw": {
                    "protocol": "file",
                    "location": self.raw_dir,
                    "stage": self.raw_dir,
                },
                "analysis": {
                    "protocol": "file",
                    "location": self.analysis_dir,
                    "stage": self.analysis_dir,
                },
            }
        }

    @property
    def _dj_custom(self) -> dict:
        self.load_config()
        return {
            "custom": {
                "debug_mode": str(self.debug_mode).lower(),
                "spyglass_dirs": {
                    "base": self.base_dir,
                    "raw": self.raw_dir,
                    "analysis": self.analysis_dir,
                    "recording": self.recording_dir,
                    "sorting": self.sorting_dir,
                    "waveforms": self.waveforms_dir,
                    "temp": self.temp_dir,
                    "video": self.video_dir,
                },
                "kachery_dirs": {
                    "cloud": self.config.get(
                        self.dir_to_var("cloud", "kachery")
                    ),
                    "storage": self.config.get(
                        self.dir_to_var("storage", "kachery")
                    ),
                    "temp": self.config.get(self.dir_to_var("tmp", "kachery")),
                },
                "kachery_zone": "franklab.default",
            }
        }

    @property
    def config(self) -> dict:
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
    def waveforms_dir(self) -> str:
        return self.config.get(self.dir_to_var("waveforms"))

    @property
    def temp_dir(self) -> str:
        return self.config.get(self.dir_to_var("temp"))

    @property
    def waveform_dir(self) -> str:
        return self.config.get(self.dir_to_var("waveform"))

    @property
    def video_dir(self) -> str:
        return self.config.get(self.dir_to_var("video"))

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
video_dir = sg_config.video_dir
debug_mode = sg_config.debug_mode
