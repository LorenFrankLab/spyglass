import json
import os
import warnings
from pathlib import Path

import datajoint as dj
import yaml
from pymysql.err import OperationalError

from spyglass.utils.dj_helper_fn import str_to_bool
from spyglass.utils.logging import logger


class SpyglassConfig:
    """Gets Spyglass dirs from dj.config or environment variables.

    Uses SpyglassConfig.relative_dirs to (a) gather user
    settings from dj.config or os environment variables or defaults relative to
    base, in that order (b) set environment variables, and (c) make dirs that
    don't exist. NOTE: when passed a base_dir, it will ignore env vars to
    facilitate testing.
    """

    def __init__(self, base_dir: str = None, **kwargs) -> None:
        """
        Initializes a new instance of the class.

        Parameters
        ----------
        base_dir (str)
            The base directory.

        Attributes
        ----------
        supplied_base_dir (str)
            The base directory passed to the class.
        config_defaults (dict)
            Default settings for the config.
        relative_dirs (dict)
            Relative dirs for each prefix (spyglass, kachery, dlc). Relative
            to respective base_dir. Created on init.
        dj_defaults (dict)
            Default settings for datajoint.
        env_defaults (dict)
            Default settings for environment variables.
        _config (dict)
            Cached config settings.
        _debug_mode (bool)
            True if debug_mode is set. Supports skipping known bugs in test env.
        _test_mode (bool)
            True if test_mode is set. Required for pytests to run without
            prompts.
        """
        self.supplied_base_dir = base_dir
        self._config = dict()
        self.config_defaults = dict(prepopulate=True)
        self._debug_mode = kwargs.get("debug_mode", False)
        self._test_mode = kwargs.get("test_mode", False)
        self._dlc_base = None
        self.load_failed = False

        self.relative_dirs = {
            # {PREFIX}_{KEY}_DIR, default dir relative to base_dir
            # NOTE: Adding new dir requires edit to HHMI hub
            "spyglass": {
                "raw": "raw",
                "analysis": "analysis",
                "recording": "recording",
                "sorting": "spikesorting",
                "waveforms": "waveforms",
                "temp": "tmp",
                "video": "video",
                "export": "export",
            },
            "kachery": {
                "cloud": ".kachery-cloud",
                "storage": "kachery_storage",
                "temp": "tmp",
            },
            "dlc": {
                "project": "projects",
                "video": "video",
                "output": "output",
            },
            "moseq": {
                "project": "projects",
                "video": "video",
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
            "HD5_USE_FILE_LOCKING": "FALSE",
        }

    def load_config(
        self,
        base_dir=None,
        force_reload=False,
        on_startup: bool = False,
        **kwargs,
    ) -> None:
        """
        Loads the configuration settings for the object.

        Order of precedence, where X is base, raw, analysis, etc.:
        1. SpyglassConfig(base_dir="string") for base dir only
        2. dj.config['custom']['spyglass_dirs']['X']
        3. dj.config['custom']['kachery_dirs']['X']
        4. os.environ['{SPYGLASS/KACHERY}_{X}_DIR']
        5. resolved_base_dir/X for non-base dirs

        Parameters
        ----------
        base_dir: str
            Optional. Default None. The base directory. If not provided, will
            use the env variable or existing config.
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
        if not force_reload and self._config:
            return self._config

        dj_custom = dj.config.get("custom", {})
        dj_spyglass = dj_custom.get("spyglass_dirs", {})
        dj_kachery = dj_custom.get("kachery_dirs", {})
        dj_dlc = dj_custom.get("dlc_dirs", {})
        dj_moseq = dj_custom.get("moseq_dirs", {})

        self._debug_mode = dj_custom.get("debug_mode", False)
        self._test_mode = kwargs.get("test_mode") or dj_custom.get(
            "test_mode", False
        )
        self._test_mode = str_to_bool(self._test_mode)
        self._debug_mode = str_to_bool(self._debug_mode)

        resolved_base = (
            base_dir
            or self.supplied_base_dir
            or dj_spyglass.get("base")
            or os.environ.get("SPYGLASS_BASE_DIR")
        )

        if resolved_base and not Path(resolved_base).exists():
            resolved_base = Path(resolved_base).expanduser()
        if not resolved_base or not Path(resolved_base).exists():
            if not on_startup:  # Only warn if not on startup
                logger.error(
                    f"Could not find SPYGLASS_BASE_DIR: {resolved_base}"
                    + "\n\tCheck dj.config['custom']['spyglass_dirs']['base']"
                    + "\n\tand os.environ['SPYGLASS_BASE_DIR']"
                )
            self.load_failed = True
            return

        self._dlc_base = (
            dj_dlc.get("base")
            or os.environ.get("DLC_BASE_DIR")
            or os.environ.get("DLC_PROJECT_PATH", "").split("projects")[0]
            or str(Path(resolved_base) / "deeplabcut")
        )
        Path(self._dlc_base).mkdir(exist_ok=True)

        self._moseq_base = (
            dj_moseq.get("base")
            or os.environ.get("MOSEQ_BASE_DIR")
            or str(Path(resolved_base) / "moseq")
        )
        Path(self._moseq_base).mkdir(exist_ok=True)

        config_dirs = {"SPYGLASS_BASE_DIR": str(resolved_base)}
        source_config_lookup = {
            "dlc": dj_dlc,
            "moseq": dj_moseq,
            "kachery": dj_kachery,
        }
        base_lookup = {"dlc": self._dlc_base, "moseq": self._moseq_base}
        for prefix, dirs in self.relative_dirs.items():
            this_base = base_lookup.get(prefix, resolved_base)
            for dir, dir_str in dirs.items():
                dir_env_fmt = self.dir_to_var(dir=dir, dir_type=prefix)

                env_loc = (  # Ignore env vars if base was passed to func
                    os.environ.get(dir_env_fmt)
                    if not self.supplied_base_dir
                    else None
                )
                source_config = source_config_lookup.get(prefix, dj_spyglass)
                dir_location = (
                    source_config.get(dir)
                    or env_loc
                    or str(Path(this_base) / dir_str)
                ).replace('"', "")

                config_dirs.update({dir_env_fmt: str(dir_location)})

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
            debug_mode=self._debug_mode,
            test_mode=self._test_mode,
            **self.config_defaults,
            **config_dirs,
            **kachery_zone_dict,
            **loaded_env,
        )

        self._set_dj_config_stores()

        return self._config

    def _load_env_vars(self) -> dict:
        loaded_dict = {}
        for var, val in self.env_defaults.items():
            loaded_dict[var] = os.getenv(var, val)
        return loaded_dict

    def _set_env_with_dict(self, env_dict) -> None:
        # NOTE: Kept for backwards compatibility. Should be removed in future
        # for custom paths. Keep self.env_defaults.
        # SPYGLASS_BASE_DIR may be used for docker assembly of export
        for var, val in env_dict.items():
            os.environ[var] = str(val)

    def _mkdirs_from_dict_vals(self, dir_dict) -> None:
        if self._debug_mode:
            return
        for dir_str in dir_dict.values():
            Path(dir_str).mkdir(exist_ok=True)

    def _set_dj_config_stores(self, check_match=True, set_stores=True) -> None:
        """
        Checks dj.config['stores'] match resolved dirs. Ensures stores set.

        Parameters
        ----------
        dir_dict: dict
            Dictionary of resolved dirs.
        check_match: bool
            Optional. Default True. Check that dj.config['stores'] match
            resolved dirs.
        set_stores: bool
            Optional. Default True. Set dj.config['stores'] to resolved dirs.
        """

        mismatch_analysis = False
        mismatch_raw = False

        if check_match:
            dj_stores = dj.config.get("stores", {})
            store_r = dj_stores.get("raw", {}).get("location")
            store_a = dj_stores.get("analysis", {}).get("location")
            mismatch_raw = store_r and Path(store_r) != Path(self.raw_dir)
            mismatch_analysis = store_a and Path(store_a) != Path(
                self.analysis_dir
            )

        if set_stores:
            if mismatch_raw or mismatch_analysis:
                logger.warning(
                    "Setting config DJ stores to resolve mismatch.\n\t"
                    + f"raw     : {self.raw_dir}\n\t"
                    + f"analysis: {self.analysis_dir}"
                )
            dj.config.update(self._dj_stores)
            return

        if mismatch_raw or mismatch_analysis:
            raise ValueError(
                "dj.config['stores'] does not match resolved dirs."
                + f"\n\tdj.config['stores']: {dj_stores}"
                + f"\n\tResolved dirs: {self._dj_stores}"
            )

        return

    def dir_to_var(self, dir: str, dir_type: str = "spyglass") -> str:
        """Converts a dir string to an env variable name."""
        return f"{dir_type.upper()}_{dir.upper()}_DIR"

    def _generate_dj_config(
        self,
        base_dir: str = None,
        database_user: str = None,
        database_password: str = None,
        database_host: str = "lmf-db.cin.ucsf.edu",
        database_port: int = 3306,
        database_use_tls: bool = True,
        **kwargs,
    ) -> dict:
        """Generate a datajoint configuration file.

        Parameters
        ----------
        database_user : str, optional
            The database user. If not provided, resulting config will not
            specify.
        database_password : str, optional
            The database password. If not provided, resulting config will not
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

        if database_user:
            kwargs.update({"database.user": database_user})
        if database_password:
            kwargs.update({"database.password": database_password})

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
        output_filename: str = None,
        base_dir=None,
        set_password=True,
        **kwargs,
    ) -> None:
        """Set the dj.config parameters, set password, and save config to file.

        Parameters
        ----------
        save_method : {'local', 'global', 'custom'}, optional
            The method to use to save the config. If either 'local' or 'global',
            datajoint builtins will be used to save.
        output_filename : str or Path, optional
            Default to datajoint global config. If save_method = 'custom', name
            of file to generate. Must end in either be either yaml or json.
        base_dir : str, optional
            The base directory. If not provided, will default to the env var
        set_password : bool, optional
            Default True. Set the database password.
        kwargs: dict, optional
            Any other valid datajoint configuration parameters, including
            database_user, database_password, database_host, database_port, etc.
            Note: python will raise error for params with `.` in name, so use
            underscores instead.
        """
        if base_dir:
            self.load_config(
                base_dir=base_dir, force_reload=True, on_startup=False
            )

        if output_filename:
            save_method = "custom"
            path = Path(output_filename).expanduser()  # Expand ~
            filepath = path if path.is_absolute() else path.absolute()
            filepath.parent.mkdir(exist_ok=True, parents=True)
            filepath = (
                filepath.with_suffix(".json")  # ensure suffix, default json
                if filepath.suffix not in [".json", ".yaml"]
                else filepath
            )
        elif save_method == "local":
            filepath = Path(".") / dj.settings.LOCALCONFIG
        elif save_method == "global":
            filepath = Path("~").expanduser() / dj.settings.GLOBALCONFIG
        else:
            raise ValueError(
                "For save_dj_config, either (a) save_method must be 'local' "
                + " or 'global' or (b) must provide custom output_filename."
            )

        dj.config.update(self._generate_dj_config(**kwargs))

        if set_password:
            try:
                dj.set_password()
            except OperationalError as e:
                warnings.warn(f"Database connection issues. Wrong pass?\n\t{e}")

        user_warn = (
            f"Replace existing file? {filepath.resolve()}\n\t"
            + "\n\t".join([f"{k}: {v}" for k, v in config.items()])
            + "\n"
        )

        if (
            not self.test_mode
            and filepath.exists()
            and dj.utils.user_choice(user_warn)[0] != "y"
        ):
            return

        if save_method == "global":
            dj.config.save_global(verbose=True)
            return

        if save_method == "local":
            dj.config.save_local(verbose=True)
            return

        with open(filepath, "w") as outfile:
            if filepath.suffix == ".yaml":
                yaml.dump(dj.config._conf, outfile, default_flow_style=False)
            else:
                json.dump(dj.config._conf, outfile, indent=2)
            logger.info(f"Saved config to {filepath}")

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
                "test_mode": str(self._test_mode).lower(),
                "spyglass_dirs": {
                    "base": self.base_dir,
                    "raw": self.raw_dir,
                    "analysis": self.analysis_dir,
                    "recording": self.recording_dir,
                    "sorting": self.sorting_dir,
                    "waveforms": self.waveforms_dir,
                    "temp": self.temp_dir,
                    "video": self.video_dir,
                    "export": self.export_dir,
                },
                "kachery_dirs": {
                    "cloud": self.config.get(
                        self.dir_to_var("cloud", "kachery")
                    ),
                    "storage": self.config.get(
                        self.dir_to_var("storage", "kachery")
                    ),
                    "temp": self.config.get(self.dir_to_var("temp", "kachery")),
                },
                "dlc_dirs": {
                    "base": self._dlc_base,
                    "project": self.dlc_project_dir,
                    "video": self.dlc_video_dir,
                    "output": self.dlc_output_dir,
                },
                "moseq_dirs": {
                    "base": self._moseq_base,
                    "project": self.moseq_project_dir,
                    "video": self.moseq_video_dir,
                },
                "kachery_zone": "franklab.default",
            }
        }

    @property
    def config(self) -> dict:
        """Dictionary of config settings."""
        self.load_config()
        return self._config

    @property
    def base_dir(self) -> str:
        """Base directory as a string."""
        return self.config.get(self.dir_to_var("base"))

    @property
    def raw_dir(self) -> str:
        """Raw data directory as a string."""
        return self.config.get(self.dir_to_var("raw"))

    @property
    def analysis_dir(self) -> str:
        """Analysis directory as a string."""
        return self.config.get(self.dir_to_var("analysis"))

    @property
    def recording_dir(self) -> str:
        """Recording directory as a string."""
        return self.config.get(self.dir_to_var("recording"))

    @property
    def sorting_dir(self) -> str:
        """Sorting directory as a string."""
        return self.config.get(self.dir_to_var("sorting"))

    @property
    def waveforms_dir(self) -> str:
        """Waveforms directory as a string."""
        return self.config.get(self.dir_to_var("waveforms"))

    @property
    def temp_dir(self) -> str:
        """Temp directory as a string."""
        return self.config.get(self.dir_to_var("temp"))

    @property
    def video_dir(self) -> str:
        """Video directory as a string."""
        return self.config.get(self.dir_to_var("video"))

    @property
    def export_dir(self) -> str:
        """Export directory as a string."""
        return self.config.get(self.dir_to_var("export"))

    @property
    def debug_mode(self) -> bool:
        """Returns True if debug_mode is set.

        Supports skipping inserts for Dockerized development.
        """
        return self._debug_mode

    @property
    def test_mode(self) -> bool:
        """Returns True if test_mode is set.

        Required for pytests to run without prompts."""
        return self._test_mode

    @property
    def dlc_project_dir(self) -> str:
        """DLC project directory as a string."""
        return self.config.get(self.dir_to_var("project", "dlc"))

    @property
    def dlc_video_dir(self) -> str:
        """DLC video directory as a string."""
        return self.config.get(self.dir_to_var("video", "dlc"))

    @property
    def dlc_output_dir(self) -> str:
        """DLC output directory as a string."""
        return self.config.get(self.dir_to_var("output", "dlc"))

    @property
    def moseq_project_dir(self) -> str:
        """Moseq project directory as a string."""
        return self.config.get(self.dir_to_var("project", "moseq"))

    @property
    def moseq_video_dir(self) -> str:
        """Moseq video directory as a string."""
        return self.config.get(self.dir_to_var("video", "moseq"))


sg_config = SpyglassConfig()
sg_config.load_config(on_startup=True)
if sg_config.load_failed:  # Failed to load
    logger.warning("Failed to load SpyglassConfig. Please set up config file.")
    config = {}  # Let __intit__ fetch empty config for first time setup
    prepopulate = False
    test_mode = False
    debug_mode = False
    base_dir = None
    raw_dir = None
    recording_dir = None
    temp_dir = None
    analysis_dir = None
    sorting_dir = None
    waveforms_dir = None
    video_dir = None
    export_dir = None
    dlc_project_dir = None
    dlc_video_dir = None
    dlc_output_dir = None
    moseq_project_dir = None
    moseq_video_dir = None
else:
    config = sg_config.config
    base_dir = sg_config.base_dir
    raw_dir = sg_config.raw_dir
    recording_dir = sg_config.recording_dir
    temp_dir = sg_config.temp_dir
    analysis_dir = sg_config.analysis_dir
    sorting_dir = sg_config.sorting_dir
    waveforms_dir = sg_config.waveforms_dir
    video_dir = sg_config.video_dir
    export_dir = sg_config.export_dir
    debug_mode = sg_config.debug_mode
    test_mode = sg_config.test_mode
    prepopulate = config.get("prepopulate", False)
    dlc_project_dir = sg_config.dlc_project_dir
    dlc_video_dir = sg_config.dlc_video_dir
    dlc_output_dir = sg_config.dlc_output_dir
    moseq_project_dir = sg_config.moseq_project_dir
    moseq_video_dir = sg_config.moseq_video_dir
