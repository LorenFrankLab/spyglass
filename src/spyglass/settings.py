import os
from pathlib import Path

import datajoint as dj

config_loaded = False
config_defaults = dict(prepopulate=True)
config = dict()
env_defaults = dict(
    FIGURL_CHANNEL="franklab2",
    DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE",
    KACHERY_CLOUD_EPHEMERAL="TRUE",
)
relative_dirs = dict(
    # {PREFIX}_{KEY}_DIR, default dir relative to base_dir
    spyglass=dict(
        raw="raw",
        analysis="analysis",
        recording="recording",
        sorting="spikesorting",  # "SPYGLASS_SORTING_DIR"
        waveforms="waveforms",
        temp="tmp",
        video="video",
    ),
    kachery=dict(
        cloud="kachery_storage",
        storage="kachery_storage",
        temp="tmp",
    ),
)


def load_config(base_dir: Path = None, force_reload: bool = False) -> dict:
    """Gets Spyglass dirs from dj.config or environment variables.

    Uses a relative_dirs dict defined in settings.py to (a) gather user
    settings from dj.config or os environment variables or defaults relative to
    base, in that order (b) set environment variables, and (c) make dirs that
    don't exist. NOTE: when passed a base_dir, it will ignore env vars to
    facilitate testing.

    Parameters
    ----------
    base_dir: Path
        Optional. Defaults to dj.config['spyglass_dirs']['base'] or
        or env SPYGLASS_BASE_DIR.
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
    global config, config_loaded
    if config_loaded and not force_reload:
        return config

    dj_custom = dj.config.get("custom", {})
    dj_spyglass = dj_custom.get("spyglass_dirs", {})
    dj_kachery = dj_custom.get("kachery_dirs", {})

    resolved_base = (
        base_dir
        or dj_spyglass.get("base")
        or os.environ.get("SPYGLASS_BASE_DIR")
    )
    if not resolved_base:
        raise ValueError(
            "SPYGLASS_BASE_DIR not defined in dj.config or os env vars"
        )

    config_dirs = {"SPYGLASS_BASE_DIR": resolved_base}
    for prefix, dirs in relative_dirs.items():
        for dir, dir_str in dirs.items():
            dir_env_fmt = f"{prefix.upper()}_{dir.upper()}_DIR"
            # Ignore env vars if base was passed to func
            env_loc = os.environ.get(dir_env_fmt) if not base_dir else None
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

    loaded_env = _load_env_vars(env_defaults)
    _set_env_with_dict({**config_dirs, **kachery_zone_dict, **loaded_env})
    _mkdirs_from_dict_vals(config_dirs)
    _set_dj_config_stores(config_dirs)

    config = dict(
        debug_mode=dj_custom.get("debug_mode", False),
        **config_defaults,
        **config_dirs,
        **kachery_zone_dict,
        **loaded_env,
    )
    config_loaded = True
    return config


def _load_env_vars(env_dict: dict) -> dict:
    """Loads env vars from dict {str: Any}."""
    loaded_dict = {}
    for var, val in env_dict.items():
        loaded_dict[var] = os.getenv(var, val)
    return loaded_dict


def _set_env_with_dict(env_dict: dict):
    """Sets env vars from dict {str: Any} where Any is convertible to str."""
    for var, val in env_dict.items():
        os.environ[var] = str(val)


def _mkdirs_from_dict_vals(dir_dict: dict):
    for dir_str in dir_dict.values():
        Path(dir_str).mkdir(exist_ok=True)


def _set_dj_config_stores(dir_dict: dict):
    raw_dir = dir_dict["SPYGLASS_RAW_DIR"]
    analysis_dir = dir_dict["SPYGLASS_ANALYSIS_DIR"]

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


# TODO: Change redundancy here to class with @properties


def load_base_dir() -> str:
    """Retrieve the base directory from the configuration.

    Returns
    -------
    str
        The base directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_BASE_DIR")


def load_raw_dir() -> str:
    """Retrieve the raw directory from the configuration.

    Returns
    -------
    str
        The raw directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_RAW_DIR")


def load_analysis_dir() -> str:
    """Retrieve the analysis directory from the configuration.

    Returns
    -------
    str
        The recording directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_ANALYSIS_DIR")


def load_recording_dir() -> str:
    """Retrieve the recording directory from the configuration.

    Returns
    -------
    str
        The recording directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_RECORDING_DIR")


def load_sorting_dir() -> str:
    """Retrieve the sorting directory from the configuration.

    Returns
    -------
    str
        The sorting directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_SORTING_DIR")


def load_temp_dir() -> str:
    """Retrieve the temp directory from the configuration.

    Returns
    -------
    str
        The temp directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_TEMP_DIR")


def load_waveform_dir() -> str:
    """Retrieve the temp directory from the configuration.

    Returns
    -------
    str
        The waveform directory path.
    """
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("SPYGLASS_WAVEFORM_DIR")


def load_debug_mode() -> bool:
    global config
    if not config_loaded or not config:
        config = load_config()
    return config.get("debug_mode", False)


base_dir = load_base_dir()
raw_dir = load_raw_dir()
recording_dir = load_recording_dir()
temp_dir = load_temp_dir()
analysis_dir = load_analysis_dir()
sorting_dir = load_sorting_dir()
waveform_dir = load_waveform_dir()
debug_mode = load_debug_mode()
