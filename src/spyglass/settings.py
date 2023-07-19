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
        spike_sorting_storage="spikesorting",
        waveforms="waveforms",
        temp="tmp",
    ),
    kachery=dict(
        storage="kachery-storage",
        temp="tmp",
    ),
)


def load_config(base_dir: Path = None, force_reload: bool = False) -> dict:
    """Gets syglass dirs from dj.config or environment variables.

    Uses a relative_dirs dict defined in settings.py to (a) gather user
    settings from dj.config or os environment variables or defaults relative to
    base, in that order (b) set environment variables, and (c) make dirs that
    don't exist.

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

    dj_spy = dj.config.get("custom", {}).get("spyglass_dirs", {})
    base_dir = (
        base_dir
        or dj_spy.get("base")
        or os.environ.get("SPYGLASS_BASE_DIR", ".")
    )
    if not base_dir:
        raise ValueError(
            "SPYGLASS_BASE_DIR not defined in dj.config or os env vars"
        )

    config_dirs = {"SPYGLASS_BASE_DIR": base_dir}
    for prefix, dirs in relative_dirs.items():
        for dir, dir_str in dirs.items():
            dir_env_fmt = f"{prefix.upper()}_{dir.upper()}_DIR"
            dir_location = (
                dj_spy.get(dir)
                or os.environ.get(dir_env_fmt)
                or base_dir + "/" + dir_str
            ).replace('"', "")
            config_dirs.update({dir_env_fmt: dir_location})

    _set_env_with_dict(config_dirs)
    _mkdirs_from_dict_vals(config_dirs)
    _set_dj_config_stores(config_dirs)

    config = dict(
        **config_defaults,
        **config_dirs,
    )
    config_loaded = True
    return config


def base_dir() -> str:
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


def _set_env_with_dict(env_dict: dict):
    env_to_set = {**env_defaults, **env_dict}
    for var, val in env_to_set.items():
        os.environ[var] = val


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
