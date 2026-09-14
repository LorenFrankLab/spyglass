"""Reuse an ingestion plan instead of parsing a file twice.

Planning reads every relevant object out of an NWB file and runs each table's
mapping over it. That is the expensive half of ingestion, and it is pure: the
same file, config and Spyglass version yield the same plan. Caching it means
a user can plan, read the report, fix what it names, and insert -- without
paying for the parse each time.

The cache is keyed on what the plan depends on, so a stale entry is not
reachable rather than merely unlikely: change the file, the config, or the
Spyglass version and the key changes with it.
"""

import json
from hashlib import md5
from pathlib import Path
from typing import Optional, Tuple

from spyglass.data_import.ingestion_plan import IngestionPlan
from spyglass.utils.logging import logger

# Bump when a change to the plan format makes older files unreadable. A plan
# written by an older Spyglass already misses via `spyglass_version`; this
# covers a format change within one version, during development.
CACHE_FORMAT = 1


def file_fingerprint(path) -> Optional[str]:
    """Return a cheap identifier for a file's current contents.

    Size and modification time, not a content hash. Hashing an NWB file costs
    about what parsing it costs, which would defeat the point of caching the
    parse. `NwbfileHasher` remains the tool for proving two files hold the
    same data; this only answers "is this the same file it was".

    The tradeoff: a file rewritten to the same byte length within the same
    nanosecond timestamp would read as unchanged. In exchange the key is free
    to compute.

    Parameters
    ----------
    path : str or Path
        The file to fingerprint.

    Returns
    -------
    str or None
        A hex digest, or None if the file cannot be read -- in which case the
        caller should decline to cache rather than cache under a wrong key.
    """
    try:
        stat = Path(path).stat()
    except OSError as err:
        logger.debug(f"Plan cache: cannot stat {path}: {err}")
        return None
    return md5(f"{stat.st_size}-{stat.st_mtime_ns}".encode()).hexdigest()


def config_fingerprint(config: dict) -> str:
    """Return a digest of a config, stable across key ordering.

    Parameters
    ----------
    config : dict
        The merged config a plan was built with.

    Returns
    -------
    str
        A hex digest. An empty or absent config hashes to a fixed value, so
        "no config" is itself a cache key rather than a wildcard.
    """
    return md5(
        json.dumps(config or dict(), sort_keys=True, default=str).encode()
    ).hexdigest()


def spyglass_version() -> str:
    """Return the running Spyglass version, or `unknown` if unavailable."""
    try:
        from spyglass import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version metadata missing
        return "unknown"


def cache_key(nwb_hash: str, config_hash: str, version: str) -> str:
    """Return the key a plan with this provenance is stored under."""
    return md5(
        f"{CACHE_FORMAT}-{nwb_hash}-{config_hash}-{version}".encode()
    ).hexdigest()[:16]


def cache_dir() -> Path:
    """Return the directory holding cached plans, creating it if needed."""
    from spyglass.settings import temp_dir

    path = Path(temp_dir) / "ingestion_plans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(nwb_file_name: str, key: str) -> Path:
    """Return the file a plan with this key is stored in.

    The file name carries the NWB file too, so the directory is legible and a
    single file's plans can be cleared without touching anyone else's.
    """
    stem = Path(nwb_file_name).stem
    return cache_dir() / f"{stem}-{key}.json"


def load_plan(
    nwb_file_name: str, nwb_hash: str, config_hash: str, version: str
) -> Optional[IngestionPlan]:
    """Return a cached plan for this exact provenance, or None.

    A cache miss is never an error: a corrupt or unreadable file is reported
    at debug level and treated as absent, so a bad cache costs a re-parse
    rather than an ingestion.

    Parameters
    ----------
    nwb_file_name : str
        The file the plan describes.
    nwb_hash : str
        The file's fingerprint when the plan was built.
    config_hash : str
        Digest of the config the plan was built with.
    version : str
        The Spyglass version that built it.

    Returns
    -------
    IngestionPlan or None
    """
    if not nwb_hash:  # unfingerprintable file: never a hit
        return None

    path = cache_path(nwb_file_name, cache_key(nwb_hash, config_hash, version))
    if not path.exists():
        return None

    try:
        plan = IngestionPlan.from_dict(json.loads(path.read_text()))
    except Exception as err:
        logger.debug(f"Plan cache: discarding unreadable {path}: {err}")
        return None

    # Belt and braces: the key is derived from these, but a hand-edited or
    # colliding file must not be served as though it described this run.
    if (
        plan.nwb_file_name != nwb_file_name
        or plan.nwb_hash != nwb_hash
        or plan.config_hash != config_hash
        or plan.spyglass_version != version
    ):
        logger.debug(f"Plan cache: provenance mismatch in {path}")
        return None

    logger.info(f"Using cached ingestion plan for {nwb_file_name}")
    return plan


def save_plan(plan: IngestionPlan) -> Optional[Path]:
    """Write a plan to the cache, keyed by its own provenance.

    Declines rather than raises when the plan cannot be keyed or written:
    caching is an optimization, and failing to cache must not fail an
    ingestion.

    Parameters
    ----------
    plan : IngestionPlan
        A plan carrying `nwb_hash`, `config_hash` and `spyglass_version`.

    Returns
    -------
    Path or None
        Where it was written, or None if it was not.
    """
    if not plan.nwb_hash:  # provenance incomplete: not safely reusable
        return None

    path = cache_path(
        plan.nwb_file_name,
        cache_key(plan.nwb_hash, plan.config_hash, plan.spyglass_version),
    )
    try:
        # Write beside, then move: a reader must never see half a plan.
        tmp = path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(plan.to_dict(), default=str))
        tmp.replace(path)
    except Exception as err:
        logger.debug(f"Plan cache: could not write {path}: {err}")
        return None

    return path


def clear_cache(nwb_file_name: str = None) -> int:
    """Delete cached plans, for one file or all of them.

    Parameters
    ----------
    nwb_file_name : str, optional
        Clear only this file's plans. Default None, clear every plan.

    Returns
    -------
    int
        How many files were removed.
    """
    pattern = (
        f"{Path(nwb_file_name).stem}-*.json" if nwb_file_name else "*.json"
    )
    removed = 0
    for path in cache_dir().glob(pattern):
        try:
            path.unlink()
            removed += 1
        except OSError as err:  # pragma: no cover - permissions
            logger.debug(f"Plan cache: could not remove {path}: {err}")
    return removed


def plan_provenance(nwb_file_path, config: dict) -> Tuple[str, str, str]:
    """Return the (nwb_hash, config_hash, version) a plan is keyed by."""
    return (
        file_fingerprint(nwb_file_path),
        config_fingerprint(config),
        spyglass_version(),
    )
