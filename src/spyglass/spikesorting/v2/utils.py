"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import datajoint as dj
import spikeinterface as si

if TYPE_CHECKING:
    from pydantic import BaseModel


class CurationLabel(str, Enum):
    """Curation labels recognized by ``CurationV2.insert_curation``.

    Members match the v1 convention list at
    ``src/spyglass/spikesorting/v1/curation.py``; v2 promotes the
    list from a docstring to an enforced enum so a typo raises at
    insert time. Free-form ``dj.Manual.insert1`` calls bypassing the
    helper remain permitted (DataJoint cannot enforce enums on
    varchar columns), and downstream filters fall back to the v1
    list for any unrecognized label that slipped in.
    """

    accept = "accept"
    mua = "mua"
    noise = "noise"
    artifact = "artifact"
    reject = "reject"


@dataclass(frozen=True)
class SourceResolution:
    """Result of ``<MasterTable>.resolve_source(key)`` on a source-part master.

    Used at the top of ``Sorting.make()`` and ``ArtifactDetection.make()``
    to dispatch on which source-part row backs the master. ``kind`` is
    the source enum; ``key`` is the source-row PK fields (e.g.
    ``{"recording_id": ...}`` or ``{"shared_artifact_group_name": ...}``)
    so the caller can pass it straight into the upstream table's
    ``get_*`` / fetch helpers.
    """

    kind: Literal[
        "recording",
        "concatenated_recording",
        "shared_artifact_group",
    ]
    key: dict


# ---------------------------------------------------------------------------
# Database-host safety guard
# ---------------------------------------------------------------------------

_SAFE_DB_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_OVERRIDE_ENV = "SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB"


def _assert_v2_db_safe() -> None:
    """Refuse to register v2 schemas against a non-test database.

    Called at module-import time by every v2 schema module before
    ``schema = dj.schema(...)`` runs. While the v2 pipeline is under
    active development we hard-fail any attempt to declare or write to
    v2 schemas on a database other than the isolated test docker
    container (host ``localhost`` / ``127.0.0.1`` / ``::1``).

    The check is intentionally narrow: it pins the database host only.
    Other isolation guarantees (the ``pytests``/``test`` schema prefix,
    the temp ``SPYGLASS_BASE_DIR`` path) are handled by
    ``bootstrap_v2_test_environment``. This guard is the last line of
    defense if some other code path repointed ``dj.config`` at the
    production server after bootstrap ran.

    Raises
    ------
    RuntimeError
        If ``dj.config['database.host']`` is set to a non-local host
        and the override env var is not set.

    Override
    --------
    Set ``SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1`` to bypass the
    guard. This requires an explicit deliberate action; do NOT export
    it for a shell session in which you might accidentally run v2
    populate / insert calls.
    """
    if os.environ.get(_OVERRIDE_ENV) == "1":
        return

    host = dj.config.get("database.host")
    if host in _SAFE_DB_HOSTS:
        return

    raise RuntimeError(
        f"v2 spike sorting refuses to register schemas against host "
        f"{host!r}. Expected one of {sorted(_SAFE_DB_HOSTS)} (the "
        f"isolated test docker container). Run "
        f"`bootstrap_v2_test_environment` from "
        f"`tests/spikesorting/v2/test_env.py` before importing v2 "
        f"runtime modules. To bypass with full understanding of the "
        f"risk, set {_OVERRIDE_ENV}=1 in the environment."
    )


def _validate_params(model_cls: type[BaseModel], payload: dict) -> dict:
    """Validate a parameter payload against a Pydantic model.

    Parameters
    ----------
    model_cls : type[pydantic.BaseModel]
        The schema to validate against.
    payload : dict
        The raw parameter dict, typically a Lookup row's ``params`` blob.

    Returns
    -------
    dict
        The validated, normalized payload (``model_dump()`` output).

    Raises
    ------
    pydantic.ValidationError
        If ``payload`` does not satisfy ``model_cls``.
    """
    return model_cls.model_validate(payload).model_dump()


def _analyzer_path(key: dict) -> Path:
    """Return the on-disk SortingAnalyzer folder for a sorting row.

    The folder holds regeneratable scratch (waveforms, templates, metric
    extensions); it is not the canonical artifact and lives outside the
    AnalysisNwbfile storage tree, under Spyglass's configured temp directory.

    Parameters
    ----------
    key : dict
        A key containing ``sorting_id``.

    Returns
    -------
    pathlib.Path
        ``{temp_dir}/spikesorting_v2/analyzers/{sorting_id}.analyzer``.
    """
    from spyglass.settings import temp_dir

    return (
        Path(temp_dir)
        / "spikesorting_v2"
        / "analyzers"
        / f"{key['sorting_id']}.analyzer"
    )


def _resolved_job_kwargs(*row_job_kwargs: dict | None) -> dict:
    """Merge SpikeInterface-global, DataJoint-config, and per-row job kwargs.

    Sources are merged in increasing precedence order: the SpikeInterface
    global defaults, then ``dj.config['custom']['spikesorting_v2_job_kwargs']``,
    then each per-row blob in the order given.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        ``job_kwargs`` blob values from the parameter rows that govern this
        compute stage, in increasing precedence order (a later argument wins
        on key conflict). ``None`` and empty-dict entries are skipped.

    Returns
    -------
    dict
        The merged kwargs, ready to splat into a compute call.
    """
    merged = dict(si.get_global_job_kwargs())
    custom = dj.config.get("custom", {}) or {}
    merged.update(custom.get("spikesorting_v2_job_kwargs", {}) or {})
    for override in row_job_kwargs:
        if override:
            merged.update(override)
    return merged


def _hash_nwb_recording(analysis_file_name: str) -> str:
    """Return a content hash of a recording's AnalysisNwbfile.

    Delegates to Spyglass's ``NwbfileHasher`` so v2 recompute verification
    uses the same hashing path as the v1 recompute machinery rather than a
    parallel implementation.

    Parameters
    ----------
    analysis_file_name : str
        Name of the AnalysisNwbfile holding the preprocessed recording.

    Returns
    -------
    str
        The ``NwbfileHasher`` digest of the file.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.utils.nwb_hash import NwbfileHasher

    abs_path = AnalysisNwbfile().get_abs_path(analysis_file_name)
    return NwbfileHasher(abs_path).hash
