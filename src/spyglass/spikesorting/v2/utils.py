"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import datajoint as dj
import spikeinterface as si

if TYPE_CHECKING:
    from pydantic import BaseModel


@contextmanager
def transaction_or_noop(connection):
    """Open a DataJoint transaction unless one is already active.

    Source-part inserts and curation inserts both want to wrap their
    master + part rows in one transaction; but the same helpers may be
    called from inside an existing populate cascade where DataJoint
    refuses nested transactions. This context manager makes the
    transaction wrap a no-op when the connection is already in one.
    """
    if connection.in_transaction:
        yield
    else:
        with connection.transaction:
            yield


class MetricsSource(str, Enum):
    """Provenance of metrics attached to a ``CurationV2`` row.

    Matches the enum on the table's ``metrics_source`` column. Promoted
    from a runtime set check so a typo at insert time raises a clear
    ``ValueError`` instead of a DataJoint enum-mismatch error from
    MySQL.
    """

    manual = "manual"
    analyzer_curation = "analyzer_curation"
    figpack = "figpack"


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


def find_orphaned_masters(master_table, part_tables: list) -> list[dict]:
    """Return master PKs whose source-part counts sum to zero.

    Shared implementation of ``ArtifactSelection.prune_orphaned_selections``
    and ``SortingSelection.prune_orphaned_selections``. ``part_tables``
    is the list of source-part tables to count against the master --
    e.g. ``[RecordingSource, SharedArtifactGroupSource]`` for artifact
    selection, ``[RecordingSource, ConcatenatedRecordingSource]`` for
    sorting selection.

    Source-part atomicity is enforced at insert time by the
    transactional ``insert_selection`` helpers, but DataJoint cannot
    enforce "exactly one source per master" across two part tables;
    an upstream cascade-delete from ``Recording`` or
    ``SharedArtifactGroup`` / ``ConcatenatedRecording`` can leave the
    master row without any source children. This helper finds those
    orphans so a maintenance script can review or remove them.
    """
    orphans: list[dict] = []
    for master in master_table.fetch("KEY", as_dict=True):
        if sum(len(part & master) for part in part_tables) == 0:
            orphans.append(master)
    return orphans


def unit_brain_region_df(unit_relation, resolution: str):
    """Join a Unit-part relation against Electrode * BrainRegion.

    Shared implementation of ``Sorting.get_unit_brain_regions`` and
    ``CurationV2.get_unit_brain_regions``. The Unit relation must
    carry an ``Electrode`` FK; the join walks it to ``BrainRegion``
    (non-null FK on ``Electrode``) and returns a DataFrame with the
    standard column set + a ``region_resolution`` literal label so
    concat-backed callers can distinguish anchor-member results.
    """
    import pandas as pd

    from spyglass.common.common_ephys import Electrode as _Electrode
    from spyglass.common.common_region import BrainRegion

    joined = (unit_relation * _Electrode * BrainRegion).fetch(
        "unit_id",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
        as_dict=True,
    )
    df = pd.DataFrame(joined)
    df["region_resolution"] = resolution
    return df


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


def _ensure_lookup_row_exists(
    lookup_table,
    restriction: dict,
    *,
    helper_name: str,
    insert_default_path: str,
) -> None:
    """Pre-check that a Lookup-row FK target exists before insert_selection.

    Without this guard, a missing Lookup row produces an opaque
    DataJoint ``IntegrityError`` ("foreign key constraint fails")
    that gives the user no hint about which Lookup table is empty or
    how to populate it. Raise a clear ``ValueError`` instead so the
    notebook user can fix the setup in one step.

    Parameters
    ----------
    lookup_table
        The Lookup table class whose row is required (e.g.
        ``PreprocessingParameters``).
    restriction
        The dict identifying the required row (e.g.
        ``{"preproc_params_name": "default_franklab"}``).
    helper_name
        Name of the insert_selection helper calling us, for the error
        message (e.g. ``"RecordingSelection.insert_selection"``).
    insert_default_path
        Importable path that loads the default rows (e.g.
        ``"PreprocessingParameters.insert_default()"``).
    """
    if not (lookup_table & restriction):
        raise ValueError(
            f"{helper_name}: required Lookup row not found in "
            f"{lookup_table.__name__} for {restriction}. "
            f"Run {insert_default_path} first to install the default "
            "rows, or insert your custom row before retrying. The "
            "one-shot `spyglass.spikesorting.v2.initialize_v2_defaults()`"
            " installs every required default in one call."
        )


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
