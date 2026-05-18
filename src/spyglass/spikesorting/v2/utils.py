"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import datajoint as dj
import spikeinterface as si

if TYPE_CHECKING:
    from pydantic import BaseModel


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
