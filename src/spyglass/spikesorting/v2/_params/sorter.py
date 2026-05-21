"""Validated parameter schemas for the sorter parameter table.

One schema per supported v2 sorter, plus a generic ``extra="allow"``
fallback for any SpikeInterface sorter the installed environment
exposes but the v2 pipeline does not ship a dedicated default for.

Strictness policy:
    MountainSort4Schema, MountainSort5Schema, Kilosort4Schema, and
    ClusterlessThresholderSchema use ``extra="forbid"`` because their
    field lists come from documented APIs (v1 mountain_default block and
    appendix.md). A typo in one of their kwargs raises at Lookup insert
    time -- the typo-catching is the value-add over the generic schema.

    SpykingCircus2Schema and Tridesclous2Schema use ``extra="allow"``
    because the v2 plan does not curate their fields; they exist as
    dedicated dispatch slots for ``_get_sorter_schema``. SI's runtime
    validates the final kwargs dict at sort time.

    GenericSorterParamsSchema is the ``extra="allow"`` fallback for any
    SI sorter outside the dedicated set.

Concurrency parameters (``n_jobs``, ``chunk_duration``, ``progress_bar``)
do NOT live on these schemas. They are stored on the per-row
``job_kwargs`` blob column and resolved by ``_resolved_job_kwargs`` at
populate time per the shared-contracts Job-Kwargs Resolution convention.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MountainSort4Schema(BaseModel):
    """Validated schema for MountainSort 4.

    Mirrors v1's ``mountain_default`` block at
    ``src/spyglass/spikesorting/v1/sorting.py`` without the runtime
    ``tempdir`` field-mutation hack. MS4 is not deterministic and the SI
    0.104 wrapper still lists it even when the runtime is not installed;
    the per-platform install evidence is recorded in the v2 resolver
    notes. ``extra="forbid"`` catches typos like ``detect_signe``
    against the v1-documented field set.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    detect_sign: Literal[-1, 0, 1] = -1
    adjacency_radius: float = Field(default=100.0, ge=0.0)
    freq_min: float = Field(default=600.0, gt=0.0, le=15000.0)
    freq_max: float = Field(default=6000.0, gt=0.0, le=15000.0)
    filter: bool = False
    whiten: bool = True
    num_workers: int = Field(default=1, ge=1)
    clip_size: int = Field(default=40, ge=1)
    detect_threshold: float = Field(default=3.0, gt=0.0)
    detect_interval: int = Field(default=10, ge=1)


class MountainSort5Schema(BaseModel):
    """Validated schema for MountainSort 5.

    Defaults follow ``appendix.md § MountainSort 5 install + params``.
    MS5 has no ``tempdir`` parameter (the v1 ``sorter_params.pop(
    'tempdir', None)`` hack is gone), and assumes the input recording
    has already been bandpass-filtered and whitened by the upstream
    recording stage. ``extra="forbid"`` catches typos against the
    documented MS5 field set.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    scheme: Literal["1", "2", "3"] = "2"
    detect_sign: Literal[-1, 0, 1] = -1
    detect_threshold: float = Field(default=5.5, gt=0.0)
    detect_time_radius_msec: float = Field(default=0.5, gt=0.0)
    snippet_T1: int = Field(default=20, ge=1)
    snippet_T2: int = Field(default=20, ge=1)
    scheme2_phase1_detect_channel_radius: float = Field(default=200.0, gt=0.0)
    scheme2_detect_channel_radius: float = Field(default=50.0, gt=0.0)


class Kilosort4Schema(BaseModel):
    """Validated schema for Kilosort 4.

    Documents the v2-relevant kwargs from ``appendix.md § Kilosort 4
    install + params``. KS4 has a large parameter surface; this schema
    types the most-used knobs and uses ``extra="forbid"`` so a typo
    fails fast. Users who need an undocumented KS4 kwarg should extend
    this schema rather than silently accept the new field. KS4 is not
    a deterministic fallback (CPU/GPU runtime differences can change
    spike times) and may require GPU for non-trivial recordings.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    Th_universal: float = Field(default=9.0, gt=0.0)
    Th_learned: float = Field(default=8.0, gt=0.0)
    nblocks: int = Field(default=1, ge=0)
    max_cluster_subset: int = Field(default=25_000, ge=1)
    do_CAR: bool = True


class ClusterlessThresholderSchema(BaseModel):
    """Validated schema for the clusterless-thresholder special case.

    Not an SpikeInterface registered sorter; ``clusterless_thresholder``
    is a Spyglass-specific peak-detection path built on
    ``spikeinterface.core.detect_peaks``. Default values mirror v1's
    ``default_clusterless`` row at
    ``src/spyglass/spikesorting/v1/sorting.py``. ``extra="forbid"``
    catches typos against the v1-documented field set.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    detect_threshold: float = Field(default=100.0, gt=0.0)
    method: Literal["locally_exclusive", "global"] = "locally_exclusive"
    peak_sign: Literal["neg", "pos", "both"] = "neg"
    exclude_sweep_ms: float = Field(default=0.1, gt=0.0)
    local_radius_um: float = Field(default=100.0, gt=0.0)
    noise_levels: list[float] = Field(default_factory=lambda: [1.0])
    random_chunk_kwargs: dict[str, Any] = Field(default_factory=dict)
    outputs: Literal["sorting"] = "sorting"


class SpykingCircus2Schema(BaseModel):
    """Validated schema for SpykingCircus 2.

    The v2 default set does not curate SC2 fields; the schema exists so
    ``_get_sorter_schema('spykingcircus2')`` dispatches to a dedicated
    class rather than the generic fallback. Users supply SC2-specific
    kwargs through ``extra="allow"``; SI validates at sort time.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1


class Tridesclous2Schema(BaseModel):
    """Validated schema for Tridesclous 2.

    Same rationale as ``SpykingCircus2Schema``: dedicated dispatch slot,
    fields delegated to SI runtime validation.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1


class GenericSorterParamsSchema(BaseModel):
    """Fallback schema for installed SI sorters with no dedicated v2 model.

    Preserves v1's "try any installed SpikeInterface sorter" escape hatch
    without auto-inserting defaults for every sorter
    ``spikeinterface.sorters.available_sorters()`` reports. Users supply
    the full SI kwargs dict; SI validates at sort time.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1


_SORTER_SCHEMAS: dict[str, type[BaseModel]] = {
    "mountainsort4": MountainSort4Schema,
    "mountainsort5": MountainSort5Schema,
    "kilosort4": Kilosort4Schema,
    "spykingcircus2": SpykingCircus2Schema,
    "tridesclous2": Tridesclous2Schema,
    "clusterless_thresholder": ClusterlessThresholderSchema,
}


def _get_sorter_schema(sorter_name: str) -> type[BaseModel]:
    """Return the Pydantic schema for a sorter name.

    Falls back to ``GenericSorterParamsSchema`` for any sorter outside the
    dedicated v2 set, so the v1 "try any installed SI sorter" escape hatch
    keeps working. ``SorterParameters.insert1`` is responsible for the
    additional check that the sorter name resolves to a known SI sorter or
    the Spyglass-specific ``clusterless_thresholder`` path; the schema
    itself does not enforce that, because the dispatch happens before any
    SpikeInterface import is required.

    Parameters
    ----------
    sorter_name : str
        Value of the ``sorter`` column on a ``SorterParameters`` row.

    Returns
    -------
    type[pydantic.BaseModel]
        The Pydantic class to validate the row's ``params`` blob against.
    """
    return _SORTER_SCHEMAS.get(sorter_name, GenericSorterParamsSchema)
