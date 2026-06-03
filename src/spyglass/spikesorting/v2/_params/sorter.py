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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MountainSort4Schema(BaseModel):
    """Validated schema for MountainSort 4.

    Mirrors v1's ``mountain_default`` block at
    ``src/spyglass/spikesorting/v1/sorting.py:145-153`` without the
    runtime ``tempdir`` field-mutation hack. ``freq_min=600`` and
    ``freq_max=6000`` defaults match v1's tetrode preset row at
    ``v1/sorting.py:158-159`` (the ``mountain_default`` block itself
    did NOT include ``freq_min`` / ``freq_max`` -- those keys came
    from the tetrode preset); v2's schema-level defaults choose the
    tetrode preset values so a user constructing
    ``MountainSort4Schema()`` without arguments gets v1's most-used
    Frank-lab production preset implicitly. MS4 is not deterministic
    and the SI 0.104 wrapper still lists it even when the runtime
    is not installed; the per-platform install evidence is recorded
    in the v2 resolver notes. ``extra="forbid"`` catches typos like
    ``detect_signe`` against the v1-documented field set.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    detect_sign: Literal[-1, 0, 1] = -1
    adjacency_radius: float = Field(
        default=100.0,
        ge=-1.0,
        description=(
            "Channel adjacency radius in microns. SpikeInterface's "
            "MountainSort4 wrapper documents -1 as the 'use all channels "
            "in the adjacency graph' sentinel; any value >= 0 sets an "
            "explicit radius. Values in the open interval (-1, 0) are "
            "invalid."
        ),
    )
    freq_min: float = Field(default=600.0, gt=0.0, le=15000.0)
    freq_max: float = Field(default=6000.0, gt=0.0, le=15000.0)
    filter: bool = False
    whiten: bool = True
    num_workers: int = Field(default=1, ge=1)
    clip_size: int = Field(default=40, ge=1)
    detect_threshold: float = Field(default=3.0, gt=0.0)
    detect_interval: int = Field(default=10, ge=1)

    @field_validator("adjacency_radius")
    @classmethod
    def _reject_open_interval_radius(cls, value: float) -> float:
        """Only -1 (sentinel) or >= 0 is meaningful; reject (-1, 0)."""
        if -1.0 < value < 0.0:
            raise ValueError(
                "adjacency_radius must be -1 (use all channels) or >= 0; "
                f"got {value}, which is in the invalid open interval "
                "(-1, 0)."
            )
        return value


class MountainSort5Schema(BaseModel):
    """Validated schema for MountainSort 5.

    Defaults follow ``appendix.md § MountainSort 5 install + params``.
    MS5 has no ``tempdir`` parameter (the v1 ``sorter_params.pop(
    'tempdir', None)`` hack is gone), and assumes the input recording
    has already been bandpass-filtered and whitened by the upstream
    recording stage. ``extra="forbid"`` catches typos against the
    documented MS5 field set.

    ``filter`` / ``whiten`` mirror ``MountainSort4Schema`` so MS5 is
    handled identically to MS4 by the runtime. The SI 0.104 MS5 wrapper
    defaults both to ``True`` (verified against
    ``Mountainsort5Sorter._default_params``); this schema overrides
    ``filter`` to ``False`` because the v2 recording stage already
    bandpass-filters the input (300-6000 Hz + median CAR) -- leaving MS5's
    internal filter on would double-filter the recording, narrowing the
    spike band twice. ``whiten`` stays ``True``: in the Spyglass runtime a
    truthy ``whiten`` routes through the external float64 whitening pin in
    ``Sorting._run_si_sorter`` (which then disables MS5's internal
    whitening so the recording is whitened exactly once), matching v1 and
    the MS4 path. Both toggles are exposed so a user feeding MS5 an
    un-preprocessed recording can re-enable the internal filter.
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
    filter: bool = False
    whiten: bool = True


class Kilosort4Schema(BaseModel):
    """Validated schema for Kilosort 4.

    Documents the v2-relevant kwargs from ``appendix.md § Kilosort 4
    install + params``. KS4 has a large parameter surface; this schema
    types the most-used knobs but uses ``extra="allow"`` so users who
    need an undocumented KS4 kwarg (``batch_size``, ``nearest_chans``,
    etc.) can pass it through to SI without an upstream schema PR.
    Matches v1's escape hatch at ``v1/sorting.py:184-189`` which
    expanded sorter contents via ``sis.get_default_sorter_params``.
    KS4 is not a deterministic fallback (CPU/GPU runtime differences
    can change spike times) and may require GPU for non-trivial
    recordings.

    Trade-off of ``extra="allow"``: typos in un-listed KS4 kwargs
    surface at SI sort time, not at Lookup insert time -- this is the
    accepted cost of not pinning KS4's large, fast-moving parameter
    surface in the schema.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    Th_universal: float = Field(default=9.0, gt=0.0)
    Th_learned: float = Field(default=8.0, gt=0.0)
    nblocks: int = Field(default=1, ge=0)
    max_cluster_subset: int = Field(default=25_000, ge=1)
    do_CAR: bool = True


class ClusterlessThresholderSchema(BaseModel):
    """Validated schema for the clusterless-thresholder special case.

    Not a SpikeInterface registered sorter; ``clusterless_thresholder``
    is a Spyglass-specific peak-detection path built on
    ``spikeinterface.sortingcomponents.peak_detection.detect_peaks``.
    Default values mirror v1's ``default_clusterless`` row at
    ``src/spyglass/spikesorting/v1/sorting.py:177``. ``extra="forbid"``
    catches typos against the v1-documented field set.

    Dropped two dead fields from v1's row shape:

    * ``outputs`` was a Spyglass routing hint; the runtime always
      treats the detector output as a sorting and never reads this
      field. Removed.
    * ``random_chunk_kwargs`` was renamed to ``random_slices_kwargs``
      and is now managed internally by SI 0.104 ``detect_peaks``; the
      v1 field had no effect in the prior strip-and-call path.
      Removed.

    ``threshold_unit`` is the primary, self-documenting knob for how
    ``detect_threshold`` is interpreted:

    * ``"mad"`` (default) -- the threshold is in MAD multiples; SI
      estimates per-channel MAD (``noise_levels`` left unset). Tracks
      the recording's actual noise floor; right for synthetic /
      low-amplitude fixtures.
    * ``"uv"`` -- the runtime derives ``noise_levels=[1.0]`` (broadcast
      across channels) so ``detect_peaks`` reads ``detect_threshold``
      directly in the recording's native amplitude units. NOTE: under
      v2's default preprocessing (bandpass + common_reference at
      float64, with NO gain scaling) the recording is in raw ADC
      counts, so ``detect_threshold`` is effectively raw counts, not
      true microvolts -- a v1-inherited unit labeling (see the
      ``detect_threshold`` units note in CHANGELOG ``[0.5.6]``). It is
      only true uV if the recording is already gain-scaled; convert via
      ``count x gain_uV_per_count``. Reproduces v1's
      ``default_clusterless`` behavior at
      ``src/spyglass/spikesorting/v1/sorting.py:177``.

    ``noise_levels`` stays available as an ADVANCED explicit override.
    Precedence: if ``noise_levels`` is set, the runtime uses it
    verbatim; if ``noise_levels is None``, the runtime derives it from
    ``threshold_unit`` as above. The runtime strips ``threshold_unit``
    before the ``detect_peaks`` call (it is not a ``detect_peaks``
    kwarg).

    ``schema_version`` is 4: 2 dropped ``outputs`` /
    ``random_chunk_kwargs``; 3 made ``noise_levels`` optional; 4 added
    ``threshold_unit``. ``extra="forbid"`` enforces the field set at
    insert time.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 4
    detect_threshold: float = Field(default=100.0, gt=0.0)
    threshold_unit: Literal["uv", "mad"] = Field(
        default="mad",
        description=(
            "How detect_threshold is interpreted when noise_levels is "
            "left unset: 'uv' derives noise_levels=[1.0] so the threshold "
            "is in the recording's native units (raw ADC counts under "
            "v2's gain-free preprocessing, not true microvolts unless "
            "gain-scaled) and 'mad' lets SpikeInterface estimate "
            "per-channel MAD. When "
            "noise_levels IS set explicitly it takes precedence and is "
            "used verbatim -- threshold_unit is then descriptive only. The "
            "two are NOT cross-validated because an explicit per-channel "
            "noise_levels is a supported advanced override that no single "
            "threshold_unit value can represent; keep them consistent "
            "(the shipped 'default' row pairs 'uv' with [1.0])."
        ),
    )
    # Only ``locally_exclusive`` is wired: the runtime always builds
    # ``radius_um`` into ``method_kwargs`` (spatial exclusion), which the
    # other SI ``detect_peaks`` methods (e.g. ``by_channel``) reject. v1
    # advertised a ``"global"`` option that was never a valid SI method --
    # it would fail at ``detect_peaks`` time -- so it is not offered here.
    method: Literal["locally_exclusive"] = "locally_exclusive"
    peak_sign: Literal["neg", "pos", "both"] = "neg"
    exclude_sweep_ms: float = Field(default=0.1, gt=0.0)
    local_radius_um: float = Field(default=100.0, gt=0.0)
    noise_levels: list[float] | None = Field(default=None)


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
