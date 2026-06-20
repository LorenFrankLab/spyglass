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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# A MAD multiplier for peak detection is conventionally ~3-15; a value
# above this with threshold_unit='mad' and no explicit noise_levels is
# almost certainly a microvolt threshold left in the wrong unit (audit
# finding #7).
_MAX_PLAUSIBLE_MAD_MULTIPLIER = 50.0


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
    surface in the schema. The one exception is ``whiten`` (see
    ``_reject_whiten``): KS4 has no such kwarg and a truthy value would
    silently double-whiten via the runtime, so it is rejected at insert.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    Th_universal: float = Field(default=9.0, gt=0.0)
    Th_learned: float = Field(default=8.0, gt=0.0)
    nblocks: int = Field(default=1, ge=0)
    max_cluster_subset: int = Field(default=25_000, ge=1)
    do_CAR: bool = True

    @model_validator(mode="after")
    def _reject_whiten(self):
        """Reject a ``whiten`` key -- KS4 whitens internally, no such kwarg.

        The v2 runtime runs its external float64 whitening whenever the
        sorter params carry a truthy ``whiten`` (``_sorting_compute.py``).
        KS4 also whitens internally (``skip_kilosort_preprocessing=False``),
        so a ``whiten`` key here -- which ``extra="allow"`` would otherwise
        pass through silently -- would double-whiten the signal. Whitening is
        controlled via ``skip_kilosort_preprocessing`` / ``whitening_range``.
        """
        if "whiten" in self.model_dump():
            raise ValueError(
                "Kilosort4 has no 'whiten' parameter; it whitens internally. "
                "Remove 'whiten' -- setting it triggers the v2 runtime's "
                "external float64 whitening on top of KS4's own, "
                "double-whitening the signal. Control whitening via "
                "skip_kilosort_preprocessing / whitening_range."
            )
        return self


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

    * ``"uv"`` (default) -- ``detect_threshold`` is a TRUE microvolt
      threshold. The runtime derives ``noise_levels=[1.0]`` (broadcast
      across channels) AND scales the recording to microvolts
      (``scale_to_uV``, using the stored NWB gain) before ``detect_peaks``,
      so a value of ``100`` means 100 uV. For Frank-lab data (gain == 1
      uV/count) this equals the raw-count value; for non-unity-gain rigs
      (e.g. Intan ~0.195 uV/count) it is the corrected behavior. This
      honors the label v1 used at ``src/spyglass/spikesorting/v1/
      sorting.py:177`` but never enforced (v1 thresholded in raw counts --
      v2 diverges here). Requires the recording to carry channel gains (it
      always does after the v2 ElectricalSeries write); otherwise the
      runtime raises. This is the production Frank-lab default (100 uV).
    * ``"mad"`` -- the threshold is in MAD multiples; SI estimates
      per-channel MAD (``noise_levels`` left unset). Tracks the
      recording's actual noise floor; right for synthetic / low-amplitude
      fixtures (the MEArec smoke row sets this EXPLICITLY).

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
    # Default 100.0 is the production/real-data clusterless threshold: a
    # microvolt / native-unit value under the default ``threshold_unit='uv'``
    # below (which derives noise_levels=[1.0]). The OLD default paired 100
    # with ``threshold_unit='mad'``, making it a 100x-MAD threshold that
    # detected almost nothing -- the bare-schema footgun fixed here (audit
    # finding #7). The synthetic/smoke fixture sets ``threshold_unit='mad'``
    # EXPLICITLY (a ~5x-MAD multiplier suits the simulation), so it does not
    # rely on this unit default.
    detect_threshold: float = Field(default=100.0, gt=0.0)
    threshold_unit: Literal["uv", "mad"] = Field(
        default="uv",
        description=(
            "How detect_threshold is interpreted when noise_levels is "
            "left unset: 'uv' (the default) derives noise_levels=[1.0] AND "
            "the runtime scales the recording to microvolts (scale_to_uV, "
            "using the stored NWB gain) before detection, so detect_threshold "
            "is a TRUE microvolt threshold (for unity-gain Frank-lab data it "
            "equals the old raw-count value); 'mad' lets SpikeInterface "
            "estimate per-channel MAD (a multiplier). The default 'uv' pairs "
            "with the "
            "default detect_threshold=100 to give the production/real-data "
            "threshold; the simulation fixture opts into 'mad' explicitly. "
            "When noise_levels IS set explicitly it takes precedence and is "
            "used verbatim -- threshold_unit is then descriptive only, and "
            "the advanced override bypasses the implausible-MAD guard below. "
            "Without an explicit noise_levels, a 'mad' multiplier above ~50 "
            "is rejected as a microvolt threshold left in the wrong unit."
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

    @model_validator(mode="after")
    def _guard_implausible_mad_threshold(self):
        """Reject a microvolt-scale threshold left in MAD units.

        ``threshold_unit='mad'`` with no explicit ``noise_levels`` makes
        SpikeInterface estimate per-channel MAD and treat
        ``detect_threshold`` as a MAD MULTIPLIER. Real multipliers are
        ~3-15; a value this large (e.g. 100, a microvolt threshold copied
        into a MAD-mode row) makes the effective threshold absurd and
        detects almost nothing -- a silent zero-unit sort. Only fires when
        ``noise_levels`` is None, so the documented explicit-``noise_levels``
        advanced override is untouched.
        """
        if (
            self.threshold_unit == "mad"
            and self.noise_levels is None
            and self.detect_threshold > _MAX_PLAUSIBLE_MAD_MULTIPLIER
        ):
            raise ValueError(
                "ClusterlessThresholderSchema: detect_threshold="
                f"{self.detect_threshold} with threshold_unit='mad' and no "
                "noise_levels is an implausibly large MAD multiplier (> "
                f"{_MAX_PLAUSIBLE_MAD_MULTIPLIER}); SpikeInterface would "
                "estimate per-channel MAD and detect almost nothing. Use "
                "threshold_unit='uv' for a microvolt/native-unit threshold, "
                "or a MAD multiplier near 5."
            )
        return self


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

# Sorters that whiten internally and have NO ``whiten`` kwarg, so a truthy
# ``whiten`` in their params would trigger the v2 runtime's external float64
# whitening (``_sorting_compute.py``) on top of the sorter's own --
# double-whitening the signal (or being rejected by SpikeInterface at sort
# time as an invalid kwarg). Kilosort4 is guarded separately by
# ``Kilosort4Schema._reject_whiten`` (it has its own typed schema); these three
# fall through to ``GenericSorterParamsSchema`` (``extra="allow"``), which
# would pass ``whiten`` through silently, so the guard must run at the
# ``SorterParameters`` insert via :func:`reject_internal_whiten`. MountainSort
# 4/5 are intentionally NOT here: they DO take ``whiten=True``, which the
# runtime routes through the external float64 whitening (whitening the signal
# exactly once).
_INTERNAL_WHITEN_NO_KWARG_SORTERS: frozenset[str] = frozenset(
    {"kilosort2_5", "kilosort3", "ironclust"}
)


def reject_internal_whiten(sorter: str, params: dict) -> None:
    """Raise if an internally-whitening sorter is given a truthy ``whiten``.

    ``kilosort2_5`` / ``kilosort3`` / ``ironclust`` whiten internally and have
    no ``whiten`` parameter, but they validate against the permissive
    ``GenericSorterParamsSchema`` (``extra="allow"``), so a ``whiten=True`` a
    user copies from a Kilosort-style mental model would slip through and
    trigger the v2 runtime's external float64 whitening on top of the sorter's
    own. This mirrors ``Kilosort4Schema._reject_whiten`` for the sorters that
    lack a dedicated typed schema. No-op for every other sorter (including
    MountainSort 4/5, which use ``whiten=True`` deliberately).

    Parameters
    ----------
    sorter : str
        The ``sorter`` column value of a ``SorterParameters`` row.
    params : dict
        The (already schema-validated) ``params`` blob.

    Raises
    ------
    ValueError
        If ``sorter`` whitens internally and ``params`` carries a truthy
        ``whiten``.
    """
    if sorter in _INTERNAL_WHITEN_NO_KWARG_SORTERS and params.get("whiten"):
        raise ValueError(
            f"Sorter {sorter!r} whitens internally and has no 'whiten' "
            "parameter; a truthy 'whiten' triggers the v2 runtime's external "
            "float64 whitening on top of the sorter's own, double-whitening "
            "the signal (or is rejected by SpikeInterface at sort time). "
            "Remove 'whiten' from the params blob."
        )


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
