"""Validated parameter schemas for the sorter parameter table.

Strictness:

- ``extra="forbid"`` (typos raise at insert): MountainSort4,
  MountainSort5, ClusterlessThresholder.
- ``extra="allow"`` (SI validates at sort time): Kilosort4 (types common
  knobs, rejects ``whiten``), SpykingCircus2, Tridesclous2, and the generic
  fallback for any other SI sorter.

Concurrency settings live in each row's ``job_kwargs`` blob.
"""

from __future__ import annotations

from typing import Literal
import difflib

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Real MAD multipliers are ~3-15; larger is a microvolt value in the wrong unit.
_MAX_PLAUSIBLE_MAD_MULTIPLIER = 50.0


class MountainSort4Schema(BaseModel):
    """Validated schema for MountainSort 4.

    Defaults are the Frank-lab tetrode preset. MS4 is not deterministic.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
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
    """Validated schema for MountainSort 5 (SpikeInterface 0.104.3 wrapper).

    Every scientific wrapper parameter is a field with the wrapper default,
    except:

    - ``delete_temporary_recording`` is rejected; ``run_si_sorter`` manages
      scratch cleanup.
    - ``filter`` defaults to ``False`` because the recording stage already
      bandpass-filters; ``freq_min`` / ``freq_max`` apply only when it is on.
    - ``whiten=True`` routes through ``run_si_sorter``'s float64 whitening,
      which disables MS5's own so the data is whitened once.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    scheme: Literal["1", "2", "3"] = "2"
    detect_sign: Literal[-1, 0, 1] = -1
    detect_threshold: float = Field(default=5.5, gt=0.0)
    detect_time_radius_msec: float = Field(default=0.5, gt=0.0)
    snippet_T1: int = Field(default=20, ge=1)
    snippet_T2: int = Field(default=20, ge=1)
    snippet_mask_radius: float = Field(default=250.0, gt=0.0)
    npca_per_channel: int = Field(default=3, ge=1)
    npca_per_subdivision: int = Field(default=10, ge=1)
    scheme1_detect_channel_radius: float = Field(default=150.0, gt=0.0)
    scheme2_phase1_detect_channel_radius: float = Field(default=200.0, gt=0.0)
    scheme2_detect_channel_radius: float = Field(default=50.0, gt=0.0)
    scheme2_max_num_snippets_per_training_batch: int = Field(default=200, ge=1)
    scheme2_training_duration_sec: float = Field(default=300.0, gt=0.0)
    scheme2_training_recording_sampling_mode: Literal["initial", "uniform"] = (
        "uniform"
    )
    scheme3_block_duration_sec: float = Field(default=1800.0, gt=0.0)
    filter: bool = False
    freq_min: float = Field(default=300.0, gt=0.0, le=15000.0)
    freq_max: float = Field(default=6000.0, gt=0.0, le=15000.0)
    whiten: bool = True

    @model_validator(mode="after")
    def _check_filter_band(self):
        """Require ``freq_min < freq_max``."""
        if self.freq_min >= self.freq_max:
            raise ValueError(
                f"MountainSort5Schema: freq_min ({self.freq_min}) must be "
                f"below freq_max ({self.freq_max})."
            )
        return self


class Kilosort4Schema(BaseModel):
    """Validated schema for Kilosort 4.

    Types the common knobs; other KS4 kwargs pass through to SI
    (``extra="allow"``), so their typos surface at sort time. KS4 is not
    deterministic across CPU/GPU.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1
    Th_universal: float = Field(default=9.0, gt=0.0)
    Th_learned: float = Field(default=8.0, gt=0.0)
    nblocks: int = Field(default=1, ge=0)
    max_cluster_subset: int = Field(default=25_000, ge=1)
    do_CAR: bool = True

    @model_validator(mode="after")
    def _reject_whiten(self):
        """Reject ``whiten``; KS4 has no such kwarg.

        The wrapper-vocabulary check misses it when ``kilosort`` is not
        installed.
        """
        if "whiten" in (self.model_extra or {}):
            raise ValueError(
                "Kilosort4 has no 'whiten' parameter; it whitens internally. "
                "Remove 'whiten' -- SpikeInterface's Kilosort4 wrapper does "
                "not accept it. "
                "Control whitening via skip_kilosort_preprocessing / "
                "whitening_range."
            )
        return self


class ClusterlessThresholderSchema(BaseModel):
    """Validated schema for the clusterless-thresholder special case.

    A Spyglass peak-detection path on SI ``detect_peaks``, not an SI sorter.

    ``threshold_unit`` sets how ``detect_threshold`` is read:

    - ``"uv"`` (default): microvolts. The runtime scales the recording to
      uV with the stored NWB gain and uses ``noise_levels=[1.0]``.
    - ``"mad"``: multiples of per-channel MAD estimated by SI; suits
      synthetic / low-amplitude data.

    An explicit ``noise_levels`` is used verbatim, in uV under ``"uv"`` and
    native units under ``"mad"``.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[4] = 4
    # 100 uV; must pair with threshold_unit='uv' (100x MAD detects nothing).
    detect_threshold: float = Field(default=100.0, gt=0.0)
    threshold_unit: Literal["uv", "mad"] = Field(
        default="uv",
        description=(
            "How detect_threshold is interpreted when noise_levels is "
            "left unset: 'uv' (the default) derives noise_levels=[1.0] AND "
            "the runtime scales the recording to microvolts (scale_to_uV, "
            "using the stored NWB gain) before detection, so detect_threshold "
            "is a TRUE microvolt threshold (for unity-gain Frank-lab data it "
            "equals the raw-count value); 'mad' lets SpikeInterface "
            "estimate per-channel MAD (a multiplier). The default 'uv' pairs "
            "with the "
            "default detect_threshold=100 to give the production/real-data "
            "threshold; the simulation fixture opts into 'mad' explicitly. "
            "When noise_levels IS set explicitly it takes precedence and is "
            "used verbatim, but threshold_unit still sets the units it is "
            "read in: 'uv' scales the recording to microvolts first, 'mad' "
            "keeps the native units. The advanced override bypasses the "
            "implausible-MAD guard below. "
            "Without an explicit noise_levels, a 'mad' multiplier above ~50 "
            "is rejected as a microvolt threshold left in the wrong unit."
        ),
    )
    # The runtime always passes ``radius_um``, which other SI methods reject.
    method: Literal["locally_exclusive"] = "locally_exclusive"
    peak_sign: Literal["neg", "pos", "both"] = "neg"
    exclude_sweep_ms: float = Field(default=0.1, gt=0.0)
    local_radius_um: float = Field(default=100.0, gt=0.0)
    noise_levels: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def _guard_implausible_mad_threshold(self):
        """Reject a microvolt-scale threshold left in MAD units.

        Such a threshold yields a silent zero-unit sort. Skipped when
        ``noise_levels`` is explicit.
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

    A dedicated dispatch slot; fields are uncurated and SI validates them.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1


class Tridesclous2Schema(BaseModel):
    """Validated schema for Tridesclous 2.

    A dedicated dispatch slot; fields are uncurated and SI validates them.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1


class GenericSorterParamsSchema(BaseModel):
    """Fallback schema for installed SI sorters with no dedicated model.

    Users supply the full SI kwargs dict; SI validates at sort time.
    """

    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1


class SorterExecutionParamsSchema(BaseModel):
    """Tracked sorter-execution backend (local vs Docker/Singularity).

    Stored on ``SorterParameters.execution_params``, never in ``params`` or
    ``job_kwargs``, so a containerized row is reproducible provenance.

    - ``backend="local"``: ``container_image`` is ``None`` and container-only
      controls keep their defaults.
    - ``backend`` docker / singularity: requires an explicit image name or
      ``.sif`` path; SI's ``True`` ("default image") is not provenance.

    ``installation_mode="auto"`` without ``spikeinterface_version`` drifts with
    the host. Reproducible rows use ``"no-install"`` with a baked image, or
    ``"pypi"`` / ``"github"`` with a pinned version. SI's ``folder`` / ``dev``
    modes are not exposed.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    backend: Literal["local", "docker", "singularity"] = "local"
    container_image: str | None = None
    delete_container_files: bool = True
    installation_mode: Literal["auto", "pypi", "github", "no-install"] = "auto"
    spikeinterface_version: str | None = None
    extra_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_backend(self):
        """Cross-check backend against image + container-only install controls."""
        if self.backend == "local":
            if self.container_image is not None:
                raise ValueError(
                    "SorterExecutionParamsSchema: backend='local' requires "
                    f"container_image=None; got {self.container_image!r}. A "
                    "containerized row must set backend='docker' or "
                    "'singularity'."
                )
            # Container-only controls would be silently inert on a local row.
            if (
                self.installation_mode != "auto"
                or self.spikeinterface_version is not None
                or self.extra_requirements
                or self.delete_container_files is not True
            ):
                raise ValueError(
                    "SorterExecutionParamsSchema: installation_mode / "
                    "spikeinterface_version / extra_requirements / "
                    "delete_container_files are container-only; backend='local' "
                    "must leave them at their defaults ('auto' / None / [] / "
                    "True). Set backend='docker' or 'singularity' to pin a "
                    "container install."
                )
        else:
            if not (
                isinstance(self.container_image, str)
                and self.container_image.strip()
            ):
                raise ValueError(
                    f"SorterExecutionParamsSchema: backend={self.backend!r} "
                    "requires a non-empty container_image string (a Docker "
                    "image name/tag/digest, or a local .sif path for "
                    f"Singularity); got {self.container_image!r}. "
                    "SpikeInterface's docker_image=True / singularity_image=True "
                    "is not accepted as tracked provenance -- pin an explicit "
                    "image."
                )
            if (
                self.installation_mode in ("pypi", "github")
                and self.spikeinterface_version is None
            ):
                raise ValueError(
                    "SorterExecutionParamsSchema: installation_mode="
                    f"{self.installation_mode!r} installs SpikeInterface into "
                    "the container at run time and so requires an explicit "
                    "spikeinterface_version (otherwise the install floats with "
                    "the host environment and the row is not reproducible by "
                    "content). Pin spikeinterface_version, or use "
                    "installation_mode='no-install' with a baked image."
                )
        return self


_SORTER_SCHEMAS: dict[str, type[BaseModel]] = {
    "mountainsort4": MountainSort4Schema,
    "mountainsort5": MountainSort5Schema,
    "kilosort4": Kilosort4Schema,
    "spykingcircus2": SpykingCircus2Schema,
    "tridesclous2": Tridesclous2Schema,
    "clusterless_thresholder": ClusterlessThresholderSchema,
}

# The params key that switches each sorter's own motion correction (SI 0.104).
_INTERNAL_MOTION_CORRECTION_KEY: dict[str, str] = {
    "spykingcircus2": "apply_motion_correction",
    "tridesclous2": "apply_motion_correction",
    "kilosort4": "do_correction",
    "kilosort2_5": "do_correction",
    "kilosort3": "do_correction",
}
# A params key that must also be true for the correction key to take effect.
_INTERNAL_MOTION_CORRECTION_GATE: dict[str, str] = {
    "tridesclous2": "apply_preprocessing",
}
# Sorters with no motion correction of their own.
_NO_INTERNAL_MOTION_CORRECTION: frozenset[str] = frozenset(
    {"mountainsort4", "mountainsort5", "clusterless_thresholder"}
)


def sorter_applies_internal_motion_correction(sorter: str, params) -> bool:
    """Return whether a sorter corrects motion itself with these params.

    Missing keys take SI's defaults. A key missing from both counts as on,
    since KS4 reports no defaults without ``kilosort`` installed. A truthy
    key counts as on even where SC2 may skip correction for the probe.

    Parameters
    ----------
    sorter : str
        The ``sorter`` of a ``SorterParameters`` row.
    params : Mapping
        That row's ``params`` blob.

    Returns
    -------
    bool
        ``True`` if the sorter would correct motion internally.

    Raises
    ------
    ValueError
        If the sorter's own motion behavior is not known here.
    """
    if sorter in _NO_INTERNAL_MOTION_CORRECTION:
        return False
    if sorter not in _INTERNAL_MOTION_CORRECTION_KEY:
        raise ValueError(
            f"The motion-correction behavior of sorter {sorter!r} is not "
            "known, so it cannot sort a motion-corrected recording. Supported "
            f"sorters: {sorted(_NO_INTERNAL_MOTION_CORRECTION)} (no internal "
            f"correction) and {sorted(_INTERNAL_MOTION_CORRECTION_KEY)} (with "
            "their internal correction turned off)."
        )
    params = params or {}
    defaults = None

    def resolved(name: str) -> bool:
        nonlocal defaults
        if name in params:
            return bool(params[name])
        if defaults is None:
            import spikeinterface.sorters as sis

            defaults = sis.get_default_sorter_params(sorter)
        return bool(defaults.get(name, True))

    gate = _INTERNAL_MOTION_CORRECTION_GATE.get(sorter)
    if gate is not None and not resolved(gate):
        return False
    return resolved(_INTERNAL_MOTION_CORRECTION_KEY[sorter])


def reject_internal_motion_correction(
    sorter: str, params, *, sorter_params_name: str
) -> None:
    """Raise if a sorter would correct an already corrected recording again.

    Parameters
    ----------
    sorter : str
        The ``sorter`` of a ``SorterParameters`` row.
    params : Mapping
        That row's ``params`` blob.
    sorter_params_name : str
        That row's name, for the message.

    Raises
    ------
    ValueError
        If :func:`sorter_applies_internal_motion_correction` is ``True``, or
        the sorter's motion behavior is not known.
    """
    if sorter_applies_internal_motion_correction(sorter, params):
        key = _INTERNAL_MOTION_CORRECTION_KEY[sorter]
        raise ValueError(
            f"SorterParameters ({sorter!r}, {sorter_params_name!r}) runs "
            f"{sorter}'s own motion correction ({key!r} is on, set or by "
            "default), so it would correct a motion-corrected recording a "
            f"second time. Insert a SorterParameters row with {key}=False "
            "and select that row."
        )


# SI execution / container-install kwargs; only valid in ``execution_params``.
EXECUTION_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "docker_image",
        "singularity_image",
        "delete_container_files",
        "installation_mode",
        "spikeinterface_version",
        "spikeinterface_folder_source",
        "extra_requirements",
    }
)


def reject_reserved_execution_keys(params, *, context: str) -> None:
    """Raise if a scientific ``params`` / ``job_kwargs`` blob holds exec keys.

    Needed because the ``extra="allow"`` sorter schemas would pass these keys
    through. Execution provenance belongs in ``execution_params``.

    Parameters
    ----------
    params : Mapping or None
        The blob to scan (a sorter ``params`` blob or a ``job_kwargs`` blob).
        ``None`` / empty is a no-op.
    context : str
        Human-readable label for the blob (e.g. ``"sorter params blob"``),
        used in the error message.

    Raises
    ------
    ValueError
        If ``params`` carries any :data:`EXECUTION_RESERVED_KEYS` member.
    """
    if not params:
        return
    present = sorted(EXECUTION_RESERVED_KEYS & set(params))
    if present:
        raise ValueError(
            f"{context} must not carry sorter-execution keys {present}: "
            "container backend and container-side install provenance are "
            "tracked on SorterParameters.execution_params (validated by "
            "SorterExecutionParamsSchema), not in the scientific params blob or "
            "job_kwargs. Move these to the execution_params blob."
        )


def sorter_wrapper_vocabulary(sorter: str) -> "frozenset[str] | None":
    """Return the scientific parameter names the installed SI wrapper accepts.

    Reads the wrapper's ``_dynamic_params``. Returns ``None`` ("cannot check")
    for a non-SI sorter or one whose runtime is not installed (Kilosort4).

    Parameters
    ----------
    sorter : str
        SpikeInterface sorter name (e.g. ``"mountainsort5"``).

    Returns
    -------
    frozenset[str] or None
        The accepted top-level parameter names, or ``None`` if unknowable.
    """
    import spikeinterface.sorters as sis

    sorter_class = sis.sorter_dict.get(sorter)
    if sorter_class is None:
        return None
    default_params, _descriptions = sorter_class._dynamic_params()
    if not default_params:
        return None
    return frozenset(default_params)


def validate_sorter_params_against_wrapper(sorter: str, params) -> None:
    """Raise if ``params`` carries a key the installed SI wrapper rejects.

    Catches typos for the ``extra="allow"`` schemas at insert and preflight
    rather than after the recording stage. A ``None`` vocabulary skips the
    check.

    Parameters
    ----------
    sorter : str
        The ``sorter`` column value of a ``SorterParameters`` row.
    params : Mapping
        The scientific ``params`` blob (``schema_version`` is ignored).

    Raises
    ------
    ValueError
        If any key is outside the wrapper's accepted parameter names.
    """
    vocabulary = sorter_wrapper_vocabulary(sorter)
    if vocabulary is None:
        return
    unknown = sorted(set(params or {}) - vocabulary - {"schema_version"})
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(vocabulary), n=1)
        hints.append(
            f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
        )
    raise ValueError(
        f"SorterParameters params for sorter {sorter!r} carry key(s) the "
        f"installed SpikeInterface {sorter!r} wrapper does not accept: "
        f"{', '.join(hints)}. Accepted parameters: {sorted(vocabulary)}."
    )


def validate_execution_params(execution_params) -> dict:
    """Validate a ``SorterParameters.execution_params`` blob.

    ``None`` resolves to the default (local execution).

    Parameters
    ----------
    execution_params : Mapping or None
        The raw ``execution_params`` blob (or ``None`` to mean default local).

    Returns
    -------
    dict
        The validated, normalized :class:`SorterExecutionParamsSchema` dump.
    """
    return SorterExecutionParamsSchema.model_validate(
        execution_params or {}
    ).model_dump()


def _get_sorter_schema(sorter_name: str) -> type[BaseModel]:
    """Return the Pydantic schema for a sorter name.

    Falls back to ``GenericSorterParamsSchema``. ``SorterParameters.insert1``
    checks that the sorter name exists.

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
