"""Motion acceptance benchmark: manifest schema, metric table and gate check.

A benchmark *manifest* is a JSON file that pins everything one benchmark run
depends on: the seeds, the recording generator, the probe, the filter, the
drift scenarios, the correction recipes, the case grid, the sorter, the
ground-truth comparison, the evaluation windows and, for a held-out run, the
numeric gates. :func:`load_manifest` validates one against
:class:`AcceptanceManifest`. A case runner turns each
``(scenario, seed, recipe)`` case into a result JSON;
:func:`case_metrics` reduces a result to one :class:`CaseMetrics` row and
:func:`check_gates` evaluates a table of rows against :class:`Gates`.

Two purposes are allowed. A ``development`` manifest uses only seeds below
:data:`HELD_OUT_SEED_MIN` and carries no gates (gates derived from development
runs would be circular for them). A ``held_out`` manifest uses only seeds at
or above it and must carry gates, so held-out seeds are never run without
gates fixed in advance.

DB-FREE and light at import (pydantic only; the v2 parameter schemas are
imported when a manifest is validated), so the gate-check unit tests run in the
per-PR unit shard.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Seeds at or above this value are reserved for held-out runs.
HELD_OUT_SEED_MIN = 1000

#: The committed development manifest (seeds 0-2, no gates).
DEVELOPMENT_MANIFEST = Path(__file__).with_name(
    "motion_acceptance_development.json"
)

#: Template displacements of step scenarios: every 1 um within +/-15 um
#: (``_motion_fixtures._step_displacement_data``'s default templates).
STEP_TEMPLATE_RANGE_UM = 15.0


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_windows(windows) -> None:
    previous_end = 0.0
    for start, end in windows:
        if not previous_end <= start < end:
            raise ValueError(
                "windows_s must be sorted, disjoint, non-empty and start at "
                f"or after 0; got {windows}."
            )
        previous_end = end


class GeneratorSpec(_Model):
    """``generate_drifting_recording`` arguments pinned by the manifest.

    Every other generator argument stays at its SpikeInterface default.
    """

    num_units: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    sampling_frequency: float = Field(gt=0)
    displacement_sampling_frequency: float = Field(gt=0)


class ProbeSpec(_Model):
    """One single-column shank (``_motion_fixtures.polymer_shank_probe``)."""

    n_contacts: int = Field(gt=1)
    pitch_um: float = Field(gt=0)
    contact_radius_um: float = Field(gt=0)


class BandpassSpec(_Model):
    """SpikeInterface ``bandpass_filter`` corner frequencies (float64)."""

    freq_min: float = Field(gt=0)
    freq_max: float = Field(gt=0)


class ArtifactSpec(_Model):
    """Planted stationary transients, masked before estimation and sorting.

    ``_motion_fixtures.plant_artifact_bursts``: inside each window a 1 ms
    biphasic transient of ``amplitude_uv`` fires at ``rate_hz`` on the top
    eight contacts; each window's frames plus 5 ms either side are masked.
    """

    windows_s: list[tuple[float, float]] = Field(min_length=1)
    rate_hz: float = Field(gt=0)
    amplitude_uv: float = Field(gt=0)

    @model_validator(mode="after")
    def _windows(self):
        _check_windows(self.windows_s)
        return self


class StaticScenario(_Model):
    """No motion: the static twin of a rigid zigzag generated with
    ``amplitude_um`` (so it shares the drifting cases' units and spikes)."""

    kind: Literal["static"]
    amplitude_um: float = Field(gt=0)


class ZigzagScenario(_Model):
    """SpikeInterface zigzag drift, one period over the recording.

    ``+/-amplitude_um`` for the deepest unit; ``non_rigid_gradient`` scales
    the most superficial unit's displacement (``None``: rigid). With
    ``masked_artifacts`` the drift must be rigid (the source-clock motion
    error of masked scenarios uses a rigid truth).
    """

    kind: Literal["zigzag"]
    amplitude_um: float = Field(gt=0)
    non_rigid_gradient: float | None = Field(default=None, ge=0, le=1)
    masked_artifacts: ArtifactSpec | None = None

    @model_validator(mode="after")
    def _rigid_when_masked(self):
        if self.masked_artifacts is not None and (
            self.non_rigid_gradient is not None
        ):
            raise ValueError(
                "a zigzag scenario with masked_artifacts must be rigid "
                "(non_rigid_gradient null)."
            )
        return self


class StepScenario(_Model):
    """Rigid displacement steps, optionally cut to windows and members.

    Displacement is ``levels_um[0]`` until ``change_times_s[0]``, then
    ``levels_um[1]``, and so on. Without ``windows_s`` the whole generated
    recording (``generator.duration_s``) is kept. With ``windows_s`` a
    recording lasting until the last window's end is generated, bandpassed
    whole, and only the windows are kept: one recording whose acquisition
    gaps are the removed time (its continuity comes from its timestamps). With
    ``members`` the windows are grouped, in order, into concatenation members
    (each member one recording with its own internal gaps; every member join
    is a continuity edge).
    """

    kind: Literal["step"]
    change_times_s: list[float]
    levels_um: list[float] = Field(min_length=2)
    windows_s: list[tuple[float, float]] | None = None
    members: list[list[int]] | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if len(self.levels_um) != len(self.change_times_s) + 1:
            raise ValueError(
                "levels_um must have one more entry than change_times_s."
            )
        if list(self.change_times_s) != sorted(self.change_times_s):
            raise ValueError("change_times_s must be sorted.")
        for level in self.levels_um:
            if level != round(level) or abs(level) > STEP_TEMPLATE_RANGE_UM:
                raise ValueError(
                    "step levels must be whole micrometres within "
                    f"+/-{STEP_TEMPLATE_RANGE_UM:g} um (the template grid); "
                    f"got {level}."
                )
        if self.windows_s is not None:
            _check_windows(self.windows_s)
        if self.members is not None:
            if self.windows_s is None:
                raise ValueError("members require windows_s.")
            flat = [i for member in self.members for i in member]
            if flat != list(range(len(self.windows_s))) or not all(
                self.members
            ):
                raise ValueError(
                    "members must group every window index once, in order, "
                    f"into non-empty members; got {self.members}."
                )
        return self


Scenario = Annotated[
    Union[StaticScenario, ZigzagScenario, StepScenario],
    Field(discriminator="kind"),
]


def _validated_interpolation(params: dict) -> dict:
    from spyglass.spikesorting.v2._params.motion_interpolation import (
        MotionInterpolationParamsSchema,
    )

    MotionInterpolationParamsSchema.model_validate(params)
    return params


class OffRecipe(_Model):
    """No correction (the artifact mask, if any, still applies)."""

    kind: Literal["off"]


class OracleRecipe(_Model):
    """The ground-truth displacement applied with ``interpolation``.

    ``max_gap_s`` caps gaps on the estimation clock the truth is placed on.
    """

    kind: Literal["oracle"]
    interpolation: dict
    max_gap_s: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _params(self):
        _validated_interpolation(self.interpolation)
        return self


class EstimateRecipe(_Model):
    """Estimate with ``estimation`` and apply with ``interpolation``.

    ``estimation`` is a ``MotionEstimationParameters`` params blob without
    ``noise_levels_seed``, which the manifest's ``noise_levels_seed`` sets.
    """

    kind: Literal["estimate"]
    estimation: dict
    interpolation: dict

    @model_validator(mode="after")
    def _params(self):
        from spyglass.spikesorting.v2._params.motion_estimation import (
            MotionEstimationParamsSchema,
        )

        if "noise_levels_seed" in self.estimation:
            raise ValueError(
                "estimation must not set noise_levels_seed; the manifest's "
                "noise_levels_seed sets it."
            )
        MotionEstimationParamsSchema.model_validate(self.estimation)
        _validated_interpolation(self.interpolation)
        return self


Recipe = Annotated[
    Union[OffRecipe, OracleRecipe, EstimateRecipe],
    Field(discriminator="kind"),
]


class SorterSpec(_Model):
    """MountainSort5 through the v2 sorting stage (external float64
    whitening seeded with ``random_seed``); ``params`` override the
    ``MountainSort5Schema`` defaults."""

    name: Literal["mountainsort5"]
    params: dict = Field(default_factory=dict)
    random_seed: int = Field(ge=0)


class ComparisonSpec(_Model):
    """``compare_sorter_to_ground_truth(exhaustive_gt=True)`` scores.

    ``oversplit_agreement``: a ground-truth unit is oversplit when two or
    more sorted units agree with it above this score.
    """

    delta_time_ms: float = Field(gt=0)
    match_score: float = Field(gt=0, le=1)
    well_detected_score: float = Field(gt=0, le=1)
    redundant_score: float = Field(gt=0, le=1)
    overmerged_score: float = Field(gt=0, le=1)
    oversplit_agreement: float = Field(gt=0, le=1)


class EvaluationSpec(_Model):
    """Corrected-signal windows and SpikeInterface job kwargs.

    ``fidelity_n_windows`` windows of ``fidelity_window_s`` evenly spaced
    from one window after the start to two windows before the end.
    """

    fidelity_n_windows: int = Field(gt=0)
    fidelity_window_s: float = Field(gt=0)
    job_kwargs: dict


class CaseSpec(_Model):
    """One scenario and the recipes run on it, for every manifest seed."""

    scenario: str
    recipes: list[str] = Field(min_length=1)


class MotionErrorGate(_Model):
    """Per case: motion RMS and 95th-percentile |error| bounds (um)."""

    rms_um: float = Field(ge=0)
    p95_um: float = Field(ge=0)


class FidelityGate(_Model):
    """Per case: pooled noise-free residual bounds (any may be null)."""

    max_residual: float | None = Field(default=None, ge=0)
    max_excess_over_oracle: float | None = None
    max_ratio_to_uncorrected: float | None = Field(default=None, ge=0)


class NoMotionSortingGate(_Model):
    """Sorting on a no-motion scenario versus ``off`` of the same seed."""

    max_accuracy_drop: float = Field(ge=0)
    max_well_detected_drop: int = Field(ge=0)
    max_false_positive_increase: int = Field(ge=0)
    max_overmerged: int = Field(ge=0)


class DriftingSortingGate(_Model):
    """Sorting on a drifting scenario versus ``off`` and ``oracle``.

    Always also checked: per seed, accuracy and well-detected count at least
    ``off``'s, and oversplit ground-truth units at most ``off``'s.
    """

    min_mean_accuracy_gain: float
    max_oracle_accuracy_gap_per_seed: float
    max_mean_oracle_accuracy_gap: float
    max_overmerged: int = Field(ge=0)
    max_false_positive_excess: int = Field(ge=0)


class SortingGates(_Model):
    no_motion: dict[str, NoMotionSortingGate] = Field(default_factory=dict)
    drifting: dict[str, DriftingSortingGate] = Field(default_factory=dict)


class CostGates(_Model):
    """Estimation wall time per recipe (s) and whole-case peak RSS (GiB)."""

    max_estimation_s: dict[str, float]
    max_peak_rss_gib: float = Field(gt=0)


class Gates(_Model):
    """Numeric acceptance gates for the recipes in ``recipes``.

    Motion, fidelity, sorting and sign gates are per scenario. Border and
    cost gates apply to every case of a gated recipe on a scenario named in
    ``motion``: a ``force_extrapolate`` recipe keeps every contact; a
    ``remove_channels`` recipe removes exactly the contacts SpikeInterface's
    rule predicts from the estimate and keeps at least one.
    """

    recipes: list[str] = Field(min_length=1)
    motion: dict[str, MotionErrorGate]
    min_sign_corr: dict[str, float] = Field(default_factory=dict)
    fidelity: dict[str, FidelityGate] = Field(default_factory=dict)
    sorting: SortingGates = Field(default_factory=SortingGates)
    cost: CostGates


class AcceptanceManifest(_Model):
    """A validated motion acceptance benchmark manifest (see module doc)."""

    name: str
    purpose: Literal["development", "held_out"]
    seeds: list[int] = Field(min_length=1)
    generator: GeneratorSpec
    probe: ProbeSpec
    bandpass: BandpassSpec
    noise_levels_seed: Literal["case_seed"] | int
    scenarios: dict[str, Scenario]
    recipes: dict[str, Recipe]
    cases: list[CaseSpec] = Field(min_length=1)
    sorter: SorterSpec
    comparison: ComparisonSpec
    evaluation: EvaluationSpec
    gates: Gates | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError(f"seeds must be unique and >= 0; {self.seeds}.")
        if self.purpose == "development":
            if max(self.seeds) >= HELD_OUT_SEED_MIN:
                raise ValueError(
                    f"a development manifest may not use seeds >= "
                    f"{HELD_OUT_SEED_MIN} (reserved for held-out runs); got "
                    f"{self.seeds}."
                )
            if self.gates is not None:
                raise ValueError(
                    "a development manifest carries no gates: gates derived "
                    "from development runs cannot test them."
                )
        else:
            if min(self.seeds) < HELD_OUT_SEED_MIN:
                raise ValueError(
                    f"a held-out manifest uses only seeds >= "
                    f"{HELD_OUT_SEED_MIN}; got {self.seeds}."
                )
            if self.gates is None:
                raise ValueError(
                    "a held-out manifest must carry its gates, fixed before "
                    "the run."
                )
        kinds = {name: r.kind for name, r in self.recipes.items()}
        if list(kinds.values()).count("off") != 1:
            raise ValueError("exactly one recipe must be of kind 'off'.")
        if list(kinds.values()).count("oracle") > 1:
            raise ValueError("at most one recipe may be of kind 'oracle'.")
        seen = set()
        for case in self.cases:
            if case.scenario not in self.scenarios:
                raise ValueError(f"case names unknown scenario {case.scenario}")
            if case.scenario in seen:
                raise ValueError(f"scenario {case.scenario} listed twice.")
            seen.add(case.scenario)
            unknown = sorted(set(case.recipes) - set(self.recipes))
            if unknown or len(set(case.recipes)) != len(case.recipes):
                raise ValueError(
                    f"case {case.scenario} names unknown or repeated recipes "
                    f"{case.recipes}."
                )
        if self.gates is not None:
            self._check_gates_reference_cases(kinds)
        return self

    def _check_gates_reference_cases(self, kinds: dict) -> None:
        gates = self.gates
        grid = {case.scenario: set(case.recipes) for case in self.cases}
        for recipe in gates.recipes:
            if kinds.get(recipe) != "estimate":
                raise ValueError(f"gated recipe {recipe} is not an estimate.")
            if recipe not in gates.cost.max_estimation_s:
                raise ValueError(f"no estimation-time gate for {recipe}.")
        needs = {s: {"self"} for s in gates.motion}
        for s in gates.min_sign_corr:
            needs.setdefault(s, set()).add("self")
        for s, g in gates.fidelity.items():
            needs.setdefault(s, set()).add("self")
            if g.max_excess_over_oracle is not None:
                needs[s].add("oracle")
        for s in gates.sorting.no_motion:
            needs.setdefault(s, set()).update({"self", "off"})
        for s in gates.sorting.drifting:
            needs.setdefault(s, set()).update({"self", "off", "oracle"})
        off = self.off_recipe
        oracle = self.oracle_recipe
        for scenario, roles in needs.items():
            if scenario not in grid:
                raise ValueError(
                    f"gates name scenario {scenario} not in cases."
                )
            required = set()
            if "self" in roles:
                required |= set(gates.recipes)
            if "off" in roles:
                required.add(off)
            if "oracle" in roles:
                if oracle is None:
                    raise ValueError(
                        f"gates on {scenario} need an oracle recipe."
                    )
                required.add(oracle)
            missing = sorted(required - grid[scenario])
            if missing:
                raise ValueError(
                    f"gates on {scenario} need cases for recipes {missing}."
                )

    @property
    def off_recipe(self) -> str:
        return next(n for n, r in self.recipes.items() if r.kind == "off")

    @property
    def oracle_recipe(self) -> str | None:
        return next(
            (n for n, r in self.recipes.items() if r.kind == "oracle"), None
        )

    def iter_cases(self):
        """Yield every ``(scenario, seed, recipe)`` case, seed-major."""
        for seed in self.seeds:
            for case in self.cases:
                for recipe in case.recipes:
                    yield case.scenario, seed, recipe


def load_manifest(path) -> AcceptanceManifest:
    """Read and validate a benchmark manifest JSON file."""
    return AcceptanceManifest.model_validate(json.loads(Path(path).read_text()))


def manifest_sha256(path) -> str:
    """SHA-256 of a manifest file's bytes (recorded in every case result)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_tag(scenario: str, seed: int, recipe: str) -> str:
    """File stem of one case's result."""
    return f"{scenario}__s{seed}__{recipe}"


class CaseMetrics(NamedTuple):
    """One case's gated quantities, reduced from its result JSON.

    Motion fields are ``None`` for ``off``; sign correlation is ``None`` when
    the truth or the estimate is constant. Fidelity sums are per output
    channel: ``fidelity_num[i]`` is ``sum_t (corrected - static)^2`` and
    ``fidelity_den[i]`` is ``sum_t static^2`` on the noise-free twins over
    the evaluation windows, for channel ``fidelity_channel_ids[i]``.
    """

    scenario: str
    seed: int
    recipe: str
    border_mode: str | None
    n_contacts: int
    motion_rms_um: float | None
    motion_p95_um: float | None
    sign_corr: float | None
    fidelity_channel_ids: tuple
    fidelity_num: tuple
    fidelity_den: tuple
    uncorrected_residual: float
    n_out_channels: int
    removed_channel_ids: tuple
    predicted_removed_channel_ids: tuple | None
    mean_accuracy: float
    n_well_detected: int
    n_overmerged: int
    n_false_positive: int
    n_gt_oversplit: int
    estimation_s: float | None
    peak_rss_gib: float


def case_metrics(result: dict) -> CaseMetrics:
    """Reduce one case result (``_motion_acceptance_run.run_case``)."""
    motion = result.get("motion") or {}
    fidelity = result["fidelity_signal"]
    border = result["border"]
    sorting = result["sorting"]
    predicted = border.get("predicted_removed_channel_ids")
    return CaseMetrics(
        scenario=result["scenario"],
        seed=int(result["seed"]),
        recipe=result["recipe"],
        border_mode=border.get("border_mode"),
        n_contacts=int(border["n_in_channels"]),
        motion_rms_um=motion.get("rms_um"),
        motion_p95_um=motion.get("p95_um"),
        sign_corr=motion.get("sign_corr"),
        fidelity_channel_ids=tuple(fidelity["corrected"]["channel_ids"]),
        fidelity_num=tuple(fidelity["corrected"]["num"]),
        fidelity_den=tuple(fidelity["corrected"]["den"]),
        uncorrected_residual=float(fidelity["uncorrected"]["pooled"]),
        n_out_channels=int(border["n_out_channels"]),
        removed_channel_ids=tuple(border["removed_channel_ids"]),
        predicted_removed_channel_ids=(
            None if predicted is None else tuple(predicted)
        ),
        mean_accuracy=float(sorting["mean_accuracy"]),
        n_well_detected=int(sorting["n_well_detected"]),
        n_overmerged=int(sorting["n_overmerged"]),
        n_false_positive=int(sorting["n_false_positive"]),
        n_gt_oversplit=int(sorting["n_gt_oversplit"]),
        estimation_s=result["timings_s"].get("estimate"),
        peak_rss_gib=float(result["peak_rss_bytes"]) / 2**30,
    )


def pooled_residual(row: CaseMetrics, channel_ids=None) -> float:
    """``sqrt(sum num / sum den)`` over ``channel_ids`` (default: the row's
    own output channels).

    Raises
    ------
    ValueError
        If a requested channel is not among the row's channels.
    """
    index = {c: i for i, c in enumerate(row.fidelity_channel_ids)}
    ids = row.fidelity_channel_ids if channel_ids is None else channel_ids
    missing = [c for c in ids if c not in index]
    if missing:
        raise ValueError(
            f"{row.scenario} seed {row.seed} {row.recipe}: no fidelity sums "
            f"for channels {missing}."
        )
    num = sum(row.fidelity_num[index[c]] for c in ids)
    den = sum(row.fidelity_den[index[c]] for c in ids)
    return math.sqrt(num / den)


class GateResult(NamedTuple):
    """One evaluated gate. ``seed`` is ``None`` for a mean over seeds."""

    check: str
    scenario: str
    recipe: str
    seed: int | None
    value: float
    limit: float
    passed: bool


def check_gates(
    rows,
    gates: Gates,
    *,
    seeds,
    off_recipe: str,
    oracle_recipe: str | None,
) -> list[GateResult]:
    """Evaluate ``gates`` on a metric table.

    Parameters
    ----------
    rows : iterable of CaseMetrics
        One row per ``(scenario, seed, recipe)`` case.
    gates : Gates
    seeds : iterable of int
        The seeds every gated scenario must have been run on.
    off_recipe, oracle_recipe : str
        Keyword-only. The recipes paired with each gated case.

    Returns
    -------
    list[GateResult]
        Every evaluated gate, passed or failed.

    Raises
    ------
    ValueError
        If a case a gate needs is missing, a case appears twice, or a gated
        quantity is missing (for example no motion error on a gated recipe).
    """
    table: dict = {}
    for row in rows:
        key = (row.scenario, row.seed, row.recipe)
        if key in table:
            raise ValueError(f"case {key} appears twice in the metric table.")
        table[key] = row
    seeds = list(seeds)
    results: list[GateResult] = []

    def get(scenario, seed, recipe) -> CaseMetrics:
        try:
            return table[(scenario, seed, recipe)]
        except KeyError:
            raise ValueError(
                f"gates need case {scenario} seed {seed} {recipe}, which is "
                "missing from the metric table."
            ) from None

    def value(x, what, row):
        if x is None:
            raise ValueError(
                f"{row.scenario} seed {row.seed} {row.recipe} has no {what}."
            )
        return float(x)

    def add(check, scenario, recipe, seed, v, limit, passed):
        results.append(
            GateResult(
                check, scenario, recipe, seed, float(v), float(limit), passed
            )
        )

    for recipe in gates.recipes:
        for scenario, g in gates.motion.items():
            for seed in seeds:
                row = get(scenario, seed, recipe)
                rms = value(row.motion_rms_um, "motion RMS", row)
                p95 = value(row.motion_p95_um, "motion p95", row)
                add(
                    "motion_rms_um",
                    scenario,
                    recipe,
                    seed,
                    rms,
                    g.rms_um,
                    rms <= g.rms_um,
                )
                add(
                    "motion_p95_um",
                    scenario,
                    recipe,
                    seed,
                    p95,
                    g.p95_um,
                    p95 <= g.p95_um,
                )
                est_s = value(row.estimation_s, "estimation time", row)
                limit = gates.cost.max_estimation_s[recipe]
                add(
                    "estimation_s",
                    scenario,
                    recipe,
                    seed,
                    est_s,
                    limit,
                    est_s <= limit,
                )
                limit = gates.cost.max_peak_rss_gib
                add(
                    "peak_rss_gib",
                    scenario,
                    recipe,
                    seed,
                    row.peak_rss_gib,
                    limit,
                    row.peak_rss_gib <= limit,
                )
                results.append(_border_result(row))

        for scenario, limit in gates.min_sign_corr.items():
            for seed in seeds:
                row = get(scenario, seed, recipe)
                corr = value(row.sign_corr, "sign correlation", row)
                add(
                    "sign_corr",
                    scenario,
                    recipe,
                    seed,
                    corr,
                    limit,
                    corr >= limit,
                )

        for scenario, g in gates.fidelity.items():
            for seed in seeds:
                row = get(scenario, seed, recipe)
                residual = pooled_residual(row)
                if g.max_residual is not None:
                    add(
                        "fidelity_residual",
                        scenario,
                        recipe,
                        seed,
                        residual,
                        g.max_residual,
                        residual <= g.max_residual,
                    )
                if g.max_excess_over_oracle is not None:
                    oracle = get(scenario, seed, oracle_recipe)
                    excess = residual - pooled_residual(
                        oracle, row.fidelity_channel_ids
                    )
                    add(
                        "fidelity_excess_over_oracle",
                        scenario,
                        recipe,
                        seed,
                        excess,
                        g.max_excess_over_oracle,
                        excess <= g.max_excess_over_oracle,
                    )
                if g.max_ratio_to_uncorrected is not None:
                    ratio = residual / row.uncorrected_residual
                    add(
                        "fidelity_ratio_to_uncorrected",
                        scenario,
                        recipe,
                        seed,
                        ratio,
                        g.max_ratio_to_uncorrected,
                        ratio <= g.max_ratio_to_uncorrected,
                    )

        for scenario, g in gates.sorting.no_motion.items():
            for seed in seeds:
                row = get(scenario, seed, recipe)
                off = get(scenario, seed, off_recipe)
                drop = off.mean_accuracy - row.mean_accuracy
                add(
                    "no_motion_accuracy_drop",
                    scenario,
                    recipe,
                    seed,
                    drop,
                    g.max_accuracy_drop,
                    drop <= g.max_accuracy_drop,
                )
                drop = off.n_well_detected - row.n_well_detected
                add(
                    "no_motion_well_detected_drop",
                    scenario,
                    recipe,
                    seed,
                    drop,
                    g.max_well_detected_drop,
                    drop <= g.max_well_detected_drop,
                )
                rise = row.n_false_positive - off.n_false_positive
                add(
                    "no_motion_false_positive_increase",
                    scenario,
                    recipe,
                    seed,
                    rise,
                    g.max_false_positive_increase,
                    rise <= g.max_false_positive_increase,
                )
                add(
                    "overmerged",
                    scenario,
                    recipe,
                    seed,
                    row.n_overmerged,
                    g.max_overmerged,
                    row.n_overmerged <= g.max_overmerged,
                )

        for scenario, g in gates.sorting.drifting.items():
            gains, gaps = [], []
            for seed in seeds:
                row = get(scenario, seed, recipe)
                off = get(scenario, seed, off_recipe)
                oracle = get(scenario, seed, oracle_recipe)
                gain = row.mean_accuracy - off.mean_accuracy
                gains.append(gain)
                add(
                    "accuracy_not_below_off",
                    scenario,
                    recipe,
                    seed,
                    gain,
                    0.0,
                    gain >= 0.0,
                )
                gain_wd = row.n_well_detected - off.n_well_detected
                add(
                    "well_detected_not_below_off",
                    scenario,
                    recipe,
                    seed,
                    gain_wd,
                    0,
                    gain_wd >= 0,
                )
                gap = oracle.mean_accuracy - row.mean_accuracy
                gaps.append(gap)
                add(
                    "oracle_accuracy_gap",
                    scenario,
                    recipe,
                    seed,
                    gap,
                    g.max_oracle_accuracy_gap_per_seed,
                    gap <= g.max_oracle_accuracy_gap_per_seed,
                )
                add(
                    "overmerged",
                    scenario,
                    recipe,
                    seed,
                    row.n_overmerged,
                    g.max_overmerged,
                    row.n_overmerged <= g.max_overmerged,
                )
                add(
                    "oversplit_not_above_off",
                    scenario,
                    recipe,
                    seed,
                    row.n_gt_oversplit,
                    off.n_gt_oversplit,
                    row.n_gt_oversplit <= off.n_gt_oversplit,
                )
                excess = row.n_false_positive - max(
                    off.n_false_positive, oracle.n_false_positive
                )
                add(
                    "false_positive_excess",
                    scenario,
                    recipe,
                    seed,
                    excess,
                    g.max_false_positive_excess,
                    excess <= g.max_false_positive_excess,
                )
            mean_gain = sum(gains) / len(gains)
            add(
                "mean_accuracy_gain",
                scenario,
                recipe,
                None,
                mean_gain,
                g.min_mean_accuracy_gain,
                mean_gain >= g.min_mean_accuracy_gain,
            )
            mean_gap = sum(gaps) / len(gaps)
            add(
                "mean_oracle_accuracy_gap",
                scenario,
                recipe,
                None,
                mean_gap,
                g.max_mean_oracle_accuracy_gap,
                mean_gap <= g.max_mean_oracle_accuracy_gap,
            )
    return results


def _border_result(row: CaseMetrics) -> GateResult:
    """The border gate of one gated case (see :class:`Gates`)."""
    if row.border_mode == "force_extrapolate":
        return GateResult(
            "border_keeps_all_channels",
            row.scenario,
            row.recipe,
            row.seed,
            float(row.n_out_channels),
            float(row.n_contacts),
            row.n_out_channels == row.n_contacts,
        )
    if row.border_mode == "remove_channels":
        if row.predicted_removed_channel_ids is None:
            raise ValueError(
                f"{row.scenario} seed {row.seed} {row.recipe}: no predicted "
                "removed channels for a remove_channels recipe."
            )
        passed = row.n_out_channels >= 1 and set(row.removed_channel_ids) == (
            set(row.predicted_removed_channel_ids)
        )
        return GateResult(
            "border_removes_predicted_channels",
            row.scenario,
            row.recipe,
            row.seed,
            float(len(row.removed_channel_ids)),
            float(len(row.predicted_removed_channel_ids)),
            passed,
        )
    raise ValueError(
        f"{row.scenario} seed {row.seed} {row.recipe}: unknown border mode "
        f"{row.border_mode!r}."
    )


def check_manifest_gates(rows, manifest: AcceptanceManifest):
    """:func:`check_gates` with the manifest's gates, seeds and recipes."""
    if manifest.gates is None:
        raise ValueError(f"manifest {manifest.name} carries no gates.")
    return check_gates(
        rows,
        manifest.gates,
        seeds=manifest.seeds,
        off_recipe=manifest.off_recipe,
        oracle_recipe=manifest.oracle_recipe,
    )
