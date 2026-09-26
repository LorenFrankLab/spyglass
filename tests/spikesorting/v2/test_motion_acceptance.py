"""Motion acceptance benchmark: manifest, gate check and opt-in runs.

The manifest-schema and gate-check tests are DB-free and fast; they run with
every unit-shard pass. The benchmark itself (every case of a manifest, each
in its own process, through the v2 motion, sorting and comparison code) and
the representative polymer-fixture run are opt-in:

- ``SPYGLASS_V2_MOTION_BENCHMARK=1`` runs them (otherwise they skip).
- ``SPYGLASS_V2_MOTION_MANIFEST`` names the manifest (default: the committed
  development manifest, seeds 0-2, no gates).
- ``SPYGLASS_V2_MOTION_BENCHMARK_OUT`` names the result directory (default: a
  pytest temporary directory). A result already there that was produced
  from the same manifest bytes is reused, so an interrupted run resumes.

A development manifest carries no gates, so its benchmark run asserts only
structural facts and writes the metric evidence; a held-out manifest's gates
are checked by ``test_benchmark_meets_manifest_gates``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.spikesorting.v2._motion_acceptance import (
    DEVELOPMENT_MANIFEST,
    AcceptanceManifest,
    CaseMetrics,
    Gates,
    case_metrics,
    case_tag,
    check_gates,
    check_manifest_gates,
    load_manifest,
    manifest_sha256,
    pooled_residual,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = os.environ.get("SPYGLASS_V2_MOTION_BENCHMARK") == "1"
BENCHMARK_SKIP = pytest.mark.skip(
    reason="the motion acceptance benchmark is opt-in: set "
    "SPYGLASS_V2_MOTION_BENCHMARK=1"
)
DRIFT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "mearec_polymer_128ch_drift_120s.nwb"
)
DRIFT_FIXTURE_GT_H5 = (
    REPO_ROOT
    / "tests"
    / "_data"
    / "spikesorting_v2"
    / "mearec_work"
    / "mearec_polymer_128ch_drift_120s.h5"
)


def _dev_dict() -> dict:
    return json.loads(DEVELOPMENT_MANIFEST.read_text())


# ---- manifest ---------------------------------------------------------------


def test_development_manifest_is_development_only():
    manifest = load_manifest(DEVELOPMENT_MANIFEST)

    assert manifest.purpose == "development"
    assert manifest.seeds == [0, 1, 2]
    assert manifest.gates is None
    assert {case.scenario for case in manifest.cases} == set(manifest.scenarios)
    assert manifest.off_recipe == "off"
    assert manifest.oracle_recipe == "oracle"


def test_development_recipes_are_the_shipped_rows_with_a_case_seed():
    """The DREDge recipes are the shipped estimation rows (the manifest sets
    only the noise seed) and every interpolation is a shipped row."""
    from spyglass.spikesorting.v2._recipe_catalog import (
        motion_estimation_default_contents,
        motion_interpolation_default_contents,
    )

    manifest = load_manifest(DEVELOPMENT_MANIFEST)
    shipped = {row[0]: row[1] for row in motion_estimation_default_contents()}
    interpolations = [
        {k: v for k, v in row[1].items() if k != "schema_version"}
        for row in motion_interpolation_default_contents()
    ]
    for recipe, row in (
        ("dredge", "dredge_v1"),
        ("dredge_fast", "dredge_fast_v1"),
    ):
        estimation = dict(shipped[row])
        assert estimation.pop("noise_levels_seed") == 0
        estimation.pop("schema_version")
        spec = manifest.recipes[recipe]
        assert {
            k: v for k, v in estimation.items() if v not in ({}, None)
        } == spec.estimation
    for spec in manifest.recipes.values():
        if spec.kind != "off":
            assert spec.interpolation in interpolations
    assert manifest.noise_levels_seed == "case_seed"


_GATES = {
    "recipes": ["dredge_fast"],
    "motion": {"rigid": {"rms_um": 1.5, "p95_um": 2.5}},
    "min_sign_corr": {"rigid": 0.95},
    "fidelity": {"rigid": {"max_excess_over_oracle": 0.01}},
    "sorting": {
        "no_motion": {
            "none": {
                "max_accuracy_drop": 0.03,
                "max_well_detected_drop": 1,
                "max_false_positive_increase": 2,
                "max_overmerged": 0,
            }
        },
        "drifting": {
            "rigid": {
                "min_mean_accuracy_gain": 0.15,
                "max_oracle_accuracy_gap_per_seed": 0.2,
                "max_mean_oracle_accuracy_gap": 0.1,
                "max_overmerged": 1,
                "max_false_positive_excess": 5,
            }
        },
    },
    "cost": {
        "max_estimation_s": {"dredge_fast": 12.0},
        "max_peak_rss_gib": 3.6,
    },
}


def _held_out(**changes) -> dict:
    """The development manifest turned held-out (schema checks only; seeds
    2000+ are placeholders, nothing is generated)."""
    data = _dev_dict()
    data.update(purpose="held_out", seeds=[2000, 2001], gates=_GATES)
    data.update(changes)
    return data


def test_held_out_manifest_with_gates_validates():
    manifest = AcceptanceManifest.model_validate(_held_out())
    assert manifest.gates.recipes == ["dredge_fast"]


def _with(path: list, value) -> dict:
    data = _dev_dict()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


@pytest.mark.parametrize(
    "data, match",
    [
        (_with(["seeds"], [0, 1, 1000]), "reserved for held-out"),
        (_with(["gates"], _GATES), "carries no gates"),
        (_held_out(gates=None), "must carry its gates"),
        (_held_out(seeds=[2, 2000]), "only seeds >= 1000"),
        (_with(["seeds"], [0, 0]), "unique"),
        (
            _with(["cases", 0, "recipes"], ["off", "unknown"]),
            "unknown or repeated",
        ),
        (
            _with(["scenarios", "jump", "levels_um"], [-15, 15.5]),
            "whole micrometres",
        ),
        (
            _with(["scenarios", "members", "members"], [[1], [0, 2]]),
            "every window index once",
        ),
        (
            _with(["scenarios", "masked_drift", "non_rigid_gradient"], 0.2),
            "must be rigid",
        ),
        (
            _with(
                ["recipes", "dredge", "estimation"],
                {"preset": "dredge", "max_gap_s": 30.0, "noise_levels_seed": 3},
            ),
            "must not set noise_levels_seed",
        ),
        (
            _with(["recipes", "dredge", "interpolation", "border_mode"], "x"),
            "border_mode",
        ),
        (
            _held_out(
                cases=[
                    {"scenario": "none", "recipes": ["off", "dredge_fast"]},
                    {"scenario": "rigid", "recipes": ["off", "dredge_fast"]},
                ]
            ),
            "need cases for recipes",
        ),
    ],
)
def test_invalid_manifest_is_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        AcceptanceManifest.model_validate(data)


def test_manifest_iterates_cases_seed_major():
    manifest = load_manifest(DEVELOPMENT_MANIFEST)
    cases = list(manifest.iter_cases())

    assert cases[0] == ("none", 0, "off")
    assert {seed for _, seed, _ in cases} == {0, 1, 2}
    assert (
        len(cases)
        == len(set(cases))
        == 3 * sum(len(case.recipes) for case in manifest.cases)
    )


# ---- gate check -------------------------------------------------------------

SEEDS = (0, 1)
CHANNELS = ("0", "1", "2")


def _row(scenario, seed, recipe, **changes) -> CaseMetrics:
    """A row that passes every gate in ``_GATES`` for its role."""
    drifting = scenario == "rigid"
    base = dict(
        scenario=scenario,
        seed=seed,
        recipe=recipe,
        border_mode=None if recipe == "off" else "force_extrapolate",
        n_contacts=3,
        motion_rms_um=None if recipe == "off" else 0.5,
        motion_p95_um=None if recipe == "off" else 1.0,
        sign_corr=0.99 if drifting and recipe != "off" else None,
        fidelity_channel_ids=CHANNELS,
        fidelity_num=(0.01, 0.01, 0.01),
        fidelity_den=(1.0, 1.0, 1.0),
        uncorrected_residual=0.4 if drifting else 0.0,
        n_out_channels=3,
        removed_channel_ids=(),
        predicted_removed_channel_ids=None,
        mean_accuracy=0.3 if recipe == "off" and drifting else 0.6,
        n_well_detected=10,
        n_overmerged=0,
        n_false_positive=2,
        n_gt_oversplit=3 if recipe == "off" else 1,
        estimation_s=5.0 if recipe == "dredge_fast" else None,
        peak_rss_gib=2.5,
    )
    if recipe == "oracle":
        base.update(mean_accuracy=0.65, fidelity_num=(0.009,) * 3)
    base.update(changes)
    return CaseMetrics(**base)


def _table(changes=None) -> list[CaseMetrics]:
    """Passing rows for none (off, dredge_fast) and rigid (off, oracle,
    dredge_fast); ``changes`` maps ``(scenario, seed, recipe)`` to field
    overrides."""
    changes = changes or {}
    keys = [
        (scenario, seed, recipe)
        for seed in SEEDS
        for scenario, recipes in (
            ("none", ("off", "dredge_fast")),
            ("rigid", ("off", "oracle", "dredge_fast")),
        )
        for recipe in recipes
    ]
    return [_row(*key, **changes.get(key, {})) for key in keys]


def _check(rows):
    return check_gates(
        rows,
        Gates.model_validate(_GATES),
        seeds=SEEDS,
        off_recipe="off",
        oracle_recipe="oracle",
    )


def test_passing_table_passes_every_gate():
    results = _check(_table())

    assert results and all(r.passed for r in results)
    assert {r.check for r in results} == {
        "motion_rms_um",
        "motion_p95_um",
        "estimation_s",
        "peak_rss_gib",
        "border_keeps_all_channels",
        "sign_corr",
        "fidelity_excess_over_oracle",
        "no_motion_accuracy_drop",
        "no_motion_well_detected_drop",
        "no_motion_false_positive_increase",
        "overmerged",
        "accuracy_not_below_off",
        "well_detected_not_below_off",
        "oracle_accuracy_gap",
        "oversplit_not_above_off",
        "false_positive_excess",
        "mean_accuracy_gain",
        "mean_oracle_accuracy_gap",
    }
    # Per-seed gates are evaluated for every seed, means once.
    per_seed = [r for r in results if r.check == "motion_rms_um"]
    assert sorted(r.seed for r in per_seed) == list(SEEDS)
    assert [r.seed for r in results if r.check == "mean_accuracy_gain"] == [
        None
    ]


R0 = ("rigid", 0, "dredge_fast")
N0 = ("none", 0, "dredge_fast")


@pytest.mark.parametrize(
    "changes, failed",
    [
        ({R0: dict(motion_rms_um=1.6)}, {"motion_rms_um"}),
        ({R0: dict(motion_p95_um=2.6)}, {"motion_p95_um"}),
        ({R0: dict(sign_corr=-0.8)}, {"sign_corr"}),
        ({R0: dict(estimation_s=12.5)}, {"estimation_s"}),
        ({R0: dict(peak_rss_gib=3.7)}, {"peak_rss_gib"}),
        ({R0: dict(n_out_channels=2)}, {"border_keeps_all_channels"}),
        # pooled residual 0.1 + 0.011 over an oracle at sqrt(0.009) = 0.0949.
        (
            {R0: dict(fidelity_num=(0.0113,) * 3)},
            {"fidelity_excess_over_oracle"},
        ),
        (
            {N0: dict(mean_accuracy=0.56)},
            {"no_motion_accuracy_drop"},
        ),
        ({N0: dict(n_well_detected=8)}, {"no_motion_well_detected_drop"}),
        (
            {N0: dict(n_false_positive=5)},
            {"no_motion_false_positive_increase"},
        ),
        ({N0: dict(n_overmerged=1)}, {"overmerged"}),
        ({R0: dict(n_overmerged=2)}, {"overmerged"}),
        # Seed 0 slightly below off; seed 1's large gain keeps the mean up.
        (
            {
                ("rigid", 0, "off"): dict(mean_accuracy=0.62),
                ("rigid", 1, "off"): dict(mean_accuracy=0.1),
            },
            {"accuracy_not_below_off"},
        ),
        # Both seeds gain 0.1: never worse than off, mean gain under 0.15.
        (
            {
                ("rigid", 0, "off"): dict(mean_accuracy=0.5),
                ("rigid", 1, "off"): dict(mean_accuracy=0.5),
            },
            {"mean_accuracy_gain"},
        ),
        ({R0: dict(n_well_detected=9)}, {"well_detected_not_below_off"}),
        # Gap 0.21 on one seed: per-seed gate fails, mean 0.13 fails too.
        (
            {R0: dict(mean_accuracy=0.44)},
            {"oracle_accuracy_gap", "mean_oracle_accuracy_gap"},
        ),
        # Both seeds at gap 0.12: only the mean gate fails.
        (
            {
                R0: dict(mean_accuracy=0.53),
                ("rigid", 1, "dredge_fast"): dict(mean_accuracy=0.53),
            },
            {"mean_oracle_accuracy_gap"},
        ),
        ({R0: dict(n_gt_oversplit=4)}, {"oversplit_not_above_off"}),
        ({R0: dict(n_false_positive=8)}, {"false_positive_excess"}),
    ],
)
def test_each_gate_fails_on_its_own_violation(changes, failed):
    results = _check(_table(changes))

    assert {r.check for r in results if not r.passed} == failed


def test_remove_channels_border_must_match_the_prediction():
    rows = _table(
        {
            R0: dict(
                border_mode="remove_channels",
                n_out_channels=2,
                removed_channel_ids=("0",),
                predicted_removed_channel_ids=("0",),
                fidelity_channel_ids=("1", "2"),
                fidelity_num=(0.01, 0.01),
                fidelity_den=(1.0, 1.0),
            )
        }
    )
    assert all(r.passed for r in _check(rows))

    index = next(i for i, r in enumerate(rows) if r[:3] == R0)
    rows[index] = rows[index]._replace(predicted_removed_channel_ids=("0", "2"))
    wrong = _check(rows)
    assert {r.check for r in wrong if not r.passed} == {
        "border_removes_predicted_channels"
    }


def test_oracle_reference_is_restricted_to_the_output_channels():
    """A remove_channels case compares with the oracle on its own output
    channels: the end contacts, which carry the largest residual, are not in
    the reference."""
    oracle = _row(
        "rigid",
        0,
        "oracle",
        fidelity_num=(0.09, 0.0025, 0.0025),
        fidelity_den=(1.0, 1.0, 1.0),
    )

    assert pooled_residual(oracle) == pytest.approx(math.sqrt(0.095 / 3))
    assert pooled_residual(oracle, ("1", "2")) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="no fidelity sums"):
        pooled_residual(oracle, ("3",))


def test_missing_or_duplicate_cases_raise():
    rows = [r for r in _table() if r[:3] != ("rigid", 1, "oracle")]
    with pytest.raises(ValueError, match="rigid seed 1 oracle"):
        _check(rows)
    with pytest.raises(ValueError, match="appears twice"):
        _check(_table() + [_row(*R0)])
    with pytest.raises(ValueError, match="no motion RMS"):
        _check(_table({R0: dict(motion_rms_um=None)}))


def test_manifest_gates_need_gates():
    with pytest.raises(ValueError, match="carries no gates"):
        check_manifest_gates(_table(), load_manifest(DEVELOPMENT_MANIFEST))


def test_case_metrics_reads_a_case_result():
    result = {
        "scenario": "rigid",
        "seed": 1,
        "recipe": "rigid_fast",
        "motion": {"rms_um": 0.7, "p95_um": 1.2, "sign_corr": 0.98},
        "fidelity_signal": {
            "corrected": {
                "channel_ids": ["1", "2"],
                "num": [0.1, 0.2],
                "den": [1.0, 2.0],
            },
            "uncorrected": {"pooled": 0.4},
        },
        "border": {
            "border_mode": "remove_channels",
            "n_in_channels": 3,
            "n_out_channels": 2,
            "removed_channel_ids": ["0"],
            "predicted_removed_channel_ids": ["0"],
        },
        "sorting": {
            "mean_accuracy": 0.5,
            "n_well_detected": 9,
            "n_overmerged": 0,
            "n_false_positive": 4,
            "n_gt_oversplit": 2,
        },
        "timings_s": {"estimate": 4.5},
        "peak_rss_bytes": 3 * 2**30,
    }

    row = case_metrics(result)

    assert row == CaseMetrics(
        scenario="rigid",
        seed=1,
        recipe="rigid_fast",
        border_mode="remove_channels",
        n_contacts=3,
        motion_rms_um=0.7,
        motion_p95_um=1.2,
        sign_corr=0.98,
        fidelity_channel_ids=("1", "2"),
        fidelity_num=(0.1, 0.2),
        fidelity_den=(1.0, 2.0),
        uncorrected_residual=0.4,
        n_out_channels=2,
        removed_channel_ids=("0",),
        predicted_removed_channel_ids=("0",),
        mean_accuracy=0.5,
        n_well_detected=9,
        n_overmerged=0,
        n_false_positive=4,
        n_gt_oversplit=2,
        estimation_s=4.5,
        peak_rss_gib=3.0,
    )
    assert pooled_residual(row) == pytest.approx(math.sqrt(0.3 / 3.0))


# ---- opt-in benchmark -------------------------------------------------------


def _manifest_path() -> Path:
    return Path(
        os.environ.get("SPYGLASS_V2_MOTION_MANIFEST", DEVELOPMENT_MANIFEST)
    )


def _benchmark_params():
    if not BENCHMARK:
        return [pytest.param(None, marks=BENCHMARK_SKIP, id="opt-in")]
    return [
        pytest.param(case, id=case_tag(*case))
        for case in load_manifest(_manifest_path()).iter_cases()
    ]


@pytest.fixture(scope="module")
def benchmark_out(tmp_path_factory) -> Path:
    out = os.environ.get("SPYGLASS_V2_MOTION_BENCHMARK_OUT")
    path = Path(out) if out else tmp_path_factory.mktemp("motion_benchmark")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_module(args: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tests.spikesorting.v2._motion_acceptance_run"]
        + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]


def _result(out: Path, case, sha: str) -> dict | None:
    path = out / f"{case_tag(*case)}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    return result if result.get("manifest_sha256") == sha else None


@pytest.mark.parametrize("case", _benchmark_params())
def test_benchmark_case(case, benchmark_out):
    """Run one case (or reuse its result) and check its structure: it ran
    through estimation, application, sorting and comparison, every metric is
    finite, and the output channels follow the border mode. No numeric
    acceptance threshold is applied here."""
    manifest_path = _manifest_path()
    sha = manifest_sha256(manifest_path)
    result = _result(benchmark_out, case, sha)
    if result is None:
        scenario, seed, recipe = case
        _run_module(
            [
                "case",
                "--manifest",
                str(manifest_path),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--recipe",
                recipe,
                "--out",
                str(benchmark_out),
            ]
        )
        result = _result(benchmark_out, case, sha)
    assert result is not None
    assert (result["scenario"], result["seed"], result["recipe"]) == case

    row = case_metrics(result)
    kind = load_manifest(manifest_path).recipes[case[2]].kind
    assert (row.motion_rms_um is None) == (kind == "off")
    if kind != "off":
        assert math.isfinite(row.motion_rms_um)
        assert math.isfinite(row.motion_p95_um)
    if kind == "estimate":
        estimation = result["estimation"]
        assert 0 < estimation["n_peaks_kept"] <= estimation["n_peaks_detected"]
        assert sum(estimation["peaks_per_continuity_span"]) == (
            estimation["n_peaks_kept"]
        )
    assert all(math.isfinite(v) for v in row.fidelity_num)
    assert all(v > 0 for v in row.fidelity_den)
    assert 1 <= row.n_out_channels <= row.n_contacts
    if row.border_mode in (None, "force_extrapolate"):
        assert row.n_out_channels == row.n_contacts
        assert row.removed_channel_ids == ()
    else:
        assert set(row.removed_channel_ids) == set(
            row.predicted_removed_channel_ids
        )
    assert list(row.fidelity_channel_ids) == [
        c
        for c in result["fidelity_signal"]["uncorrected"]["channel_ids"]
        if c not in row.removed_channel_ids
    ]
    assert 0.0 <= row.mean_accuracy <= 1.0
    assert result["sorting"]["n_gt_units"] == (
        load_manifest(manifest_path).generator.num_units
    )


@pytest.mark.parametrize(
    "enabled",
    (
        [pytest.param(True, id="gates")]
        if BENCHMARK
        else [pytest.param(True, marks=BENCHMARK_SKIP, id="opt-in")]
    ),
)
def test_benchmark_meets_manifest_gates(enabled, benchmark_out):
    """Every case result of the manifest meets its gates (held-out only)."""
    manifest_path = _manifest_path()
    manifest = load_manifest(manifest_path)
    if manifest.gates is None:
        pytest.skip(
            f"manifest {manifest.name} carries no gates (development "
            "evidence, not an acceptance test)"
        )
    sha = manifest_sha256(manifest_path)
    rows = []
    for case in manifest.iter_cases():
        result = _result(benchmark_out, case, sha)
        assert result is not None, f"no result for case {case}"
        rows.append(case_metrics(result))
    failed = [r for r in check_manifest_gates(rows, manifest) if not r.passed]
    assert not failed, "\n".join(map(str, failed))


@pytest.mark.parametrize(
    "enabled",
    (
        [pytest.param(True, id="representative")]
        if BENCHMARK
        else [pytest.param(True, marks=BENCHMARK_SKIP, id="opt-in")]
    ),
)
def test_representative_polymer_drift_fixture(enabled, benchmark_out):
    """Paired off / dredge / dredge_fast on one shank of the MEArec polymer
    drift fixture, estimated, applied and sorted: it runs, and the estimates
    and corrected traces are finite with every contact kept. Structural only:
    the fixture's drift (10 um) is below one contact pitch and its
    drift-vector sign convention relative to SpikeInterface is unconfirmed,
    so no scientific threshold applies. Quality, border, runtime and memory
    figures are written to the result directory."""
    if not DRIFT_FIXTURE.exists():
        pytest.skip(
            "representative polymer evidence is MISSING: the MEArec drift "
            f"fixture {DRIFT_FIXTURE} is not present"
        )
    args = [
        "representative",
        "--nwb",
        str(DRIFT_FIXTURE),
        "--out",
        str(benchmark_out),
        "--shank",
        "2",
        "--sort",
    ]
    if DRIFT_FIXTURE_GT_H5.exists():
        args += ["--gt-h5", str(DRIFT_FIXTURE_GT_H5)]
    _run_module(args)
    result = json.loads(
        (benchmark_out / "representative_shank2.json").read_text()
    )

    assert result["n_channels"] == 32
    assert set(result["recipes"]) == {"dredge_v1", "dredge_fast_v1"}
    for entry in result["recipes"].values():
        assert entry["n_temporal_bins"] > 0
        assert entry["displacement_finite"]
        assert entry["corrected_traces_finite"]
        assert entry["n_out_channels"] == 32
        assert entry["removed_channel_ids"] == []
    assert {"off", "dredge_v1", "dredge_fast_v1"} <= set(result["sorting"])
