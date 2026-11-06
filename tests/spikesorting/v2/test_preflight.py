"""Tests for ``preflight_v2_pipeline`` and ``run_v2_pipeline(preflight=...)``.

Preflight is a read-only check that upstream rows, preset parameter rows and
the sorter exist, returning a :class:`PreflightReport`; ``run_v2_pipeline``
raises :class:`PreflightError` on a failing report.

- ``unit`` tier (no DB): frozen selection ids and input checks that fail
  before any query.
- ``database`` tier: per-check behavior, read-only-ness, speed, the
  ``expected_ids`` round-trip and ``run_v2_pipeline`` wiring on a configured,
  unrun smoke session.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import datajoint as dj
import pytest

from spyglass.spikesorting.v2._core.selection_identity import (
    artifact_detection_identity_payload,
    deterministic_id,
    recording_identity_payload,
    sorting_identity_payload,
)
from spyglass.spikesorting.v2.exceptions import PreflightError
from spyglass.spikesorting.v2.pipeline import (
    PreflightCheck,
    PreflightReport,
    preflight_v2_pipeline,
    run_v2_concat_pipeline,
    run_v2_pipeline,
)
from tests.spikesorting.v2._ingest_helpers import (
    configure_v2_run_inputs,
    copy_and_insert_nwb,
)

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_TEAM = "preflight_test_team"
_INTERVAL = "raw data valid times"


# ---------------------------------------------------------------------------
# unit tier (DB-free)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("blocked", [False, True])
def test_preflight_summary_distinguishes_selection_from_completed_output(
    monkeypatch, blocked
):
    """The displayed plan must not promise reuse for an unfinished stage."""

    def no_query(*args, **kwargs):
        pytest.fail("Rendering a preflight report must not query the DB")

    monkeypatch.setattr(dj.Connection, "query", no_query)
    report = PreflightReport(
        warnings=["Review the reference choice."],
        resolved_pipeline_preset="test_recipe",
        expected_ids={
            "recording_id": {
                "id": "recording",
                "exists": True,
                "computed_exists": True,
            },
            "artifact_detection_id": {
                "id": None,
                "exists": False,
                "computed_exists": False,
            },
            "sorting_id": {
                "id": "sorting",
                "exists": True,
                "computed_exists": False,
            },
        },
        checks=[
            PreflightCheck(
                "sorter_installed",
                not blocked,
                "Install the sorter runtime." if blocked else "",
            )
        ],
        effective_config={
            "sorter": "mountainsort5",
            "execution_backend": "local",
            "job_kwargs": {"n_jobs": 2, "chunk_duration": "1s"},
        },
        resource_notes=["Sorter scratch: /scratch/sorting"],
    )
    summary = report.summary()
    assert (
        f"Preflight {'blocked' if blocked else 'ready'}: test_recipe" in summary
    )
    assert "recording: reuse completed output" in summary
    assert "artifact_detection: skip" in summary
    assert "sorting: compute" in summary
    assert "Warning: Review the reference choice." in summary
    assert "mountainsort5" in summary and "local" in summary
    assert "'n_jobs': 2" in summary and "'chunk_duration': '1s'" in summary
    assert "Sorter scratch: /scratch/sorting" in summary
    assert ("ERROR: Install the sorter runtime." in summary) == blocked


@pytest.mark.unit
def test_installed_sorters_probed_once_per_reuse_block():
    """A session preflight probes the installed sorters once, not per group.

    Outside a reuse block every lookup probes afresh, so a long-lived kernel
    sees a sorter installed after an earlier preflight.
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._orchestration.preflight_runtime import (
        _installed_sorters,
        _installed_sorters_reused,
    )

    probes = []

    def installed_sorters():
        probes.append(None)
        return ["mountainsort5"]

    sis = SimpleNamespace(installed_sorters=installed_sorters)
    with _installed_sorters_reused():
        assert _installed_sorters(sis) == {"mountainsort5"}
        assert _installed_sorters(sis) == {"mountainsort5"}
    assert len(probes) == 1
    _installed_sorters(sis)
    _installed_sorters(sis)
    assert len(probes) == 3


@pytest.mark.unit
def test_identity_payload_extraction_stable():
    """The extracted payload builders reproduce the frozen selection ids.

    Preflight and ``insert_selection`` share these builders; pinning the
    ``uuid5`` outputs to literals catches drift.
    """
    cfg = {
        "nwb_file_name": "preflight_pin_test_.nwb",
        "sort_group_id": 0,
        "interval_list_name": _INTERVAL,
        "preprocessing_params_name": "default",
        "team_name": "pin_team",
    }
    # A fixed input hash stands in for the live sort group's; the DB-tier
    # round-trip test covers the real one.
    input_hash = "0" * 64
    recording_id = deterministic_id(
        "recording",
        {
            **recording_identity_payload(cfg),
            "recording_input_hash": input_hash,
        },
    )
    artifact_detection_id = deterministic_id(
        "artifact_detection",
        artifact_detection_identity_payload(
            artifact_detection_params_name="default", recording_id=recording_id
        ),
    )
    sorting_id = deterministic_id(
        "sorting",
        sorting_identity_payload(
            recording_id=recording_id,
            sorter="mountainsort5",
            sorter_params_name="franklab_30khz_ms5_2026_06",
            artifact_detection_id=artifact_detection_id,
        ),
    )
    # Regenerate (do not loosen) these if a shipped row name or the id
    # derivation changes.
    assert recording_id == uuid.UUID("189a9ec3-d299-5450-8712-437aa56e2d68")
    assert artifact_detection_id == uuid.UUID(
        "aa3b4fbe-013c-583a-8da2-be359777d535"
    )
    assert sorting_id == uuid.UUID("bfe079a9-98d4-5db6-9c2c-e52f68511407")


@pytest.mark.unit
def test_preflight_unknown_pipeline_preset(monkeypatch):
    """A bogus pipeline preset fails fast with no database access at all.

    ``Connection.query`` is patched to fail if preflight touches the DB.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "preflight_v2_pipeline queried the DB on an unknown pipeline preset"
        )

    monkeypatch.setattr(dj.Connection, "query", _boom)
    report = preflight_v2_pipeline(
        "x.nwb", 0, _INTERVAL, "team", pipeline_preset="not_a_real_preset"
    )
    assert isinstance(report, PreflightReport)
    assert report.ok is False
    assert bool(report) is False
    assert report.expected_ids == {}
    assert [c.name for c in report.checks] == ["pipeline_preset_known"]
    (msg,) = report.errors
    assert "not_a_real_preset" in msg
    assert "describe_pipeline_presets()" in msg


_CONTRADICTORY_MOTION_REQUESTS = [
    pytest.param("off", "dredge_fast_v1", "motion_mode='off'", id="recipe-off"),
    pytest.param("estimate", None, "requires", id="estimate-no-recipe"),
    pytest.param("apply", None, "requires", id="apply-no-recipe"),
    pytest.param("correct", "dredge_fast_v1", "unknown motion_mode", id="bad"),
]


@pytest.mark.unit
@pytest.mark.parametrize("mode, name, match", _CONTRADICTORY_MOTION_REQUESTS)
def test_contradictory_motion_request_fails_before_any_query(
    monkeypatch, mode, name, match
):
    """A motion recipe with ``off``, a mode without a recipe, or an unknown
    mode fails preflight and the run with no database access."""
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    def _boom(*args, **kwargs):
        raise AssertionError("a contradictory motion request queried the DB")

    monkeypatch.setattr(dj.Connection, "query", _boom)
    report = preflight_v2_pipeline(
        "x.nwb",
        0,
        _INTERVAL,
        "team",
        motion_mode=mode,
        motion_correction_params_name=name,
    )
    assert report.ok is False
    assert report.expected_ids == {}
    assert [c.name for c in report.checks] == [
        "pipeline_preset_known",
        "motion_request_valid",
    ]
    (message,) = report.errors
    assert match in message
    with pytest.raises(PipelineInputError, match=match):
        run_v2_pipeline(
            nwb_file_name="x.nwb",
            sort_group_id=0,
            interval_list_name=_INTERVAL,
            team_name="team",
            motion_mode=mode,
            motion_correction_params_name=name,
        )
    with pytest.raises(PipelineInputError, match=match):
        run_v2_concat_pipeline(
            session_group_owner="owner",
            session_group_name="group",
            motion_mode=mode,
            motion_correction_params_name=name,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode, name",
    [("off", None), ("estimate", "any_recipe"), ("apply", "any_recipe")],
)
def test_valid_motion_requests_have_no_problem(mode, name):
    from spyglass.spikesorting.v2._orchestration.preflight import (
        motion_request_problem,
    )

    assert motion_request_problem(mode, name) is None


_ESTIMATE_ID = "9b8c7f5e-2d1a-4c3b-8e6f-0a1b2c3d4e5f"


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode, name, estimate_id, match",
    [
        pytest.param(
            "off", None, _ESTIMATE_ID, "only motion_mode='apply'", id="off"
        ),
        pytest.param(
            "estimate",
            "dredge_fast_v1",
            _ESTIMATE_ID,
            "only motion_mode='apply'",
            id="estimate",
        ),
        pytest.param(
            "apply", None, _ESTIMATE_ID, "requires", id="apply-no-recipe"
        ),
        pytest.param(
            "apply", "dredge_fast_v1", "not-a-uuid", "is not a UUID", id="bad"
        ),
    ],
)
def test_motion_estimate_id_outside_apply_fails_before_any_query(
    monkeypatch, mode, name, estimate_id, match
):
    """A saved estimate id with a mode other than ``apply``, without a
    recipe, or that is not a UUID fails preflight and the run (either input
    mode) with no database access."""
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    def _boom(*args, **kwargs):
        raise AssertionError("a bad motion_estimate_id request queried the DB")

    monkeypatch.setattr(dj.Connection, "query", _boom)
    motion = {
        "motion_mode": mode,
        "motion_correction_params_name": name,
        "motion_estimate_id": estimate_id,
    }
    report = preflight_v2_pipeline("x.nwb", 0, _INTERVAL, "team", **motion)
    assert report.ok is False
    assert [c.name for c in report.checks][-1] == "motion_request_valid"
    (message,) = report.errors
    assert match in message
    with pytest.raises(PipelineInputError, match=match):
        run_v2_pipeline(
            nwb_file_name="x.nwb",
            sort_group_id=0,
            interval_list_name=_INTERVAL,
            team_name="team",
            **motion,
        )
    with pytest.raises(PipelineInputError, match=match):
        run_v2_concat_pipeline(
            session_group_owner="owner",
            session_group_name="group",
            **motion,
        )


@pytest.mark.unit
def test_apply_with_a_saved_estimate_id_is_a_valid_request():
    from spyglass.spikesorting.v2._orchestration.preflight import (
        motion_request_problem,
    )

    assert (
        motion_request_problem(
            "apply", "dredge_fast_v1", uuid.UUID(_ESTIMATE_ID)
        )
        is None
    )


_SINGLE_SESSION = {
    "nwb_file_name": "x.nwb",
    "sort_group_id": 0,
    "interval_list_name": _INTERVAL,
    "team_name": "team",
}


@pytest.mark.unit
@pytest.mark.parametrize(
    "inputs, match",
    [
        pytest.param(
            {
                **_SINGLE_SESSION,
                "pipeline_preset": "not_a_real_preset",
                "motion_correction_params_name": "dredge_fast_v1",
            },
            "estimate_motion: unknown pipeline_preset",
            id="unknown-preset",
        ),
    ],
)
def test_estimate_motion_rejects_a_bad_request_before_any_query(
    monkeypatch, inputs, match
):
    """``estimate_motion`` with an unknown preset fails with
    ``PipelineInputError`` naming itself, before any database access."""
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import estimate_motion

    def _boom(*args, **kwargs):
        raise AssertionError("a bad estimate_motion request queried the DB")

    monkeypatch.setattr(dj.Connection, "query", _boom)
    with pytest.raises(PipelineInputError, match=match):
        estimate_motion(**inputs)


@pytest.mark.unit
def test_estimate_motion_options_are_keyword_only():
    """Only the single-session source fields are positional, so a preset or
    a preflight flag passed by position is refused instead of landing on
    another parameter."""
    from spyglass.spikesorting.v2.pipeline import estimate_motion

    with pytest.raises(TypeError, match="positional"):
        estimate_motion(
            "x.nwb", 0, _INTERVAL, "team", "franklab_clusterless_2026_06"
        )


def _resolved_motion_recipe(name: str) -> dict:
    from spyglass.spikesorting.v2._motion.estimation import (
        resolve_estimation_params,
    )
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        motion_estimation_default_contents,
    )

    (params,) = [
        row[1] for row in motion_estimation_default_contents() if row[0] == name
    ]
    return resolve_estimation_params(params)


def _column(n_contacts: int, pitch_um: float = 26.0):
    """``(n, 3)`` single-column contact positions, rel_z constant."""
    import numpy as np

    return np.column_stack(
        [
            np.zeros(n_contacts),
            np.arange(n_contacts) * pitch_um,
            np.zeros(n_contacts),
        ]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "case, match",
    [
        ("polymer_shank", None),
        ("partial_geometry", None),
        ("legacy_tetrode", "detection radius_um"),
        ("positioned_tetrode", "detection radius_um"),
        ("four_contacts", "detection radius_um"),
        ("two_shanks", "2 shanks"),
        ("sixteen_contacts", "nonrigid windows"),
    ],
)
def test_motion_geometry_eligibility_from_registered_contacts(case, match):
    """Preflight runs the estimator's eligibility check on the group's
    effective planar geometry.

    - Passes: a 32-contact polymer shank (806 um).
    - Refused: tetrodes (including a positioned ``tetrode_12.5``, which the
      recording stage replaces with the 12.5 um square), two shanks, and a
      16-contact shank (too short for 400 um windows).
    - Incomplete geometry is left to the recording-geometry check.
    """
    import numpy as np

    from spyglass.spikesorting.v2._orchestration.preflight import (
        motion_geometry_problem_from_contacts,
    )

    n = {
        "legacy_tetrode": 4,
        "positioned_tetrode": 4,
        "four_contacts": 4,
        "sixteen_contacts": 16,
    }.get(case, 32)
    positions = _column(n)
    probe_types = ["128c-4s6mm6cm-15um-26um-sl"] * n
    groups = ["0"] * n
    shanks = [0] * n
    if case == "legacy_tetrode":
        positions = np.full((4, 3), np.nan)
        probe_types = ["tetrode_12.5"] * 4
    elif case == "positioned_tetrode":
        positions = _column(4, pitch_um=300.0)
        probe_types = ["tetrode_12.5"] * 4
    elif case == "partial_geometry":
        positions[3, 2] = np.nan
    elif case == "two_shanks":
        shanks = [0] * 16 + [1] * 16
    problem = motion_geometry_problem_from_contacts(
        list(range(n)),
        positions,
        probe_types,
        groups,
        shanks,
        _resolved_motion_recipe("dredge_fast_v1"),
    )
    if match is None:
        assert problem is None
    else:
        assert match in problem


@pytest.mark.parametrize(
    "name, refused", [("no_filter", True), ("default", False)]
)
def test_motion_needs_a_filtering_preprocessing_recipe(name, refused):
    """Motion is estimated on filtered traces: a shipped recipe without a
    temporal filter is refused, naming it; a bandpass recipe is not."""
    from spyglass.spikesorting.v2._motion.estimation import (
        unfiltered_source_problem,
    )
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        preprocessing_default_contents,
    )

    (params,) = [
        row[1] for row in preprocessing_default_contents() if row[0] == name
    ]
    problem = unfiltered_source_problem(name, params)
    if not refused:
        assert problem is None
        return
    assert f"{name!r} applies no temporal filter" in problem
    assert "filtered, unwhitened traces" in problem


def _catalog_motion_recipe(interpolation_name="kriging_force_extrapolate_v1"):
    """A ``MotionRecipe`` built from the shipped rows, without the DB."""
    from spyglass.spikesorting.v2._orchestration.preflight import MotionRecipe
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        motion_estimation_default_contents,
        motion_interpolation_default_contents,
    )

    (estimation,) = [
        row[1]
        for row in motion_estimation_default_contents()
        if row[0] == "dredge_fast_v1"
    ]
    (interpolation,) = [
        row[1]
        for row in motion_interpolation_default_contents()
        if row[0] == interpolation_name
    ]
    return MotionRecipe(
        recipe={
            "motion_correction_params_name": "dredge_fast_v1",
            "motion_estimation_params_name": "dredge_fast_v1",
            "motion_interpolation_params_name": interpolation_name,
        },
        estimation_params=estimation,
        interpolation_params=interpolation,
        resolved_estimation=_resolved_motion_recipe("dredge_fast_v1"),
    )


@pytest.mark.unit
@pytest.mark.parametrize("concat", [False, True])
def test_motion_setup_states_mode_recipe_preset_border_and_status(concat):
    """The scientific setup names the selected motion mode, recipe, resolved
    estimation preset, applied border mode and experimental status, in both
    run shapes; ``off`` says the sort reads the uncorrected source."""
    from spyglass.spikesorting.v2._orchestration.preflight_scientific_setup import (
        MOTION_RECIPE_STATUS,
        _describe_motion,
    )

    off = _describe_motion("off", None, concat=concat)
    assert off["mode"] == "off"
    assert "uncorrected" in off["description"]
    assert "not yet available" not in off["description"]

    recipe = _catalog_motion_recipe("kriging_remove_channels_v1")
    estimate = _describe_motion("estimate", recipe, concat=concat)
    assert estimate["mode"] == "estimate"
    assert estimate["recipe"] == "dredge_fast_v1"
    assert estimate["estimation_preset"] == "dredge_fast"
    assert estimate["status"] == MOTION_RECIPE_STATUS
    assert "border_mode" not in estimate
    assert "same sort as 'off'" in estimate["description"]

    applied = _describe_motion("apply", recipe, concat=concat)
    assert applied["border_mode"] == "remove_channels"
    assert applied["interpolation_recipe"] == "kriging_remove_channels_v1"
    assert applied["estimation_preset"] == "dredge_fast"
    assert "motion-corrected recording" in applied["description"]
    source = "concatenation" if concat else "recording under its mask"
    assert source in applied["description"]


# ---------------------------------------------------------------------------
# database tier: fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def preflight_session(dj_conn, smoke_nwb):
    """Ingest the smoke fixture under a preflight-isolated session name.

    Module-scoped; a distinct ``nwb_file_name`` keeps the cleanup isolated
    from other modules.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(
        _FIXTURE_PATH, dest_name="mearec_preflight_smoke.nwb"
    )
    return nwb_file_name


def _configure_inputs(nwb_file_name: str) -> dict:
    """Ensure defaults + team + sort group (no populate); return run inputs.

    Idempotent, so it also rebuilds the sort group after
    ``_clean_session_v2`` drops it.
    """
    return configure_v2_run_inputs(
        nwb_file_name,
        _TEAM,
        interval_list_name=_INTERVAL,
        team_description="preflight tests",
    )


@pytest.fixture
def preflight_inputs(preflight_session):
    """A cleanly-configured-but-not-yet-run session (defaults + team + group).

    Seeds defaults, the LabTeam and a sort group without ``populate``. Yields
    the ``run_v2_pipeline`` inputs for the default pipeline preset.
    """
    return _configure_inputs(preflight_session)


# ---------------------------------------------------------------------------
# database tier: preflight behavior
# ---------------------------------------------------------------------------


@pytest.mark.database
def test_preflight_all_pass(preflight_inputs):
    """A fully-configured session passes every check."""
    report = preflight_v2_pipeline(**preflight_inputs)
    assert report.ok is True
    assert bool(report) is True
    assert report.errors == []
    assert all(c.ok for c in report.checks)
    assert "raw_valid_times_exists" in {c.name for c in report.checks}
    # The default 'default' artifact params do real detection -> no warning.
    assert report.warnings == []
    # Resolved by the dispatcher's own resolve_sort_config.
    from spyglass.spikesorting.v2._sorting.dispatch import resolve_sort_config
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    assert "sorter_params_valid" in {c.name for c in report.checks}
    config = report.effective_config
    assert config is not None
    assert config["sorter"] == "mountainsort4"
    assert config["external_whiten"] is True
    assert config["si_sorter_params"]["whiten"] is False
    assert config["execution_backend"] == "local"
    rows = (SorterParameters & {"sorter": "mountainsort4"}).fetch(as_dict=True)
    matching = [
        r
        for r in rows
        if resolve_sort_config(
            r["sorter"],
            r["params"],
            job_kwargs=_resolved_job_kwargs(r["job_kwargs"]),
            execution_params=r.get("execution_params"),
        ).as_dict()
        == config
    ]
    assert matching, "effective_config must equal the dispatcher's resolution"
    # Per unit: the unit count is unknown before the sort.
    notes = "\n".join(report.resource_notes)
    assert "spikes/unit" in notes and "MiB per unit" in notes
    assert "analyzer cache root" in notes
    assert "sorter scratch" in notes
    assert "n_jobs=" in notes and "external_whiten=True" in notes


@pytest.mark.database
def test_preflight_missing_raw_valid_times(preflight_inputs, monkeypatch):
    """A session missing the 'raw data valid times' interval fails preflight.

    ``Recording.make_fetch`` reads it for the raw sample bounds.
    """
    import spyglass.common as common

    real_interval = common.IntervalList

    class _NoRawValidTimes:
        def __and__(self, restriction):
            if (
                isinstance(restriction, dict)
                and restriction.get("interval_list_name")
                == "raw data valid times"
            ):
                return []  # simulate the missing raw interval
            return real_interval & restriction

    monkeypatch.setattr(common, "IntervalList", _NoRawValidTimes())

    report = preflight_v2_pipeline(**preflight_inputs)
    assert report.ok is False
    (check,) = [c for c in report.checks if c.name == "raw_valid_times_exists"]
    assert check.ok is False
    assert "raw data valid times" in check.fix


@pytest.mark.database
def test_preflight_gates_analyzer_waveform_params_row(
    preflight_inputs, monkeypatch
):
    """Preflight checks the region-resolved analyzer-waveform recipe row.

    ``analyzer_waveform_params_exist`` passes with the defaults installed and
    fails when the region resolver names an uninstalled row.
    """
    report = preflight_v2_pipeline(**preflight_inputs)
    (check,) = [
        c for c in report.checks if c.name == "analyzer_waveform_params_exist"
    ]
    assert check.ok is True

    import spyglass.spikesorting.v2._core.recipe_catalog as catalog_mod

    monkeypatch.setattr(
        catalog_mod,
        "waveform_params_for_preprocessing",
        lambda _name: (
            "missing_waveform_preflight",
            "missing_metric_waveform_preflight",
        ),
    )
    missing = preflight_v2_pipeline(**preflight_inputs)
    assert missing.ok is False
    (missing_check,) = [
        c for c in missing.checks if c.name == "analyzer_waveform_params_exist"
    ]
    assert missing_check.ok is False
    assert "missing_waveform_preflight" in missing_check.fix
    assert "initialize_v2_defaults()" in missing_check.fix


@pytest.mark.database
def test_preflight_missing_raw(preflight_inputs, monkeypatch):
    """A session present but Raw absent fails raw_exists (not a false-negative).

    ``RecordingSelection`` references ``Raw``, not ``Session``.
    """
    import spyglass.common as common

    class _NoRaw:
        def __and__(self, restriction):
            return []  # empty -> falsy: simulates a missing Raw row

    monkeypatch.setattr(common, "Raw", _NoRaw())

    report = preflight_v2_pipeline(**preflight_inputs)
    assert report.ok is False
    (raw_check,) = [c for c in report.checks if c.name == "raw_exists"]
    assert raw_check.ok is False
    assert "Raw" in raw_check.fix
    (session_check,) = [c for c in report.checks if c.name == "session_exists"]
    assert session_check.ok is True


@pytest.mark.database
def test_preflight_missing_team(preflight_inputs):
    """A missing LabTeam fails team_exists; the other checks still run."""
    inputs = {**preflight_inputs, "team_name": "no_such_team_preflight"}
    report = preflight_v2_pipeline(**inputs)
    assert report.ok is False
    failed = {c.name for c in report.checks if not c.ok}
    assert failed == {"team_exists"}
    (team_check,) = [c for c in report.checks if c.name == "team_exists"]
    assert "no_such_team_preflight" in team_check.fix
    # Every check ran, not just up to the first failure.
    assert {c.name for c in report.checks} >= {
        "session_exists",
        "raw_exists",
        "interval_exists",
        "team_exists",
        "sort_group_exists",
        "sorter_installed",
    }


@pytest.mark.database
def test_preflight_missing_sort_group(preflight_inputs):
    """A missing SortGroupV2 fails with a set_group_by_shank pointer."""
    inputs = {**preflight_inputs, "sort_group_id": 987654}
    report = preflight_v2_pipeline(**inputs)
    assert report.ok is False
    (sg_check,) = [c for c in report.checks if c.name == "sort_group_exists"]
    assert sg_check.ok is False
    assert "set_group_by_shank" in sg_check.fix


@pytest.mark.database
def test_preflight_empty_sort_group(preflight_inputs):
    """A SortGroupV2 master with ZERO electrode members fails preflight.

    ``sort_group_has_electrodes`` fails while ``sort_group_exists`` passes.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = preflight_inputs["nwb_file_name"]
    empty_sg_id = 987001
    restr = {"nwb_file_name": nwb_file_name, "sort_group_id": empty_sg_id}
    SortGroupV2.insert1(
        {**restr, "reference_mode": "none"}, skip_duplicates=True
    )
    try:
        report = preflight_v2_pipeline(
            **{**preflight_inputs, "sort_group_id": empty_sg_id}
        )
        assert report.ok is False
        (exists_check,) = [
            c for c in report.checks if c.name == "sort_group_exists"
        ]
        assert exists_check.ok is True
        (members_check,) = [
            c for c in report.checks if c.name == "sort_group_has_electrodes"
        ]
        assert members_check.ok is False
        assert "zero electrode" in members_check.fix
    finally:
        (SortGroupV2 & restr).delete_quick()


def _make_sort_group(nwb_file_name: str, sort_group_id: int, n_channels: int):
    """Create a SortGroupV2 with exactly ``n_channels`` real members.

    The ``tetrode_12.5`` repair depends on the channel count, so the members
    are real ``Electrode`` rows.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    electrode_keys = sorted(
        (Electrode & {"nwb_file_name": nwb_file_name}).fetch("KEY"),
        key=lambda k: int(k["electrode_id"]),
    )[:n_channels]
    assert len(electrode_keys) == n_channels
    restr = {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    SortGroupV2.insert1(
        {**restr, "reference_mode": "none"}, skip_duplicates=True
    )
    SortGroupV2.SortGroupElectrode.insert(
        [{**key, "sort_group_id": sort_group_id} for key in electrode_keys],
        skip_duplicates=True,
    )
    return restr, electrode_keys


@pytest.mark.database
def test_preflight_rejects_coincident_contacts_after_repair(
    preflight_inputs, monkeypatch
):
    """Preflight reports geometry that collapses AFTER the tetrode repair.

    Preflight repeats the recording stage: pick the plane that keeps contacts
    distinct, then apply the ``tetrode_12.5`` repair. Contacts collapse only if
    both fail.
    """
    import numpy as np

    import spyglass.spikesorting.v2._recording.geometry as geometry

    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = preflight_inputs["nwb_file_name"]
    five_restr, _ = _make_sort_group(nwb_file_name, 987101, 5)
    four_restr, four_electrodes = _make_sort_group(nwb_file_name, 987102, 4)

    def _geometry_check(sort_group_id):
        report = preflight_v2_pipeline(
            **{**preflight_inputs, "sort_group_id": sort_group_id}
        )
        (check,) = [
            c for c in report.checks if c.name == "sort_group_geometry_distinct"
        ]
        return check

    def _patch(positions_for, probe_type):
        monkeypatch.setattr(
            geometry,
            "fetch_sort_group_contact_positions",
            lambda _f, channel_ids: positions_for(len(channel_ids)),
        )
        monkeypatch.setattr(
            geometry,
            "fetch_sort_group_probe_info",
            lambda _f, channel_ids: (
                (probe_type,) * len(channel_ids),
                ("0",) * len(channel_ids),
            ),
        )

    try:
        # 1. All-zero, five channels: no plane separates them and the repair
        #    needs exactly four.
        _patch(lambda n: np.zeros((n, 3)), "tetrode_12.5")
        failing = _geometry_check(987101)
        assert failing.ok is False
        assert "sort_group_id=987101" in failing.fix
        assert nwb_file_name in failing.fix
        assert "Probe.Electrode" in failing.fix
        # The positions are quoted so the operator sees WHAT collapsed.
        assert "0.0" in failing.fix

        # 2. All-zero four-channel tetrode_12.5: the repair spreads it.
        assert _geometry_check(987102).ok is True

        # 3. Frank-lab tetrode (rel_y == 0, +-6.25 um in x and z), not
        #    tetrode_12.5: passes only because the x-z plane is selected.
        xz_positions = np.array(
            [
                [6.25, 0.0, 6.25],
                [-6.25, 0.0, 6.25],
                [-6.25, 0.0, -6.25],
                [6.25, 0.0, -6.25],
            ]
        )
        assert (
            len(np.unique(xz_positions[:, :2], axis=0)) == 2
        ), "fixture must be degenerate in x-y for this case to mean anything"
        _patch(lambda _n: xz_positions, "not_a_tetrode")
        assert _geometry_check(987102).ok is True

        # 4. Four-channel tetrode_12.5 with finite x, y and NULL rel_z: only
        #    an entirely non-finite set counts as unpositioned, so the repair
        #    does not apply and this is reported, as at make.
        nan_z_positions = np.array(xz_positions)
        nan_z_positions[:, 2] = np.nan
        _patch(lambda _n: nan_z_positions, "tetrode_12.5")
        partial = _geometry_check(987102)
        assert partial.ok is False, (
            "finite x/y with a NaN rel_z is a partially positioned probe, "
            "not the legacy all-zero geometry the repair rescues"
        )
        assert "sort_group_id=987102" in partial.fix
        electrode_ids = sorted(
            int(key["electrode_id"]) for key in four_electrodes
        )
        for electrode_id in electrode_ids:
            assert (
                f"electrode {electrode_id} (rel_z)" in partial.fix
            ), "name each electrode AND the coordinate it is missing"
    finally:
        for restr in (five_restr, four_restr):
            (SortGroupV2.SortGroupElectrode & restr).delete_quick()
            (SortGroupV2 & restr).delete_quick()


@pytest.mark.database
def test_preflight_missing_params_row_points_to_initialize_defaults(
    preflight_inputs, monkeypatch
):
    """A preset referencing an uninstalled parameter row fails the matching
    ``*_params_exist`` check and routes to ``initialize_v2_defaults()``.

    A copy of the default preset with a bogus ``PreprocessingParameters``
    name, so no shared row is deleted.
    """
    import spyglass.spikesorting.v2._orchestration.preflight as preflight_mod
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    base = _PIPELINE_PRESETS["franklab_tetrode_hippocampus_30khz_ms5_2026_06"]
    bogus = base.model_copy(
        update={"preprocessing_params_name": "missing_preproc_preflight"}
    )
    monkeypatch.setattr(
        preflight_mod,
        "_PIPELINE_PRESETS",
        {**_PIPELINE_PRESETS, "bogus_missing_row_preset": bogus},
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": "bogus_missing_row_preset"}
    )
    assert report.ok is False
    (check,) = [
        c for c in report.checks if c.name == "preprocessing_params_exist"
    ]
    assert check.ok is False
    assert "missing_preproc_preflight" in check.fix
    assert "initialize_v2_defaults()" in check.fix


@pytest.mark.database
def test_preflight_sorter_not_installed(preflight_inputs, monkeypatch):
    """A spelled-valid sorter whose binary is absent fails the install gate.

    The internal ``clusterless_thresholder`` is never an SI binary, so its
    pipeline preset still passes the sorter check under the same patch.
    """
    import spikeinterface.sorters as sis

    real_installed = set(sis.installed_sorters())
    monkeypatch.setattr(
        sis,
        "installed_sorters",
        lambda: sorted(real_installed - {"mountainsort4"}),
    )

    report = preflight_v2_pipeline(**preflight_inputs)  # default = native ms4
    (sorter_check,) = [c for c in report.checks if c.name == "sorter_installed"]
    assert sorter_check.ok is False
    assert "installed_sorters()" in sorter_check.fix

    clusterless = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "franklab_clusterless_2026_06",
        }
    )
    (cl_check,) = [
        c for c in clusterless.checks if c.name == "sorter_installed"
    ]
    assert cl_check.ok is True


@pytest.mark.database
def test_preflight_sampling_rate_mismatch(preflight_inputs):
    """A 20 kHz preset on the 30 kHz smoke recording fails the rate check.

    The MS4 sorter row's snippet window is tuned for 20 kHz.
    """
    report = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "franklab_probe_cortex_20khz_ms4_2026_06",
        }
    )
    (rate_check,) = [
        c for c in report.checks if c.name == "sampling_rate_matches"
    ]
    assert rate_check.ok is False
    assert report.ok is False
    assert "20000" in rate_check.fix  # the preset's tuned rate
    assert "describe_pipeline_presets()" in rate_check.fix

    # The rate-matched 30 kHz cortex preset passes the same check.
    matched = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "franklab_probe_cortex_30khz_ms4_2026_06",
        }
    )
    (matched_check,) = [
        c for c in matched.checks if c.name == "sampling_rate_matches"
    ]
    assert matched_check.ok is True


@pytest.mark.database
def test_preflight_sorter_misspelled(preflight_inputs, monkeypatch):
    """A sorter that is not even a known SI sorter fails with a spelling hint.

    The hint names the spelling or preset, not ``installed_sorters()``.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    bogus_preset = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="default",
        sorter="not_a_real_sorter_xyz",
        sorter_params_name="default",
        metric_params_name="minimal",
        auto_curation_rules_name="none",
    )
    monkeypatch.setitem(
        pl._PIPELINE_PRESETS, "_preflight_bogus_sorter", bogus_preset
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": "_preflight_bogus_sorter"}
    )
    (check,) = [c for c in report.checks if c.name == "sorter_installed"]
    assert check.ok is False
    assert "not a known SpikeInterface sorter" in check.fix
    assert "installed_sorters()" not in check.fix


@pytest.mark.database
def test_preflight_sorter_runtime_backend_missing(
    preflight_inputs, monkeypatch
):
    """A sorter whose runtime backend can't import fails sorter_runtime_available.

    installed_sorters() can find a runtime it cannot import, so
    sorter_installed still passes.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    default_sorter = pl._PIPELINE_PRESETS[
        "franklab_probe_hippocampus_30khz_ms4_2026_06"
    ].sorter
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._orchestration.preflight_runtime._SORTER_RUNTIME_BACKENDS",
        {default_sorter: ("definitely_absent_backend_xyz",)},
    )

    report = preflight_v2_pipeline(**preflight_inputs)
    assert report.ok is False
    (rt_check,) = [
        c for c in report.checks if c.name == "sorter_runtime_available"
    ]
    assert rt_check.ok is False
    assert "definitely_absent_backend_xyz" in rt_check.fix
    (inst,) = [c for c in report.checks if c.name == "sorter_installed"]
    assert inst.ok is True


@pytest.mark.database
def test_preflight_ms4_preset_gets_runtime_check(preflight_inputs):
    """A native MS4 preset checks its bundled algorithm in the v2 environment."""
    import mountainsort4
    import spikeinterface.sorters as sis

    assert "mountainsort4" in set(sis.installed_sorters())
    assert callable(mountainsort4.mountainsort4)

    report = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "franklab_tetrode_hippocampus_30khz_ms4_2026_06",
        }
    )
    (rt_check,) = [
        c for c in report.checks if c.name == "sorter_runtime_available"
    ]

    assert rt_check.ok is True
    assert report.ok, report.errors


@pytest.mark.database
def test_preflight_warns_on_none_artifact_params(preflight_inputs, monkeypatch):
    """artifact_detection_params_name='none' raises a non-blocking advisory, not an error.

    Uses a temporary preset; ``ok`` stays True.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    none_preset = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="none",
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="minimal",
        auto_curation_rules_name="none",
    )
    monkeypatch.setitem(
        pl._PIPELINE_PRESETS, "_preflight_none_artifact", none_preset
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": "_preflight_none_artifact"}
    )
    assert report.ok is True
    assert any(
        "artifact" in w and "none" in w for w in report.warnings
    ), report.warnings


@pytest.mark.database
def test_preflight_skips_artifact_when_params_none(
    preflight_inputs, monkeypatch
):
    """A preset with no artifact stage passes preflight without an artifact row.

    ``artifact_detection_params_name=None``: no row check and a None expected
    artifact_detection_id.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    no_artifact = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name=None,
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="minimal",
        auto_curation_rules_name="none",
    )
    monkeypatch.setitem(
        pl._PIPELINE_PRESETS, "_preflight_no_artifact", no_artifact
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": "_preflight_no_artifact"}
    )
    assert report.ok is True
    # The artifact-row existence check is omitted entirely.
    assert not any(
        c.name == "artifact_detection_params_exist" for c in report.checks
    )
    # No artifact id is expected.
    assert report.expected_ids["artifact_detection_id"]["id"] is None
    assert report.expected_ids["artifact_detection_id"]["exists"] is False


@pytest.mark.database
@pytest.mark.parametrize("artifact_params_name", [None, "none"])
def test_preflight_no_masking_advisory_skipped_with_manual_exclusions(
    preflight_inputs, monkeypatch, artifact_params_name
):
    """Manual exclusions mask the sort even without automatic detection, so
    the 'no artifact masking' advisory must not fire (matches concat)."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2 import pipeline as pl

    preset = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name=artifact_params_name,
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="minimal",
        auto_curation_rules_name="none",
    )
    monkeypatch.setitem(pl._PIPELINE_PRESETS, "_preflight_manual_only", preset)
    start = float(
        (
            IntervalList
            & {
                "nwb_file_name": preflight_inputs["nwb_file_name"],
                "interval_list_name": preflight_inputs["interval_list_name"],
            }
        ).fetch1("valid_times")[0][0]
    )

    report = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "_preflight_manual_only",
            "manual_excluded_times": [[start + 1.0, start + 2.0]],
        }
    )
    assert report.ok is True
    assert not any(
        "no artifact masking" in w for w in report.warnings
    ), report.warnings


@pytest.mark.database
def test_concat_preset_preflights_in_single_session_mode(preflight_inputs):
    """The input fields, not the preset, choose the mode: the shipped concat
    preset runs every single-session check (no motion gate short-circuits it)
    and passes, and so does the whole-session preflight."""
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline_session,
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": _CONCAT_PRESET}
    )
    assert report.ok is True, report.errors
    assert "sorter_params_valid" in {c.name for c in report.checks}
    assert report.expected_ids
    session_report = preflight_v2_pipeline_session(
        nwb_file_name=preflight_inputs["nwb_file_name"],
        interval_list_name=preflight_inputs["interval_list_name"],
        team_name=preflight_inputs["team_name"],
        pipeline_preset=_CONCAT_PRESET,
        sort_group_ids=[preflight_inputs["sort_group_id"]],
    )
    assert session_report.ok is True, session_report.errors


@pytest.mark.database
def test_preflight_auto_curate_adds_prereq_checks(preflight_inputs):
    """``auto_curate=True`` adds the auto-curation prerequisite checks.

    The metric, auto-curation rule and metric analyzer rows are checked only
    when opted in.
    """
    auto_check_names = {
        "metric_params_exist",
        "auto_curation_rules_exist",
        "metric_waveform_params_exist",
    }

    base = preflight_v2_pipeline(**preflight_inputs)
    assert auto_check_names.isdisjoint(c.name for c in base.checks)

    curated = preflight_v2_pipeline(**preflight_inputs, auto_curate=True)
    names = {c.name for c in curated.checks}
    assert auto_check_names <= names
    # The fixture seeds the default rows, so the new checks pass.
    assert all(c.ok for c in curated.checks if c.name in auto_check_names)
    assert curated.ok is True


@pytest.mark.database
def test_preflight_auto_curate_flags_missing_metric_row(
    preflight_inputs, monkeypatch
):
    """A missing auto-curation metric row fails preflight only when opted in."""
    import inspect

    from spyglass.spikesorting.v2 import pipeline as pl

    # Copy the function's default preset so only the metric row is bogus.
    default_preset = (
        inspect.signature(pl.preflight_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
    )
    base = pl._PIPELINE_PRESETS[default_preset]
    bogus = base.model_copy(update={"metric_params_name": "missing_metric_xyz"})
    monkeypatch.setitem(pl._PIPELINE_PRESETS, "_preflight_bad_metric", bogus)
    inputs = {**preflight_inputs, "pipeline_preset": "_preflight_bad_metric"}

    flagged = preflight_v2_pipeline(**inputs, auto_curate=True)
    assert flagged.ok is False
    assert "metric_params_exist" in {c.name for c in flagged.checks if not c.ok}

    # Not opted in -> the missing metric row is not checked.
    default = preflight_v2_pipeline(**inputs)
    assert not any(c.name == "metric_params_exist" for c in default.checks)


@pytest.mark.database
def test_preflight_is_read_only(preflight_inputs, monkeypatch):
    """Preflight inserts nothing: every Selection/Lookup count is unchanged."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    tables = [
        RecordingSelection,
        RecordingArtifactSelection,
        SortingSelection,
        PreprocessingParameters,
        ArtifactDetectionParameters,
        SorterParameters,
        LabTeam,
        SortGroupV2,
    ]
    before = [len(t()) for t in tables]
    real_query = dj.Connection.query
    queries = []

    def read_only_query(connection, query, *args, **kwargs):
        command = query.lstrip().split(None, 1)[0].upper()
        assert command not in {
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
            "CREATE",
            "ALTER",
            "DROP",
            "TRUNCATE",
        }, f"Preflight mutated the database: {query}"
        queries.append(command)
        return real_query(connection, query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(dj.Connection, "query", read_only_query)
        preflight_v2_pipeline(**preflight_inputs)
    assert (
        "SELECT" in queries
    ), "the guard must observe the real preflight reads"
    after = [len(t()) for t in tables]
    assert before == after


@pytest.mark.database
def test_preflight_speed(preflight_inputs):
    """Preflight returns well under 1 s (guards against an accidental populate).

    Best of several runs against 1.5 s, so load jitter does not fail it; a
    real populate takes seconds to minutes.
    """
    preflight_v2_pipeline(**preflight_inputs)  # warm lazy imports
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        preflight_v2_pipeline(**preflight_inputs)
        best = min(best, time.perf_counter() - start)
    assert (
        best < 1.5
    ), f"preflight best-of-5 took {best:.2f}s (expected < 1.5 s)"


@pytest.mark.database
@pytest.mark.slow
def test_preflight_expected_ids_round_trip(preflight_session):
    """``expected_ids`` equals the PKs ``run_v2_pipeline`` actually produces.

    On a clean session ``exists`` is False pre-run; the deterministic ids match
    the run_summary the run returns, and a post-run preflight sees them as
    existing.
    """
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    # The cleanup cascade drops SortGroupV2, so configure after cleaning.
    _clean_session_v2({"nwb_file_name": preflight_session})
    preflight_inputs = _configure_inputs(preflight_session)

    pre = preflight_v2_pipeline(**preflight_inputs)
    assert pre.ok is True
    for stage in ("recording_id", "artifact_detection_id", "sorting_id"):
        # Pre-run: neither the selection nor the computed output exists.
        assert pre.expected_ids[stage]["exists"] is False
        assert pre.expected_ids[stage]["computed_exists"] is False

    run_summary = run_v2_pipeline(**preflight_inputs)
    assert pre.expected_ids["recording_id"]["id"] == run_summary["recording_id"]
    assert (
        pre.expected_ids["artifact_detection_id"]["id"]
        == run_summary["artifact_detection_id"]
    )
    assert pre.expected_ids["sorting_id"]["id"] == run_summary["sorting_id"]

    post = preflight_v2_pipeline(**preflight_inputs)
    for stage in ("recording_id", "artifact_detection_id", "sorting_id"):
        # Post-run: both the selection and the computed output exist.
        assert post.expected_ids[stage]["exists"] is True
        assert post.expected_ids[stage]["computed_exists"] is True

    _clean_session_v2({"nwb_file_name": preflight_inputs["nwb_file_name"]})


# ---------------------------------------------------------------------------
# database tier: run_v2_pipeline wiring
# ---------------------------------------------------------------------------


@pytest.mark.database
def test_run_pipeline_preflight_raises(preflight_inputs):
    """``preflight=True`` raises PreflightError and inserts no Selection rows."""
    from spyglass.spikesorting.v2.recording import RecordingSelection

    inputs = {**preflight_inputs, "team_name": "no_such_team_preflight"}
    before = len(RecordingSelection())
    with pytest.raises(PreflightError) as exc:
        run_v2_pipeline(**inputs)
    assert "no_such_team_preflight" in str(exc.value)
    assert len(RecordingSelection()) == before


@pytest.mark.database
@pytest.mark.slow
def test_run_pipeline_preflight_bypass(preflight_inputs):
    """``preflight=False`` bypasses the guard and hits the raw failure.

    The missing team surfaces as the underlying insert error.
    """
    inputs = {
        **preflight_inputs,
        "team_name": "no_such_team_preflight",
        "preflight": False,
    }
    with pytest.raises(dj.errors.IntegrityError) as exc:
        run_v2_pipeline(**inputs)
    assert not isinstance(exc.value, PreflightError)


# ---------------------------------------------------------------------------
# database tier: containerized sorter execution backend
# ---------------------------------------------------------------------------

# The one shipped container preset (Singularity, 30 kHz). The execution backend
# lives on its referenced SorterParameters row, not the preset.
_SINGULARITY_PRESET = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"


@pytest.mark.database
@pytest.mark.unit
def test_container_runtime_probes_return_bool_detail_unmocked():
    """The real probes return ``(bool, str)``.

    Other tests patch them; this catches a renamed SI symbol. Either truth
    value is fine.
    """
    from spyglass.spikesorting.v2._orchestration.preflight_runtime import (
        _container_runtime_available,
    )

    for backend in ("docker", "singularity"):
        ok, detail = _container_runtime_available(backend)
        assert isinstance(ok, bool)
        assert isinstance(detail, str) and detail


def test_preflight_container_runtime_errors(preflight_inputs, monkeypatch):
    """The container preset fails clearly when its runtime is absent.

    ``container_runtime_available`` names the backend and image; the local
    ``sorter_installed`` check does not run.
    """
    from spyglass.spikesorting.v2._orchestration import preflight_runtime as pf

    monkeypatch.setattr(
        pf,
        "_container_runtime_available",
        lambda backend: (False, "Singularity/Apptainer was not found"),
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": _SINGULARITY_PRESET}
    )
    (chk,) = [
        c for c in report.checks if c.name == "container_runtime_available"
    ]
    assert chk.ok is False
    assert "singularity" in chk.fix
    assert "does not fall back to local" in chk.fix
    # The failed container check is a blocking error.
    assert chk.fix in report.errors
    assert not any(c.name == "sorter_installed" for c in report.checks)
    assert not any(c.name == "sorter_runtime_available" for c in report.checks)


@pytest.mark.database
def test_preflight_reports_container_ms4_modern_host_path(
    preflight_inputs, monkeypatch
):
    """A runnable container MS4 preset reports the host-stays-on-numpy>=2 path.

    The advisory is non-blocking.
    """
    from spyglass.spikesorting.v2._orchestration import preflight_runtime as pf

    monkeypatch.setattr(
        pf,
        "_container_runtime_available",
        lambda backend: (
            True,
            "Singularity + Python `spython` package available",
        ),
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": _SINGULARITY_PRESET}
    )
    (chk,) = [
        c for c in report.checks if c.name == "container_runtime_available"
    ]
    assert chk.ok is True
    assert any(
        "numpy>=2" in w and "container" in w for w in report.warnings
    ), report.warnings


@pytest.mark.database
def test_preflight_matlab_local_backend_errors(
    preflight_inputs, monkeypatch, request
):
    """A MATLAB-sorter row on a local backend fails with the container message.

    MATLAB sorters ship only as containers; the error matches the dispatch's.
    The backend comes from the SorterParameters row.
    """
    from spyglass.spikesorting.v2 import pipeline as pl
    from spyglass.spikesorting.v2.sorting import SorterParameters

    request.addfinalizer(
        lambda: (
            SorterParameters
            & {"sorter": "kilosort2_5", "sorter_params_name": "preflight_local"}
        ).delete(safemode=False)
    )
    SorterParameters().insert1(
        {
            "sorter": "kilosort2_5",
            "sorter_params_name": "preflight_local",
            "params": {},
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )

    matlab_local = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="default",
        sorter="kilosort2_5",
        sorter_params_name="preflight_local",
        metric_params_name="minimal",
        auto_curation_rules_name="none",
    )
    monkeypatch.setitem(
        pl._PIPELINE_PRESETS, "_preflight_matlab_local", matlab_local
    )

    report = preflight_v2_pipeline(
        **{**preflight_inputs, "pipeline_preset": "_preflight_matlab_local"}
    )
    (chk,) = [c for c in report.checks if c.name == "sorter_execution_backend"]
    assert chk.ok is False
    assert "container" in chk.fix
    # The local-install checks do not run for a MATLAB local row.
    assert not any(c.name == "sorter_installed" for c in report.checks)


# --------------------------------------------------------------------------- #
# assert_preset_compute_rows: compute-row and sorter checks shared with
# the concat preflight.
# --------------------------------------------------------------------------- #

_CONCAT_PRESET = "franklab_concat_hippocampus_30khz_ms5_2026_09"


@pytest.mark.database
def test_assert_preset_compute_rows_passes_for_seeded_concat_preset(dj_conn):
    """A seeded concat preset's compute rows + local sorter pass the check."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET]
    # No raise: every param row is seeded and mountainsort5 is a local SI sorter.
    assert_preset_compute_rows(bundle)


@pytest.mark.database
def test_assert_preset_compute_rows_raises_on_missing_sorter_row(dj_conn):
    """A missing SorterParameters row fails fast with the exact fix."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET].model_copy(
        update={"sorter_params_name": "does_not_exist_xyz"}
    )
    with pytest.raises(PreflightError, match="SorterParameters row"):
        assert_preset_compute_rows(bundle)


@pytest.mark.database
def test_assert_preset_compute_rows_raises_on_missing_preprocessing_row(
    dj_conn,
):
    """A missing PreprocessingParameters row fails fast with the exact fix."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET].model_copy(
        update={"preprocessing_params_name": "does_not_exist_xyz"}
    )
    with pytest.raises(PreflightError, match="PreprocessingParameters row"):
        assert_preset_compute_rows(bundle)


_CONTAINER_PRESET = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"


@pytest.mark.database
def test_assert_concat_preflight_raises_on_missing_group(dj_conn):
    """The concat preflight fails fast when the SessionGroup does not exist."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_concat_preflight,
    )
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET]
    with pytest.raises(PreflightError, match="does not exist"):
        assert_concat_preflight("no_owner_xyz", "no_group_xyz", bundle)


@pytest.mark.database
def test_assert_preset_compute_rows_container_backend_checks_runtime(
    dj_conn, monkeypatch
):
    """A container-backend preset's runtime is a blocking check, as in
    single-session."""
    import spyglass.spikesorting.v2._orchestration.preflight_runtime as preflight_mod
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.preflight import (
        assert_preset_compute_rows,
    )
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONTAINER_PRESET]

    # Patched for every backend, so the test is backend-agnostic.
    monkeypatch.setattr(
        preflight_mod,
        "_container_runtime_available",
        lambda backend: (False, f"{backend} not installed"),
    )
    with pytest.raises(
        PreflightError, match="^run_v2_concat_pipeline: .*not runnable here"
    ):
        assert_preset_compute_rows(bundle)

    # Runtime available -> the container check passes (no local install needed).
    monkeypatch.setattr(
        preflight_mod,
        "_container_runtime_available",
        lambda backend: (True, "ok"),
    )
    assert_preset_compute_rows(bundle)


@pytest.mark.database
def test_scientific_setup_uses_execution_rows(preflight_inputs):
    from spyglass.spikesorting.v2._orchestration import presets as pl
    from spyglass.spikesorting.v2._orchestration.preflight import (
        describe_scientific_setup,
    )
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        DEFAULT_PIPELINE_PRESET,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    bundle = pl._PIPELINE_PRESETS[DEFAULT_PIPELINE_PRESET]
    group = {
        field: preflight_inputs[field]
        for field in ("nwb_file_name", "sort_group_id")
    }
    setup = describe_scientific_setup(bundle, [group])
    assert setup["preprocessing"] == (
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name}
    ).fetch1("params")
    assert setup["artifact_detection"] == (
        ArtifactDetectionParameters
        & {
            "artifact_detection_params_name": bundle.artifact_detection_params_name
        }
    ).fetch1("params")
    assert len(setup["references"]) == 1
    no_mask = bundle.model_copy(
        update={"sorter": "kilosort4", "artifact_detection_params_name": None}
    )
    setup = describe_scientific_setup(
        no_mask, [group], {"si_sorter_params": {"nblocks": 0}}
    )
    assert setup["artifact_detection"] is None
    assert setup["motion"]["mode"] == "off"
    assert setup["motion"]["sorter_correction"] == {
        "sorter": "kilosort4",
        "nblocks": 0,
    }
    assert "artifact_application" not in setup


@pytest.mark.database
@pytest.mark.parametrize("artifact_recipe", [None, "none"])
def test_concat_scientific_setup_reports_explicit_no_mask(
    preflight_inputs, artifact_recipe
):
    from spyglass.spikesorting.v2._orchestration import presets as pl
    from spyglass.spikesorting.v2._orchestration.preflight import (
        describe_scientific_setup,
    )

    bundle = pl._PIPELINE_PRESETS[
        "franklab_concat_hippocampus_30khz_ms5_2026_09"
    ].model_copy(update={"artifact_detection_params_name": artifact_recipe})
    group = {
        field: preflight_inputs[field]
        for field in ("nwb_file_name", "sort_group_id")
    }
    setup = describe_scientific_setup(bundle, [group], concat=True)
    assert setup["artifact_application"] == "No artifact masking selected."
    assert setup["motion"]["mode"] == "off"
    assert "not yet available" not in str(setup["motion"])
