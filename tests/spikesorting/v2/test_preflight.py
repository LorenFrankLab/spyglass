"""Tests for ``preflight_v2_pipeline`` and ``run_v2_pipeline(preflight=...)``.

Preflight is a read-only, pre-populate configuration check: it verifies the
    upstream rows + pipeline-preset param rows exist and the sorter binary is
    installed, returning a structured :class:`PreflightReport` (and
    ``run_v2_pipeline`` turns a failing report into a :class:`PreflightError`)
    so a misconfigured run fails in seconds instead of minutes into
    ``populate``.

Two tiers:

    * ``unit`` (DB-free): the deterministic-identity payload extraction is
      behavior-preserving (frozen to literal UUIDs), and an unknown
      pipeline preset short-circuits before any database access.
* ``database``: the per-check pass/fail behavior, read-only-ness, speed, the
  ``expected_ids`` round-trip, and the ``run_v2_pipeline`` raise/bypass wiring,
  against a cleanly-configured (but not-yet-run) smoke session.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import datajoint as dj
import pytest

from spyglass.spikesorting.v2._selection_identity import (
    artifact_detection_identity_payload,
    deterministic_id,
    recording_identity_payload,
    sorting_identity_payload,
)
from spyglass.spikesorting.v2.exceptions import PreflightError
from spyglass.spikesorting.v2.pipeline import (
    PreflightReport,
    preflight_v2_pipeline,
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
def test_identity_payload_extraction_stable():
    """The extracted payload builders reproduce the frozen selection ids.

    The recording/artifact-detection/sorting identity payloads were extracted
    from the three ``insert_selection`` methods into shared builders so
    preflight cannot derive a different id than the real run. The builders
    feed ``uuid.uuid5``, a pure hash, so a fixed logical identity must always
    map to the same UUID: pinning those UUIDs to literals catches any drift in
    the extraction.
    """
    cfg = {
        "nwb_file_name": "preflight_pin_test_.nwb",
        "sort_group_id": 0,
        "interval_list_name": _INTERVAL,
        "preprocessing_params_name": "default",
        "team_name": "pin_team",
    }
    recording_id = deterministic_id(
        "recording", recording_identity_payload(cfg)
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
    # The literals derive from the dated catalog row names above
    # (preproc "default", artifact "default", sorter
    # "franklab_30khz_ms5_2026_06"); correcting/renaming a shipped row
    # changes every derived id, so these are regenerated -- not loosened --
    # whenever the catalog names change.
    assert recording_id == uuid.UUID("6264f54c-7315-518e-b86e-e83903725387")
    assert artifact_detection_id == uuid.UUID(
        "1364ddfe-466a-56ea-8f6a-12d122b523b7"
    )
    assert sorting_id == uuid.UUID("89b487d4-0b61-57d3-afbc-2d20521ec46c")


@pytest.mark.unit
def test_preflight_unknown_pipeline_preset(monkeypatch):
    """A bogus pipeline preset fails fast with no database access at all.

    The param-row / existence checks need the resolved bundle's names, so an
    unknown pipeline preset short-circuits before any query. Patch
    ``Connection.query`` to fail loudly if preflight touches the DB.
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


# ---------------------------------------------------------------------------
# database tier — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def preflight_session(dj_conn):
    """Ingest the smoke fixture under a preflight-isolated session name.

    Module-scoped: NWB ingestion is the heaviest setup. Ingested under a
    DISTINCT ``nwb_file_name`` so the missing-prerequisite tests (which never
    touch other modules' rows) and the round-trip cleanup stay isolated.
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

    Idempotent ensure-exists, so it both builds the initial config and
    rebuilds the sort group after a ``_clean_session_v2`` cascade (which
    drops ``SortGroupV2``) in the round-trip test.
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

    Ensure-exists and idempotent: seeds the default Lookup rows, the LabTeam,
    and a sort group, but runs no ``populate``. Yields the ``run_v2_pipeline``
    input dict for the default pipeline preset.
    """
    return _configure_inputs(preflight_session)


# ---------------------------------------------------------------------------
# database tier — preflight behavior
# ---------------------------------------------------------------------------


@pytest.mark.database
def test_preflight_all_pass(preflight_inputs):
    """A fully-configured session passes every check."""
    report = preflight_v2_pipeline(**preflight_inputs)
    assert report.ok is True
    assert bool(report) is True
    assert report.errors == []
    assert all(c.ok for c in report.checks)
    # The default 'default' artifact params do real detection -> no warning.
    assert report.warnings == []


@pytest.mark.database
def test_preflight_missing_raw(preflight_inputs, monkeypatch):
    """A session present but Raw absent fails raw_exists (not a false-negative).

    ``RecordingSelection`` FKs ``Raw``, not ``Session``, so a partial ingestion
    (Session row but no Raw) would otherwise pass preflight and crash at the
    recording insert. Simulate the missing Raw by patching ``spyglass.common.Raw``
    so its restriction is empty; ``Session`` is untouched, so ``session_exists``
    still passes and ``raw_exists`` is the lone failure.
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
    # session_exists used the real Session table, so it still passes -- proving
    # raw_exists is a distinct check, not a duplicate of session_exists.
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
    # Report is complete: every check ran, not just up to the first failure.
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
        lambda: sorted(real_installed - {"mountainsort5"}),
    )

    report = preflight_v2_pipeline(**preflight_inputs)  # default = ms5
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

    The smoke fixture samples at 30 kHz; the 20 kHz MS4 preset's rate-keyed
    sorter row holds its snippet window at 20 kHz, so preflight must flag the
    mismatch (rather than letting the sort run with a mistuned window).
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

    Distinct from the 'known but binary not installed' branch
    (test_preflight_sorter_not_installed): the fix message points at the
    spelling / pipeline preset, not at installing a binary -- so it must NOT
    mention ``installed_sorters()``.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    bogus_preset = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="default",
        sorter="not_a_real_sorter_xyz",
        sorter_params_name="default",
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
def test_preflight_sorter_runtime_backend_missing(preflight_inputs, monkeypatch):
    """A sorter whose runtime backend can't import fails sorter_runtime_available.

    installed_sorters() only checks the thin SI wrapper; some sorters call a
    separate backend at run time (mountainsort4 -> ml_ms4alg). Map the default
    sorter to a guaranteed-absent backend so the check fires deterministically,
    and confirm sorter_installed still passes -- proving this is a distinct,
    stricter gate, not a duplicate of sorter_installed.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    default_sorter = pl._PIPELINE_PRESETS[
        "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    ].sorter
    monkeypatch.setattr(
        pl,
        "_SORTER_RUNTIME_BACKENDS",
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
    """An MS4 preset is gated on its real ml_ms4alg backend (map wiring).

    The mountainsort4 -> ml_ms4alg entry must actually be consulted: when the
    mountainsort4 wrapper is installed, an MS4 preset gets a
    sorter_runtime_available check whose result tracks whether ml_ms4alg
    actually imports (not just find_spec -- a present-but-broken numpy<2 backend
    must fail too). Under the v2 numpy>=2 baseline ml_ms4alg does not install,
    so the check fails there; tolerate an environment that has it. The check is
    gated on sorter_installed, so skip if the wrapper itself is absent.
    """
    import importlib

    import spikeinterface.sorters as sis

    if "mountainsort4" not in set(sis.installed_sorters()):
        pytest.skip(
            "mountainsort4 wrapper not installed; the runtime-backend check is "
            "gated on the sorter_installed check passing."
        )

    report = preflight_v2_pipeline(
        **{
            **preflight_inputs,
            "pipeline_preset": "franklab_tetrode_hippocampus_30khz_ms4_2026_06",
        }
    )
    (rt_check,) = [
        c for c in report.checks if c.name == "sorter_runtime_available"
    ]

    def _importable(mod: str) -> bool:
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False

    has_backend = _importable("ml_ms4alg")
    assert rt_check.ok == has_backend
    if not has_backend:
        assert "ml_ms4alg" in rt_check.fix


@pytest.mark.database
def test_preflight_warns_on_none_artifact_params(preflight_inputs, monkeypatch):
    """artifact_detection_params_name='none' raises a non-blocking advisory, not an error.

    The three built-ins use 'default' (real amplitude-threshold detection, no
    warning -- see test_preflight_all_pass); register a temporary 'none'-
    artifact-detection pipeline preset to drive the advisory branch and
    confirm it does not flip ``ok`` to False.
    """
    from spyglass.spikesorting.v2 import pipeline as pl

    none_preset = pl._PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="none",
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
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
def test_preflight_is_read_only(preflight_inputs):
    """Preflight inserts nothing: every Selection/Lookup count is unchanged."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
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
        ArtifactDetectionSelection,
        SortingSelection,
        PreprocessingParameters,
        ArtifactDetectionParameters,
        SorterParameters,
        LabTeam,
        SortGroupV2,
    ]
    before = [len(t()) for t in tables]
    preflight_v2_pipeline(**preflight_inputs)
    after = [len(t()) for t in tables]
    assert before == after


@pytest.mark.database
def test_preflight_speed(preflight_inputs):
    """Preflight returns well under 1 s (guards against an accidental populate).

    A real populate/materialization of the smoke recording costs seconds to
    minutes, so a small fixed bound catches an accidental one. We assert the
    *best* of several runs rather than a single shot: on a loaded box (e.g. the
    full suite running in parallel) scheduler/GC/DB-contention jitter can add
    tens of ms to any one call without reflecting preflight's real cost. The
    bound is 1.5 s -- preflight does a handful of read-only restrictions plus a
    ``Raw`` sampling-rate lookup, all sub-second on a warm connection -- and
    still blows past it if preflight ever starts doing real work.
    """
    preflight_v2_pipeline(**preflight_inputs)  # warm lazy imports
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        preflight_v2_pipeline(**preflight_inputs)
        best = min(best, time.perf_counter() - start)
    assert best < 1.5, f"preflight best-of-5 took {best:.2f}s (expected < 1.5 s)"


@pytest.mark.database
@pytest.mark.slow
def test_preflight_expected_ids_round_trip(preflight_session):
    """``expected_ids`` equals the PKs ``run_v2_pipeline`` actually produces.

    On a clean session ``exists`` is False pre-run; the deterministic ids match
    the run_summary the run returns, and a post-run preflight sees them as
    existing.
    """
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    # Clean any prior run for this session, then rebuild the config: the
    # cascade also drops SortGroupV2, so configure AFTER cleaning.
    _clean_session_v2({"nwb_file_name": preflight_session})
    preflight_inputs = _configure_inputs(preflight_session)

    pre = preflight_v2_pipeline(**preflight_inputs)
    assert pre.ok is True
    for stage in ("recording_id", "artifact_detection_id", "sorting_id"):
        assert pre.expected_ids[stage]["exists"] is False

    run_summary = run_v2_pipeline(**preflight_inputs)
    assert pre.expected_ids["recording_id"]["id"] == run_summary["recording_id"]
    assert (
        pre.expected_ids["artifact_detection_id"]["id"]
        == run_summary["artifact_detection_id"]
    )
    assert pre.expected_ids["sorting_id"]["id"] == run_summary["sorting_id"]

    post = preflight_v2_pipeline(**preflight_inputs)
    for stage in ("recording_id", "artifact_detection_id", "sorting_id"):
        assert post.expected_ids[stage]["exists"] is True

    _clean_session_v2({"nwb_file_name": preflight_inputs["nwb_file_name"]})


# ---------------------------------------------------------------------------
# database tier — run_v2_pipeline wiring
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

    The same missing-team misconfiguration that ``preflight=True`` catches as a
    PreflightError instead surfaces as the underlying insert/FK error, proving
    the bypass actually skips the check.
    """
    inputs = {
        **preflight_inputs,
        "team_name": "no_such_team_preflight",
        "preflight": False,
    }
    with pytest.raises(Exception) as exc:
        run_v2_pipeline(**inputs)
    assert not isinstance(exc.value, PreflightError)
