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
    # recording_id folds in the resolved-input hash (membership + reference +
    # interpolate bad-channels), which insert_selection / preflight resolve from
    # the live sort group. This DB-free test pins the derivation with a FIXED
    # hash so the folding + downstream (artifact/sorting) chain stay stable; the
    # real-run match (preflight id == insert_selection id, real hash) is covered
    # by the DB-tier round-trip test.
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
    # The literals derive from the dated catalog row names above
    # (preproc "default", artifact "default", sorter
    # "franklab_30khz_ms5_2026_06") plus the fixed input_hash; correcting/
    # renaming a shipped row or changing the id-folding changes every derived
    # id, so these are regenerated -- not loosened -- when that happens.
    assert recording_id == uuid.UUID("189a9ec3-d299-5450-8712-437aa56e2d68")
    assert artifact_detection_id == uuid.UUID(
        "aa3b4fbe-013c-583a-8da2-be359777d535"
    )
    assert sorting_id == uuid.UUID("bfe079a9-98d4-5db6-9c2c-e52f68511407")


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
    # The recording build reads the 'raw data valid times' interval; preflight
    # gates it up front (in addition to the sort interval).
    assert "raw_valid_times_exists" in {c.name for c in report.checks}
    # The default 'default' artifact params do real detection -> no warning.
    assert report.warnings == []


@pytest.mark.database
def test_preflight_missing_raw_valid_times(preflight_inputs, monkeypatch):
    """A session missing the 'raw data valid times' interval fails preflight.

    ``Recording.make_fetch`` reads ``'raw data valid times'`` for the raw sample
    bounds in addition to the sort interval, so a partial ingest that misses it
    would otherwise crash late on a bare ``fetch1``. Simulate it absent
    (``IntervalList`` returns empty only for that name) and assert the
    ``raw_valid_times_exists`` check fails up front.
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

    The display analyzer recipe is FK-required on the Sorting row, so a missing
    AnalyzerWaveformParameters row would crash deep in make_fetch. Preflight
    must gate it up front like the other params rows: assert the
    ``analyzer_waveform_params_exist`` check is present, passes when the
    defaults are installed, and fails clearly when the region resolver points at
    an uninstalled row.
    """
    report = preflight_v2_pipeline(**preflight_inputs)
    (check,) = [
        c for c in report.checks if c.name == "analyzer_waveform_params_exist"
    ]
    assert check.ok is True

    import spyglass.spikesorting.v2._recipe_catalog as catalog_mod

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
def test_preflight_empty_sort_group(preflight_inputs):
    """A SortGroupV2 master with ZERO electrode members fails preflight.

    Checking only the master is a false-green: the master exists, but
    ``Recording.populate`` raises 'has zero electrodes' minutes into the run.
    The ``sort_group_has_electrodes`` check catches it up front. Audit
    finding #2."""
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
        # The master EXISTS -> sort_group_exists passes (this is the
        # false-green the membership check closes)...
        (exists_check,) = [
            c for c in report.checks if c.name == "sort_group_exists"
        ]
        assert exists_check.ok is True
        # ...but sort_group_has_electrodes catches the empty group.
        (members_check,) = [
            c for c in report.checks if c.name == "sort_group_has_electrodes"
        ]
        assert members_check.ok is False
        assert "zero electrode" in members_check.fix
    finally:
        (SortGroupV2 & restr).delete_quick()


@pytest.mark.database
def test_preflight_missing_params_row_points_to_initialize_defaults(
    preflight_inputs, monkeypatch
):
    """A preset referencing an uninstalled parameter row fails the matching
    ``*_params_exist`` check and routes to ``initialize_v2_defaults()``.

    Reproduces "I forgot to install the defaults": the preset points at a
    ``PreprocessingParameters`` row that is not in the DB. Preflight must
    flag the specific missing-row check up front (instead of crashing deep
    in ``populate``) and tell the user how to fix it. A synthetic preset
    (a copy of the runnable default with one bogus row name) drives the
    missing-row branch without deleting any shared default row.
    """
    import spyglass.spikesorting.v2._pipeline_preflight as preflight_mod
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
        "spyglass.spikesorting.v2._pipeline_preflight._SORTER_RUNTIME_BACKENDS",
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

    ``artifact_detection_params_name=None`` means the sort runs no artifact
    detection: preflight must not check for an ArtifactDetectionParameters row
    (there is none to require) and must report a None expected
    artifact_detection_id rather than deriving one.
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
    # The DB-free fail-fast rejection of a motion-pinned (concat) preset is
    # regressed in tests/spikesorting/v2/test_pipeline_presets.py
    # (test_preflight_rejects_motion_pinned_preset_without_db).


@pytest.mark.database
def test_preflight_auto_curate_adds_prereq_checks(preflight_inputs):
    """``auto_curate=True`` adds the auto-curation prerequisite checks.

    Auto-curation scores the root curation with the preset's metric +
    auto-curation rule rows on the whitened metric analyzer recipe, so preflight
    must gate those rows up front -- but only when opted in, so a default
    preflight is unchanged.
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
    """A missing auto-curation metric row fails preflight only when opted in.

    Without ``auto_curate`` the row is irrelevant and not checked; with it, a
    missing ``QualityMetricParameters`` row is a blocking error caught before
    the upstream compute.
    """
    import inspect

    from spyglass.spikesorting.v2 import pipeline as pl

    # preflight_inputs omits pipeline_preset (it relies on the function
    # default); base the bogus preset on that same default so only the metric
    # row is corrupted -- its preproc/sorter checks still pass.
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

    # Clean any prior run for this session, then rebuild the config: the
    # cascade also drops SortGroupV2, so configure AFTER cleaning.
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
    # The bypass path skips the team-existence check, so the missing
    # ``team_name`` surfaces as the raw DataJoint FK violation from
    # ``RecordingSelection.insert_selection``'s ``insert1`` (an
    # ``IntegrityError``) -- specifically NOT a curated ``PreflightError``.
    with pytest.raises(dj.errors.IntegrityError) as exc:
        run_v2_pipeline(**inputs)
    assert not isinstance(exc.value, PreflightError)


# ---------------------------------------------------------------------------
# database tier — containerized sorter execution backend
# ---------------------------------------------------------------------------

# The one shipped container preset (Singularity, 30 kHz). The execution backend
# lives on its referenced SorterParameters row, not the preset.
_SINGULARITY_PRESET = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"


@pytest.mark.database
@pytest.mark.unit
def test_container_runtime_probes_return_bool_detail_unmocked():
    """The real probes return ``(bool, str)`` -- catches an SI symbol rename.

    Every other container preflight test monkeypatches these probes, so the real
    code path (which lazily imports SI's ``has_docker`` / ``has_docker_python`` /
    ``has_singularity`` / ``has_spython``) is never exercised. If SI renamed or
    moved one of those symbols, the unavailable-backend protection would break in
    production (an ``ImportError`` instead of a clean ``(bool, detail)``), and
    every mocked test would stay green. This calls the probes for real and only
    asserts the contract shape (either truth value is fine in any environment).
    """
    from spyglass.spikesorting.v2._pipeline_preflight import (
        _docker_runtime_available,
        _singularity_runtime_available,
    )

    for probe in (_docker_runtime_available, _singularity_runtime_available):
        ok, detail = probe()
        assert isinstance(ok, bool)
        assert isinstance(detail, str) and detail


def test_preflight_container_runtime_errors(preflight_inputs, monkeypatch):
    """The container preset fails clearly when its runtime is absent.

    A Singularity preset is gated by the container runtime (engine + Python
    package), not the local sorter install. With the probe reporting
    "unavailable", preflight raises an actionable ``container_runtime_available``
    error naming the backend + image -- and never falls back to a local
    ``sorter_installed`` check.
    """
    from spyglass.spikesorting.v2 import _pipeline_preflight as pf

    monkeypatch.setattr(
        pf,
        "_singularity_runtime_available",
        lambda: (False, "Singularity/Apptainer was not found"),
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
    # No silent fallback to local execution: the local sorter-install checks do
    # not run for a container backend.
    assert not any(c.name == "sorter_installed" for c in report.checks)
    assert not any(c.name == "sorter_runtime_available" for c in report.checks)


@pytest.mark.database
def test_preflight_reports_container_ms4_modern_host_path(
    preflight_inputs, monkeypatch
):
    """A runnable container MS4 preset reports the host-stays-on-numpy>=2 path.

    When the container runtime is available, the container check passes and the
    report carries a non-blocking advisory making clear the host can stay on the
    v2 numpy>=2 environment because the MS4 runtime lives inside the container.
    """
    from spyglass.spikesorting.v2 import _pipeline_preflight as pf

    monkeypatch.setattr(
        pf,
        "_singularity_runtime_available",
        lambda: (True, "Singularity + Python `spython` package available"),
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

    MATLAB-backed sorters (Kilosort 2.5/3, IronClust) ship only as container
    images; a local execution backend for one of them must surface the tracked
    container-backend error -- the SAME message the dispatch raises -- not the
    local ``sorter_installed`` checks. The backend is read from the
    SorterParameters row (the single source of truth), so the test inserts a
    kilosort2_5 row with default (local) execution and a preset referencing it.
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
# assert_preset_compute_rows -- the mode-independent compute-row / sorter
# checks the concat preflight reuses.
# --------------------------------------------------------------------------- #

_CONCAT_PRESET = "franklab_concat_hippocampus_30khz_ms5_2026_06"


@pytest.mark.database
def test_assert_preset_compute_rows_passes_for_seeded_concat_preset(dj_conn):
    """A seeded concat preset's compute rows + local sorter pass the check."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET]
    # No raise: every param row is seeded and mountainsort5 is a local SI sorter.
    assert_preset_compute_rows(bundle, with_artifact=False)


@pytest.mark.database
def test_assert_preset_compute_rows_raises_on_missing_sorter_row(dj_conn):
    """A missing SorterParameters row fails fast with the exact fix.

    Guards the concat-preflight gap: before this check a concat run with a
    missing sorter row failed only deep in the member/concat populate.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET].model_copy(
        update={"sorter_params_name": "does_not_exist_xyz"}
    )
    with pytest.raises(PreflightError, match="SorterParameters row"):
        assert_preset_compute_rows(bundle, with_artifact=False)


@pytest.mark.database
def test_assert_preset_compute_rows_raises_on_missing_preprocessing_row(
    dj_conn,
):
    """A missing PreprocessingParameters row fails fast with the exact fix."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_preflight import (
        assert_preset_compute_rows,
    )

    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONCAT_PRESET].model_copy(
        update={"preprocessing_params_name": "does_not_exist_xyz"}
    )
    with pytest.raises(PreflightError, match="PreprocessingParameters row"):
        assert_preset_compute_rows(bundle, with_artifact=False)


_CONTAINER_PRESET = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"


@pytest.mark.database
def test_assert_concat_preflight_raises_on_missing_group(dj_conn):
    """The concat preflight fails fast when the SessionGroup does not exist."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_preflight import (
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
    """A container-backend preset's runtime is a BLOCKING preflight check
    (matching single-session), not silently skipped."""
    import spyglass.spikesorting.v2._pipeline_preflight as preflight_mod
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_preflight import (
        assert_preset_compute_rows,
    )
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    initialize_v2_defaults()
    bundle = _PIPELINE_PRESETS[_CONTAINER_PRESET]

    # Runtime unavailable -> a blocking PreflightError (no silent local
    # fallback). Both engines are patched so the test is backend-agnostic.
    monkeypatch.setattr(
        preflight_mod,
        "_singularity_runtime_available",
        lambda: (False, "singularity not installed"),
    )
    monkeypatch.setattr(
        preflight_mod,
        "_docker_runtime_available",
        lambda: (False, "docker not installed"),
    )
    with pytest.raises(PreflightError, match="not runnable here"):
        assert_preset_compute_rows(bundle, with_artifact=False)

    # Runtime available -> the container check passes (no local install needed).
    monkeypatch.setattr(
        preflight_mod, "_singularity_runtime_available", lambda: (True, "ok")
    )
    monkeypatch.setattr(
        preflight_mod, "_docker_runtime_available", lambda: (True, "ok")
    )
    assert_preset_compute_rows(bundle, with_artifact=False)
