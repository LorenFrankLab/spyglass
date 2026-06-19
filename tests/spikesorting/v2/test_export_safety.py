"""Export-safety tests for the v2 spike-sorting pipeline.

A paper export over a v2 ``merge_id`` must capture every analysis file
needed to reproduce it -- the curated units NWB and the upstream
preprocessed-recording cache -- exactly as the v1 pipeline does. v1 and
v2 both achieve this through the upward foreign-key cascade run inside
``Export.populate_paper`` (NOT through per-fetch accessor logging: the
``get_sorting`` / ``get_recording`` accessors on both pipelines read
their files directly via ``AnalysisNwbfile.get_abs_path`` and log
nothing). These tests pin that the standard supported path
(``start_export`` -> ``SpikeSortingOutput.fetch_nwb`` -> ``stop_export``
-> ``Export.populate_paper``) leaves the v2 recording cache in the final
``Export.File``, and that the zero-unit curation exports without error.

Each test asserts a two-sided invariant -- the recording cache is ABSENT
from the selection-stage ``ExportSelection.File`` but PRESENT in the
post-populate ``Export.File`` -- so the test exercises (and would fail on
a regression of) the cascade that actually does the capture, rather than
trivially passing.

All tests are ``slow``: each runs a full sort plus an ``Export``
populate against the Docker MySQL container.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
from tests.spikesorting.v2.test_single_session_pipeline import (
    _clean_session_v2,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
_TEAM = "v2_export_team"


@pytest.fixture(scope="module")
def export_smoke_session(dj_conn):
    """Ingest the MEArec polymer smoke fixture for this module."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}


def _run_pipeline(session, pipeline_preset):
    """Run ``run_v2_pipeline`` end-to-end and return its manifest.

    Cleans every v2 row for the session first so the sort is rebuilt
    deterministically regardless of test order.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = session["nwb_file_name"]
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": _TEAM, "team_description": "v2 export tests"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
    )
    return run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=_TEAM,
        pipeline_preset=pipeline_preset,
        description="v2 export test",
    )


def _file_names(manifest):
    """Return (units_nwb, recording_nwb) AnalysisNwbfile names for a merge."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording

    cur_part = SpikeSortingOutput.CurationV2 & {
        "merge_id": manifest["merge_id"]
    }
    cur_key = (CurationV2 & cur_part).fetch1("KEY")
    units_nwb = (CurationV2 & cur_key).fetch1("analysis_file_name")
    recording_nwb = (
        Recording & {"recording_id": manifest["recording_id"]}
    ).fetch1("analysis_file_name")
    return units_nwb, recording_nwb


def _export_and_populate(merge_id, paper_id, *, also_spike_times=False):
    """Run the supported export path and return (selection, final) file sets.

    Returns two sets of AnalysisNwbfile *basenames*: those logged at the
    selection stage (``ExportSelection.File``) and those in the final
    populated export (``Export.File``, which folds in the FK cascade).
    """
    from spyglass.common.common_usage import Export, ExportSelection
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    es = ExportSelection()
    # Idempotent clean of any prior run for this paper.
    (es & {"paper_id": paper_id}).super_delete(warn=False, safemode=False)
    (Export() & {"paper_id": paper_id}).super_delete(warn=False, safemode=False)

    es.start_export(paper_id=paper_id, analysis_id=1)
    SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
    if also_spike_times:
        # Exercised for the zero-unit case: must not raise
        # KeyError: 'spike_times' on the empty (no-column) units table.
        SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    es.stop_export()

    eid = es._max_export_id(paper_id)
    selection = {
        Path(p).name
        for p in (es.File & {"export_id": eid}).fetch("analysis_file_name")
    }

    Export().populate_paper(paper_id=paper_id)
    export_id = (Export() & {"paper_id": paper_id}).fetch("export_id")[0]
    final = {
        Path(p).name
        for p in (Export.File & {"export_id": export_id}).fetch("file_path")
    }
    return selection, final


def _cleanup_export(paper_id):
    from spyglass.common.common_usage import Export, ExportSelection

    (Export() & {"paper_id": paper_id}).super_delete(warn=False, safemode=False)
    (ExportSelection() & {"paper_id": paper_id}).super_delete(
        warn=False, safemode=False
    )


@pytest.mark.slow
def test_v2_export_captures_curation_and_recording_files(export_smoke_session):
    """A v2 ``merge_id`` export captures the units NWB AND the recording
    cache in the final ``Export.File`` (v1 parity, via the FK cascade)."""
    paper_id = "v2_export_complete"
    manifest = _run_pipeline(
        export_smoke_session, "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    )
    assert manifest["n_units"] >= 1, "expected a populated sort"
    units_nwb, recording_nwb = _file_names(manifest)
    assert units_nwb != recording_nwb, "units and recording must be distinct"

    try:
        selection, final = _export_and_populate(manifest["merge_id"], paper_id)

        # The supported path captures BOTH analysis files in the final export.
        assert units_nwb in final, (
            f"curated units NWB {units_nwb!r} missing from Export.File "
            f"{sorted(final)}"
        )
        assert recording_nwb in final, (
            f"recording cache {recording_nwb!r} missing from Export.File "
            f"{sorted(final)} -- the populate_paper FK cascade no longer "
            "reaches the v2 Recording table (v1-parity regression)."
        )

        # Two-sided invariant: the recording cache is captured by the
        # cascade at populate time, NOT by selection-stage fetch logging.
        # This pins the mechanism so the test fails if the cascade breaks
        # rather than passing trivially.
        assert recording_nwb not in selection, (
            f"recording cache {recording_nwb!r} unexpectedly logged at the "
            "selection stage; the capture mechanism changed -- revisit "
            "whether this still matches v1 behavior."
        )
        assert units_nwb in selection, (
            f"units NWB {units_nwb!r} should be logged by fetch_nwb at the "
            f"selection stage; got {sorted(selection)}"
        )
    finally:
        _cleanup_export(paper_id)


@pytest.mark.slow
def test_v2_zero_unit_export_path(export_smoke_session):
    """A zero-unit curation exports without error and its empty-but-real
    units NWB is captured; the recording cache is still captured too."""
    paper_id = "v2_export_zero_unit"
    manifest = _run_pipeline(
        export_smoke_session, "franklab_clusterless_2026_06"
    )
    assert manifest["n_units"] == 0, (
        "shipped clusterless default should find zero units on the smoke "
        f"fixture; got {manifest['n_units']}"
    )
    assert (
        manifest["merge_id"] is not None
    ), "zero-unit sort must be merge-keyable"
    units_nwb, recording_nwb = _file_names(manifest)

    try:
        # also_spike_times exercises get_spike_times on the empty units
        # table -- must not raise KeyError: 'spike_times'.
        selection, final = _export_and_populate(
            manifest["merge_id"], paper_id, also_spike_times=True
        )

        # The empty-but-real units NWB is logged at the selection stage:
        # fetch_nwb records the curation's analysis file regardless of unit
        # count. The recording cache is captured only by the populate
        # cascade (same two-sided invariant as the completeness test).
        assert units_nwb in selection, (
            f"empty units NWB {units_nwb!r} should be logged by fetch_nwb at "
            f"the selection stage; got {sorted(selection)}"
        )
        assert units_nwb in final, (
            f"empty-but-real units NWB {units_nwb!r} missing from "
            f"Export.File {sorted(final)}"
        )
        assert recording_nwb not in selection, (
            f"recording cache {recording_nwb!r} unexpectedly logged at the "
            "selection stage for a zero-unit sort"
        )
        assert recording_nwb in final, (
            f"recording cache {recording_nwb!r} missing from zero-unit "
            f"Export.File {sorted(final)}"
        )
    finally:
        _cleanup_export(paper_id)
