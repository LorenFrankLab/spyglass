"""Execute the user-facing spike-sorting notebooks end-to-end on a fixture.

These are smoke tests: they run every code cell of the published notebooks in
order against a small ingested fixture, so a cell that no longer runs (a renamed
API, a wrong argument, a stale example) fails here instead of in a user's hands.

Execution is in-process -- each code cell's source is ``exec``'d in one shared
namespace, papermill-style parameter overrides injected right after the
``parameters``-tagged cell -- rather than in an isolated Jupyter kernel. That
keeps the notebook on the test's own DataJoint connection + fixtures (no
subprocess credential plumbing) and renders plots to a headless Agg backend.

The notebooks gate their optional-extra cells on import checks, and neither
optional extra is in the default v2 test env. 10_'s browser-curation cells need
the ``spikesorting-v2-curation`` extra (figpack), so they self-skip in the
default job and run in the separate curation CI lane. UnitMatch's bundle
extraction needs the ``spikesorting-v2-matching`` extra (UnitMatchPy), so 14_'s
match (Part B) runs only in the matching lane -- the cross-session test below
stands in a lightweight fixture matcher there and exercises Part A (concat)
everywhere.

Heavy (real MountainSort5 sorts + curation-evaluation PCA), hence
``@pytest.mark.slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NOTEBOOKS = _REPO_ROOT / "notebooks"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


def _require_fixture():
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )


@pytest.fixture(autouse=True)
def _restore_pipeline_preset_registry():
    """Undo any in-session pipeline-preset registration a notebook makes.

    ``_execute_notebook`` runs cells IN-PROCESS (see the module docstring), so
    the presets notebook's ``register_pipeline_preset`` / ``clone_pipeline_preset``
    calls mutate the module-level ``_PIPELINE_PRESETS`` dict (adding ``my_lab_*``)
    for the whole pytest process. Without restoring it those registrations leak
    into later tests that iterate the registry -- the preset-enumeration and
    recipe-parity tests then see a preset with no ``intended_use`` or a
    non-shipped sorter row. Snapshot and restore around every test in this
    module so a notebook's in-session registrations stay local to it.
    """
    from spyglass.spikesorting.v2 import _pipeline_presets as _pp

    snapshot = dict(_pp._PIPELINE_PRESETS)
    try:
        yield
    finally:
        _pp._PIPELINE_PRESETS.clear()
        _pp._PIPELINE_PRESETS.update(snapshot)


def _execute_notebook(ipynb_path: Path, parameters: dict) -> dict:
    """Run a notebook's code cells in-process; return the final namespace.

    Cells run in order in one namespace. After the ``parameters``-tagged cell
    runs (installing the notebook's defaults), the ``parameters`` overrides are
    assigned on top of them -- the same contract papermill uses -- so downstream
    cells see the test's session / preset values. A cell that raises propagates,
    failing the test with the offending cell's traceback.
    """
    import matplotlib

    matplotlib.use("Agg")

    notebook = json.loads(ipynb_path.read_text())
    namespace: dict = {"__name__": "__main__"}
    override_src = "\n".join(
        f"{key} = {value!r}" for key, value in parameters.items()
    )

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(
            compile(source, f"{ipynb_path.name}#cell{index}", "exec"), namespace
        )
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            exec(
                compile(override_src, "<injected-parameters>", "exec"),
                namespace,
            )
    return namespace


def _prepare_notebook_session(dj_conn, dest_name):
    """Ingest the smoke fixture + build sort groups; return (nwb, sort_group_id).

    The notebooks pick a sort group only when exactly one exists; build the
    groups here and hand the notebook the first id so the multi-shank guard does
    not stop the run.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    _require_fixture()
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH, dest_name=dest_name)
    if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted(
            (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
                "sort_group_id"
            )
        )[0]
    )
    return nwb_file_name, sort_group_id


def _notebook_params(nwb_file_name, sort_group_id):
    return {
        "nwb_file_name": nwb_file_name,
        "team_name": "notebook_exec_team",
        "interval_list_name": "raw data valid times",
        "pipeline_preset": "franklab_probe_hippocampus_30khz_ms5_2026_06",
        "sort_group_id": sort_group_id,
    }


@pytest.mark.slow
def test_single_session_notebook_runs(dj_conn):
    """``10_Spike_SortingV2`` (the lean first-sort path) runs end-to-end.

    Exercises the published first-sort walkthrough on one fixture: setup,
    preset choice, preflight, the run, one-call auto-curation, and the
    downstream accessor. (Browser/step-by-step curation moved to the Curation
    how-to; preset customization + the whole-session sweep to the Presets how-to,
    each with its own test below.)
    """
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, "notebook_single.nwb"
    )
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2.ipynb",
        _notebook_params(nwb_file_name, sort_group_id),
    )
    # The walkthrough produced a real, merge-keyable auto-curated result. The
    # notebook keys downstream off analysis_merge_id (the analysis-ready handle
    # auto-curation fills), not the uncurated root_merge_id.
    assert namespace["run_summary"]["n_units"] >= 0
    assert namespace["auto_summary"]["analysis_merge_id"] is not None
    assert namespace["merge_id"] is not None


@pytest.mark.slow
def test_curation_notebook_runs(dj_conn):
    """``10_Spike_SortingV2_Curation`` runs end-to-end on the smoke session.

    Self-contained: it sets up, sorts to a root curation, then exercises the
    FigPack browser path (self-skips without the curation extra) and the
    step-by-step evaluate -> merge -> re-evaluate loop, producing a final
    curated, merge-keyable result.
    """
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, "notebook_curation.nwb"
    )
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2_Curation.ipynb",
        _notebook_params(nwb_file_name, sort_group_id),
    )
    assert namespace["run_summary"]["n_units"] >= 0
    assert namespace["final_merge_id"] is not None


@pytest.mark.slow
def test_presets_notebook_runs(dj_conn):
    """``10_Spike_SortingV2_Presets`` runs end-to-end on the smoke session.

    Self-contained: setup, then customize a preset (clone + register) and sort
    the whole session at once with ``run_v2_pipeline_session``.
    """
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, "notebook_presets.nwb"
    )
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2_Presets.ipynb",
        _notebook_params(nwb_file_name, sort_group_id),
    )
    # The clone is registered, and the whole-session sweep returns per-group rows.
    assert "my_lab_ms5_lower_threshold" in namespace["list_pipeline_presets"]()
    assert isinstance(namespace["session_results"], list)
    assert namespace["session_results"]


class _NotebookMatcherParams(BaseModel):
    """Params schema for the test-only notebook fixture matcher."""

    model_config = ConfigDict(extra="forbid")
    tracked_unit_threshold: float = 0.5
    max_strict_nodes: int = 2000
    probability: float = 0.99
    schema_version: int = 1


class _NotebookFixtureMatcher:
    """Match the first matchable unit of each session, deterministically.

    A real sort of the small fixture recovers too few units for UnitMatchPy's
    metric path, so this lightweight matcher stands in: it reads each session's
    matchable units (no hardcoded ids) and emits one cross-session pair, exactly
    the surface ``UnitMatch`` / ``TrackedUnit`` consume.
    """

    name = "notebook_fixture_matcher"

    def match(self, session_inputs, params):
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.matcher_protocol import MatchPair

        if len(session_inputs) < 2:
            return []
        a, b = session_inputs[0], session_inputs[1]
        units_a = CurationV2().get_matchable_unit_ids(a.curation_key)
        units_b = CurationV2().get_matchable_unit_ids(b.curation_key)
        if not len(units_a) or not len(units_b):
            return []
        return [
            MatchPair(
                session_a_sorting_id=str(a.curation_key["sorting_id"]),
                session_a_curation_id=int(a.curation_key["curation_id"]),
                unit_a_id=int(units_a[0]),
                session_b_sorting_id=str(b.curation_key["sorting_id"]),
                session_b_curation_id=int(b.curation_key["curation_id"]),
                unit_b_id=int(units_b[0]),
                match_probability=float(params.get("probability", 0.99)),
            )
        ]


@pytest.mark.slow
def test_cross_session_notebook_runs(dj_conn):
    """``10_Spike_SortingV2_CrossSession`` runs both workflows on two sessions.

    Ingests the polymer smoke fixture twice (identical, same-day sessions) and
    runs the notebook: Part A concatenates and sorts them; Part B sorts each
    independently and matches units across them.

    UnitMatch's bundle extraction needs the optional ``UnitMatchPy`` package, so
    Part B runs only where it is installed (the default v2 test env excludes the
    matching extra). When it IS present, a real sort of the tiny fixture recovers
    too few units for UnitMatchPy's metric path, so the notebook's matcher is
    pointed (via its ``matcher_params_name`` parameter) at a registered
    lightweight fixture matcher -- the same substrate the unit-match table tests
    use -- so the full match + tracked-unit chain runs in-process. Without
    UnitMatchPy, only Part A (concat) is exercised here; the match API is covered
    by the unit-match table tests.
    """
    import importlib.util

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.matcher_protocol import register_matcher
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    _require_fixture()
    unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None
    members = []
    for dest in ("notebook_xsession_a.nwb", "notebook_xsession_b.nwb"):
        nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH, dest_name=dest)
        if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
            SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
        sort_group_id = int(
            sorted(
                (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
                    "sort_group_id"
                )
            )[0]
        )
        members.append(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": "raw data valid times",
            }
        )

    parameters = {
        "team_name": "notebook_xsession_team",
        "session_group_owner": "notebook_xsession_team",
        "same_day_members": members,
        "concat_group_name": "notebook_concat",
        "concat_preset": "franklab_concat_hippocampus_30khz_ms5_2026_06",
        "match_members": members,
        "match_group_name": "notebook_match",
        "single_preset": "franklab_probe_hippocampus_30khz_ms5_2026_06",
        "run_concat": True,
        "run_unit_match": unitmatch_available,
    }

    matcher_params_name = "notebook_fixture_matcher_params"
    registry = None
    try:
        if unitmatch_available:
            registry = (
                dict(mp._MATCHER_REGISTRY),
                dict(mp._SCHEMA_REGISTRY),
            )
            register_matcher(_NotebookFixtureMatcher(), _NotebookMatcherParams)
            MatcherParameters().insert1(
                {
                    "matcher_params_name": matcher_params_name,
                    "matcher": "notebook_fixture_matcher",
                    "params": {"probability": 0.99},
                },
                skip_duplicates=True,
            )
            parameters["matcher_params_name"] = matcher_params_name
        namespace = _execute_notebook(
            _NOTEBOOKS / "10_Spike_SortingV2_CrossSession.ipynb", parameters
        )
    finally:
        if registry is not None:
            (
                MatcherParameters & {"matcher_params_name": matcher_params_name}
            ).super_delete(warn=False)
            mp._MATCHER_REGISTRY.clear()
            mp._MATCHER_REGISTRY.update(registry[0])
            mp._SCHEMA_REGISTRY.clear()
            mp._SCHEMA_REGISTRY.update(registry[1])

    # Part A concatenated both members into one sort (runs everywhere).
    assert len(namespace["concat_summary"]["member_recording_ids"]) == 2
    # Part B matched units into tracked units (only where UnitMatchPy is present).
    if unitmatch_available:
        assert namespace["match_summary"]["n_tracked_units"] >= 1
