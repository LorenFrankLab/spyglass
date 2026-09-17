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

import importlib.util
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
            compile(source, f"{ipynb_path.name}#cell{index}", "exec"),
            namespace,
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
    # The walkthrough produced a real auto-labeled child and ended with the
    # explicit unit-selection handoff: a receipt pinned to that child's
    # generation whose group is what downstream reads, and (when the FigPack
    # extra is installed) a review that was reopened from its persisted id.
    assert namespace["run_summary"]["n_units"] >= 0
    assert namespace["auto_summary"]["auto_labeled_merge_id"] is not None
    receipt = namespace["receipt"]
    assert receipt.curation == namespace["auto_summary"].auto_labeled_curation
    assert receipt.policy_name == "v2_unflagged_units"
    assert set(receipt.included_unit_ids).isdisjoint(receipt.excluded_units)
    assert len(namespace["spike_times"]) == len(receipt.included_unit_ids)
    assert namespace["receipt"].group_key["unit_filter_params_name"] == (
        "v2_unflagged_units"
    )
    if importlib.util.find_spec("figpack") is not None:
        assert namespace["reopened"].review_id == namespace["review"].review_id


@pytest.mark.slow
def test_curation_notebook_runs(dj_conn):
    """``10_Spike_SortingV2_Curation`` runs end-to-end on the smoke session.

    Self-contained: it sets up, sorts to a root curation, starts the FigPack
    browser review (self-skips without the curation extra), previews it
    (no browser edits in run-all, no commit), and hands ONE deliberate
    ``final_curation`` -- here the reviewed parent itself -- to
    ``select_units_for_analysis``. The opt-in scripted appendix runs too and
    must not replace that result.
    """
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, "notebook_curation.nwb"
    )
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2_Curation.ipynb",
        {
            **_notebook_params(nwb_file_name, sort_group_id),
            "run_scripted_curation_example": True,
        },
    )
    assert namespace["run_summary"]["n_units"] >= 0
    assert namespace["final_merge_id"] is not None
    final_curation = namespace["final_curation"]
    assert final_curation == namespace["run_summary"].root_curation
    selection = namespace["selection"]
    assert selection.curation == final_curation
    assert selection.policy_name == "v2_accepted_single_units"
    assert selection.included_unit_ids == ()  # unreviewed: nothing accepted
    assert "no unit carries a required label" in selection.summary()
    # The scripted appendix produced its own, distinct result.
    scripted = namespace["scripted_curation"]
    assert scripted is not None and scripted != final_curation
    assert namespace["final_curation"] == final_curation


@pytest.mark.slow
@pytest.mark.parametrize("subset", [False, True], ids=["all-groups", "subset"])
def test_presets_notebook_runs(dj_conn, subset, monkeypatch):
    """``10_Spike_SortingV2_Presets`` runs end-to-end on the smoke session.

    Self-contained: setup, then customize a preset (clone + register) and sort
    the whole session at once with ``run_v2_pipeline_session``.
    """
    # Exercise the production label filter; shared v1 fixtures normally
    # disable it under test_mode, which would invalidate population receipts.
    from spyglass.spikesorting.analysis.v1 import group as group_module

    monkeypatch.setattr(group_module, "test_mode", False)
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, f"notebook_presets_{'subset' if subset else 'merged_all'}.nwb"
    )
    if not subset:
        import numpy as np
        from spikeinterface.core import NumpySorting

        from spyglass.spikesorting.v2.sorting import Sorting

        def plant_units(sorter, sorter_params, recording, sorting_id, **kwargs):
            frames = np.arange(1000, recording.get_num_samples() - 1000, 1000)
            return NumpySorting.from_unit_dict(
                {unit: frames + unit * 100 for unit in range(3)},
                recording.get_sampling_frequency(),
            )

        monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(plant_units))
    parameters = _notebook_params(nwb_file_name, sort_group_id)
    parameters.pop("sort_group_id")  # a whole-session run needs no single ID
    parameters.update(
        use_auto_labels_only=True, analysis_policy="v2_unflagged_units"
    )
    if subset:
        parameters["sort_group_ids"] = [sort_group_id]
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2_Presets.ipynb",
        parameters,
    )
    # The clone is registered, and the whole-session sweep returns per-group rows.
    assert "my_lab_ms5_lower_threshold" in namespace["list_pipeline_presets"]()
    assert isinstance(namespace["session_results"], list)
    assert namespace["session_results"]
    expected_ids = (
        [sort_group_id]
        if subset
        else sorted(namespace["sort_groups"]["sort_group_id"].tolist())
    )
    assert namespace["target_sort_group_ids"] == expected_ids
    assert [
        row["sort_group_id"] for row in namespace["session_results"]
    ] == expected_ids
    assert all(row["outcome"] == "ok" for row in namespace["session_results"])
    assert namespace["population_key"] is not None
    if not subset:
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.curation_api import CurationRef

        assert len(expected_ids) >= 2
        run = namespace["successful_runs"][expected_ids[0]]
        child = CurationV2.create_merged_curation(
            sorting_key={"sorting_id": run["sorting_id"]},
            parent_curation_id=run.root_curation.curation_id,
            merge_groups=[[0, 1]],
        )
        ref = CurationRef.from_key(child)
        namespace["final_curations"][expected_ids[0]] = ref
        with pytest.raises(ValueError, match="different members or policy"):
            namespace["assemble_population"]()
        namespace["population_name"] = "v2_session_after_merge"
        key, _, _, identities = namespace["assemble_population"]()
        merged_unit = (
            set((CurationV2.Unit & child).fetch("unit_id")) - {0, 1, 2}
        ).pop()
        assert {
            "spikesorting_merge_id": ref.merge_id,
            "unit_id": merged_unit,
        } in identities
        namespace["population_key"] = key
        namespace["population_unit_ids"] = identities
    first_key, _, _, first_ids = namespace["assemble_population"]()
    assert first_key == namespace["population_key"]
    assert {
        (row["spikesorting_merge_id"], row["unit_id"]) for row in first_ids
    } == {
        (row["spikesorting_merge_id"], row["unit_id"])
        for row in namespace["population_unit_ids"]
    }
    # A failed group makes the handoff pending until explicitly omitted.
    missing = max(expected_ids) + 1
    namespace["target_sort_group_ids"].append(missing)
    namespace["session_results"].append(
        {
            "sort_group_id": missing,
            "outcome": "failed",
            "error": "test failure",
        }
    )
    assert namespace["assemble_population"]()[0] is None
    namespace["omitted_sort_group_ids"].append(missing)
    assert namespace["assemble_population"]()[0] == first_key
    table = namespace["population_review_table"]().set_index("sort_group_id")
    assert (
        table.loc[missing, "omitted"]
        and table.loc[missing, "outcome"] == "failed"
    )
    namespace["analysis_policy"] = "all_units"
    with pytest.raises(ValueError, match="different members or policy"):
        namespace["assemble_population"]()


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
    member_merge_ids = namespace["concat_summary"]["member_merge_ids"]
    assert len(member_merge_ids) == 2
    # The automatic-only concat result is handed to analysis through an
    # explicit unflagged policy over the auto-labeled curation, one group per
    # member on that member's own merge id.
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    selection = namespace["concat_selection"]
    assert selection.policy_name == "v2_unflagged_units"
    assert (
        selection.curation == namespace["concat_summary"].auto_labeled_curation
    )
    assert {g.merge_id for g in selection.groups} == set(
        member_merge_ids.values()
    )
    for member_group in selection.groups:
        assert SortedSpikesGroup & dict(member_group.group_key)
    # Part B matched units into tracked units (only where UnitMatchPy is present).
    if unitmatch_available:
        assert namespace["match_summary"]["n_tracked_units"] >= 1


@pytest.mark.slow
def test_targeted_inspection_and_metric_filter_use_final_units(
    planted_three_unit_sort,
):
    """Execute the published inspection/filter cells over an actual merged child."""
    import matplotlib
    import numpy as np
    import pandas as pd
    from IPython.display import display

    from spyglass.spikesorting.v2 import visualization as ssviz
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import select_units_for_analysis
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    matplotlib.use("Agg")
    clear_curations_for(planted_three_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=planted_three_unit_sort)
        child = CurationV2.create_merged_curation(
            sorting_key=planted_three_unit_sort,
            parent_curation_id=root["curation_id"],
            merge_groups=[[0, 1]],
        )
        ref = CurationRef.from_key(child)
        units = sorted((CurationV2.Unit & child).fetch("unit_id"))
        spikes, identities = select_units_for_analysis(
            ref, policy="all_units"
        ).fetch_spike_data(return_unit_ids=True)
        namespace = {
            "np": np,
            "pd": pd,
            "display": display,
            "ssviz": ssviz,
            "final_curation": ref,
            "inspection_unit_ids": units,
            "inspection_pair": units,
            "spike_times": spikes,
            "selected_unit_ids": identities,
            "snr_threshold": 0.0,
            "analysis_policy": "all_units",
            "select_units_for_analysis": select_units_for_analysis,
        }
        book = json.loads(
            (_NOTEBOOKS / "10_Spike_SortingV2_Curation.ipynb").read_text()
        )
        active = False
        for cell in book["cells"]:
            source = "".join(cell["source"])
            if "### Is this unit neural" in source:
                active = True
            if "## Appendix A." in source:
                break
            if active and cell["cell_type"] == "code":
                exec(  # noqa: S102 - execute repository-owned notebook cells
                    compile(source, "<targeted-inspection>", "exec"), namespace
                )
        assert namespace["final_evaluation"].curation == ref
        assert set(namespace["final_evaluation"].metrics.index) == set(units)
        selection = namespace["metric_selection"]
        _, selected = selection.fetch_spike_data(return_unit_ids=True)
        assert selected == namespace["filtered_unit_ids"]
        assert selection.selection_provenance["evaluation_id"] == str(
            namespace["final_evaluation"].evaluation_id
        )
        assert namespace["analysis_provenance"]["curation_uuid"] == str(
            ref.curation_uuid
        )
    finally:
        clear_curations_for(planted_three_unit_sort)
