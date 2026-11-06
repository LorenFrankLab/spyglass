"""Execute the user-facing spike-sorting notebooks end-to-end on a fixture.

Cells are ``exec``'d in-process in one namespace so they share the test's
DataJoint connection; parameter overrides follow the ``parameters`` cell as
in papermill. Cells needing the curation (figpack) or matching (UnitMatchPy)
extras self-skip when those are absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
from tests.spikesorting.v2._sorter_stub import plant_sorter

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

    In-process cells mutate the module-level ``_PIPELINE_PRESETS``, which
    would leak ``my_lab_*`` presets into later registry tests.
    """
    from spyglass.spikesorting.v2._orchestration import presets as _pp

    snapshot = dict(_pp._PIPELINE_PRESETS)
    try:
        yield
    finally:
        _pp._PIPELINE_PRESETS.clear()
        _pp._PIPELINE_PRESETS.update(snapshot)


def _execute_notebook(ipynb_path: Path, parameters: dict) -> dict:
    """Run a notebook's code cells in-process; return the final namespace.

    ``parameters`` overrides are assigned right after the
    ``parameters``-tagged cell, as in papermill. A raising cell propagates.
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

    The first id is passed so the notebook's multi-shank guard does not stop.
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
        "pipeline_preset": "franklab_probe_hippocampus_30khz_ms4_2026_06",
        "sort_group_id": sort_group_id,
    }


@pytest.mark.slow
def test_single_session_notebook_runs(dj_conn, smoke_nwb):
    """``10_Spike_SortingV2`` (the lean first-sort path) runs end-to-end."""
    nwb_file_name, sort_group_id = _prepare_notebook_session(
        dj_conn, "notebook_single.nwb"
    )
    namespace = _execute_notebook(
        _NOTEBOOKS / "10_Spike_SortingV2.ipynb",
        _notebook_params(nwb_file_name, sort_group_id),
    )
    # An auto-labeled child, a receipt pinned to it, and (with FigPack) a
    # review reopened from its persisted id.
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
def test_curation_notebook_runs(dj_conn, smoke_nwb):
    """``10_Spike_SortingV2_Curation`` runs end-to-end on the smoke session.

    The reviewed parent is the ``final_curation`` handed to
    ``select_units_for_analysis``; the scripted appendix must not replace it.
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
def test_presets_notebook_runs(dj_conn, subset, monkeypatch, smoke_nwb):
    """``10_Spike_SortingV2_Presets`` runs end-to-end on the smoke session."""
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

        def plant_units(sorter, sorter_params, recording, sorting_id, **kwargs):
            frames = np.arange(1000, recording.get_num_samples() - 1000, 1000)
            return NumpySorting.from_unit_dict(
                {unit: frames + unit * 100 for unit in range(3)},
                recording.get_sampling_frequency(),
            )

        plant_sorter(monkeypatch, plant_units)
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
    assert "my_lab_ms4_lower_threshold" in namespace["list_pipeline_presets"]()
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

        assert len(expected_ids) >= 2
        run = namespace["successful_runs"][expected_ids[0]]
        ref = run.root_curation.commit_merges([[0, 1]])
        namespace["final_curations"][expected_ids[0]] = ref
        with pytest.raises(ValueError, match="different members or policy"):
            namespace["assemble_population"]()
        namespace["population_name"] = "v2_session_after_merge"
        key, _, _, identities = namespace["assemble_population"]()
        merged_unit = (
            set((CurationV2.Unit & ref.as_key()).fetch("unit_id")) - {0, 1, 2}
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

    The small fixture yields too few units for UnitMatchPy's metric path.
    """

    name = "notebook_fixture_matcher"

    def match(self, session_inputs, params):
        from spyglass.spikesorting.v2.matcher_protocol import MatchPair

        if len(session_inputs) < 2:
            return []
        a, b = session_inputs[0], session_inputs[1]
        from tests.spikesorting.v2._ingest_helpers import matched_unit_ids

        units_a = matched_unit_ids(a.curation_key)
        units_b = matched_unit_ids(b.curation_key)
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
def test_cross_session_notebook_runs(dj_conn, smoke_nwb):
    """``10_Spike_SortingV2_CrossSession`` runs all three workflows.

    Sessions a, b and c are copies of the smoke fixture. Part A concatenates
    a and b; Part B matches a and c; Part C matches a+b against c. Parts B
    and C need UnitMatchPy and use the fixture matcher.
    """
    import importlib.util

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.matcher_protocol import register_matcher
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    _require_fixture()
    unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None
    members = []
    for dest in (
        "notebook_xsession_a.nwb",
        "notebook_xsession_b.nwb",
        "notebook_xsession_c.nwb",
    ):
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
        "same_day_members": members[:2],
        "concat_group_name": "notebook_concat",
        "concat_preset": "franklab_probe_hippocampus_30khz_ms4_2026_06",
        "match_members": [members[0], members[2]],
        "match_group_name": "notebook_match",
        "single_preset": "franklab_probe_hippocampus_30khz_ms4_2026_06",
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

    assert len(namespace["concat_summary"]["member_recording_ids"]) == 2
    member_merge_ids = namespace["concat_summary"]["member_merge_ids"]
    assert len(member_merge_ids) == 2
    # One unflagged-policy group per member, on its own merge id.
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
    if unitmatch_available:
        assert namespace["match_summary"]["n_tracked_units"] >= 1
        daily_summary = namespace["daily_summary"]
        inputs = {item.source_kind: item for item in daily_summary["inputs"]}
        assert len(daily_summary["inputs"]) == 2
        assert set(inputs) == {"concatenated_recording", "recording"}
        concat_input = inputs["concatenated_recording"]
        single_input = inputs["recording"]
        assert str(concat_input.sorting_id) == str(
            namespace["concat_summary"]["sorting_id"]
        )
        assert concat_input.nwb_file_names == (
            members[0]["nwb_file_name"],
            members[1]["nwb_file_name"],
        )
        assert str(single_input.sorting_id) == str(
            namespace["member_summaries"][1]["sorting_id"]
        )
        assert single_input.nwb_file_names == (members[2]["nwb_file_name"],)
        assert daily_summary["n_tracked_units"] >= 1


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
        ref = CurationRef.from_key(root).commit_merges([[0, 1]])
        units = sorted((CurationV2.Unit & ref.as_key()).fetch("unit_id"))
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
