"""Shared fixtures and opt-in gating for the v2 end-to-end acceptance probes.

The probes in this directory run the whole v2 lifecycle on the MEArec tetrode
fixture (sort, curate, review, decode, restore) and take minutes each, so they
are marked ``acceptance`` and skipped unless pytest is given
``--run-acceptance``.
"""

from pathlib import Path

import datajoint as dj
import pytest

#: The recording every lifecycle probe sorts.
TETRODE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "mearec_tetrode_60s.nwb"
)


def pytest_collection_modifyitems(config, items):
    """Skip ``acceptance``-marked items unless ``--run-acceptance`` is given.

    A conftest-level ``pytest_collection_modifyitems`` receives every item of
    the session, not only this directory's, so items are selected by marker.
    """
    if config.getoption("run_acceptance"):
        return
    skip = pytest.mark.skip(
        reason="acceptance probe: pass --run-acceptance to run"
    )
    for item in items:
        if item.get_closest_marker("acceptance") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def workflow(dj_conn, tmp_path_factory):
    """Sort the tetrode fixture with a 20-21 s exclusion, then merge two units.

    Yields the pipeline run, its root curation, a merged-and-labelled child
    curation, the ingested NWB name, the root's sorted unit ids, a scratch
    directory and a saved DataJoint config for fresh-process checks.
    """
    from spyglass.common import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not TETRODE_FIXTURE.exists():
        pytest.fail(
            f"The acceptance probes require the MEArec tetrode fixture "
            f"{TETRODE_FIXTURE}, which is absent. Fetch it with "
            "`python tests/spikesorting/v2/fixtures/_fetch.py "
            "mearec_tetrode_60s`.",
            pytrace=False,
        )
    name = copy_and_insert_nwb(
        TETRODE_FIXTURE, dest_name="lifecycle_tetrode.nwb"
    )
    initialize_v2_defaults()
    LabTeam.insert1({"team_name": "lifecycle_audit"}, skip_duplicates=True)
    SortGroupV2.set_group_by_shank(nwb_file_name=name)
    group = min((SortGroupV2 & {"nwb_file_name": name}).fetch("sort_group_id"))
    run = run_v2_pipeline(
        nwb_file_name=name,
        sort_group_id=int(group),
        interval_list_name="raw data valid times",
        team_name="lifecycle_audit",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
        manual_excluded_times=[[20.0, 21.0]],
        require_units=True,
    )
    root = run.root_curation
    ids = sorted(map(int, (Sorting.Unit & root.as_key()).fetch("unit_id")))
    assert len(ids) >= 2, "A real merge needs at least two sorted units"
    child = CurationRef.from_key(
        CurationV2.insert_curation(
            {"sorting_id": root.sorting_id},
            parent_curation_id=root.curation_id,
            labels={u: ["accept"] for u in ids},
            merge_groups=[ids[:2]],
            apply_merge=True,
            description="Lifecycle audit: controlled merge, not a biological judgment",
        )
    )
    # Verify and label the resulting unit namespace after the merge, including
    # the newly allocated merged unit; contributor IDs are no longer units.
    merged_ids = (CurationV2.Unit & child.as_key()).fetch("unit_id")
    child = CurationRef.from_key(
        CurationV2.insert_curation(
            {"sorting_id": root.sorting_id},
            parent_curation_id=child.curation_id,
            labels={int(u): ["accept"] for u in merged_ids},
            description="Lifecycle audit: verified merged population",
        )
    )
    out = tmp_path_factory.mktemp("lifecycle")
    config_file = out / "db.json"
    dj.config.save(str(config_file))
    config_file.chmod(0o600)
    yield {
        "run": run,
        "root": root,
        "child": child,
        "name": name,
        "unit_ids": ids,
        "out": out,
        "config": config_file,
    }
    config_file.unlink(missing_ok=True)
