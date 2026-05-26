"""Regenerate the pre-refactor baseline bundle for v2 spike-sorting
regression tests.

This test exists so the baseline bundle can be regenerated via a single
pytest invocation, but the test itself is intentionally not part of the
default suite. It runs the full v2 ``Recording -> Artifact -> Sort
-> Curation`` chain on the 60s MEArec polymer fixture and writes a
small bundle to ``tests/spikesorting/v2/_fixtures/phase1_baseline/``.

Workflow
--------
This test must be run on **unmodified pre-refactor code**, before any
v2 refactor edit lands. The artifacts it writes are the "what the
pre-refactor code produced" reference that the validation slice
compares against.

Workflow:

    1. ``git stash`` any in-progress refactor edits.
    2. ``rm -rf tests/spikesorting/v2/_fixtures/phase1_baseline/``.
    3. ``pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q``.
    4. Commit the resulting bundle.
    5. ``git stash pop`` to restore in-progress edits.

The test marker keeps it out of the default CI run; trigger it explicitly
with ``-m regenerate`` or by selecting the test by path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
from tests.spikesorting.v2._fixtures import phase1_baseline as _baseline

_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_128ch_60s.nwb"
)


@pytest.mark.regenerate
@pytest.mark.slow
def test_regenerate_phase1_baseline(dj_conn):
    """Run the clusterless preset on the 60s polymer fixture and write the bundle.

    Requires the DB (``dj_conn``) and the 60s MEArec polymer NWB on disk.
    Skips cleanly if either is missing -- in particular the fixture file
    is large and not under git, so a fresh checkout without
    ``generate_mearec.py`` having been run gets a clear pointer.
    """
    if not _POLYMER_60S_PATH.exists():
        pytest.skip(
            f"60s polymer fixture {_POLYMER_60S_PATH.name} missing; "
            "run `python tests/spikesorting/v2/fixtures/generate_mearec.py` "
            "(no --smoke) first."
        )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import SorterParameters

    # The polymer 60s fixture is large; copy_and_insert_nwb is idempotent.
    # Ground-truth units live in the sidecar processing module --
    # ``ImportedSpikeSorting`` only reads ``nwbfile.units`` and is
    # not needed here.
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    session_key = {"nwb_file_name": nwb_file_name}

    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_baseline_team",
            "team_description": "v2 pre-refactor baseline capture",
        },
        skip_duplicates=True,
    )

    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )

    # The shipped clusterless_thresholder ``default`` row has a 100 uV
    # threshold which finds no peaks on the polymer fixture. Tune it down
    # in place (and restore on exit) so the baseline captures a
    # non-empty unit set. This mirrors the existing pipeline test.
    default_key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }
    original_default = (SorterParameters & default_key).fetch1()
    # NOTE: the pre-refactor baseline was originally captured with
    # ``outputs="sorting"`` and ``params_schema_version=1``. The v2
    # tri-part refactor dropped that field from
    # ``ClusterlessThresholderSchema`` (now schema_version 2). The
    # regen workflow recommends running this on pre-refactor tip
    # code BEFORE the refactor lands -- if you are regenerating
    # against a fresh checkout that already includes the schema
    # edit, drop the ``outputs`` key and bump
    # ``params_schema_version`` to 2 (already done below). The
    # runtime strip path tolerates either shape, so the sort output
    # is unchanged.
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "params": {
                "detect_threshold": 5.0,
                "method": "locally_exclusive",
                "peak_sign": "neg",
                "exclude_sweep_ms": 0.1,
                "local_radius_um": 100.0,
            },
            "params_schema_version": 2,
            "job_kwargs": None,
        },
        skip_duplicates=False,
        replace=True,
    )

    try:
        bundle = _baseline.regenerate(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            team_name="v2_baseline_team",
            interval_list_name="raw data valid times",
            description="phase1 baseline (clusterless_thresholder)",
        )
    finally:
        # update1 rather than insert1(replace=True) because the
        # populate above created SortingSelection rows that FK back to
        # the row we mutated.
        SorterParameters.update1(original_default)

    assert _baseline.baseline_present(), (
        "regenerate() returned but the on-disk bundle is incomplete."
    )
    assert bundle.recording.traces.size > 0
    assert bundle.recording.timestamps.size > 0
    assert bundle.sorting.spike_samples_per_unit, (
        "regenerate() produced an empty unit set; refusing to commit an "
        "uninformative baseline."
    )
