"""``CurationV2.get_unit_brain_regions``.

- A multi-region sort group reports the correct region per unit_id.
- ``include_labels`` returns only units carrying a matching label.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_P60_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


def _clear_curations(sorting_key):
    """Drop a sorting's CurationV2 rows + merge masters (shared helper)."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(sorting_key)


def _sort_group_electrodes(sorting_key):
    """Return ``(nwb_file_name, sort_group_id, [electrode PK dicts])`` for
    the single-session recording behind ``sorting_key``."""
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source(
        {"sorting_id": sorting_key["sorting_id"]}
    )
    assert source.kind == "recording", (
        "multi-region attribution test expects a single-session sorting; "
        f"got source kind {source.kind!r}."
    )
    rsel = (
        RecordingSelection & {"recording_id": source.key["recording_id"]}
    ).fetch1()
    nwb = rsel["nwb_file_name"]
    sort_group_id = int(rsel["sort_group_id"])
    elecs = (
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb, "sort_group_id": sort_group_id}
    ).fetch("electrode_group_name", "electrode_id", as_dict=True)
    elec_pks = [
        {
            "nwb_file_name": nwb,
            "electrode_group_name": e["electrode_group_name"],
            "electrode_id": int(e["electrode_id"]),
        }
        for e in elecs
    ]
    return nwb, sort_group_id, elec_pks


@pytest.mark.slow
@pytest.mark.integration
def test_multi_region_unit_attribution(populated_sorting):
    """Each unit on a multi-region sort group reports its own peak
    electrode's region.

    Temporarily remaps the shared fixture's ``common.Electrode`` regions so
    the sort group spans two regions; restores them in teardown.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})

    unit_rows = (CurationV2.Unit & pk).fetch(
        "unit_id",
        "electrode_group_name",
        "electrode_id",
        as_dict=True,
        order_by="unit_id",
    )
    nwb, _sgid, sg_elec_pks = _sort_group_electrodes(populated_sorting)
    if len(sg_elec_pks) < 2:
        pytest.skip("need >=2 channels in the sort group to span two regions")

    # Each curated unit maps to exactly one peak electrode (its FK). Key
    # by the (electrode_group_name, electrode_id) pair to match sg PKs.
    unit_peak = {
        int(r["unit_id"]): (
            r["electrode_group_name"],
            int(r["electrode_id"]),
        )
        for r in unit_rows
    }

    region_a = int(BrainRegion.fetch_add(region_name="v1test_region_A"))
    region_b = int(BrainRegion.fetch_add(region_name="v1test_region_B"))
    name_by_id = {region_a: "v1test_region_A", region_b: "v1test_region_B"}

    # Snapshot original region_id for every sort-group channel BEFORE any
    # mutation so teardown always restores, even if an update raises.
    snapshot: dict[tuple, int] = {}
    for epk in sg_elec_pks:
        key = (epk["electrode_group_name"], epk["electrode_id"])
        snapshot[key] = int((Electrode & epk).fetch1("region_id"))

    # Alternate distinct peak electrodes across the two regions and put
    # non-peak channels in A, so different units expect different regions.
    # A single unit's peak lands in B, so a fetch returning A still fails.
    distinct_peaks = sorted(set(unit_peak.values()))
    peak_region = {
        pk: (region_b if i % 2 == 0 else region_a)
        for i, pk in enumerate(distinct_peaks)
    }
    assign = {
        (epk["electrode_group_name"], epk["electrode_id"]): peak_region.get(
            (epk["electrode_group_name"], epk["electrode_id"]), region_a
        )
        for epk in sg_elec_pks
    }

    try:
        for epk in sg_elec_pks:
            key = (epk["electrode_group_name"], epk["electrode_id"])
            Electrode.update1({**epk, "region_id": assign[key]})

        # Precondition: the sort group spans two regions.
        sg_regions = {
            (Electrode * BrainRegion & epk).fetch1("region_name")
            for epk in sg_elec_pks
        }
        assert sg_regions == {
            "v1test_region_A",
            "v1test_region_B",
        }, f"sort group should span both regions; got {sg_regions}."

        expected = {
            uid: name_by_id[assign[peak]] for uid, peak in unit_peak.items()
        }
        df = CurationV2().get_unit_brain_regions(pk)
        # One row per unit (Unit -> Electrode -> BrainRegion are all
        # single-valued FKs); a duplicate would corrupt the dict below.
        assert len(df) == len(unit_peak)
        got = {
            int(u): region
            for u, region in zip(df["unit_id"], df["region_name"])
        }
        # The correct region for each unit_id.
        assert got == expected, (
            f"per-unit region attribution wrong: got {got}, expected "
            f"{expected}."
        )
    finally:
        for epk in sg_elec_pks:
            key = (epk["electrode_group_name"], epk["electrode_id"])
            Electrode.update1({**epk, "region_id": snapshot[key]})


@pytest.mark.slow
@pytest.mark.integration
def test_get_unit_brain_regions_include_labels(populated_sorting):
    """``include_labels`` restricts the result to units carrying one of
    the given labels (the ``UnitLabel``-join branch)."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    if len(unit_ids) < 1:
        pytest.skip("need >=1 unit to exercise the include_labels branch")

    labeled_unit = unit_ids[0]
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={labeled_unit: ["mua"]},
    )

    # Only the mua-labeled unit is returned.
    filtered = CurationV2().get_unit_brain_regions(pk, include_labels=["mua"])
    assert {int(u) for u in filtered["unit_id"]} == {labeled_unit}

    # Unfiltered returns every unit; the mua filter is a strict subset
    # whenever more than one unit exists (proving it is not a no-op).
    unfiltered = CurationV2().get_unit_brain_regions(pk)
    assert {int(u) for u in unfiltered["unit_id"]} == set(unit_ids)
    if len(unit_ids) > 1:
        assert {int(u) for u in filtered["unit_id"]} < set(unit_ids)

    # A label no unit carries returns an empty frame with full columns.
    none_match = CurationV2().get_unit_brain_regions(
        pk, include_labels=["noise"]
    )
    assert len(none_match) == 0
    assert "region_name" in none_match.columns


def test_get_unit_brain_regions_empty_include_labels_returns_all(
    populated_sorting,
):
    """An empty ``include_labels`` means no filter (all units), not "match
    nothing" (``& []`` would return zero units)."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    if len(unit_ids) < 1:
        pytest.skip("need >=1 unit to exercise the include_labels branch")

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={unit_ids[0]: ["mua"]},
    )

    all_units = CurationV2().get_unit_brain_regions(pk)
    empty_filter = CurationV2().get_unit_brain_regions(pk, include_labels=[])
    assert {int(u) for u in empty_filter["unit_id"]} == set(
        unit_ids
    ), "include_labels=[] must return all units (no filter), not zero"
    assert len(empty_filter) == len(all_units)


@pytest.fixture(scope="module")
def region_60s_sort(dj_conn):
    """Sort shank 0 of the 60s polymer fixture and yield its sorting PK.

    The 60s shank yields several units with distinct peak electrodes (the
    smoke fixture can yield one). Ingested under a unique session name and
    cleaned on teardown.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    if not _P60_PATH.exists():
        pytest.skip(f"60s fixture {_P60_PATH.name} not found.")

    nwb = copy_and_insert_nwb(_P60_PATH, dest_name="mearec_region60.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 region attr"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)

    yield sort_pk

    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_multi_region_attribution_through_merge_dispatch(region_60s_sort):
    """Per-unit multi-region attribution on a multi-unit sort, through both
    ``CurationV2.get_unit_brain_regions`` and the merge-level
    ``SpikeSortingOutput.get_unit_brain_regions``.

    Requires >= 2 units with distinct peak electrodes and >= 2 expected
    regions. Not checked against MEArec soma positions: on this geometry the
    peak channel is the soma-nearest contact only ~50% of the time.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = region_60s_sort
    _clear_curations(sort_pk)
    pk = CurationV2.insert_curation(sorting_key=sort_pk, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    unit_rows = (CurationV2.Unit & pk).fetch(
        "unit_id",
        "electrode_group_name",
        "electrode_id",
        as_dict=True,
        order_by="unit_id",
    )
    unit_peak = {
        int(r["unit_id"]): (r["electrode_group_name"], int(r["electrode_id"]))
        for r in unit_rows
    }
    distinct_peaks = sorted(set(unit_peak.values()))
    if len(distinct_peaks) < 2:
        pytest.skip(
            "need >=2 distinct peak electrodes for multi-region "
            f"discrimination; got {len(distinct_peaks)}"
        )

    nwb, _sgid, sg_elec_pks = _sort_group_electrodes(sort_pk)
    # fetch_add is idempotent; these two test-only BrainRegion rows are left
    # in the shared common.BrainRegion (cleaning them would require restoring
    # the Electrode FK first). Reused across reruns, never duplicated.
    region_a = int(BrainRegion.fetch_add(region_name="v2_60s_region_A"))
    region_b = int(BrainRegion.fetch_add(region_name="v2_60s_region_B"))
    name_by_id = {region_a: "v2_60s_region_A", region_b: "v2_60s_region_B"}

    snapshot: dict[tuple, int] = {}
    for epk in sg_elec_pks:
        key = (epk["electrode_group_name"], epk["electrode_id"])
        snapshot[key] = int((Electrode & epk).fetch1("region_id"))

    # Alternate distinct peak electrodes across the two regions; every
    # non-peak channel goes to region A.
    peak_region = {
        peak: (region_b if i % 2 == 0 else region_a)
        for i, peak in enumerate(distinct_peaks)
    }
    assign = {
        (epk["electrode_group_name"], epk["electrode_id"]): peak_region.get(
            (epk["electrode_group_name"], epk["electrode_id"]), region_a
        )
        for epk in sg_elec_pks
    }

    try:
        for epk in sg_elec_pks:
            Electrode.update1(
                {
                    **epk,
                    "region_id": assign[
                        (epk["electrode_group_name"], epk["electrode_id"])
                    ],
                }
            )

        expected = {
            uid: name_by_id[assign[peak]] for uid, peak in unit_peak.items()
        }
        # Units map to more than one region.
        assert len(set(expected.values())) >= 2, (
            "alternating assignment should yield >=2 distinct per-unit "
            f"regions; got {set(expected.values())}"
        )

        # Path 1: CurationV2 accessor directly.
        df = CurationV2().get_unit_brain_regions(pk)
        assert len(df) == len(unit_peak)
        got = {int(u): r for u, r in zip(df["unit_id"], df["region_name"])}
        assert got == expected, f"CurationV2 path: got {got}, want {expected}"

        # Path 2: the consumer-facing merge dispatcher.
        df2 = SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})
        got2 = {int(u): r for u, r in zip(df2["unit_id"], df2["region_name"])}
        assert (
            got2 == expected
        ), f"merge-dispatch path: got {got2}, want {expected}"
    finally:
        # Restore every electrode even if one update raises.
        restore_errors = []
        for epk in sg_elec_pks:
            try:
                Electrode.update1(
                    {
                        **epk,
                        "region_id": snapshot[
                            (epk["electrode_group_name"], epk["electrode_id"])
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - aggregate + re-raise
                restore_errors.append((epk["electrode_id"], exc))
        _clear_curations(sort_pk)
        if restore_errors:
            raise RuntimeError(
                f"failed to restore region_id for electrodes {restore_errors}"
            )
