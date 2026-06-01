"""Per-unit brain-region attribution coverage for CurationV2.

Two behavioral surfaces of ``CurationV2.get_unit_brain_regions`` that
the pipeline suite left untested:

- **Multi-region attribution**: a sort group whose electrodes span more
  than one ``BrainRegion`` must report the correct region *per unit_id*
  (the load-bearing claim of the v1 multi-region under-reporting fix).
  The existing merge-table registration test would pass even if every
  region were wrong, because it checks row counts, not the region a unit
  actually maps to.
- **``include_labels`` filter**: restricting to units carrying a given
  curation label must exercise the ``UnitLabel``-join branch and return
  only the matching units.

Both build a root curation on the shared package-scoped
``populated_sorting`` fixture (conftest) and clear any prior curation
first so they are order-independent.
"""

from __future__ import annotations

import pytest


def _clear_curations(sorting_key):
    """Drop CurationV2 rows for a sorting + their merge-table masters.

    Master-before-part: the merge insert wraps the ``CurationV2`` part FK
    in a transaction with the ``SpikeSortingOutput`` master, so the
    master must go first or DataJoint refuses the part delete.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    curation_keys = (CurationV2 & sorting_key).fetch("KEY", as_dict=True)
    if curation_keys:
        merge_ids = (
            SpikeSortingOutput.CurationV2 & curation_keys
        ).fetch("merge_id")
        for mid in merge_ids:
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & sorting_key).super_delete(warn=False)


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
    """A unit on a multi-region sort group reports the region of ITS OWN
    peak electrode -- correct per unit_id, not just the right row count.

    Mutates the ingested ``common.Electrode`` -> ``BrainRegion`` mapping
    so the sort group's channels span two regions (each unit's peak
    electrode in one, the remaining channels in another), then asserts
    ``get_unit_brain_regions`` returns each unit's TRUE region rather
    than an arbitrary region of the multi-region sort group (the v1
    under-reporting bug). Restores the original ``region_id`` values in
    teardown (the fixture is shared package-scoped state). No recording
    numerics change.
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
        pytest.skip(
            "need >=2 channels in the sort group to span two regions"
        )

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

    # Assign every unit's peak electrode to region B and all other
    # channels to region A. The sort group then spans {A, B} (a genuine
    # multi-region group), and each unit's expected region is B (its own
    # peak), NOT A (the distractor channels a buggy single-region fetch
    # might return).
    peak_keys = set(unit_peak.values())
    assign = {
        (epk["electrode_group_name"], epk["electrode_id"]): (
            region_b
            if (epk["electrode_group_name"], epk["electrode_id"]) in peak_keys
            else region_a
        )
        for epk in sg_elec_pks
    }

    try:
        for epk in sg_elec_pks:
            key = (epk["electrode_group_name"], epk["electrode_id"])
            Electrode.update1({**epk, "region_id": assign[key]})

        # Verify the precondition: the sort group genuinely spans two
        # regions after the mutation.
        sg_regions = {
            (
                Electrode * BrainRegion & epk
            ).fetch1("region_name")
            for epk in sg_elec_pks
        }
        assert sg_regions == {"v1test_region_A", "v1test_region_B"}, (
            f"sort group should span both regions; got {sg_regions}."
        )

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
        # Load-bearing assertion: the correct region for EACH unit_id.
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

    # The filter returns ONLY the mua-labeled unit (the UnitLabel-join
    # branch restricts to label-carrying units).
    filtered = CurationV2().get_unit_brain_regions(pk, include_labels=["mua"])
    assert {int(u) for u in filtered["unit_id"]} == {labeled_unit}

    # Unfiltered returns every unit; the mua filter is a strict subset
    # whenever more than one unit exists (proving it is not a no-op).
    unfiltered = CurationV2().get_unit_brain_regions(pk)
    assert {int(u) for u in unfiltered["unit_id"]} == set(unit_ids)
    if len(unit_ids) > 1:
        assert {int(u) for u in filtered["unit_id"]} < set(unit_ids)

    # A label no unit carries returns an empty frame (with full columns) --
    # confirms the filter genuinely selects on the label, not a constant.
    none_match = CurationV2().get_unit_brain_regions(
        pk, include_labels=["noise"]
    )
    assert len(none_match) == 0
    assert "region_name" in none_match.columns
