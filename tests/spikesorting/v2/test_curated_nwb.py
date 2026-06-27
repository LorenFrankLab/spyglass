"""CNEP-1: curated-units NWBs carry per-unit ``obs_intervals``.

The sort-time writer wrote a per-unit observation window, but the curated
writer dropped it and the combined reader never read it back -- so any NWB-only
firing-rate / presence-ratio / duration denominator over a curated export
silently assumed the full session. The reader now returns ``obs`` and the
curated writer carries it forward (intersection for merged contributors).

Hermetic NWB-IO tests + one DB integration round-trip.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest


def _write_units_nwb(path, specs, *, with_obs=True):
    """Write a units NWB. ``specs`` is ``[(id, spike_times, obs_intervals)]``;
    ``with_obs=False`` omits the obs_intervals column (legacy file)."""
    from pynwb import NWBFile, NWBHDF5IO

    nwbf = NWBFile(
        "s", "i", datetime.datetime.now(datetime.timezone.utc)
    )
    for uid, st, obs in specs:
        kwargs = {"spike_times": st, "id": uid}
        if with_obs:
            kwargs["obs_intervals"] = obs
        nwbf.add_unit(**kwargs)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbf)


# ---------- intersect_interval_sets (pure) ----------------------------------


def test_intersect_interval_sets():
    from spyglass.spikesorting.v2._signal_math import intersect_interval_sets

    full = np.array([[0.0, 1.0]])
    # identical inputs -> the shared window (common single-sort case).
    np.testing.assert_array_equal(intersect_interval_sets([full, full]), full)
    # differing -> the conservative intersection.
    gapped = np.array([[0.0, 0.5], [0.7, 1.0]])
    np.testing.assert_array_equal(
        intersect_interval_sets([full, gapped]), gapped
    )
    # disjoint -> empty.
    assert (
        intersect_interval_sets(
            [np.array([[0.0, 0.4]]), np.array([[0.6, 1.0]])]
        ).size
        == 0
    )
    # degenerate inputs.
    assert intersect_interval_sets([]).shape == (0, 2)
    np.testing.assert_array_equal(intersect_interval_sets([full]), full)


# ---------- _curated_obs_intervals (merge rule) -----------------------------


def test_curated_obs_intervals_merge_rule():
    from spyglass.spikesorting.v2._units_nwb import _curated_obs_intervals

    obs = {
        0: np.array([[0.0, 1.0]]),
        1: np.array([[0.0, 0.5], [0.7, 1.0]]),
    }
    # Merged contributors with DIFFERING windows -> intersection.
    np.testing.assert_array_equal(
        _curated_obs_intervals(0, [0, 1], True, obs),
        np.array([[0.0, 0.5], [0.7, 1.0]]),
    )
    # Singleton / preview -> the unit's own window.
    np.testing.assert_array_equal(
        _curated_obs_intervals(1, [1], False, obs), obs[1]
    )
    # Legacy source (no obs column) -> None (no obs written).
    assert _curated_obs_intervals(0, [0, 1], True, None) is None


# ---------- reader returns obs ----------------------------------------------


def test_reader_returns_obs_intervals(tmp_path):
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
    )

    p = tmp_path / "obs.nwb"
    _write_units_nwb(
        p, [(0, [0.1], [[0.0, 1.0]]), (1, [0.2], [[0.0, 0.5]])]
    )
    _abs, _samp, obs = read_units_abs_times_and_sample_indices(str(p))
    assert obs is not None
    np.testing.assert_array_equal(obs[0], [[0.0, 1.0]])
    np.testing.assert_array_equal(obs[1], [[0.0, 0.5]])

    # A legacy file without the obs column -> obs is None.
    p_legacy = tmp_path / "noobs.nwb"
    _write_units_nwb(p_legacy, [(0, [0.1], None)], with_obs=False)
    _abs2, _samp2, obs2 = read_units_abs_times_and_sample_indices(str(p_legacy))
    assert obs2 is None


# ---------- DB integration: curated export carries obs ----------------------


@pytest.mark.slow
@pytest.mark.integration
def test_curated_nwb_carries_merge_lineage(planted_two_unit_sort):
    """The curated NWB embeds the kept->contributor merge lineage + a header.

    An applied merge writes the kept-unit->contributors map (matching
    ``CurationV2.MergeGroup``) flagged applied; a preview curation writes the
    proposed groups flagged not-applied. A small header records the curation
    identity/source so the file is interpretable without the DB.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        CURATION_MERGE_LINEAGE,
        CURATION_PROVENANCE,
        read_long_provenance,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sort = planted_two_unit_sort
    unit_ids = sorted(int(u) for u in (Sorting.Unit & sort).fetch("unit_id"))

    def _merge_group_pairs(curation_key):
        # The lineage table records actual MERGES -- kept units with >1
        # contributor. MergeGroup also stores singleton self-rows
        # (unit -> itself) for unmerged units, which are identity, not lineage;
        # exclude them to compare against the merge structure.
        from collections import Counter

        units, contribs = (CurationV2.MergeGroup & curation_key).fetch(
            "unit_id", "contributor_unit_id"
        )
        counts = Counter(int(u) for u in units)
        return {
            (int(u), int(c))
            for u, c in zip(units, contribs)
            if counts[int(u)] > 1
        }

    def _lineage_pairs(curation_key):
        abs_path = AnalysisNwbfile.get_abs_path(
            (CurationV2 & curation_key).fetch1("analysis_file_name")
        )
        rows = read_long_provenance(abs_path, CURATION_MERGE_LINEAGE)
        header = read_provenance_values(abs_path, CURATION_PROVENANCE)
        return rows, header, abs_path

    clear_curations_for(sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sort)

        # Applied merge -> kept->contributor map matching MergeGroup, applied.
        merged = CurationV2.insert_curation(
            sorting_key=sort,
            parent_curation_id=root["curation_id"],
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
            description="merged pair",
        )
        rows, header, _ = _lineage_pairs(merged)
        assert {(r["kept_unit_id"], r["contributor_unit_id"]) for r in rows} == (
            _merge_group_pairs(merged)
        )
        assert all(r["applied"] is True for r in rows)
        assert header["sorting_id"] == str(sort["sorting_id"])
        assert header["curation_id"] == int(merged["curation_id"])
        assert header["parent_curation_id"] == int(root["curation_id"])
        assert header["merges_applied"] is True
        assert header["description"] == "merged pair"
        merged_row = (CurationV2 & merged).fetch1()
        assert header["curation_source"] == merged_row["curation_source"]

        # Preview (proposed, not applied).
        preview = CurationV2.insert_curation(
            sorting_key=sort,
            parent_curation_id=root["curation_id"],
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=False,
            description="proposed pair",
        )
        rows_p, header_p, _ = _lineage_pairs(preview)
        assert {
            (r["kept_unit_id"], r["contributor_unit_id"]) for r in rows_p
        } == _merge_group_pairs(preview)
        assert all(r["applied"] is False for r in rows_p)
        assert header_p["merges_applied"] is False
    finally:
        clear_curations_for(sort)


@pytest.mark.slow
@pytest.mark.integration
def test_curated_units_carry_obs_intervals(planted_two_unit_sort):
    """A curated export's per-unit obs_intervals match the source sort, and a
    merged unit gets the intersection of its contributors' windows."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._signal_math import intersect_interval_sets
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sort = planted_two_unit_sort
    unit_ids = sorted(int(u) for u in (Sorting.Unit & sort).fetch("unit_id"))

    src_abs = AnalysisNwbfile.get_abs_path(
        (Sorting & sort).fetch1("analysis_file_name")
    )
    _a, _s, src_obs = read_units_abs_times_and_sample_indices(src_abs)
    assert src_obs is not None, "source sort must carry obs_intervals"

    clear_curations_for(sort)
    try:
        # Root curation: every unit's curated obs matches the source.
        root = CurationV2.insert_curation(sorting_key=sort)
        root_abs = AnalysisNwbfile.get_abs_path(
            (CurationV2 & root).fetch1("analysis_file_name")
        )
        _a2, _s2, root_obs = read_units_abs_times_and_sample_indices(root_abs)
        assert root_obs is not None
        for uid in unit_ids:
            np.testing.assert_array_equal(root_obs[uid], src_obs[uid])

        # Merged curation: the merged unit's obs is the contributors'
        # intersection (= the shared window for one sort).
        merged = CurationV2.insert_curation(
            sorting_key=sort,
            parent_curation_id=root["curation_id"],
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
        )
        merged_abs = AnalysisNwbfile.get_abs_path(
            (CurationV2 & merged).fetch1("analysis_file_name")
        )
        _a3, _s3, merged_obs = read_units_abs_times_and_sample_indices(
            merged_abs
        )
        assert merged_obs is not None
        expected = intersect_interval_sets(
            [src_obs[unit_ids[0]], src_obs[unit_ids[1]]]
        )
        merged_uid = max(unit_ids) + 1  # apply_merge mints max+1
        np.testing.assert_array_equal(merged_obs[merged_uid], expected)
    finally:
        clear_curations_for(sort)
