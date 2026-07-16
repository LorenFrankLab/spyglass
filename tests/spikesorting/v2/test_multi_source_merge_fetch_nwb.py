"""Real-merge-master test for multi_source SpikeSortingOutput.fetch_nwb.

The hermetic two-source tests in ``tests/utils/test_merge_consumer_boundary.py``
exercise the generic ``_Merge.fetch_nwb`` rewrite on a synthetic merge table
with single-int-PK leaves. This module is the pre-merge gate the code review
called for: it builds a REAL ``SpikeSortingOutput`` with rows from two
DISTINCT source part types and exercises:

- the multi_source MASTER path on real parts -- a ``merge_id``-spanning
  restriction over the two sources, aligned per file (``..._aligned``);
- the parent-attribute JOIN path on a real COMPOSITE-PK parent -- restricting
  by ``CurationV2``'s ``(sorting_id, curation_id)`` PK routes through
  ``self * part * parent`` and the ``parent.primary_key`` / merge_id-by-PK
  mapping (``..._parent_key_join_branch``);
- the migrated ``SortedSpikesGroup`` consumer over a 2-source group.

Under SI 0.104 the only two source types whose ``fetch_nwb`` works are
``CurationV2`` (the v2 pipeline) and ``ImportedSpikeSorting`` (reads
``nwb_file.units`` directly; only its ``get_recording``/``get_sorting`` raise
NotImplementedError). v0/v1 sources are import-incompatible with SI 0.104.
The ``ImportedSpikeSorting`` source needs an NWB with a populated ``units``
table -- the MEArec fixtures deliberately keep ``nwbfile.units`` empty, so we
plant a small units table onto a copy and ingest it as its own session.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import (
    clear_curations_for as _clear_curations_for,
)

_SMOKE = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


def _build_units_nwb(dst: Path) -> Path:
    """Copy the smoke NWB and plant a small ``units`` table.

    The smoke NWB carries all Spyglass-required session metadata
    (subject/devices/electrodes); only ``units`` is missing, which is what
    ``ImportedSpikeSorting`` ingests.
    """
    import pynwb

    shutil.copy(_SMOKE, dst)
    with pynwb.NWBHDF5IO(str(dst), "a", load_namespaces=True) as io:
        nwb = io.read()
        for i in range(3):
            base = 0.1 * i
            nwb.add_unit(spike_times=[base + 0.01, base + 0.05, base + 0.09])
        io.write(nwb)
    return dst


@pytest.fixture(scope="module")
def two_source_output(populated_sorting, dj_conn, tmp_path_factory):
    """A SpikeSortingOutput holding one CurationV2 row and one
    ImportedSpikeSorting row (two distinct source part types).

    Yields the two merge ids, the imported source's nwb_file_name, and the
    v2 session's nwb_file_name (for the SortedSpikesGroup consumer test).
    """
    if not _SMOKE.exists():
        pytest.skip("smoke fixture missing")

    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    from spyglass.spikesorting.imported import ImportedSpikeSorting
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # --- v2 source (CurationV2) ---
    _clear_curations_for(populated_sorting)
    cur_v2 = CurationV2.insert_curation(sorting_key=populated_sorting)
    mid_v2 = (SpikeSortingOutput.CurationV2 & cur_v2).fetch1("merge_id")
    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    v2_nwb_file_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")

    # --- imported source (ImportedSpikeSorting) ---
    units_nwb = _build_units_nwb(
        tmp_path_factory.mktemp("imported") / "mearec_imported_smoke.nwb"
    )
    imported_name = copy_and_insert_nwb(units_nwb)
    imp_rows = (ImportedSpikeSorting & {"nwb_file_name": imported_name}).fetch(
        "KEY", as_dict=True
    )
    assert imp_rows, "ImportedSpikeSorting was not auto-populated"
    SpikeSortingOutput.insert(
        imp_rows, part_name="ImportedSpikeSorting", skip_duplicates=True
    )
    mid_imported = (
        SpikeSortingOutput.ImportedSpikeSorting & imp_rows[0]
    ).fetch1("merge_id")

    yield {
        "mid_v2": mid_v2,
        "cur_v2": cur_v2,
        "mid_imported": mid_imported,
        "imported_name": imported_name,
        "v2_nwb_file_name": v2_nwb_file_name,
    }

    # cleanup: drop the merge rows (master-first), then the v2 curation.
    for mid in (mid_v2, mid_imported):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    _clear_curations_for(populated_sorting)


@pytest.mark.slow
@pytest.mark.integration
def test_spikesortingoutput_multi_source_fetch_nwb_aligned(
    two_source_output, caplog
):
    """A SpikeSortingOutput restriction spanning CurationV2 +
    ImportedSpikeSorting: default WARNS and fetches across both;
    multi_source=True returns one merge_id per file, each aligned to its
    owning source's file (and silences the warning)."""
    import logging

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_source_output
    merge_keys = [
        {"merge_id": ctx["mid_v2"]},
        {"merge_id": ctx["mid_imported"]},
    ]

    # Default (multi_source=False): a 2-source restriction now WARNS (not
    # raises) and fetches across both sources.
    with caplog.at_level(logging.WARNING):
        nwb_default, mids_default = (SpikeSortingOutput & merge_keys).fetch_nwb(
            return_merge_ids=True
        )
    assert "multi_source=True" in caplog.text
    assert len(nwb_default) == 2
    assert set(mids_default) == {ctx["mid_v2"], ctx["mid_imported"]}

    # multi_source=True: aligned (nwb_list, merge_ids) across the two sources.
    nwb_list, merge_ids = (SpikeSortingOutput & merge_keys).fetch_nwb(
        return_merge_ids=True, multi_source=True
    )
    assert len(nwb_list) == 2
    assert len(merge_ids) == 2
    assert set(merge_ids) == {ctx["mid_v2"], ctx["mid_imported"]}

    # The two ids resolve to the two distinct source part types.
    sources = {
        (SpikeSortingOutput & {"merge_id": m}).fetch1("source")
        for m in merge_ids
    }
    assert sources == {"CurationV2", "ImportedSpikeSorting"}

    # Ownership/alignment: the imported source's file (a raw NWB whose
    # nwb_file_name is the planted-units session) must pair with
    # mid_imported; the other with mid_v2.
    imported_idx = [
        i
        for i, f in enumerate(nwb_list)
        if f.get("nwb_file_name") == ctx["imported_name"]
    ]
    assert len(imported_idx) == 1, "imported file not uniquely identifiable"
    assert merge_ids[imported_idx[0]] == ctx["mid_imported"]
    assert merge_ids[1 - imported_idx[0]] == ctx["mid_v2"]


@pytest.mark.slow
@pytest.mark.integration
def test_spikesortingoutput_parent_key_join_branch(two_source_output):
    """Restricting by CurationV2's composite primary key
    ``(sorting_id, curation_id)`` routes through the parent-attribute join
    branch (``self * part * parent``) on a REAL composite-PK parent and
    resolves the single owning file + merge_id.

    The merge_id-spanning test above stays on the master path; this is the
    only case that drives the join branch / ``parent.primary_key`` mapping
    against a real (non-synthetic, multi-field-PK) parent.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_source_output
    # cur_v2 is the CurationV2 PK dict {sorting_id, curation_id} -- both are
    # parent (not master) attributes, so this enters the join branch.
    nwb_list, merge_ids = SpikeSortingOutput().fetch_nwb(
        ctx["cur_v2"], return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert merge_ids == [ctx["mid_v2"]]


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_group_fetch_spike_data_spans_sources(
    two_source_output,
):
    """SortedSpikesGroup.fetch_spike_data over a group whose units span
    CurationV2 + ImportedSpikeSorting works end-to-end (the migrated
    multi_source=True caller), returning spikes from both sources."""
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )

    ctx = two_source_output
    UnitSelectionParams.insert_default()
    group_name = "two_source_group"
    group_key = {
        "nwb_file_name": ctx["v2_nwb_file_name"],
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "all_units",
    }
    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=ctx["v2_nwb_file_name"],
        unit_filter_params_name="all_units",
        keys=[
            {"spikesorting_merge_id": ctx["mid_v2"]},
            {"spikesorting_merge_id": ctx["mid_imported"]},
        ],
    )

    spike_times = SortedSpikesGroup.fetch_spike_data(group_key)
    # Units from BOTH sources are returned (the imported NWB planted 3 units;
    # the v2 sort contributes its curated units), and nothing raised.
    assert len(spike_times) >= 3

    (SortedSpikesGroup & group_key).super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_unit_annotation_hard_blocks_multi_source(two_source_output):
    """UnitAnnotation.fetch_unit_spikes raises on a restriction spanning >1
    SpikeSortingOutput source, but a single-source restriction still fetches.

    Merge.fetch_nwb now only WARNS on a multi-source restriction (it used to
    raise), so annotations -- which are single-analysis by design -- carry the
    hard block in fetch_unit_spikes itself. Unlike SortedSpikesGroup (above),
    which intentionally spans sources, UnitAnnotation must not silently mix
    spike-time namespaces.
    """
    from spyglass.spikesorting.analysis.v1.group import (
        _get_nwb_unit_ids,
        _get_spike_obj_name,
    )
    from spyglass.spikesorting.analysis.v1.unit_annotation import (
        UnitAnnotation,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ctx = two_source_output

    def _valid_unit_id(merge_id):
        nwb = (SpikeSortingOutput & {"merge_id": merge_id}).fetch_nwb()[0]
        field = _get_spike_obj_name(nwb)
        return int(sorted(_get_nwb_unit_ids(nwb, field))[0])

    ua = UnitAnnotation()
    inserted = []
    try:
        for mid in (ctx["mid_v2"], ctx["mid_imported"]):
            unit_id = _valid_unit_id(mid)
            ua.add_annotation(
                {
                    "spikesorting_merge_id": mid,
                    "unit_id": unit_id,
                    "annotation": "cell_type",
                    "label": "test",
                },
                skip_duplicates=True,
            )
            inserted.append({"spikesorting_merge_id": mid, "unit_id": unit_id})

        # Single source: the guard must NOT false-raise -- spikes are returned.
        single = (
            UnitAnnotation & {"spikesorting_merge_id": ctx["mid_v2"]}
        ).fetch_unit_spikes()
        assert len(single) >= 1

        # Two sources: hard-blocked with an actionable error.
        with pytest.raises(ValueError, match="multiple SpikeSortingOutput"):
            (
                UnitAnnotation
                & [
                    {"spikesorting_merge_id": ctx["mid_v2"]},
                    {"spikesorting_merge_id": ctx["mid_imported"]},
                ]
            ).fetch_unit_spikes()
    finally:
        for key in inserted:
            (UnitAnnotation & key).delete(safemode=False)
