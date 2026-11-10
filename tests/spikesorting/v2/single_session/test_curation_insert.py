"""CurationV2 insert, label, and validation tests for the v2 single-session pipeline."""

from __future__ import annotations

import uuid

import pytest

from tests.spikesorting.v2.single_session._helpers import _clear_curations
from tests.spikesorting.v2._ingest_helpers import matched_unit_ids


@pytest.mark.slow
def test_curation_v2_insert_root_unlabeled(populated_sorting):
    """``insert_curation(labels={})`` writes a root curation with no
    ``UnitLabel`` rows and a ``CurationV2.Unit`` row per sorted unit."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    # Stable curation_id.
    _clear_curations(populated_sorting)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    assert pk == {**populated_sorting, "curation_id": 0}
    assert len(CurationV2 & pk) == 1
    # Pin the id set, not just the count.
    expected_unit_ids = set(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    curated_unit_ids = set(
        int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
    )
    assert curated_unit_ids == expected_unit_ids, (
        f"CurationV2.Unit unit_ids {sorted(curated_unit_ids)} differ "
        f"from Sorting.Unit unit_ids {sorted(expected_unit_ids)} for "
        "a root curation with no merges -- units should pass through "
        "1:1."
    )
    assert len(CurationV2.UnitLabel & pk) == 0

    row = (CurationV2 & pk).fetch1()
    assert row["parent_curation_id"] == -1
    assert row["merges_applied"] == 0
    assert row["curation_source"] == "manual"
    assert isinstance(row["curation_uuid"], uuid.UUID)


@pytest.mark.slow
def test_curation_uuid_immutable_and_unique(populated_sorting):
    """Each curation generation gets one DB-unique, immutable UUID."""
    import datajoint as dj

    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    try:
        root = CurationV2.insert_curation(populated_sorting)
        child = CurationV2.insert_curation(
            populated_sorting,
            parent_curation_id=root["curation_id"],
            description="uuid child",
        )
        rows = (CurationV2 & [root, child]).fetch(as_dict=True)
        uuids = [row["curation_uuid"] for row in rows]
        assert len(uuids) == len(set(uuids)) == 2
        assert all(isinstance(value, uuid.UUID) for value in uuids)

        root_row = (CurationV2 & root).fetch1()
        with pytest.raises(dj.errors.DataJointError, match="In-place update1"):
            CurationV2.update1({**root_row, "curation_uuid": uuid.uuid4()})

        # The UNIQUE INDEX also blocks a factory bypass reusing a token.
        child_row = (CurationV2 & child).fetch1()
        with pytest.raises(dj.errors.DuplicateError):
            CurationV2.insert1(
                {
                    **child_row,
                    "curation_id": child["curation_id"] + 1,
                    "parent_curation_id": root["curation_id"],
                },
                allow_direct_insert=True,
            )
    finally:
        _clear_curations(populated_sorting)


@pytest.mark.slow
def test_curation_id_reuse_yields_new_uuid(populated_sorting):
    """Delete/recreate may reuse the numeric id but never its generation UUID."""
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    try:
        root = CurationV2.insert_curation(populated_sorting)
        first = CurationV2.insert_curation(
            populated_sorting,
            parent_curation_id=root["curation_id"],
            description="replaceable child",
        )
        first_uuid = (CurationV2 & first).fetch1("curation_uuid")

        (CurationV2 & first).delete(safemode=False)
        replacement = CurationV2.insert_curation(
            populated_sorting,
            parent_curation_id=root["curation_id"],
            description="replaceable child",
        )
        replacement_uuid = (CurationV2 & replacement).fetch1("curation_uuid")
        assert replacement["curation_id"] == first["curation_id"]
        assert replacement_uuid != first_uuid
    finally:
        _clear_curations(populated_sorting)


@pytest.mark.slow
def test_reuse_existing_keeps_uuid(populated_sorting):
    """Idempotent child reuse returns the exact existing generation."""
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    try:
        root = CurationV2.insert_curation(populated_sorting)
        child = CurationV2.insert_curation(
            populated_sorting,
            parent_curation_id=root["curation_id"],
            description="reusable child",
            reuse_existing=True,
        )
        first_uuid = (CurationV2 & child).fetch1("curation_uuid")
        reused = CurationV2.insert_curation(
            populated_sorting,
            parent_curation_id=root["curation_id"],
            description="reusable child",
            reuse_existing=True,
        )
        assert reused == child
        assert (CurationV2 & reused).fetch1("curation_uuid") == first_uuid
    finally:
        _clear_curations(populated_sorting)


@pytest.mark.slow
def test_curation_source_invalid_preserves_cause(populated_sorting):
    """A bogus ``curation_source`` re-raise keeps the enum error as
    ``__cause__``. Needs a real Sorting row to reach the enum check.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    with pytest.raises(ValueError) as excinfo:
        CurationV2.insert_curation(
            sorting_key=populated_sorting, curation_source="bogus"
        )
    assert "curation_source" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert excinfo.value.__cause__ is not excinfo.value


@pytest.mark.slow
def test_curation_source_invalid_rejected_with_existing_root(populated_sorting):
    """A bogus ``curation_source`` raises even when a root already exists,
    instead of returning the existing root.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    with pytest.raises(ValueError) as excinfo:
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            curation_source="not_a_real_source",
        )
    assert "curation_source" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.slow
def test_curation_v2_insert_with_labels(populated_sorting):
    """Labels round-trip into ``CurationV2.UnitLabel`` and unknown
    labels raise before any DB rows are written."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)

    units = (Sorting.Unit & populated_sorting).fetch("unit_id")
    assert len(units) > 0
    target_unit = int(units[0])
    labels = {target_unit: ["mua", "accept"]}

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=labels,
        description="manual labels test",
    )
    label_rows = (CurationV2.UnitLabel & pk).fetch(as_dict=True)
    label_set = {(r["unit_id"], r["curation_label"]) for r in label_rows}
    assert (target_unit, "mua") in label_set
    assert (target_unit, "accept") in label_set
    assert (CurationV2 & pk).fetch1("description") == "manual labels test"

    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: ["good"]},  # v1-era label not in v2 enum
        )


@pytest.mark.slow
def test_curation_v2_parent_validation_and_nonincreasing_ids(populated_sorting):
    """parent_curation_id must reference an existing row for the same
    sort; auto-incremented curation_id starts at 0 and increments.

    ``labels=None`` means no labels, as in v1.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    pk0 = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    assert pk0["curation_id"] == 0

    pk1 = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={},
        parent_curation_id=0,
    )
    assert pk1["curation_id"] == 1
    assert (CurationV2 & pk1).fetch1("parent_curation_id") == 0

    with pytest.raises(ValueError, match="does not exist for sorting_id"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={},
            parent_curation_id=999,
        )

    # ``labels=None`` is accepted (equivalent to ``labels={}``).
    pk_none = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=None,
        parent_curation_id=0,
    )
    assert pk_none["curation_id"] == 2


@pytest.mark.slow
def test_curation_v2_get_sort_group_info_returns_all_electrodes(
    populated_sorting,
):
    """``get_sort_group_info`` returns a DataJoint relation covering
    every electrode in the sort group, not one row like v1."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    rel = CurationV2().get_sort_group_info(pk)
    rec_row = (
        RecordingSelection
        & (SortingSelection.RecordingSource & populated_sorting)
    ).fetch1()
    expected = len(
        SortGroupV2.SortGroupElectrode
        & {
            "nwb_file_name": rec_row["nwb_file_name"],
            "sort_group_id": rec_row["sort_group_id"],
        }
    )
    assert len(rel) == expected
    assert expected > 1  # must be a multi-row relation, NOT v1's limit=1


@pytest.mark.slow
def test_curation_v2_get_sorting_round_trips_spike_times(populated_sorting):
    """``get_sorting`` returns a SI BaseSorting whose spike times match
    the upstream Sorting's after the curation is applied (no merges,
    no labels => identical spike trains)."""
    import numpy as np

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    src_sorting = Sorting().get_sorting(populated_sorting)
    cur_sorting = CurationV2().get_sorting(pk)
    assert set(cur_sorting.unit_ids) == set(src_sorting.unit_ids)
    for uid in src_sorting.unit_ids:
        src_times = src_sorting.get_unit_spike_train(uid, return_times=True)
        cur_times = cur_sorting.get_unit_spike_train(
            int(uid), return_times=True
        )
        assert len(src_times) == len(cur_times)
        if len(src_times):
            assert np.allclose(np.sort(src_times), np.sort(cur_times))


@pytest.mark.slow
def test_curation_v2_auto_registers_in_merge_table(populated_sorting):
    """``insert_curation`` writes the matching
    ``SpikeSortingOutput.CurationV2`` part row in the same transaction;
    a merge_id resolves back through ``get_unit_brain_regions``."""
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    assert "CurationV2" in source_class_dict
    assert hasattr(SpikeSortingOutput, "CurationV2")

    _clear_curations(populated_sorting)
    # Also clear stale merge rows for this sort.
    stale = (SpikeSortingOutput.CurationV2 & populated_sorting).fetch(
        "merge_id"
    )
    for mid in stale:
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_rows = (SpikeSortingOutput.CurationV2 & pk).fetch(as_dict=True)
    assert len(merge_rows) == 1
    merge_id = merge_rows[0]["merge_id"]
    assert (SpikeSortingOutput & {"merge_id": merge_id}).fetch1(
        "source"
    ) == "CurationV2"

    # Pin row count and unit ids: an empty frame would mean a broken join.
    df = SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})
    assert "unit_id" in df.columns
    assert "region_resolution" in df.columns
    expected_unit_ids = set(
        int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
    )
    assert len(df) == len(expected_unit_ids), (
        f"get_unit_brain_regions returned {len(df)} rows; expected "
        f"{len(expected_unit_ids)} (one per CurationV2.Unit row)."
    )
    assert set(int(u) for u in df["unit_id"]) == expected_unit_ids, (
        f"get_unit_brain_regions unit_ids {sorted(df['unit_id'])} "
        f"differ from CurationV2.Unit unit_ids "
        f"{sorted(expected_unit_ids)}; the dispatch returned the "
        "wrong rows."
    )


def test_unit_brain_region_df_empty_keeps_full_schema(dj_conn):
    """An empty unit-region lookup returns the full column schema.

    Empty and non-empty results share a shape.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2._recording.unit_metadata import (
        unit_brain_region_df,
    )

    empty_units = CurationV2.Unit & "unit_id = -1"  # no unit has id -1
    df = unit_brain_region_df(empty_units, "single_session")
    assert len(df) == 0
    assert {
        "unit_id",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
        "region_resolution",
    } <= set(df.columns)


@pytest.mark.slow
def test_curation_v2_all_units_labeled_noise(populated_sorting):
    """All-noise label set: the matching population is empty AND
    ``insert_curation`` succeeds without crashing.

    The curated Units NWB still has every unit: labels filter at the
    consumer (``resolve_unit_selection``), not at write time.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._core.enums import CurationLabel

    _clear_curations(populated_sorting)

    unit_ids = (Sorting.Unit & populated_sorting).fetch("unit_id")
    assert len(unit_ids) > 0, (
        "populated_sorting fixture should have at least one unit; "
        "test cannot assert noise-labeled behavior on an empty source."
    )
    noise_labels = {int(uid): [CurationLabel.noise] for uid in unit_ids}

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels=noise_labels,
        description="all-noise curation regression",
    )

    matchable = matched_unit_ids(pk)
    assert list(matchable) == [], (
        f"Expected zero matchable unit ids after labeling every unit "
        f"noise; got {list(matchable)}."
    )

    n_curated_units = len(CurationV2.Unit & pk)
    assert n_curated_units == len(unit_ids), (
        f"Curated NWB has {n_curated_units} units; expected "
        f"{len(unit_ids)} (labels do NOT filter the written units, "
        "only the matchable-unit accessor)."
    )


@pytest.mark.slow
def test_curation_v2_stages_empty_units_nwb_on_zero_kept_units(
    populated_sorting, monkeypatch
):
    """``write_curated_units_nwb`` writes an empty ``pynwb.misc.Units``
    when no units are kept, so ``nwbf.units.object_id`` does not raise.

    Monkeypatched because ``Sorting.make`` rejects zero-unit sorts.
    """
    from spyglass.spikesorting.v2._curation import plan as _curation_plan
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    # Patch where build_curation_insert_plan looks it up. apply_merge=True:
    # only then does the writer use kept_unit_to_contributors.
    def _empty_rows(
        sorting_id, sorting_units, merge_groups, curation_id, apply_merge
    ):
        return [], {}

    monkeypatch.setattr(_curation_plan, "build_curated_unit_rows", _empty_rows)

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={},
        description="empty-kept rollback guard regression",
        apply_merge=True,
    )
    assert len(CurationV2 & pk) == 1
    assert len(CurationV2.Unit & pk) == 0


@pytest.mark.slow
def test_curation_v2_insert_rollback_cleans_units_nwb(
    populated_sorting, monkeypatch
):
    """``insert_curation`` unlinks the staged Units NWB on rollback.

    ``SpikeSortingOutput._merge_insert`` is patched to raise mid-transaction.
    """
    import pathlib

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    from spyglass.settings import analysis_dir as ad

    analysis_dir = pathlib.Path(ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    original_merge_insert = SpikeSortingOutput._merge_insert

    def _broken_merge_insert(cls, *args, **kwargs):
        raise RuntimeError("simulated merge_insert failure")

    monkeypatch.setattr(
        SpikeSortingOutput, "_merge_insert", classmethod(_broken_merge_insert)
    )

    with pytest.raises(RuntimeError, match="simulated merge_insert failure"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={},
            description="rollback regression test",
        )

    monkeypatch.setattr(
        SpikeSortingOutput, "_merge_insert", original_merge_insert
    )

    assert len(CurationV2 & populated_sorting) == 0, (
        "CurationV2 row present after a transaction that should "
        "have rolled back."
    )
    assert len(SpikeSortingOutput.CurationV2 & populated_sorting) == 0, (
        "SpikeSortingOutput.CurationV2 merge-part row present after "
        "a transaction that should have rolled back."
    )

    # No orphan analysis NWB on disk.
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"CurationV2.insert_curation rollback left orphan analysis "
        f"files: {new_files}. The except-block in insert_curation "
        f"must unlink the staged file when the transaction rolls back."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_dataframe_includes_curation_label(populated_sorting):
    """``CurationV2.get_sorting(as_dataframe=True)`` carries the per-unit
    ``curation_label`` column from UnitLabel, as v1 did.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    units = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    labels = {units[0]: ["mua"]}
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels=labels
    )
    df = CurationV2().get_sorting(pk, as_dataframe=True)
    assert (
        "curation_label" in df.columns
    ), "DataFrame missing the curation_label column joined from UnitLabel."
    # The labeled unit carries ["mua"]; any other unit carries [].
    assert df.loc[units[0], "curation_label"] == ["mua"]
    for uid in units[1:]:
        assert df.loc[uid, "curation_label"] == [], (
            f"Unit {uid} unexpectedly carries non-empty curation_label "
            f"{df.loc[uid, 'curation_label']!r}; expected empty list "
            "for unlabeled units."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_curation_label_post_curation_is_indexed_ragged_list(populated_sorting):
    """Post-curation NWB writes ``curation_label`` as an indexed (ragged)
    list column, matching v1's shape.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    units = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    labels = {units[0]: ["mua", "artifact"]}
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels=labels
    )
    row = (CurationV2 & pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        df = nwbf.units.to_dataframe()

    assert (
        "curation_label" in df.columns
    ), "Curated NWB missing curation_label column."
    # pynwb may return a list or an object ndarray; a scalar would mean a
    # non-indexed column.
    import numpy as np

    labeled_value = df.loc[units[0], "curation_label"]
    assert isinstance(labeled_value, (list, np.ndarray)), (
        f"curation_label for labeled unit is {type(labeled_value)}; "
        "expected list / ndarray (ragged column from index=True). "
        "Regression to a non-indexed column."
    )
    assert set(labeled_value) == {"mua", "artifact"}


@pytest.mark.slow
def test_curation_labels_raise_on_stray_unit_ids(
    populated_sorting, monkeypatch
):
    """``insert_curation`` raises when labels reference a unit_id not in
    the sorting (the stray label would otherwise vanish silently);
    ``allow_unknown_unit_ids=True`` restores the warn-and-drop behavior.
    """
    from spyglass.spikesorting.v2 import curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)

    # Default: a stray unit_id is a hard error (no DB rows written).
    with pytest.raises(ValueError, match="999999"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={999999: ["noise"]},  # not a real unit_id
            description="stray-label test",
        )

    # Opt-in permissive: warn and drop the stray label instead.
    captured = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg),
    )
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={999999: ["noise"]},
        description="stray-label permissive test",
        allow_unknown_unit_ids=True,
    )
    assert any("999999" in m for m in captured), captured
    # The stray label was dropped, not persisted.
    assert 999999 not in {
        int(u) for u in (CurationV2.UnitLabel & pk).fetch("unit_id")
    }


@pytest.mark.slow
def test_curation_insert_idempotent_rejects_new_args(populated_sorting):
    """Re-inserting a root curation with non-default args raises (the
    args would be silently ignored); ``reuse_existing=True`` allows it.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})

    # description, not labels: the smoke sort's unit count varies.
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting, description="changed description"
        )

    # reuse_existing=True returns the existing root without raising.
    reused = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        description="changed description",
        reuse_existing=True,
    )
    assert reused["curation_id"] == pk["curation_id"]


@pytest.mark.slow
def test_curation_label_validation_all_paths(populated_sorting):
    """Curation labels are validated on EVERY insert path.

    Both ``CurationV2.insert_curation(labels=...)`` and a direct
    ``CurationV2.UnitLabel.insert1`` reject a typo'd label, and
    ``allow_custom_labels=True`` accepts a label outside the canonical set
    through both paths.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])

    # Path 1: insert_curation rejects a typo'd label.
    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: ["noies"]},
        )

    # Path 1 with the escape hatch: a custom label is accepted.
    pk_custom = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        labels={target_unit: ["my_custom_tag"]},
        allow_custom_labels=True,
    )
    custom_rows = {
        r["curation_label"]
        for r in (CurationV2.UnitLabel & pk_custom).fetch(as_dict=True)
    }
    assert "my_custom_tag" in custom_rows

    # Path 2: a direct UnitLabel.insert1 rejects a typo'd label (the
    # part-table override, not only insert_curation).
    direct_row = {
        **pk_custom,
        "unit_id": target_unit,
        "curation_label": "noies",
    }
    with pytest.raises(ValueError, match="not in CurationLabel"):
        CurationV2.UnitLabel.insert1(direct_row)

    # Path 2 with the escape hatch: the direct insert accepts a custom
    # label when allow_custom_labels=True.
    CurationV2.UnitLabel.insert1(
        {**pk_custom, "unit_id": target_unit, "curation_label": "another_tag"},
        allow_custom_labels=True,
    )
    assert (
        CurationV2.UnitLabel
        & pk_custom
        & {"unit_id": target_unit, "curation_label": "another_tag"}
    )


@pytest.mark.slow
def test_unit_label_insert_rejects_ordered_row(populated_sorting):
    """A direct ordered-sequence (tuple) UnitLabel insert is validated.

    A tuple row must not slip a bogus label past the mapping-based check.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])
    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={target_unit: ["mua"]}
    )

    ordered = (
        pk["sorting_id"],
        pk["curation_id"],
        target_unit,
        "noies",  # not in CurationLabel
    )
    with pytest.raises(
        ValueError, match="requires mapping|not in CurationLabel"
    ):
        CurationV2.UnitLabel.insert1(ordered)
    with pytest.raises(
        ValueError, match="requires mapping|not in CurationLabel"
    ):
        CurationV2.UnitLabel.insert([ordered])
    # The type guard precedes the custom-label opt-out.
    with pytest.raises(ValueError, match="requires mapping"):
        CurationV2.UnitLabel.insert1(ordered, allow_custom_labels=True)


@pytest.mark.slow
def test_insert_curation_rejects_scalar_string_label(populated_sorting):
    """A scalar string label value is rejected, not split per-character.

    Checked with and without ``allow_custom_labels``, which skips the
    per-value check.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    target_unit = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])

    with pytest.raises(ValueError, match="must be a list of labels"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: "mua"},  # scalar string, not ["mua"]
        )
    with pytest.raises(ValueError, match="must be a list of labels"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            labels={target_unit: "custom_tag"},
            allow_custom_labels=True,
        )


def test_insert_curation_rejects_invalid_curation_source(populated_sorting):
    """An unknown ``curation_source`` raises naming the valid options.

    Fails in Python, before the DataJoint enum check.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    with pytest.raises(ValueError, match="CurationSource"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            curation_source="not_a_real_source",
        )


def test_insert_curation_idempotent_root_rejects_nondefault_args(
    populated_sorting_with_curation, populated_sorting
):
    """A second root insert with non-default args raises unless
    ``reuse_existing=True``, which returns the existing key.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    existing = populated_sorting_with_curation
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting,
            description="a second description that would be ignored",
        )
    reused = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        description="ignored but explicitly tolerated",
        reuse_existing=True,
    )
    assert reused["curation_id"] == existing["curation_id"]


def test_insert_curation_idempotent_root_rejects_apply_merge_and_metrics(
    populated_sorting_with_curation, populated_sorting
):
    """The existing-root guard also covers apply_merge / curation_source.

    Raises unless ``reuse_existing=True``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    existing = populated_sorting_with_curation
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting, apply_merge=True
        )
    with pytest.raises(ValueError, match="already exists"):
        CurationV2.insert_curation(
            sorting_key=populated_sorting, curation_source="figpack"
        )
    reused = CurationV2.insert_curation(
        sorting_key=populated_sorting,
        apply_merge=True,
        reuse_existing=True,
    )
    assert reused["curation_id"] == existing["curation_id"]


def test_insert_curation_root_reuse_deterministic_with_multiple_roots(
    populated_sorting_with_curation,
):
    """Reuse picks the canonical lowest-curation_id root.

    A second root can only come from a raw insert, so one is planted.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    root = populated_sorting_with_curation
    sid = root["sorting_id"]
    full = (CurationV2 & root).fetch1()
    existing_ids = [
        int(c) for c in (CurationV2 & {"sorting_id": sid}).fetch("curation_id")
    ]
    planted_id = max(existing_ids) + 1
    planted = {
        **full,
        "curation_id": planted_id,
        "curation_uuid": uuid.uuid4(),
        "parent_curation_id": -1,
    }
    CurationV2.insert1(planted, allow_direct_insert=True)
    try:
        reused = CurationV2.insert_curation(
            sorting_key={"sorting_id": sid}, reuse_existing=True
        )
        assert reused["curation_id"] == min(root["curation_id"], planted_id)
        assert reused["curation_id"] == root["curation_id"]
    finally:
        (
            CurationV2 & {"sorting_id": sid, "curation_id": planted_id}
        ).delete_quick()


def test_get_analyzer_accepts_non_sorting_id_restriction(populated_sorting):
    """get_analyzer resolves sorting_id from the matched row.

    A restriction that selects a single ``Sorting`` row WITHOUT a literal
    ``sorting_id`` key (here the unique ``object_id``) must load / rebuild
    the analyzer, not ``KeyError`` on ``key["sorting_id"]``.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    object_id, n_units = (Sorting & populated_sorting).fetch1(
        "object_id", "n_units"
    )
    if int(n_units) == 0:
        pytest.skip("zero-unit sort has no analyzer to load")
    analyzer = Sorting().get_analyzer({"object_id": object_id})
    assert analyzer is not None
    assert analyzer.get_num_units() == int(n_units)


def test_insert_curation_rejects_missing_sorting_id(dj_conn):
    """A ``sorting_id`` not in ``Sorting`` raises a clear ValueError.

    Not a raw FK IntegrityError.
    """
    import uuid

    from spyglass.spikesorting.v2.curation import CurationV2

    with pytest.raises(ValueError, match="not in Sorting"):
        CurationV2.insert_curation(sorting_key={"sorting_id": uuid.uuid4()})


@pytest.mark.usefixtures("dj_conn")
def test_curation_get_unit_brain_regions_concat_raises():
    """The CurationV2 accessor raises ``ConcatBrainRegionAmbiguousError``
    for a concat-backed sorting (mirror of the Sorting accessor).

    A planted concat ``SortingSelection`` suffices: the guard reads the
    source before ``CurationV2.Unit``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ConcatBrainRegionAmbiguousError):
            CurationV2().get_unit_brain_regions(
                {"sorting_id": sid, "curation_id": 0},
                allow_anchor_member=False,
            )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_curation_get_unit_brain_regions_concat_anchor_member_df(
    populated_sorting,
):
    """``CurationV2.get_unit_brain_regions(..., allow_anchor_member=True)``
    returns the ``unit_brain_region_df`` DataFrame labeled ``anchor_member``.

    Same shape as the ``Sorting`` accessor. A ``CurationV2.Unit`` row with a
    real Electrode FK makes the BrainRegion join non-empty.
    """
    import datetime as dt
    import uuid

    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2._recording.source import SourceResolution

    template_unit = (Sorting.Unit & populated_sorting).fetch(as_dict=True)[0]
    # CurationV2.Unit carries the same Electrode FK + amplitude/spike columns
    # as Sorting.Unit; drop the Sorting PK and re-key onto the planted curation.
    unit_fields = {k: v for k, v in template_unit.items() if k != "sorting_id"}

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a28_concat_fake.nwb",
                "object_id": "a28-concat-object-id",
                "n_units": 1,
                "time_of_sort": dt.datetime(2020, 1, 1),
                # NOT NULL; FK checks are off so the row need not exist.
                "display_waveform_params_name": (
                    "franklab_cortex_actual_waveforms"
                ),
                # NOT NULL; a real sort sets si.__version__.
                "spikeinterface_version": "0.0.0",
            },
            allow_direct_insert=True,
        )
        CurationV2.insert1(
            {
                "sorting_id": sid,
                "curation_id": 0,
                "curation_uuid": uuid.uuid4(),
                "analysis_file_name": "a28_concat_curation_fake.nwb",
                "object_id": "a28-concat-curation-object-id",
                "description": "a28 concat anchor probe",
            },
            allow_direct_insert=True,
        )
        CurationV2.Unit.insert1(
            {**unit_fields, "sorting_id": sid, "curation_id": 0, "unit_id": 0},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        result = CurationV2().get_unit_brain_regions(
            {"sorting_id": sid, "curation_id": 0}, allow_anchor_member=True
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, SourceResolution)
        assert "region_resolution" in result.columns
        assert len(result) == 1, "anchor-member df should carry the one unit"
        assert (result["region_resolution"] == "anchor_member").all()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (CurationV2.Unit & {"sorting_id": sid}).delete_quick()
            (CurationV2 & {"sorting_id": sid}).delete_quick()
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_insert_curation_empty_labels_skip_unit_label_insert(
    populated_sorting_with_curation,
):
    """With empty labels the ``UnitLabel`` insert is skipped, while the
    ``MergeGroup`` self-entries are still written.

    Every unit gets a self-entry so ``Unit * MergeGroup`` keeps all units.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    key = populated_sorting_with_curation
    n_units = len(CurationV2.Unit & key)
    assert n_units >= 1, "fixture curation must have >= 1 unit"
    assert len(CurationV2.UnitLabel & key) == 0
    assert len(CurationV2.MergeGroup & key) == n_units


def test_curation_label_column_added_only_with_nonempty_labels(
    populated_sorting,
):
    """The units NWB carries a ``curation_label`` column iff at least one
    unit has a non-empty label.

    An all-empty column would trip pynwb's empty-list dtype inference.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    def _units_colnames(curation_key):
        analysis_file_name = (CurationV2 & curation_key).fetch1(
            "analysis_file_name"
        )
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            return list(nwbf.units.colnames) if nwbf.units is not None else []

    _clear_curations(populated_sorting)
    empty_key = CurationV2.insert_curation(sorting_key=populated_sorting)
    assert "curation_label" not in _units_colnames(empty_key)

    _clear_curations(populated_sorting)
    uid = int((Sorting.Unit & populated_sorting).fetch("unit_id")[0])
    labeled_key = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={uid: ["mua"]}
    )
    try:
        assert "curation_label" in _units_colnames(labeled_key)
    finally:
        _clear_curations(populated_sorting)
