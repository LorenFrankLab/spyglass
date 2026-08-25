from types import SimpleNamespace

import datajoint as dj
import numpy as np
import pytest
from pandas import DataFrame, Index

from spyglass.common.common_interval import IntervalList


@pytest.fixture(scope="module")
def lfp_raw(lfp_analysis_raw):
    lfp_raw = lfp_analysis_raw.scratch["filtered data"]
    yield lfp_raw.data, Index(lfp_raw.timestamps, name="time")


def test_lfp_fetch1_dataframe(lfp, lfp_raw):
    lfp_data, lfp_index = lfp_raw
    df_fetch = lfp.v1.LFPV1().fetch1_dataframe()
    df_raw = DataFrame(lfp_data, index=lfp_index)

    assert df_raw.equals(df_fetch), "LFP dataframe not match."


def test_lfp_dataframe(lfp, lfp_raw, lfp_merge_key):
    lfp_data, lfp_index = lfp_raw
    df_raw = DataFrame(lfp_data, index=lfp_index)
    df_fetch = (lfp.LFPOutput & lfp_merge_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFP dataframe not match."


def test_lfp_band_dataframe(lfp_band_analysis_raw, lfp_band, lfp_band_key):
    lfp_band_raw = (
        lfp_band_analysis_raw.processing["ecephys"]
        .fields["data_interfaces"]["LFP"]
        .electrical_series["filtered data"]
    )
    lfp_band_index = Index(lfp_band_raw.timestamps, name="time")
    df_raw = DataFrame(lfp_band_raw.data, index=lfp_band_index)
    df_fetch = (lfp_band.LFPBandV1 & lfp_band_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFPBand dataframe not match."


def test_lfp_band_compute_signal_invalid(lfp_band_v1):
    with pytest.raises(ValueError):
        lfp_band_v1.compute_analytic_signal([4])


def test_lfp_band_compute_signal(lfp_band_v1):
    signal_sum = lfp_band_v1.compute_analytic_signal([0]).iloc[:, 0].sum()
    assert (
        pytest.approx(signal_sum, 0.0001) == 189
    ), "LFPBand hilbert signal off."


def test_lfp_band_compute_phase(lfp_band_v1):
    phase_sum = lfp_band_v1.compute_signal_phase([0]).iloc[:, 0].sum()
    assert (
        pytest.approx(phase_sum, 0.0001) == 2857.9293
    ), "LFPBand phase signal off."


def test_lfp_band_compute_power(lfp_band_v1):
    power_sum = lfp_band_v1.compute_signal_power([0]).iloc[:, 0].sum()
    assert pytest.approx(power_sum) == 5_391_437, "LFPBand power signal off."


def test_invalid_band_selection(
    lfp_band,
    mini_copy_name,
    mini_dict,
    lfp_merge_key,
    add_interval,
    lfp_constants,
    add_band_filter,
    add_electrode_group,
):
    valid = dict(
        nwb_file_name=mini_copy_name,
        lfp_merge_id=lfp_merge_key.get("merge_id"),
        electrode_list=lfp_constants.get("lfp_band_electrode_ids"),
        filter_name=add_band_filter,
        interval_list_name=add_interval,
        reference_electrode_list=[-1],
        lfp_band_sampling_rate=lfp_constants.get("lfp_band_sampling_rate"),
    )
    set_elec = lfp_band.LFPBandSelection().set_lfp_band_electrodes
    with pytest.raises(ValueError):
        set_elec(**valid | {"electrode_list": [3]})
    with pytest.raises(ValueError):
        set_elec(**valid | {"filter_name": "invalid_filter"})
    with pytest.raises(ValueError):  # ref list size > electrode list size
        set_elec(**valid | {"reference_electrode_list": [1, 2]})
    with pytest.raises(ValueError):  # ref not in electrode list
        set_elec(**valid | {"reference_electrode_list": [3]})


def test_artifact_param_defaults(art_params, art_param_defaults):
    assert set(art_params.fetch("artifact_params_name")).issubset(
        set(art_param_defaults)
    ), "LFPArtifactDetectionParameters missing default item."


@pytest.mark.skip(reason="See #850")
def test_artifact_detection(lfp, pop_art_detection):
    pass


def test_pop_imported_lfp(lfp, common, mini_dict):
    # check that populated from populate_all_common
    assert len(lfp.lfp_imported.ImportedLFP()) == 1
    assert (
        len(
            lfp.lfp_imported.LFPElectrodeGroup
            & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        )
        == 1
    )
    # Re-ingesting an already-ingested file raises rather than skipping:
    # ImportedLFP does not expect duplicates, and nothing new is added.
    with pytest.raises(dj.errors.DuplicateError):
        lfp.lfp_imported.ImportedLFP().insert_from_nwbfile(
            mini_dict["nwb_file_name"]
        )
    assert len(lfp.lfp_imported.ImportedLFP()) == 1


def test_imported_lfp_dry_run_writes_nothing(lfp, common, mini_dict):
    """Resolving an electrode group is a read; the write waits for the plan.

    `cautious_insert` used to run during entry generation, before the
    `dry_run` gate, so a dry run created an LFPElectrodeGroup row.
    """
    group_tbl = lfp.lfp_imported.LFPElectrodeGroup()
    before = len(group_tbl)

    table = lfp.lfp_imported.ImportedLFP()
    table._planned_groups, table._planned_names = dict(), set()

    # An electrode set no group holds yet, so a new group must be planned
    electrode_ids = sorted((common.Electrode & mini_dict).fetch("electrode_id"))
    _, entries = table._plan_electrode_group(dict(mini_dict), electrode_ids[:3])

    assert entries, "A novel electrode set should plan a new group"
    assert len(group_tbl) == before, "Planning a group wrote to the database"

    table.insert_from_nwbfile(mini_dict["nwb_file_name"], dry_run=True)

    assert len(group_tbl) == before, "A dry run wrote to the database"


def test_imported_lfp_group_name_skips_gaps(lfp, mini_dict):
    """The next group name comes from the max suffix, not the group count."""
    table = lfp.lfp_imported.ImportedLFP()
    table._planned_groups, table._planned_names = dict(), set()

    stored = set(
        (
            lfp.lfp_imported.LFPElectrodeGroup
            & mini_dict
            & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        ).fetch("lfp_electrode_group_name")
    )
    assert stored == {"imported_lfp_000"}, "Expected one group from ingestion"

    assert (
        table._next_group_name(dict(mini_dict)) == "imported_lfp_001"
    ), "The next name should follow the stored group"

    # Suffixes {0, 2}: test incrementing over highest existing entry
    table._planned_names = {"imported_lfp_002"}

    assert (
        table._next_group_name(dict(mini_dict)) == "imported_lfp_003"
    ), "A name in use must not be handed out again"


def test_imported_lfp_skipped_series_keeps_file_position(lfp, monkeypatch):
    """A series with no timestamps must not renumber the ones after it.

    `interval_list_name` is part of the primary key, so numbering by
    position among *ingested* series would give a file's second series a
    different identity depending on whether the first had data.
    """
    from spyglass.utils.mixins import ingestion

    monkeypatch.setattr(
        ingestion, "is_nwb_obj_type", lambda obj, obj_type: True
    )

    empty = SimpleNamespace(object_id="empty-series", get_timestamps=list)
    valid = SimpleNamespace(
        object_id="valid-series", get_timestamps=lambda: [0.0, 1.0]
    )
    container = SimpleNamespace(
        electrical_series={"empty": empty, "valid": valid}
    )
    nwb_file = SimpleNamespace(objects={"lfp": container})

    table = lfp.lfp_imported.ImportedLFP()
    series = table.get_nwb_objects(nwb_file)

    assert series == [empty, valid], "Both series should be selected"

    skipped = table.generate_entries_from_nwb_object(empty, dict())
    assert not any(
        skipped.values()
    ), "A series with no timestamps yields no entry"
    assert (
        table.enumerated_interval_name(valid) == "imported lfp 1 valid times"
    ), "The second series keeps its file position after the first is skipped"

    # A skipped first series still has to seat the parents ahead of `self`:
    # emission order is insert order, and it is fixed by the first series.
    group_tbl = lfp.lfp_imported.LFPElectrodeGroup
    keys = list(skipped)
    assert keys.index(table) == len(keys) - 1, "self must be emitted last"
    for parent in (group_tbl, group_tbl.LFPElectrode, IntervalList):
        assert keys.index(parent) < keys.index(
            table
        ), f"{parent.__name__} must precede ImportedLFP in insert order"


def test_remove_null_from_dicts_keeps_arrays(lfp):
    """A blob-valued parent row must survive the no-adjustment fallback.

    `v not in [None, ""]` compares an array elementwise and then coerces the
    result to bool, raising "truth value of an array is ambiguous". Only
    tables without `_adjust_keys_for_entry` take this path, so no live plan
    reaches it with an array today -- but ImportedLFP emits an IntervalList
    row whose valid_times is exactly such an array.
    """
    valid_times = np.array([[0.0, 1.0], [2.0, 3.0]])
    row = {
        "nwb_file_name": "fake_.nwb",
        "valid_times": valid_times,
        "pipeline": "imported_lfp",
        "dropped": None,
        "also_dropped": "",
        "kept_empty_list": [],
    }

    (cleaned,) = lfp.lfp_imported.ImportedLFP()._remove_null_from_dicts([row])

    assert np.array_equal(
        cleaned["valid_times"], valid_times
    ), "An array value should pass through unchanged"
    assert set(cleaned) == {
        "nwb_file_name",
        "valid_times",
        "pipeline",
        "kept_empty_list",
    }, "None and empty-string values should still be dropped"
