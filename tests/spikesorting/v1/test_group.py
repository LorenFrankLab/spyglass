import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def units_df():
    """Minimal units table: two curated units per label, varied metrics"""
    return pd.DataFrame(
        {
            "curation_label": [[], ["noise"], ["mua"], []],
            "brain_region": ["CA1", "CA1", "CA3", "CA3"],
            "snr": [5.0, 1.0, 4.0, np.nan],
            "isi_violation": [0.001, 0.5, 0.02, 0.0],
            "n_spikes": [1000, 10, 500, 300],
        }
    )


def test_criteria_comparisons(spike_v1_group, units_df):
    """Numeric comparisons select units on either side of the threshold"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    assert np.array_equal(
        filter_units_by_criteria(units_df, {"snr": {">": 3}}),
        [True, False, True, False],
    ), "Unexpected mask for 'greater than'"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"snr": {"<=": 4.0}}),
        [False, True, True, False],
    ), "Unexpected mask for 'less than or equal'"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"snr": {">=": 4.0}}),
        [True, False, True, False],
    ), "Unexpected mask for 'greater than or equal'"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"n_spikes": {"==": 10}}),
        [False, True, False, False],
    ), "Unexpected mask for 'equal'"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"n_spikes": {"!=": 10}}),
        [True, False, True, True],
    ), "Unexpected mask for 'not equal'"


def test_criteria_missing_value_excluded(spike_v1_group, units_df):
    """A missing value fails every criterion, negated operators included

    A unit missing the value a criterion gates on must not pass it: "!=" and
    "notin" would otherwise admit exactly the units a criterion meant to drop.
    """
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    for criterion in ({"<": 1e9}, {"!=": 1.0}, {"notin": [1.0]}):
        mask = filter_units_by_criteria(units_df, {"snr": criterion})
        assert not mask[3], f"NaN snr not excluded by {criterion}"

    # None only reaches us through ImportedSpikeSorting, whose units table is
    # written by another pipeline; an empty list is a unit with no labels
    missing = units_df.assign(curation_label=[[], ["noise"], np.nan, None])
    assert np.array_equal(
        filter_units_by_criteria(
            missing, {"curation_label": {"notin": ["noise"]}}
        ),
        [True, False, False, False],
    ), "Missing label not excluded by 'notin' on an object column"


def test_criteria_range(spike_v1_group, units_df):
    """'between' and 'outside' bounds are inclusive"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    assert np.array_equal(
        filter_units_by_criteria(units_df, {"snr": {"between": [4.0, 5.0]}}),
        [True, False, True, False],
    ), "Unexpected mask for 'between'"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"snr": {"outside": [4.0, 5.0]}}),
        [False, True, False, False],
    ), "Unexpected mask for 'outside'"


def test_criteria_range_needs_pair(spike_v1_group, units_df):
    """A range operator given one value says so, rather than TypeError"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    for value in (3, [3], [1, 2, 3]):
        with pytest.raises(ValueError, match=r"takes a \[low, high\] pair"):
            filter_units_by_criteria(units_df, {"snr": {"between": value}})


def test_criteria_membership(spike_v1_group, units_df):
    """Bare values are shorthand for membership"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    assert np.array_equal(
        filter_units_by_criteria(units_df, {"brain_region": "CA1"}),
        [True, True, False, False],
    ), "Unexpected mask for scalar membership"
    assert np.array_equal(
        filter_units_by_criteria(units_df, {"brain_region": ["CA1", "CA3"]}),
        [True, True, True, True],
    ), "Unexpected mask for list membership"
    assert np.array_equal(
        filter_units_by_criteria(
            units_df, {"brain_region": {"notin": ["CA1"]}}
        ),
        [False, False, True, True],
    ), "Unexpected mask for 'notin'"


def test_criteria_combined(spike_v1_group, units_df):
    """Multiple criteria and multiple operators are ANDed"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df,
        {
            "snr": {">": 3, "<": 4.5},
            "isi_violation": {"<": 0.1},
            "brain_region": ["CA1", "CA3"],
        },
    )
    assert np.array_equal(
        mask, [False, False, True, False]
    ), "Unexpected mask for combined criteria"


def test_criteria_empty(spike_v1_group, units_df):
    """Empty or null criteria include every unit"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )
    for criteria in (None, {}):
        assert filter_units_by_criteria(
            units_df, criteria
        ).all(), f"Units dropped by empty criteria {criteria}"


def test_criteria_missing_column_strict(spike_v1_group, units_df):
    """A criterion on a missing column raises rather than failing open"""
    with pytest.raises(ValueError, match="not_a_column"):
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
            units_df, {"not_a_column": {">": 1}}
        )


def test_criteria_missing_column(spike_v1_group, units_df, caplog):
    """Opting out of strict skips a missing column, keeping those units"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df, {"not_a_column": {">": 1}}, strict=False
    )
    assert mask.all(), "Units dropped for a missing criteria column"
    assert "not_a_column" in caplog.text, "No warning for missing column"


def test_criteria_missing_column_others_applied(spike_v1_group, units_df):
    """Skipping a missing column still applies the remaining criteria"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df, {"not_a_column": {">": 1}, "snr": {">": 3}}, strict=False
    )
    assert np.array_equal(
        mask, [True, False, True, False]
    ), "Remaining criteria not applied alongside a missing column"


def test_criteria_invalid_operator(spike_v1_group, units_df):
    """An unrecognized operator raises rather than silently passing"""
    with pytest.raises(ValueError, match="Invalid unit criteria operator"):
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
            units_df, {"snr": {"greater_than": 3}}
        )


def test_criteria_empty_operator_dict(spike_v1_group, units_df):
    """A criterion with no operator raises rather than passing every unit"""
    with pytest.raises(ValueError, match="Empty criterion for column 'snr'"):
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
            units_df, {"snr": {}}
        )


def test_criteria_membership_list_column(spike_v1_group, units_df):
    """Membership matches any item of a column holding a list per unit"""
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    assert np.array_equal(
        filter_units_by_criteria(units_df, {"curation_label": "noise"}),
        [False, True, False, False],
    ), "Unexpected mask for membership in a list-valued column"
    assert np.array_equal(
        filter_units_by_criteria(
            units_df, {"curation_label": {"notin": ["noise", "mua"]}}
        ),
        [True, False, False, True],
    ), "Unexpected mask for 'notin' on a list-valued column"


def test_criteria_comparison_on_list_column(spike_v1_group, units_df):
    """Only isin/notin may filter a column holding a list per unit

    Comparing the whole list to the criterion masks every unit in or out at
    once (e.g. "!=" would keep the noise unit and "==" would drop every unit)
    so the criterion raises rather than returning a meaningless mask.
    """
    filter_units_by_criteria = (
        spike_v1_group.SortedSpikesGroup.filter_units_by_criteria
    )

    for operator in ("==", "!=", ">", "between"):
        value = [0, 1] if operator == "between" else "noise"
        with pytest.raises(ValueError, match="holds a list per unit"):
            filter_units_by_criteria(
                units_df, {"curation_label": {operator: value}}
            )


def test_skipped_criteria_error_message(spike_v1_group):
    """The error separates a typo'd column from a heterogeneous group"""
    skipped_criteria_error = spike_v1_group._skipped_criteria_error

    skipped_id, applied_id = "skipped-merge-id", "applied-merge-id"

    typo = str(skipped_criteria_error({"snr_": [skipped_id]}, {"snr_": []}))
    assert "no sorting in this group" in typo, "Typo case not called out"
    assert "snr_" in typo, "Missing column not named"

    mixed = str(
        skipped_criteria_error(
            {"snr": [skipped_id]}, {"snr": [applied_id, applied_id]}
        )
    )
    assert "missing from 1 of 3 units tables" in mixed, "Wrong sorting counts"
    assert skipped_id in mixed, "Skipped sorting not named"
    assert applied_id not in mixed, "Named a sorting that was not skipped"


def test_params_round_trip(spike_v1_group):
    """New column round-trips through the table"""
    tbl = spike_v1_group.UnitSelectionParams()
    name = "test_unit_criteria"
    params = {
        "unit_filter_params_name": name,
        "include_labels": [],
        "exclude_labels": ["noise", "mua"],
        "unit_criteria": {"snr": {">=": 3}},
    }
    tbl.insert1(params, skip_duplicates=True)
    assert (tbl & {"unit_filter_params_name": name}).fetch1() == params


def test_fetch_spike_data_criteria_drops_units(
    spike_v1_group, pop_spikes_group, pop_spike_merge, mini_dict
):
    """A criterion drops units from the fetched spike data

    The other tests cover the mask itself; this covers the rest of the path -
    the criteria reaching each sorting's units table, and the units failing
    them being dropped from the returned spike times and unit ids.

    Grouped on the metric-curated sorting alone: the session's imported
    sorting has no metric columns, and a criterion that cannot be applied to
    every sorting of a group raises rather than filtering only some of them.
    """
    name = "test_snr_criterion"
    # sorts after the shared "test_group" so that fixtures fetching the first
    # group still get theirs when a prior run left this one in the database
    criteria_group = "test_group_criteria"
    spike_v1_group.UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": name,
            "include_labels": [],
            "exclude_labels": [],
            "unit_criteria": {"snr": {">": np.inf}},  # no unit can pass
        },
        skip_duplicates=True,
    )

    group_tbl = spike_v1_group.SortedSpikesGroup()
    group_tbl.create_group(
        group_name=criteria_group,
        nwb_file_name=mini_dict["nwb_file_name"],
        keys=[{"spikesorting_merge_id": pop_spike_merge["merge_id"]}],
        unit_filter_params_name=name,
    )

    key = {**mini_dict, "sorted_spikes_group_name": criteria_group}
    all_spikes = group_tbl.fetch_spike_data(
        {**key, "unit_filter_params_name": "all_units"}
    )
    kept, kept_ids = group_tbl.fetch_spike_data(
        {**key, "unit_filter_params_name": name}, return_unit_ids=True
    )

    assert len(all_spikes) > 0, "No spike data to filter"
    assert len(kept) == 0, "Criterion dropped no units"
    assert len(kept_ids) == 0, "Unit ids kept a unit the spike times dropped"


def test_fetch_spike_data_unknown_column(
    spike_v1_group, pop_spikes_group, mini_dict, group_name
):
    """Criteria on a column no sorting has raise instead of filtering nothing

    Silently passing every unit would let a typo'd column name disable the
    criterion, so the group reports that it never applied it.
    """
    name = "test_unknown_column"
    spike_v1_group.UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": name,
            "include_labels": [],
            "exclude_labels": [],
            "unit_criteria": {"not_a_column": {">": 1}},
        },
        skip_duplicates=True,
    )

    key = {**mini_dict, "sorted_spikes_group_name": group_name}
    all_spikes = spike_v1_group.SortedSpikesGroup().fetch_spike_data(key)
    assert len(all_spikes) > 0, "No spike data to filter"

    with pytest.raises(ValueError, match="no sorting in this group"):
        spike_v1_group.SortedSpikesGroup().fetch_spike_data(
            {**key, "unit_filter_params_name": name}
        )
