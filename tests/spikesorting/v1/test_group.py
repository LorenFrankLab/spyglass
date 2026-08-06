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
        filter_units_by_criteria(units_df, {"n_spikes": {"!=": 10}}),
        [True, False, True, True],
    ), "Unexpected mask for 'not equal'"


def test_criteria_nan_excluded(spike_v1_group, units_df):
    """NaN fails every numeric comparison, so those units are excluded"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df, {"snr": {"<": 1e9}}
    )
    assert not mask[3], "Unit with NaN snr was not excluded"


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


def test_criteria_missing_column(spike_v1_group, units_df, caplog):
    """A criterion on a missing column is skipped, keeping those units"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df, {"not_a_column": {">": 1}}
    )
    assert mask.all(), "Units dropped for a missing criteria column"
    assert "not_a_column" in caplog.text, "No warning for missing column"


def test_criteria_missing_column_others_applied(spike_v1_group, units_df):
    """Skipping a missing column still applies the remaining criteria"""
    mask = spike_v1_group.SortedSpikesGroup.filter_units_by_criteria(
        units_df, {"not_a_column": {">": 1}, "snr": {">": 3}}
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


def test_label_columns_appended(spike_v1_group, units_df):
    """Values of label_columns join the curation labels for each unit"""
    labels = spike_v1_group._compile_unit_labels(
        units_df, label_columns=["brain_region"]
    )
    assert labels == [
        ["CA1"],
        ["noise", "CA1"],
        ["mua", "CA3"],
        ["CA3"],
    ], "Unexpected compiled labels"


def test_label_columns_cast_to_str(spike_v1_group, units_df):
    """Non-string label columns are cast to strings"""
    labels = spike_v1_group._compile_unit_labels(
        units_df, label_columns=["n_spikes"]
    )
    assert labels[0] == ["1000"], "Numeric label not cast to string"


def test_label_columns_missing(spike_v1_group, units_df, caplog):
    """A missing label column warns but leaves curation labels intact"""
    labels = spike_v1_group._compile_unit_labels(
        units_df, label_columns=["not_a_column"]
    )
    assert labels[1] == ["noise"], "Curation labels lost"
    assert "not_a_column" in caplog.text, "No warning for missing column"


def test_label_columns_no_labels(spike_v1_group):
    """A sorting with no label columns at all filters no units"""
    no_labels = spike_v1_group._compile_unit_labels(
        pd.DataFrame({"snr": [1.0]})
    )
    assert (
        no_labels is None
    ), "Expected None for units table without label columns"


def test_filter_units_with_label_columns(spike_v1_group, units_df):
    """Extra labels are usable by include_labels/exclude_labels"""
    labels = spike_v1_group._compile_unit_labels(
        units_df, label_columns=["brain_region"]
    )
    mask = spike_v1_group.SortedSpikesGroup.filter_units(
        labels, include_labels=["CA1"], exclude_labels=["noise", "mua"]
    )
    assert np.array_equal(
        mask, [True, False, False, False]
    ), "Unexpected mask combining region and curation labels"


def test_params_round_trip(spike_v1_group):
    """New columns round-trip through the table"""
    tbl = spike_v1_group.UnitSelectionParams()
    name = "test_label_columns"
    params = {
        "unit_filter_params_name": name,
        "include_labels": [],
        "exclude_labels": ["noise", "mua"],
        "label_columns": ["brain_region"],
        "unit_criteria": {"snr": {">=": 3}},
    }
    tbl.insert1(params, skip_duplicates=True)
    assert (tbl & {"unit_filter_params_name": name}).fetch1() == params


def test_fetch_spike_data_unknown_column(
    spike_v1_group, pop_spikes_group, mini_dict, group_name
):
    """Criteria on a column no sorting has leave the fetched units untouched

    The units tables of a group need not share columns, so an unknown column
    must not silently empty the group.
    """
    name = "test_unknown_column"
    spike_v1_group.UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": name,
            "include_labels": [],
            "exclude_labels": [],
            "label_columns": [],
            "unit_criteria": {"not_a_column": {">": 1}},
        },
        skip_duplicates=True,
    )

    key = {**mini_dict, "sorted_spikes_group_name": group_name}
    all_spikes = spike_v1_group.SortedSpikesGroup().fetch_spike_data(key)
    assert len(all_spikes) > 0, "No spike data to filter"

    filtered = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        {**key, "unit_filter_params_name": name}
    )
    assert len(filtered) == len(all_spikes), "Units dropped for unknown column"
