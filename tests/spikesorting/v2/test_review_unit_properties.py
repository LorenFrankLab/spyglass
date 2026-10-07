"""Review columns -> SpikeInterface unit-table arrays (DB-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spyglass.spikesorting.v2._review_unit_properties import (
    display_column_array,
    review_unit_properties,
)


def test_numeric_columns_keep_values_and_gaps_as_nan():
    ints = pd.Series([3, 7], dtype="int64")
    assert display_column_array(ints).dtype == np.float64
    assert display_column_array(ints).tolist() == [3.0, 7.0]
    nullable = pd.Series([1, pd.NA], dtype="Int64")
    out = display_column_array(nullable)
    assert out.dtype == np.float64 and out[0] == 1.0 and np.isnan(out[1])
    floats = pd.Series([0.25, np.nan])
    out = display_column_array(floats)
    assert out[0] == 0.25 and np.isnan(out[1])


def test_bool_columns_stay_bool_only_when_complete():
    complete = pd.Series([True, False])
    assert display_column_array(complete).dtype == np.bool_
    gappy = pd.Series([True, None], dtype="boolean")
    out = display_column_array(gappy)
    # A missing annotation is rendered as text, never as False.
    assert out.dtype.kind == "U" and out.tolist() == ["True", ""]


def test_text_and_object_columns_become_unicode_with_empty_gaps():
    strings = pd.Series(["stable", None, np.nan], dtype="object")
    out = display_column_array(strings)
    assert out.dtype.kind == "U" and out.tolist() == ["stable", "", ""]
    pandas_strings = pd.Series(["a", "b"], dtype="string")
    assert display_column_array(pandas_strings).dtype.kind == "U"


def test_review_unit_properties_aligns_to_analyzer_order_and_keeps_order():
    table = pd.DataFrame(
        {
            "snr": [5.0, 6.0, 7.0],
            "proposed_labels": ["accept", "", "noise"],
            "flag": [True, False, True],
        },
        index=pd.Index([1, 2, 3], name="unit_id"),
    )
    props = review_unit_properties(table, unit_ids=np.array([3, 1, 2]))
    assert list(props) == ["snr", "proposed_labels", "flag"]
    assert props["snr"].tolist() == [7.0, 5.0, 6.0]
    assert props["proposed_labels"].tolist() == ["noise", "accept", ""]
    assert props["flag"].tolist() == [True, True, False]
    # SI accepts only 1-D arrays of these dtype kinds.
    assert all(a.ndim == 1 and a.dtype.kind in "iuUSfb" for a in props.values())

    # A unit the analyzer shows but the table lacks is a gap, not an error.
    partial = review_unit_properties(table, unit_ids=[1, 2, 3, 4])
    assert np.isnan(partial["snr"][3]) and partial["proposed_labels"][3] == ""
    assert partial["flag"].dtype.kind == "U" and partial["flag"][3] == ""
    # A unit the analyzer does NOT show cannot be described.
    with pytest.raises(ValueError, match="absent from the analyzer"):
        review_unit_properties(table, unit_ids=[1, 2])


def test_missing_rule_inputs_follow_enabled_rules_and_unit_ids():
    from spyglass.spikesorting.v2._review_unit_properties import (
        missing_rule_metrics,
    )

    metrics = pd.DataFrame(
        {"snr": [8.0, np.nan], "unused": [np.nan, np.nan]}, index=[9, 3]
    )
    coverage = missing_rule_metrics(metrics, ["snr", "isi_violation", "snr"])
    assert coverage.to_dict() == {9: "isi_violation", 3: "snr, isi_violation"}
    assert missing_rule_metrics(metrics, []).to_dict() == {9: "", 3: ""}
    table = metrics.join(coverage)
    properties = review_unit_properties(table, [3, 9])
    assert properties["unavailable_qc"].tolist() == [
        "snr, isi_violation",
        "isi_violation",
    ]
    assert np.isnan(properties["snr"][0])
