"""DB-free unit tests for the ``describe_run`` receipt and ``CurationLabel``.

``describe_run`` is a pure transform over the dict / list that
``run_v2_pipeline`` and ``run_v2_pipeline_session`` return, so it needs no
DataJoint connection. These tests pin the receipt's row schema -- the property
that makes a zero-unit sort or a failed group a first-class row instead of a
value buried in a nested dict.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spyglass.spikesorting.v2 import CurationLabel
from spyglass.spikesorting.v2._pipeline_reporting import (
    _RUN_COLUMNS,
    describe_run,
)


def _run_summary(*, n_units=3, warnings=None):
    """A minimal run_v2_pipeline-shaped summary."""
    return {
        "pipeline_preset": "preset_x",
        "recording_id": "rec",
        "artifact_detection_id": "art",
        "sorting_id": "sort",
        "curation_id": 0,
        "merge_id": "merge-1",
        "n_units": n_units,
        "recording_status": "computed",
        "artifact_detection_status": "computed",
        "sorting_status": "reused",
        "curation_status": "computed",
        "stage_seconds": {
            "recording": 1.0,
            "artifact_detection": 2.0,
            "sorting": 0.0,
            "curation": 0.5,
        },
        "warnings": list(warnings or []),
    }


def test_describe_run_single_columns_and_rows():
    frame = describe_run(_run_summary())
    assert list(frame.columns) == _RUN_COLUMNS

    by_type = frame["row_type"].tolist()
    assert by_type[0] == "summary"
    # one stage row per stage_seconds entry
    assert by_type.count("stage") == 4
    assert "warning" not in by_type

    summary = frame.iloc[0]
    assert summary["n_units"] == 3
    assert summary["merge_id"] == "merge-1"
    assert summary["seconds"] == pytest.approx(3.5)  # 1 + 2 + 0 + 0.5

    stage_rows = frame[frame["row_type"] == "stage"].set_index("stage")
    assert stage_rows.loc["sorting", "status"] == "reused"
    assert stage_rows.loc["recording", "seconds"] == pytest.approx(1.0)


def test_describe_run_single_warning_is_its_own_row():
    frame = describe_run(
        _run_summary(n_units=0, warnings=["zero units found on this shank"])
    )
    warn_rows = frame[frame["row_type"] == "warning"]
    assert len(warn_rows) == 1
    assert "zero units" in warn_rows.iloc[0]["warning"]
    # the zero-unit count is visible on the summary row, not hidden
    assert frame.iloc[0]["n_units"] == 0


def test_describe_run_session_counts_and_group_rows():
    ok = {**_run_summary(n_units=5), "sort_group_id": 0, "outcome": "ok"}
    zero = {
        **_run_summary(n_units=0, warnings=["zero units"]),
        "sort_group_id": 1,
        "outcome": "ok",
    }
    failed = {
        "sort_group_id": 2,
        "pipeline_preset": "preset_x",
        "outcome": "failed",
        "error": "ZeroUnitSortError: ...",
        "partial_run_summary": None,
    }
    frame = describe_run([ok, zero, failed])

    header = frame.iloc[0]
    assert header["row_type"] == "summary"
    assert header["status"] == "2 ok, 1 failed, 1 zero-unit, 1 with warnings"

    groups = frame[frame["row_type"] == "group"]
    assert groups["sort_group_id"].tolist() == [0, 1, 2]
    failed_row = groups[groups["sort_group_id"] == 2].iloc[0]
    assert failed_row["status"] == "failed"
    assert "ZeroUnitSortError" in failed_row["error"]
    # seconds tolerated as missing on a partial-less failed group (NaN, not a
    # spurious 0.0)
    assert pd.isna(failed_row["seconds"])

    # the zero-unit group's warning is surfaced as its own row
    warn_rows = frame[frame["row_type"] == "warning"]
    assert warn_rows["sort_group_id"].tolist() == [1]


def test_describe_run_session_surfaces_partial_summary_seconds():
    # A failed group that completed some stages carries a partial summary; its
    # seconds and stable metadata should still aggregate.
    failed_partial = {
        "sort_group_id": 0,
        "outcome": "failed",
        "error": "boom",
        "partial_run_summary": {
            "stage_seconds": {"recording": 4.0},
            "n_units": 0,
            "merge_id": "merge-before-failure",
            "warnings": ["zero units before curation failure"],
        },
    }
    frame = describe_run([failed_partial])
    group = frame[frame["row_type"] == "group"].iloc[0]
    assert group["seconds"] == pytest.approx(4.0)
    assert group["n_units"] == 0
    assert group["merge_id"] == "merge-before-failure"
    header = frame.iloc[0]
    assert header["status"] == "0 ok, 1 failed, 1 zero-unit, 1 with warnings"
    warning = frame[frame["row_type"] == "warning"].iloc[0]
    assert "zero units" in warning["warning"]


def test_describe_run_rejects_unexpected_type():
    with pytest.raises(TypeError):
        describe_run("not a run summary")


def test_describe_run_rejects_list_entry_without_outcome():
    # A raw run summary wrapped in a list (no 'outcome' key) must not be
    # silently counted as an ok group -- that would inflate the ok tally.
    with pytest.raises(ValueError, match="outcome"):
        describe_run([_run_summary()])


def test_curation_label_export_and_order():
    # DB-free: CurationLabel is re-exported from the stdlib-only _enums module.
    assert [label.value for label in CurationLabel] == [
        "accept",
        "mua",
        "noise",
        "artifact",
        "reject",
    ]
    # lowercase, as documented (CurationLabel.mua, not CurationLabel.MUA)
    assert CurationLabel.mua.value == "mua"
