"""DB-free round-trip tests for analyzer-curation NWB (de)serialization.

Builds a minimal NWB file on a temp path (no DataJoint, no AnalysisNwbfile
registration) and exercises ``_metric_curation_nwb`` write/read for the
normal, no-label, and zero-unit cases -- including the #1625 omit-empty-label
rule and the NaN->None read-path coercion (#1556).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pynwb import NWBHDF5IO, NWBFile

from spyglass.spikesorting.v2._metric_curation_nwb import (
    read_merge_suggestions,
    read_proposed_labels,
    read_quality_metrics,
    write_analyzer_curation_tables,
)


@pytest.fixture
def empty_nwb_path(tmp_path):
    """An on-disk NWB file the writer can append scratch tables to."""
    path = str(tmp_path / "analyzer_curation_test.nwb")
    nwbfile = NWBFile("desc", "id", datetime.now(timezone.utc))
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    return path


def _write(path, metrics_df, merge_groups, labels_by_unit, unit_ids):
    return write_analyzer_curation_tables(
        path,
        metrics_df=metrics_df,
        merge_groups=merge_groups,
        labels_by_unit=labels_by_unit,
        unit_ids=unit_ids,
    )


def test_round_trip_metrics_merges_labels(empty_nwb_path):
    """Metrics (NaN->None), merge groups, and labels all round-trip."""
    metrics = pd.DataFrame(
        {"snr": [2.5, np.nan], "isi_violation": [0.0, 0.3]},
        index=pd.Index([1, 2], name="unit_id"),
    )
    qm_oid, ms_oid, pl_oid = _write(
        empty_nwb_path,
        metrics,
        merge_groups=[[1, 2]],
        labels_by_unit={2: ["noise", "reject"]},
        unit_ids=[1, 2],
    )

    qm = read_quality_metrics(empty_nwb_path, qm_oid)
    assert qm.loc[1, "snr"] == 2.5
    # Non-finite metric surfaces as None on the read path (on-disk is NaN).
    assert qm.loc[2, "snr"] is None
    assert qm.loc[2, "isi_violation"] == 0.3

    assert read_merge_suggestions(empty_nwb_path, ms_oid) == [[1, 2]]
    assert read_proposed_labels(empty_nwb_path, pl_oid) == {
        2: ["noise", "reject"]
    }


def test_no_labels_omits_column_and_reads_empty(empty_nwb_path):
    """A sort with units but no labels saves cleanly; get_labels is empty.

    The ragged ``curation_label`` column is omitted (not an all-empty
    list-of-lists), so no hdmf dtype-inference crash (#1625).
    """
    metrics = pd.DataFrame(
        {"snr": [5.0, 6.0]}, index=pd.Index([1, 2], name="unit_id")
    )
    qm_oid, ms_oid, pl_oid = _write(
        empty_nwb_path,
        metrics,
        merge_groups=[],
        labels_by_unit={},
        unit_ids=[1, 2],
    )
    assert read_proposed_labels(empty_nwb_path, pl_oid) == {}
    assert read_merge_suggestions(empty_nwb_path, ms_oid) == []
    assert list(read_quality_metrics(empty_nwb_path, qm_oid).index) == [1, 2]


def test_zero_unit_writes_empty_tables(empty_nwb_path):
    """A zero-unit curation writes empty tables and reads back empty."""
    qm_oid, ms_oid, pl_oid = _write(
        empty_nwb_path,
        pd.DataFrame(),
        merge_groups=[],
        labels_by_unit={},
        unit_ids=[],
    )
    assert read_quality_metrics(empty_nwb_path, qm_oid).empty
    assert read_merge_suggestions(empty_nwb_path, ms_oid) == []
    assert read_proposed_labels(empty_nwb_path, pl_oid) == {}


def test_round_trip_preserves_template_columns(empty_nwb_path):
    """Surfaced waveform-shape columns flow through the column-generic writer.

    The write -> ``read_quality_metrics`` path (the same one ``get_metrics``
    reads through) is column-generic, so a frame carrying template-shape
    columns alongside the quality metrics round-trips with its values intact --
    the metric table surfaces them for downstream cell typing.
    """
    metrics = pd.DataFrame(
        {
            "firing_rate": [3.0, 12.0],
            "trough_half_width": [0.00040, 0.00025],
            "peak_to_trough_duration": [0.00055, 0.00042],
        },
        index=pd.Index([1, 2], name="unit_id"),
    )
    qm_oid, _, _ = _write(
        empty_nwb_path,
        metrics,
        merge_groups=[],
        labels_by_unit={},
        unit_ids=[1, 2],
    )
    qm = read_quality_metrics(empty_nwb_path, qm_oid)
    assert {
        "firing_rate",
        "trough_half_width",
        "peak_to_trough_duration",
    } <= set(qm.columns)
    assert qm.loc[2, "trough_half_width"] == pytest.approx(0.00025)
    assert qm.loc[1, "peak_to_trough_duration"] == pytest.approx(0.00055)


def test_build_table_guards_nonscalar_column():
    """A non-scalar / non-numeric metric cell is coerced to NaN, not crashed.

    Every validated single-channel template column is scalar, so this is
    belt-and-suspenders against a future SI column whose cell is an array or a
    string: the wide one-float-per-column write must NaN it rather than raise.
    """
    from spyglass.spikesorting.v2._metric_curation_nwb import (
        build_quality_metrics_table,
    )

    metrics = pd.DataFrame(
        {
            "snr": [4.0, 5.0],
            "weird": [np.array([1.0, 2.0]), "not_a_number"],
        },
        index=pd.Index([1, 2], name="unit_id"),
    )
    table = build_quality_metrics_table(metrics)
    frame = table.to_dataframe().set_index("unit_id")
    # The scalar column is preserved; the non-scalar / non-numeric cells NaN.
    assert frame.loc[1, "snr"] == 4.0
    assert np.isnan(frame.loc[1, "weird"])
    assert np.isnan(frame.loc[2, "weird"])
