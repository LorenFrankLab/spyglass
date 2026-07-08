"""Validation slice for chronic-EEG ingestion (``common_eeg.ImportedEEG``).

Exercises the end-to-end path against a synthetic NWB mirroring the
Gonzalez-Sulser chronic-EEG structure in ``DANDI:001888``: a probe-less
multi-channel ``ElectricalSeries`` in ``acquisition`` whose channels reuse the
standard ``common_ephys`` electrode ingestion, referenced (not re-filtered) by
``ImportedEEG``.
"""

from pathlib import Path

import pytest

from tests.common import _eeg_fixture as fx


class _WarnRecorder:
    """Stand-in ``logger`` that records ``warning`` messages, no-ops the rest."""

    def __init__(self):
        self.messages = []

    def warning(self, msg, *args, **kwargs):
        self.messages.append(str(msg))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def insert_eeg(raw_dir, common, data_import):
    """Factory: build + write + ingest an EEG NWB, with auto-cleanup.

    Returns ``insert(filename, **kwargs) -> key`` where ``key`` restricts to the
    ingested copy's ``nwb_file_name``.
    """
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    inserted = []

    def _insert(filename, raise_err=True, **builder_kwargs):
        key = {"nwb_file_name": get_nwb_copy_filename(filename)}
        inserted.append(key)
        fx.write(Path(raw_dir) / filename, **builder_kwargs)
        data_import.insert_sessions(filename, raise_err=raise_err)
        return key

    yield _insert

    for key in inserted:
        (common.Nwbfile & key).delete(safemode=False)


def test_electrodes_reuse_existing_ingestion(insert_eeg, common):
    """The probe-less EEG/EMG channels ingest through the standard common_ephys
    Electrode/ElectrodeGroup tables (no bespoke EEG electrode schema)."""
    key = insert_eeg("mock_eeg_reuse.nwb")

    groups = (common.ElectrodeGroup & key).fetch("electrode_group_name")
    assert set(groups) == {"EEGArray", "EMGArray"}
    assert len(common.Electrode & key) == fx.N_CHANNELS


def test_imported_eeg_ingests_acquisition_series(insert_eeg, common):
    """ImportedEEG references the raw acquisition ElectricalSeries by object_id,
    keeping the trace in the NWB file (no filtering)."""
    key = insert_eeg("mock_eeg_series.nwb")
    common.ImportedEEG().populate(key)

    rows = (common.ImportedEEG & key).fetch(as_dict=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == fx.EEG_SERIES_NAME
    assert row["num_samples"] == fx.N_TIME
    assert row["unit"] == "volts"
    assert row["eeg_sampling_rate"] == pytest.approx(fx.EEG_RATE)


def test_imported_eeg_electrode_part_maps_channels(insert_eeg, common):
    """Each series column maps, via the .Electrode part, to the common Electrode
    row it records -- carrying the EEG/EMG group split without duplication."""
    key = insert_eeg("mock_eeg_part.nwb")
    common.ImportedEEG().populate(key)

    part = common.ImportedEEG.Electrode & key
    assert len(part) == fx.N_CHANNELS
    # Region indices are the 0-based column order.
    assert set(part.fetch("region_index")) == set(range(fx.N_CHANNELS))
    # The part rows join back to real Electrode rows across both groups.
    joined = part * common.Electrode
    assert set(joined.fetch("electrode_group_name")) == {"EEGArray", "EMGArray"}


def test_imported_eeg_no_filtering_step(insert_eeg, common):
    """ImportedEEG stores a reference only -- it must not create any AnalysisNwbfile
    (i.e. no re-filtered derivative like the LFP pipeline)."""
    key = insert_eeg("mock_eeg_nofilter.nwb")
    n_analysis_before = len(common.AnalysisNwbfile & key)
    common.ImportedEEG().populate(key)
    assert len(common.AnalysisNwbfile & key) == n_analysis_before


def test_populate_all_common_ingests_eeg(insert_eeg, common):
    """insert_sessions -> populate_all_common auto-ingests ImportedEEG (no manual
    populate needed), proving the table is wired into the standard pipeline."""
    key = insert_eeg("mock_eeg_pac.nwb")
    assert len(common.ImportedEEG & key) == 1
    assert len(common.ImportedEEG.Electrode & key) == fx.N_CHANNELS


def test_imported_eeg_noop_on_non_eeg_file(mini_insert, common, mini_copy_name):
    """A standard ephys file (only a Raw-named acquisition series) yields no
    ImportedEEG rows -- the selection heuristic excludes the wideband series, so
    wiring ImportedEEG into populate_all_common is a safe no-op there."""
    key = {"nwb_file_name": mini_copy_name}
    common.ImportedEEG().populate(key)
    assert len(common.ImportedEEG & key) == 0


def test_imported_eeg_timestamps_series(insert_eeg, common):
    """An explicit-``timestamps`` series (no ``rate``) ingests without crashing;
    the rate is estimated and valid_times spans the first/last timestamp."""
    key = insert_eeg("mock_eeg_ts.nwb", builder=fx.build_eeg_timestamps)
    common.ImportedEEG().populate(key)

    row = (common.ImportedEEG & key).fetch1()
    assert row["num_samples"] == fx.N_TIME
    # rate is estimated from the timestamps (estimate_sampling_rate rounds), so
    # only assert it lands near the nominal rate, not exactly.
    assert row["eeg_sampling_rate"] == pytest.approx(fx.EEG_RATE, rel=1e-2)

    valid_times = (
        common.IntervalList
        & key
        & {"interval_list_name": row["interval_list_name"]}
    ).fetch1("valid_times")
    # Gapless -> one interval spanning [first, last] (get_valid_intervals pads
    # the edges by a fraction of a sample, so assert with an absolute tolerance).
    assert len(valid_times) == 1
    assert valid_times[0][0] == pytest.approx(0.0, abs=1e-3)
    assert valid_times[0][-1] == pytest.approx(
        (fx.N_TIME - 1) / fx.EEG_RATE, abs=1e-3
    )


def test_noncontiguous_ids_map_correctly(insert_eeg, common):
    """With non-consecutive electrode ids and a permuted/subset region, each
    region_index maps to the correct electrode -- the position->id translation
    (not a coincidental identity) is what recovers the mapping."""
    key = insert_eeg(
        "mock_eeg_noncontig.nwb", builder=fx.build_eeg_noncontiguous
    )
    common.ImportedEEG().populate(key)

    part = (common.ImportedEEG.Electrode & key).fetch(
        "region_index", "electrode_group_name", "electrode_id", as_dict=True
    )
    got = {
        r["region_index"]: (r["electrode_group_name"], r["electrode_id"])
        for r in part
    }
    # region [2, 0, 4] over ids [10..14]: pos2->id12 (EEG), pos0->id10 (EEG),
    # pos4->id14 (EMG).
    assert got == {
        0: ("EEGArray", 12),
        1: ("EEGArray", 10),
        2: ("EMGArray", 14),
    }


def test_region_column_mismatch_warns(insert_eeg, common, monkeypatch):
    """A region whose length disagrees with the trace's column count is warned
    about (mis-mapping risk), not silently ingested."""
    rec = _WarnRecorder()
    monkeypatch.setattr(common.common_eeg, "logger", rec)
    insert_eeg("mock_eeg_mismatch.nwb", builder=fx.build_eeg_col_mismatch)
    assert any(
        "region" in m and "column" in m for m in rec.messages
    ), rec.messages


def test_excludes_non_eeg_group_series(insert_eeg, common):
    """An acquisition ElectricalSeries whose electrodes are in a non-EEG group
    (analog/aux) is NOT ingested -- the group-name gate keeps ImportedEEG from
    claiming a stray acquisition series when wired into populate_all_common."""
    key = insert_eeg("mock_eeg_analog.nwb", builder=fx.build_non_eeg_series)
    assert len(common.ImportedEEG & key) == 0


def test_gappy_timestamps_split_valid_times(insert_eeg, common):
    """A dropped-packet gap in a timestamps series splits valid_times into two
    intervals (the gap is excluded, not marked as valid recording time)."""
    key = insert_eeg("mock_eeg_gappy.nwb", builder=fx.build_eeg_gappy)
    common.ImportedEEG().populate(key)

    row = (common.ImportedEEG & key).fetch1()
    valid_times = (
        common.IntervalList
        & key
        & {"interval_list_name": row["interval_list_name"]}
    ).fetch1("valid_times")
    assert len(valid_times) == 2


def test_nwb_object_round_trip(insert_eeg, common):
    """nwb_object() re-fetches the referenced ElectricalSeries by object_id --
    the core "trace stays in the NWB file, this row indexes it" contract."""
    import pynwb

    key = insert_eeg("mock_eeg_roundtrip.nwb")
    common.ImportedEEG().populate(key)

    row_key = (common.ImportedEEG & key).fetch1("KEY")
    series = common.ImportedEEG().nwb_object(row_key)
    assert isinstance(series, pynwb.ecephys.ElectricalSeries)
    assert series.object_id == row_key["eeg_object_id"]
    assert series.data.shape[0] == fx.N_TIME
