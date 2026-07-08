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
    assert row["num_samples"] == 500
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
