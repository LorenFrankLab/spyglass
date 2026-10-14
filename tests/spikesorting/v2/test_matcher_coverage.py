"""Exact original-sample coverage without a database, traces, or full clocks."""

import sys
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self, times, fs=10):
        self.times, self.fs = np.asarray(times), fs

    def get_num_segments(self):
        return 1

    def get_num_samples(self, segment_index=0):
        return len(self.times)

    def get_sampling_frequency(self):
        return self.fs

    def sample_index_to_time(self, frames, segment_index=0):
        return self.times[frames]

    def get_time_info(self, segment_index=0):
        return {"time_vector": self.times}

    def get_times(self, **kwargs):
        raise AssertionError("Full timestamp vectors must not be materialized.")

    def get_traces(self, **kwargs):
        raise AssertionError("Signal traces must not be read for coverage.")


def test_coverage_maps_jittered_disjoint_selected_samples_exactly():
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock(np.arange(50) / 10 + np.sin(np.arange(50)) * 0.003)
    selected = Clock(raw.times[np.r_[2:8, 19:34]])
    coverage = recording_source_coverage(selected, raw, source_object_id="raw")
    assert coverage["sample_ranges"] == [[2, 8], [19, 34]]
    assert coverage["time_ranges"] == [
        [raw.times[2], raw.times[7]],
        [raw.times[19], raw.times[33]],
    ]


def test_coverage_preserves_single_samples_and_contiguous_chunk_boundaries():
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock(np.arange(50) / 10)
    assert recording_source_coverage(
        Clock(raw.times[1:42]), raw, source_object_id="raw"
    )["sample_ranges"] == [[1, 42]]
    assert recording_source_coverage(
        Clock(raw.times[42:43]), raw, source_object_id="raw"
    )["sample_ranges"] == [[42, 43]]


@pytest.mark.parametrize("fs", [1, 2, 4])
def test_coverage_rejects_duplicate_clocks_independent_of_chunk_size(fs):
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock([0.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="Duplicate acquisition timestamps"):
        recording_source_coverage(
            Clock(raw.times, fs=fs), raw, source_object_id="raw"
        )


@pytest.mark.parametrize("selected", [[1.0], [0.0, 1.0], [1.0, 2.0]])
def test_coverage_rejects_ambiguous_raw_range_boundaries(selected):
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock([0.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="Duplicate acquisition timestamps"):
        recording_source_coverage(Clock(selected), raw, source_object_id="raw")


def test_coverage_checks_ambiguity_at_each_disjoint_range_boundary():
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="Duplicate acquisition timestamps"):
        recording_source_coverage(
            Clock([0.0, 1.0, 3.0, 5.0]), raw, source_object_id="raw"
        )


@pytest.mark.parametrize("fs", [1, 2, 4])
def test_coverage_allows_unique_epochs_with_duplicates_elsewhere(fs):
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    assert recording_source_coverage(
        Clock([0.0, 2.0, 3.0, 5.0], fs=fs), raw, source_object_id="raw"
    )["sample_ranges"] == [[0, 1], [3, 5], [7, 8]]


def test_coverage_rejects_changed_source_clock_instead_of_rounding():
    from spyglass.spikesorting.v2._matching.coverage import (
        recording_source_coverage,
    )

    raw = Clock(np.arange(20) / 10)
    with pytest.raises(ValueError, match="no longer map exactly"):
        recording_source_coverage(
            Clock(raw.times[2:8] + 1e-8), raw, source_object_id="raw"
        )


def test_missing_plan_cache_does_not_rebuild_or_open_nwb(monkeypatch, tmp_path):
    from spyglass.spikesorting.v2._matching.coverage import open_base_recording
    from spyglass.spikesorting.v2._matching import coverage

    monkeypatch.setattr(
        coverage, "stored_analysis_path", lambda name: str(tmp_path / name)
    )

    class Table:
        def __and__(self, key):
            return self

        def fetch1(self, field=None):
            return (
                "day.nwb"
                if field
                else {
                    "content_hash": "a" * 64,
                    "analysis_file_name": "missing.nwb",
                    "electrical_series_path": "acquisition/traces",
                }
            )

        def __call__(self):
            raise AssertionError("Plan must not instantiate a healing table.")

    common = ModuleType("spyglass.common.common_nwbfile")

    class Analysis:
        @staticmethod
        def get_abs_path(name, from_schema=False):
            assert (
                from_schema
            ), "Missing artifact must not reach checksum/file fallback."
            return str(tmp_path / name)

    common.AnalysisNwbfile = Analysis
    tables = ModuleType("spyglass.spikesorting.v2.recording")
    tables.Recording = tables.RecordingSelection = Table()
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, tables.__name__, tables)
    with pytest.raises(FileNotFoundError, match="Rebuild it explicitly"):
        open_base_recording(
            {
                "recording_id": "one",
                "nwb_file_name": "day.nwb",
                "recording_content_hash": "a" * 64,
            }
        )


@pytest.mark.parametrize("ranges", [[[0.5, 2]], [[0, 0]], [[0, 4], [3, 8]]])
def test_coverage_rejects_invalid_integer_ranges(ranges):
    from spyglass.spikesorting.v2._matching.coverage import (
        validate_source_coverage,
    )

    with pytest.raises((ValueError, TypeError)):
        validate_source_coverage(
            {
                "version": 1,
                "source_object_id": "raw",
                "sample_ranges": ranges,
                "time_ranges": [[0, 1]] * len(ranges),
            }
        )


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 2])
def test_coverage_version_requires_the_exact_integer_contract(version):
    from spyglass.spikesorting.v2._matching.coverage import (
        validate_source_coverage,
    )

    with pytest.raises(ValueError, match="coverage version"):
        validate_source_coverage(
            {
                "version": version,
                "source_object_id": "raw",
                "sample_ranges": [[0, 1]],
                "time_ranges": [[0, 0]],
            }
        )


def test_close_adapter_releases_owned_pynwb_reader():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from spyglass.spikesorting.v2._storage.spikeinterface import (
        close_nwb_recording,
    )

    reader = Mock()
    recording = SimpleNamespace(
        _nwbfile=SimpleNamespace(get_read_io=lambda: reader)
    )
    close_nwb_recording(recording)
    reader.close.assert_called_once_with()


@pytest.mark.parametrize(
    "paths, succeeds",
    [
        (["d/new_123.nwb", "d/newX123.nwb"], True),
        ([], False),
        (["d/new_123.nwb", "e/new_123.nwb"], False),
    ],
)
def test_readonly_analysis_path_has_no_directory_fallback(
    monkeypatch, tmp_path, paths, succeeds
):
    from spyglass.spikesorting.v2._matching.coverage import stored_analysis_path

    class External:
        def __and__(self, restriction):
            return self

        def fetch(self, field):
            assert field == "filepath"
            return paths

    class Analysis:
        _ext_tbl = External()
        _analysis_dir = str(tmp_path / "absent_analysis_dir")

        @staticmethod
        def get_abs_path(*args, **kwargs):
            raise AssertionError(
                "Read-only resolution must never enter fallback."
            )

    common = ModuleType("spyglass.common.common_nwbfile")
    common.AnalysisNwbfile = Analysis
    monkeypatch.setitem(sys.modules, common.__name__, common)
    if succeeds:
        assert stored_analysis_path("new_123.nwb") == str(
            tmp_path / "absent_analysis_dir/d/new_123.nwb"
        )
    else:
        with pytest.raises(FileNotFoundError, match="registered analysis path"):
            stored_analysis_path("new_123.nwb")
    assert not (tmp_path / "absent_analysis_dir").exists()
