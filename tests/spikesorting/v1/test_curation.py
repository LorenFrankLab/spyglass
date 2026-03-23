import numpy as np
import pytest
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor


def test_curation_rec(spike_v1, pop_curation):
    rec = spike_v1.CurationV1.get_recording(pop_curation)
    assert isinstance(
        rec, NwbRecordingExtractor
    ), "CurationV1.get_recording failed to return a RecordingExtractor"

    sample_freq = rec.get_sampling_frequency()
    assert np.isclose(
        29_959.3, sample_freq
    ), "CurqtionV1.get_sampling_frequency unexpected value"

    times = rec.get_times()
    assert np.isclose(
        1687474805.4, np.mean((times[0], times[-1]))
    ), "CurationV1.get_times unexpected value"


@pytest.mark.slow
def test_curation_sort(spike_v1, pop_curation):
    sort = spike_v1.CurationV1.get_sorting(pop_curation)
    sort_dict = sort.to_dict()
    assert isinstance(
        sort, BaseSorting
    ), "CurationV1.get_sorting failed to return a BaseSorting"

    expected = {
        "class": "spikeinterface.core.numpyextractors.NumpySorting",
        "module": "spikeinterface",
        "relative_paths": False,
    }
    for k in expected:
        assert (
            sort_dict[k] == expected[k]
        ), f"CurationV1.get_sorting unexpected value: {k}"


def test_curation_sort_info(spike_v1, pop_curation):
    sort_info = spike_v1.CurationV1.get_sort_group_info(pop_curation).fetch1()
    exp = {
        "bad_channel": "False",
        "curation_id": 0,
        "electrode_group_name": "0",
        "electrode_id": 0,
        "filtering": "None",
        "impedance": 0.0,
        "merges_applied": 0,
        "name": "0",
        "nwb_file_name": "minirec20230622_.nwb",
        "original_reference_electrode": 0,
        "parent_curation_id": -1,
        "probe_electrode": 0,
        "probe_id": "tetrode_12.5",
        "probe_shank": 0,
        "region_id": 1,
        "sort_group_id": 0,
        "subregion_name": None,
        "subsubregion_name": None,
        "x": 0.0,
        "x_warped": 0.0,
        "y": 0.0,
        "y_warped": 0.0,
        "z": 0.0,
        "z_warped": 0.0,
    }

    for k in exp:
        assert (
            sort_info[k] == exp[k]
        ), f"CurationV1.get_sort_group_info unexpected value: {k}"


def test_curation_sort_metric(spike_v1, pop_curation, pop_curation_metric):
    sort_metric = spike_v1.CurationV1.get_sort_group_info(
        pop_curation_metric
    ).fetch1()
    expected = {
        "bad_channel": "False",
        "contacts": "",
        "description": "after metric curation",
        "electrode_group_name": "0",
        "electrode_id": 0,
        "filtering": "None",
        "impedance": 0.0,
        "merges_applied": 0,
        "name": "0",
        "nwb_file_name": "minirec20230622_.nwb",
        "original_reference_electrode": 0,
        "parent_curation_id": 0,
        "probe_electrode": 0,
        "probe_id": "tetrode_12.5",
        "probe_shank": 0,
        "region_id": 1,
        "sort_group_id": 0,
        "subregion_name": None,
        "subsubregion_name": None,
        "x": 0.0,
        "x_warped": 0.0,
        "y": 0.0,
        "y_warped": 0.0,
        "z": 0.0,
        "z_warped": 0.0,
    }
    for k in expected:
        assert (
            sort_metric[k] == expected[k]
        ), f"CurationV1.get_sort_group_info unexpected value: {k}"


# ============================================================================
# No-Spikes Case Tests (Issue #1532)
# ============================================================================


@pytest.fixture
def empty_units_nwb(tmp_path):
    """Create NWB file with empty units table for testing no-spikes case."""
    from datetime import datetime
    from uuid import uuid4

    import pynwb

    nwb_path = tmp_path / "empty_units.nwb"

    nwbfile = pynwb.NWBFile(
        session_description="Test session with no spikes",
        identifier=str(uuid4()),
        session_start_time=datetime.now(),
    )
    # Create empty units table (no spike_times column)
    nwbfile.units = pynwb.misc.Units(
        name="units", description="Empty units table."
    )

    with pynwb.NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    return nwb_path


@pytest.fixture
def curation_mocks(tmp_path):
    """Fixture providing mocked dependencies for _write_sorting_to_nwb_with_curation."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    class CurationMocks:
        def __init__(self):
            self.tmp_path = tmp_path
            self.patches = []
            self.write_nwbf = None

        def setup(self, units_df=None):
            """Setup mocks with given units DataFrame."""
            if units_df is None:
                units_df = pd.DataFrame()  # Empty DataFrame, no spike_times

            mock_nwbf = MagicMock()
            mock_nwbf.units.to_dataframe.return_value = units_df

            class _WriteNWBMock(MagicMock):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._units = None

                @property
                def units(self):
                    return self._units

                @units.setter
                def units(self, value):
                    self._units = value

            self.write_nwbf = _WriteNWBMock()
            mock_io_read = MagicMock()
            mock_io_read.read.return_value = mock_nwbf
            mock_io_read.__enter__ = MagicMock(return_value=mock_io_read)
            mock_io_read.__exit__ = MagicMock(return_value=False)

            mock_io_write = MagicMock()
            mock_io_write.read.return_value = self.write_nwbf
            mock_io_write.__enter__ = MagicMock(return_value=mock_io_write)
            mock_io_write.__exit__ = MagicMock(return_value=False)

            def mock_nwbhdf5io(*args, **kwargs):
                if kwargs.get("mode") == "r" or (args and "r" in str(args)):
                    return mock_io_read
                return mock_io_write

            self.patches = [
                patch(
                    "spyglass.spikesorting.v1.curation.SpikeSortingSelection",
                    self._mock_table("test.nwb"),
                ),
                patch(
                    "spyglass.spikesorting.v1.curation.SpikeSorting",
                    self._mock_table("test_analysis.nwb"),
                ),
                patch(
                    "spyglass.spikesorting.v1.curation.AnalysisNwbfile",
                    self._mock_analysis_nwb(),
                ),
                patch(
                    "spyglass.spikesorting.v1.curation.pynwb.NWBHDF5IO",
                    side_effect=mock_nwbhdf5io,
                ),
            ]
            return self

        def _mock_table(self, return_value):
            mock = MagicMock()
            mock_instance = MagicMock()
            mock_instance.fetch1.return_value = return_value
            mock.__and__.return_value = mock_instance
            return mock

        def _mock_analysis_nwb(self):
            mock = MagicMock()
            mock_instance = MagicMock()
            mock_instance.create.return_value = "new_analysis.nwb"
            mock.return_value = mock_instance
            mock.get_abs_path.return_value = str(self.tmp_path / "test.nwb")
            return mock

        def __enter__(self):
            for p in self.patches:
                p.__enter__()
            return self

        def __exit__(self, *args):
            for p in reversed(self.patches):
                p.__exit__(*args)

    return CurationMocks()


def test_write_sorting_no_spikes(curation_mocks):
    """Test _write_sorting_to_nwb_with_curation handles missing spike_times."""
    import pynwb

    from spyglass.spikesorting.v1.curation import (
        _write_sorting_to_nwb_with_curation,
    )

    with curation_mocks.setup():
        result = _write_sorting_to_nwb_with_curation(
            sorting_id="test_sorting_id",
            labels=None,
            merge_groups=[["unit1", "unit2"]],
            metrics=None,
            apply_merge=True,  # Also tests apply_merge guard
        )

        assert result is not None
        assert len(result) == 2
        assert isinstance(curation_mocks.write_nwbf._units, pynwb.misc.Units)


def test_empty_units_nwb_readable(empty_units_nwb):
    """Test that NWB file with empty units table is readable."""
    import pynwb

    with pynwb.NWBHDF5IO(str(empty_units_nwb), "r") as io:
        nwbf = io.read()
        assert nwbf.units is not None
        units_df = nwbf.units.to_dataframe()
        assert len(units_df) == 0
