"""``UnitWaveformFeatures.fetch_data`` on feature sets with no units."""

import pandas as pd
import pytest


class _NoUnitFeatures:
    """Stand-in relation whose NWB feature tables hold no units."""

    def __init__(self, nwb_files):
        self.nwb_files = nwb_files

    def fetch_nwb(self):
        return self.nwb_files

    def close_nwb(self):
        """No-op: the payloads are in memory."""


@pytest.mark.parametrize(
    "nwb_files",
    [[], [{"object_id": pd.DataFrame()}]],
    ids=["no_rows", "empty_units_table"],
)
def test_fetch_data_without_units_unpacks(nwb_files):
    """No units still unpack into empty spike times and features."""
    from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures

    relation = _NoUnitFeatures(nwb_files)
    relation._convert_data = UnitWaveformFeatures._convert_data

    spike_times, features = UnitWaveformFeatures.fetch_data(relation)

    assert len(spike_times) == 0
    assert len(features) == 0
